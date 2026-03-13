from __future__ import annotations

import argparse
import base64
import io
import json
import multiprocessing
import random
import string
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageFont

from .config import BridgeConfig, load_config, save_config
from .paths import claude_settings_paths, config_file, system_name
from .proxy import serve
from .service import install_service, restart_installed_service, service_status, uninstall_service


def pretty_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def health_url(config: BridgeConfig) -> str:
    return f"http://{config.listen_host}:{config.listen_port}/health"


def messages_url(config: BridgeConfig) -> str:
    return f"http://{config.listen_host}:{config.listen_port}/v1/messages?beta=true"


def build_validation_image():
    code = "VISION-" + "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    image = Image.new("RGB", (960, 240), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((40, 90), code, fill="black", font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return code, base64.b64encode(buffer.getvalue()).decode()


def post_json(url: str, payload: dict, token: str) -> tuple[int, dict]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": token,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def wait_for_health(config: BridgeConfig, timeout: int = 10) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(health_url(config), timeout=1) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def run_temp_server(config: BridgeConfig) -> multiprocessing.Process:
    process = multiprocessing.Process(target=serve, args=(config,), daemon=True)
    process.start()
    return process


def validate_through_bridge(config: BridgeConfig, require_image: bool) -> dict:
    process = run_temp_server(config)
    try:
        if not wait_for_health(config):
            raise ValueError("temporary bridge did not become healthy")

        text_status, text_body = post_json(
            messages_url(config),
            {
                "model": "claude-opus-4-6",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Reply with exactly pong."}],
            },
            config.local_auth_token,
        )
        if text_status != 200:
            raise ValueError(f"text validation failed with status {text_status}")
        text = text_body.get("content", [{}])[0].get("text", "").strip().lower()
        if text != "pong":
            raise ValueError(f"text validation returned unexpected content: {text_body}")

        tool_status, tool_body = post_json(
            messages_url(config),
            {
                "model": "claude-opus-4-6",
                "max_tokens": 64,
                "tool_choice": {"type": "any"},
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo text",
                        "input_schema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    }
                ],
                "messages": [{"role": "user", "content": "Use the echo tool with text hello."}],
            },
            config.local_auth_token,
        )
        if tool_status != 200:
            raise ValueError(f"tool validation failed with status {tool_status}")
        if tool_body.get("stop_reason") != "tool_use":
            raise ValueError(f"tool validation did not trigger tool_use: {tool_body}")

        image_ok = False
        if require_image:
            expected_code, image_b64 = build_validation_image()
            image_status, image_body = post_json(
                messages_url(config),
                {
                    "model": "claude-opus-4-6",
                    "max_tokens": 24,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Read the text in the image and reply with exactly {expected_code}."},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": image_b64,
                                    },
                                },
                            ],
                        }
                    ],
                },
                config.local_auth_token,
            )
            if image_status == 200:
                image_text = image_body.get("content", [{}])[0].get("text", "").strip()
                image_ok = image_text == expected_code
        return {"ok": True, "image_ok": image_ok}
    finally:
        process.terminate()
        process.join(timeout=5)


def detect_api_mode(base_url: str, api_key: str, model: str, requested_mode: str) -> tuple[str, bool]:
    candidates = [requested_mode] if requested_mode != "auto" else ["hybrid", "responses", "chat_completions"]
    errors = {}
    best_text_only = None
    for candidate in candidates:
        config = BridgeConfig(
            listen_host="127.0.0.1",
            listen_port=random.randint(20000, 40000),
            local_auth_token="claude-openai-bridge-test",
            upstream_base=base_url.rstrip("/"),
            upstream_api_key=api_key,
            upstream_model=model,
            upstream_api_format=candidate,
            upstream_image_support=True,
            upstream_timeout=120,
        )
        try:
            result = validate_through_bridge(config, require_image=True)
        except Exception as exc:
            errors[candidate] = str(exc)
            continue
        if result["image_ok"]:
            return candidate, True
        best_text_only = candidate
    if best_text_only:
        return best_text_only, False
    raise ValueError("auto detect failed: " + "; ".join(f"{k} => {v}" for k, v in errors.items()))


def parse_models_payload(payload: dict) -> list[tuple[str, str]]:
    raw_items = payload.get("data") or payload.get("models") or []
    models = []
    seen = set()
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id") or item.get("slug") or item.get("name")
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        models.append((str(model_id), str(item.get("display_name") or item.get("description") or "")))
    return models


def fetch_models(base_url: str, api_key: str) -> list[tuple[str, str]]:
    request = Request(base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {api_key}"}, method="GET")
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return parse_models_payload(payload)


def recommended_model(models: list[str], current: str) -> str:
    if current in models:
        return current
    preferred = ["gpt-5.4", "gpt-5.3-codex", "gpt-5.2", "gpt-5.1", "gpt-5", "gpt-4.1", "gpt-4o"]
    for name in preferred:
        for model in models:
            if model == name or model.startswith(name + "-"):
                return model
    return models[0]


def gui_available() -> bool:
    if system_name() == "linux":
        return bool(sys.platform) and bool(Path("/usr/bin/xdg-open").exists() or Path("/usr/bin/zenity").exists() or True)
    return True


def prompt_gui(defaults: dict) -> dict:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Claude OpenAI Bridge")
    root.resizable(False, False)
    values = {
        "base_url": tk.StringVar(value=defaults["base_url"]),
        "api_key": tk.StringVar(value=defaults["api_key"]),
        "api_mode": tk.StringVar(value=defaults["api_mode"]),
    }
    result = {}
    labels = [("Base URL", "base_url"), ("API Key", "api_key"), ("API Mode", "api_mode")]
    for row, (label, key) in enumerate(labels):
        ttk.Label(root, text=label).grid(row=row, column=0, sticky="w", padx=10, pady=6)
        entry = ttk.Entry(root, textvariable=values[key], width=64, show="*" if key == "api_key" else "")
        entry.grid(row=row, column=1, padx=10, pady=6)
    def submit():
        result.update({key: var.get().strip() for key, var in values.items()})
        root.destroy()
    ttk.Button(root, text="Continue", command=submit).grid(row=3, column=0, columnspan=2, pady=10)
    root.mainloop()
    if not result:
        raise SystemExit(1)
    return result


def pick_model_gui(models: list[tuple[str, str]], default_model: str) -> str:
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("Choose Model")
    root.resizable(True, True)
    choice = tk.StringVar(value=default_model)
    ttk.Label(root, text="Pick the upstream model").pack(anchor="w", padx=10, pady=8)
    listbox = tk.Listbox(root, width=80, height=min(18, max(6, len(models))))
    for model_id, description in models:
        suffix = f"  {description}" if description else ""
        listbox.insert(tk.END, f"{model_id}{suffix}")
    selected_index = next((index for index, (model_id, _) in enumerate(models) if model_id == default_model), 0)
    listbox.selection_set(selected_index)
    listbox.activate(selected_index)
    listbox.pack(fill="both", expand=True, padx=10, pady=6)

    def submit():
        current = listbox.curselection()
        if current:
            choice.set(models[current[0]][0])
        root.destroy()

    ttk.Button(root, text="Use selected model", command=submit).pack(pady=10)
    root.mainloop()
    return choice.get()


def prompt_cli(defaults: dict) -> dict:
    def ask(label: str, default: str) -> str:
        value = input(f"{label} [{default}]: ").strip()
        return value or default

    return {
        "base_url": ask("Base URL", defaults["base_url"]),
        "api_key": ask("API Key", defaults["api_key"]),
        "api_mode": ask("API mode (auto/hybrid/responses/chat_completions)", defaults["api_mode"]),
    }


def choose_inputs(args, current: BridgeConfig) -> dict:
    defaults = {
        "base_url": current.upstream_base or "https://api.openai.com/v1",
        "api_key": current.upstream_api_key,
        "api_mode": current.upstream_api_format or "auto",
        "model": current.upstream_model or "gpt-5.4",
    }
    if args.base_url and args.api_key:
        return {
            "base_url": args.base_url,
            "api_key": args.api_key,
            "api_mode": args.api_mode or defaults["api_mode"],
            "model": args.model or "",
        }
    values = prompt_gui(defaults) if args.gui else prompt_cli(defaults)
    values["model"] = args.model or ""
    return values


def resolve_model(values: dict, current: BridgeConfig, gui: bool) -> str:
    if values.get("model"):
        return values["model"]
    try:
        models = fetch_models(values["base_url"], values["api_key"])
    except Exception:
        return current.upstream_model or "gpt-5.4"
    if not models:
        return current.upstream_model or "gpt-5.4"
    model_ids = [model_id for model_id, _ in models]
    default_model = recommended_model(model_ids, current.upstream_model or "gpt-5.4")
    if len(models) == 1:
        return model_ids[0]
    if gui:
        return pick_model_gui(models, default_model)
    if not sys.stdin.isatty():
        return default_model
    print("Available models:")
    for index, (model_id, description) in enumerate(models, start=1):
        print(f"{index}. {model_id}" + (f"  {description}" if description else ""))
    answer = input(f"Choose model [default {default_model}]: ").strip()
    if not answer:
        return default_model
    if answer.isdigit() and 1 <= int(answer) <= len(models):
        return models[int(answer) - 1][0]
    return answer


def update_claude_settings(config: BridgeConfig) -> None:
    for path in claude_settings_paths():
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {}
        data.setdefault("env", {})
        data["env"]["ANTHROPIC_AUTH_TOKEN"] = config.local_auth_token
        data["env"]["ANTHROPIC_BASE_URL"] = f"http://{config.listen_host}:{config.listen_port}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pretty_json(data) + "\n", encoding="utf-8")


def configure_command(args) -> int:
    current = load_config()
    values = choose_inputs(args, current)
    gui = args.gui
    model = resolve_model(values, current, gui)
    api_mode, image_support = detect_api_mode(values["base_url"], values["api_key"], model, values["api_mode"])
    config = BridgeConfig(
        listen_host=current.listen_host,
        listen_port=current.listen_port,
        local_auth_token=current.local_auth_token,
        upstream_base=values["base_url"].rstrip("/"),
        upstream_api_key=values["api_key"],
        upstream_model=model,
        upstream_api_format=api_mode,
        upstream_image_support=image_support,
        upstream_timeout=current.upstream_timeout,
        debug_log=current.debug_log,
    )
    save_config(config)
    update_claude_settings(config)
    restart_installed_service()
    print(pretty_json({"status": "ok", "config_path": str(config_file()), "api_mode": api_mode, "image_support": image_support, "model": model}))
    return 0


def run_command(args) -> int:
    config = load_config(Path(args.config) if args.config else None)
    if not config.upstream_base or not config.upstream_api_key:
        raise SystemExit("No upstream configured. Run `claude-openai-bridge configure` first.")
    serve(config)
    return 0


def status_command(_args) -> int:
    config = load_config()
    info = {
        "config_path": str(config_file()),
        "config": config.to_dict(),
        "service": service_status(),
    }
    try:
        with urlopen(health_url(config), timeout=1) as response:
            info["health"] = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        info["health_error"] = str(exc)
    print(pretty_json(info))
    return 0


def verify_command(_args) -> int:
    config = load_config()
    result = validate_through_bridge(config, require_image=True)
    print(pretty_json(result))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAI-compatible upstreams inside Claude Code.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the local bridge server")
    run_parser.add_argument("--config", help="Optional path to a config.json file")
    run_parser.set_defaults(func=run_command)

    configure_parser = subparsers.add_parser("configure", help="Configure the bridge and update Claude settings")
    configure_parser.add_argument("--base-url")
    configure_parser.add_argument("--api-key")
    configure_parser.add_argument("--model")
    configure_parser.add_argument("--api-mode", default="auto", choices=["auto", "hybrid", "responses", "chat_completions"])
    configure_parser.add_argument("--gui", action="store_true", help="Use a small Tkinter GUI")
    configure_parser.set_defaults(func=configure_command)

    install_parser = subparsers.add_parser("install-service", help="Install a background service / startup item")
    install_parser.set_defaults(func=lambda _args: (print(install_service()), 0)[1])

    uninstall_parser = subparsers.add_parser("uninstall-service", help="Remove the background service / startup item")
    uninstall_parser.set_defaults(func=lambda _args: (print(uninstall_service()), 0)[1])

    status_parser = subparsers.add_parser("status", help="Show config and service status")
    status_parser.set_defaults(func=status_command)

    verify_parser = subparsers.add_parser("verify", help="Verify the currently saved upstream")
    verify_parser.set_defaults(func=verify_command)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
