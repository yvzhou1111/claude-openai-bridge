from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .config import BridgeConfig


MODEL_ALIASES = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-3-5-haiku-latest",
]
CONFIG: BridgeConfig | None = None


def json_compact(value) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def get_config() -> BridgeConfig:
    if CONFIG is None:
        raise RuntimeError("Bridge config not loaded")
    return CONFIG


def debug_log(*parts) -> None:
    if get_config().debug_log:
        print("[claude-openai-bridge]", *parts, flush=True)


def require_local_auth(handler: "BridgeHandler") -> bool:
    incoming = handler.headers.get("x-api-key") or handler.headers.get("authorization", "")
    if incoming.startswith("Bearer "):
        incoming = incoming[7:]
    if get_config().local_auth_token and incoming != get_config().local_auth_token:
        handler.send_json(
            401,
            {
                "type": "error",
                "error": {"type": "authentication_error", "message": "Invalid local bridge token"},
            },
        )
        return False
    return True


def extract_system_text(system_value):
    if isinstance(system_value, str):
        return system_value
    if isinstance(system_value, list):
        parts = []
        for block in system_value:
            if isinstance(block, str):
                parts.append(block)
            elif block.get("type") in {"text", "thinking"}:
                parts.append(block.get("text", ""))
        return "\n".join(part for part in parts if part)
    return ""


def request_contains_images(request_json: dict) -> bool:
    for message in request_json.get("messages", []):
        content = message.get("content", [])
        if isinstance(content, str):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                return True
    return False


def image_block_to_responses(block: dict) -> dict:
    source = block.get("source") or {}
    if source.get("type") != "base64":
        raise ValueError(f"Unsupported image source type: {source.get('type')}")
    media_type = source.get("media_type")
    data = source.get("data")
    if not media_type or not data:
        raise ValueError("Image blocks require base64 media_type and data")
    return {"type": "input_image", "image_url": {"url": f"data:{media_type};base64,{data}"}}


def image_block_to_chat(block: dict) -> dict:
    source = block.get("source") or {}
    if source.get("type") != "base64":
        raise ValueError(f"Unsupported image source type: {source.get('type')}")
    media_type = source.get("media_type")
    data = source.get("data")
    if not media_type or not data:
        raise ValueError("Image blocks require base64 media_type and data")
    return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}


def normalize_text_block(role: str, text: str) -> dict:
    return {"type": "output_text" if role == "assistant" else "input_text", "text": text}


def stringify_tool_result_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif block.get("type") in {"text", "thinking"}:
                parts.append(block.get("text", ""))
            else:
                parts.append(json_compact(block))
        return "\n".join(part for part in parts if part)
    return json_compact(content)


def anthropic_message_to_responses_items(message: dict) -> list[dict]:
    role = message.get("role", "user")
    content = message.get("content", [])
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]

    items = []
    inline_blocks = []
    for block in content:
        block_type = block.get("type", "text")
        if block_type in {"text", "thinking"}:
            inline_blocks.append(normalize_text_block(role, block.get("text", "")))
            continue
        if role == "user" and block_type == "image":
            inline_blocks.append(image_block_to_responses(block))
            continue
        if inline_blocks:
            items.append({"role": role, "content": inline_blocks})
            inline_blocks = []
        if block_type == "tool_use":
            items.append(
                {
                    "type": "function_call",
                    "call_id": block.get("id"),
                    "name": block.get("name"),
                    "arguments": json_compact(block.get("input", {})),
                }
            )
            continue
        if block_type == "tool_result":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": block.get("tool_use_id"),
                    "output": stringify_tool_result_content(block.get("content", "")),
                }
            )
            continue
        raise ValueError(f"Unsupported content block type: {block_type}")
    if inline_blocks:
        items.append({"role": role, "content": inline_blocks})
    return items


def anthropic_messages_to_chat_messages(request_json: dict) -> list[dict]:
    chat_messages = []
    system_text = extract_system_text(request_json.get("system"))
    if system_text:
        chat_messages.append({"role": "system", "content": system_text})

    for message in request_json.get("messages", []):
        role = message.get("role", "user")
        content = message.get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]

        user_parts = []
        assistant_text_parts = []
        assistant_tool_calls = []
        for block in content:
            block_type = block.get("type", "text")
            if block_type in {"text", "thinking"}:
                if role == "user":
                    user_parts.append({"type": "text", "text": block.get("text", "")})
                else:
                    assistant_text_parts.append(block.get("text", ""))
                continue
            if role == "user" and block_type == "image":
                user_parts.append(image_block_to_chat(block))
                continue
            if block_type == "tool_use":
                assistant_tool_calls.append(
                    {
                        "id": block.get("id"),
                        "type": "function",
                        "function": {
                            "name": block.get("name"),
                            "arguments": json_compact(block.get("input", {})),
                        },
                    }
                )
                continue
            if block_type == "tool_result":
                if user_parts:
                    chat_messages.append({"role": "user", "content": user_parts})
                    user_parts = []
                elif assistant_text_parts:
                    chat_messages.append({"role": role, "content": "\n".join(part for part in assistant_text_parts if part)})
                    assistant_text_parts = []
                chat_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id"),
                        "content": stringify_tool_result_content(block.get("content", "")),
                    }
                )
                continue
            raise ValueError(f"Unsupported content block type: {block_type}")

        if role == "assistant" and assistant_tool_calls:
            message_obj = {"role": "assistant", "tool_calls": assistant_tool_calls}
            text = "\n".join(part for part in assistant_text_parts if part)
            message_obj["content"] = text or None
            chat_messages.append(message_obj)
            continue
        if role == "user" and user_parts:
            if len(user_parts) == 1 and user_parts[0].get("type") == "text":
                chat_messages.append({"role": "user", "content": user_parts[0]["text"]})
            else:
                chat_messages.append({"role": "user", "content": user_parts})
            continue
        if assistant_text_parts:
            chat_messages.append({"role": role, "content": "\n".join(part for part in assistant_text_parts if part)})
    return chat_messages


def anthropic_tools_to_responses(tools):
    converted = []
    for tool in tools or []:
        converted.append(
            {
                "type": "function",
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
            }
        )
    return converted


def anthropic_tools_to_chat(tools):
    converted = []
    for tool in tools or []:
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
        )
    return converted


def anthropic_tool_choice_to_responses(choice):
    if not choice:
        return None
    if isinstance(choice, str):
        return choice
    mapping = {"auto": "auto", "any": "required", "none": "none"}
    if choice.get("type") in mapping:
        return mapping[choice["type"]]
    if choice.get("type") == "tool":
        return {"type": "function", "name": choice.get("name")}
    return None


def anthropic_tool_choice_to_chat(choice):
    if not choice:
        return None
    if isinstance(choice, str):
        return choice
    mapping = {"auto": "auto", "any": "required", "none": "none"}
    if choice.get("type") in mapping:
        return mapping[choice["type"]]
    if choice.get("type") == "tool":
        return {"type": "function", "function": {"name": choice.get("name")}}
    return None


def build_responses_payload(request_json: dict) -> dict:
    payload = {
        "model": get_config().upstream_model,
        "input": [],
        "store": False,
        "stream": True,
    }
    if request_json.get("max_tokens") is not None:
        payload["max_output_tokens"] = request_json["max_tokens"]
    if request_json.get("temperature") is not None:
        payload["temperature"] = request_json["temperature"]
    instructions = extract_system_text(request_json.get("system"))
    if instructions:
        payload["instructions"] = instructions
    for message in request_json.get("messages", []):
        payload["input"].extend(anthropic_message_to_responses_items(message))
    tools = anthropic_tools_to_responses(request_json.get("tools"))
    if tools:
        payload["tools"] = tools
    tool_choice = anthropic_tool_choice_to_responses(request_json.get("tool_choice"))
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return payload


def build_chat_payload(request_json: dict) -> dict:
    payload = {
        "model": get_config().upstream_model,
        "messages": anthropic_messages_to_chat_messages(request_json),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if request_json.get("max_tokens") is not None:
        payload["max_tokens"] = request_json["max_tokens"]
    if request_json.get("temperature") is not None:
        payload["temperature"] = request_json["temperature"]
    tools = anthropic_tools_to_chat(request_json.get("tools"))
    if tools:
        payload["tools"] = tools
    tool_choice = anthropic_tool_choice_to_chat(request_json.get("tool_choice"))
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return payload


def extract_usage(usage: dict | None) -> dict:
    usage = usage or {}
    return {
        "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens", 0),
    }


def anthropic_message_from_responses_response(response_obj: dict, requested_model: str) -> dict:
    content = []
    stop_reason = "end_turn"
    for item in response_obj.get("output", []):
        item_type = item.get("type")
        if item_type == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    content.append({"type": "text", "text": part.get("text", "")})
        elif item_type == "function_call":
            stop_reason = "tool_use"
            try:
                parsed_arguments = json.loads(item.get("arguments") or "{}")
            except json.JSONDecodeError:
                parsed_arguments = {"raw": item.get("arguments")}
            content.append(
                {
                    "type": "tool_use",
                    "id": item.get("call_id"),
                    "name": item.get("name"),
                    "input": parsed_arguments,
                }
            )
    return {
        "id": response_obj.get("id", "msg_bridge"),
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": extract_usage(response_obj.get("usage")),
    }


def anthropic_message_from_chat_result(chat_result: dict, requested_model: str) -> dict:
    content = []
    if chat_result["text"]:
        content.append({"type": "text", "text": chat_result["text"]})
    stop_reason = "end_turn"
    for tool_call in chat_result["tool_calls"]:
        stop_reason = "tool_use"
        try:
            parsed_arguments = json.loads("".join(tool_call["arguments_parts"]) or "{}")
        except json.JSONDecodeError:
            parsed_arguments = {"raw": "".join(tool_call["arguments_parts"])}
        content.append(
            {
                "type": "tool_use",
                "id": tool_call["id"] or f"tool_{tool_call['index']}",
                "name": tool_call["name"] or "tool",
                "input": parsed_arguments,
            }
        )
    return {
        "id": chat_result.get("id", "chat_bridge"),
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": extract_usage(chat_result.get("usage")),
    }


def iter_sse_data(stream):
    data_lines = []
    while True:
        raw_line = stream.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                yield "\n".join(data_lines)
            data_lines = []
            continue
        if line.startswith(":") or line.startswith("event:"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
            continue
        data_lines.append(line)


def read_upstream_message(upstream_response, requested_model: str, api_format: str) -> dict:
    if api_format == "responses":
        completed = None
        error_message = None
        for raw_data in iter_sse_data(upstream_response):
            if raw_data == "[DONE]":
                break
            try:
                payload = json.loads(raw_data)
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "response.completed":
                completed = payload.get("response")
                break
            if payload.get("type") == "error":
                error_message = payload.get("error", {}).get("message") or raw_data
        if completed is None:
            raise ValueError(error_message or "Upstream stream ended without response.completed")
        return anthropic_message_from_responses_response(completed, requested_model)

    text_parts = []
    tool_calls = {}
    usage = {}
    response_id = "chat_bridge"
    for raw_data in iter_sse_data(upstream_response):
        if raw_data == "[DONE]":
            break
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            continue
        response_id = payload.get("id", response_id)
        usage = payload.get("usage") or usage
        for choice in payload.get("choices", []):
            delta = choice.get("delta", {})
            if isinstance(delta.get("content"), str):
                text_parts.append(delta["content"])
            for tool_call in delta.get("tool_calls", []) or []:
                index = tool_call.get("index", 0)
                entry = tool_calls.setdefault(index, {"index": index, "id": None, "name": None, "arguments_parts": []})
                if tool_call.get("id"):
                    entry["id"] = tool_call["id"]
                function_part = tool_call.get("function") or {}
                if function_part.get("name"):
                    entry["name"] = function_part["name"]
                if function_part.get("arguments"):
                    entry["arguments_parts"].append(function_part["arguments"])
    return anthropic_message_from_chat_result(
        {"id": response_id, "text": "".join(text_parts), "tool_calls": [tool_calls[k] for k in sorted(tool_calls)], "usage": usage},
        requested_model,
    )


class BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def send_json(self, status_code: int, payload: dict) -> None:
        raw = json_compact(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
        self.wfile.flush()

    def send_sse(self, event_name: str, payload: dict) -> None:
        self.wfile.write(f"event: {event_name}\n".encode("utf-8"))
        self.wfile.write(f"data: {json_compact(payload)}\n\n".encode("utf-8"))
        self.wfile.flush()

    def emit_streaming_message(self, message: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.send_sse(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": message.get("id", "msg_bridge"),
                    "type": "message",
                    "role": "assistant",
                    "model": message.get("model"),
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )
        for index, block in enumerate(message.get("content", [])):
            if block.get("type") == "text":
                self.send_sse("content_block_start", {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}})
                self.send_sse("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": block.get("text", "")}})
                self.send_sse("content_block_stop", {"type": "content_block_stop", "index": index})
            elif block.get("type") == "tool_use":
                self.send_sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "tool_use", "id": block.get("id"), "name": block.get("name"), "input": {}},
                    },
                )
                self.send_sse("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "input_json_delta", "partial_json": json_compact(block.get("input", {}))}})
                self.send_sse("content_block_stop", {"type": "content_block_stop", "index": index})
        self.send_sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": message.get("stop_reason", "end_turn"), "stop_sequence": message.get("stop_sequence")},
                "usage": message.get("usage", {"input_tokens": 0, "output_tokens": 0}),
            },
        )
        self.send_sse("message_stop", {"type": "message_stop"})
        self.close_connection = True

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/health":
            self.send_json(
                200,
                {
                    "status": "ok",
                    "listen": f"{get_config().listen_host}:{get_config().listen_port}",
                    "upstream_base": get_config().upstream_base,
                    "upstream_model": get_config().upstream_model,
                    "upstream_api_format": get_config().upstream_api_format,
                    "upstream_image_support": get_config().upstream_image_support,
                },
            )
            return
        if path == "/v1/models":
            self.send_json(200, {"data": [{"type": "model", "id": m, "display_name": m} for m in MODEL_ALIASES]})
            return
        self.send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path == "/v1/messages/count_tokens":
            if not require_local_auth(self):
                return
            body = self.read_json_body()
            if body is None:
                return
            self.send_json(200, {"input_tokens": max(1, len(json.dumps(body, ensure_ascii=False)) // 4)})
            return
        if path != "/v1/messages":
            self.send_json(404, {"error": "Not found"})
            return
        if not require_local_auth(self):
            return
        request_json = self.read_json_body()
        if request_json is None:
            return
        if request_contains_images(request_json) and not get_config().upstream_image_support:
            self.send_json(
                400,
                {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "Current upstream configuration does not provide reliable image understanding. Switch to a verified vision-capable provider before uploading images.",
                    },
                },
            )
            return

        try:
            api_format = self.resolve_api_format(request_json)
            upstream_payload = self.build_upstream_payload(request_json, api_format)
            debug_log("payload", json_compact(upstream_payload)[:2000])
            with self.open_upstream(upstream_payload, api_format) as response:
                message = read_upstream_message(response, request_json.get("model", MODEL_ALIASES[0]), api_format)
        except HTTPError as exc:
            self.forward_http_error(exc)
            return
        except URLError as exc:
            self.send_json(502, {"type": "error", "error": {"type": "api_error", "message": f"Upstream connection failed: {exc}"}})
            return
        except Exception as exc:
            self.send_json(500, {"type": "error", "error": {"type": "api_error", "message": f"Bridge failure: {exc}"}})
            return

        if request_json.get("stream"):
            self.emit_streaming_message(message)
        else:
            self.send_json(200, message)

    def read_json_body(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "Request body must be valid JSON"}})
            return None

    def resolve_api_format(self, request_json: dict) -> str:
        configured = get_config().upstream_api_format
        if configured != "hybrid":
            return configured
        if request_contains_images(request_json):
            return "chat_completions"
        return "responses"

    def build_upstream_payload(self, request_json: dict, api_format: str) -> dict:
        if api_format == "responses":
            return build_responses_payload(request_json)
        if api_format == "chat_completions":
            return build_chat_payload(request_json)
        raise ValueError(f"Unsupported upstream api format: {api_format}")

    def open_upstream(self, payload: dict, api_format: str):
        endpoint = "/responses" if api_format == "responses" else "/chat/completions"
        request = Request(
            f"{get_config().upstream_base.rstrip('/')}{endpoint}",
            data=json_compact(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {get_config().upstream_api_key}"},
            method="POST",
        )
        return urlopen(request, timeout=get_config().upstream_timeout)

    def forward_http_error(self, exc: HTTPError) -> None:
        body = exc.read()
        try:
            parsed = json.loads(body.decode("utf-8"))
            if isinstance(parsed.get("error"), dict):
                message = parsed["error"].get("message") or parsed["error"]
            else:
                message = parsed.get("error") or parsed.get("message") or parsed
        except Exception:
            message = body.decode("utf-8", errors="replace")
        self.send_json(exc.code, {"type": "error", "error": {"type": "api_error" if exc.code >= 500 else "invalid_request_error", "message": str(message)}})


def serve(config: BridgeConfig) -> None:
    global CONFIG
    CONFIG = config
    server = ThreadingHTTPServer((config.listen_host, config.listen_port), BridgeHandler)
    print(
        f"claude-openai-bridge listening on http://{config.listen_host}:{config.listen_port} -> "
        f"{config.upstream_base} ({config.upstream_model}, {config.upstream_api_format}, image_support={config.upstream_image_support})",
        flush=True,
    )
    server.serve_forever()
