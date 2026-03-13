from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .paths import (
    SERVICE_NAME,
    linux_systemd_user_dir,
    macos_launch_agents_dir,
    system_name,
    windows_startup_dir,
)


def service_artifact_path() -> Path:
    system = system_name()
    if system == "linux":
        return linux_systemd_user_dir() / f"{SERVICE_NAME}.service"
    if system == "darwin":
        return macos_launch_agents_dir() / f"com.github.yvzhou1111.{SERVICE_NAME}.plist"
    return windows_startup_dir() / "ClaudeOpenAIBridge.cmd"


def install_service() -> Path:
    system = system_name()
    path = service_artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if system == "linux":
        path.write_text(
            "\n".join(
                [
                    "[Unit]",
                    "Description=Claude OpenAI Bridge",
                    "After=network.target",
                    "",
                    "[Service]",
                    "Type=simple",
                    f"ExecStart={sys.executable} -m claude_openai_bridge.cli run",
                    "Restart=always",
                    "RestartSec=2",
                    "",
                    "[Install]",
                    "WantedBy=default.target",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        subprocess.run(["systemctl", "--user", "enable", "--now", f"{SERVICE_NAME}.service"], check=False)
        return path

    if system == "darwin":
        path.write_text(
            "\n".join(
                [
                    '<?xml version="1.0" encoding="UTF-8"?>',
                    '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">',
                    "<plist version=\"1.0\">",
                    "<dict>",
                    "  <key>Label</key>",
                    f"  <string>com.github.yvzhou1111.{SERVICE_NAME}</string>",
                    "  <key>ProgramArguments</key>",
                    "  <array>",
                    f"    <string>{sys.executable}</string>",
                    "    <string>-m</string>",
                    "    <string>claude_openai_bridge.cli</string>",
                    "    <string>run</string>",
                    "  </array>",
                    "  <key>RunAtLoad</key>",
                    "  <true/>",
                    "  <key>KeepAlive</key>",
                    "  <true/>",
                    "</dict>",
                    "</plist>",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        subprocess.run(["launchctl", "unload", str(path)], check=False)
        subprocess.run(["launchctl", "load", str(path)], check=False)
        return path

    pythonw = Path(sys.executable)
    pythonw_candidate = pythonw.with_name("pythonw.exe")
    if pythonw_candidate.exists():
        pythonw = pythonw_candidate
    path.write_text(
        "\n".join(
            [
                "@echo off",
                "setlocal",
                f"start \"ClaudeOpenAIBridge\" /min \"{pythonw}\" -m claude_openai_bridge.cli run",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def uninstall_service() -> Path:
    system = system_name()
    path = service_artifact_path()
    if system == "linux":
        subprocess.run(["systemctl", "--user", "disable", "--now", f"{SERVICE_NAME}.service"], check=False)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    elif system == "darwin":
        subprocess.run(["launchctl", "unload", str(path)], check=False)
    if path.exists():
        path.unlink()
    return path


def restart_installed_service() -> None:
    system = system_name()
    path = service_artifact_path()
    if not path.exists():
        return
    if system == "linux":
        subprocess.run(["systemctl", "--user", "restart", f"{SERVICE_NAME}.service"], check=False)
    elif system == "darwin":
        subprocess.run(["launchctl", "unload", str(path)], check=False)
        subprocess.run(["launchctl", "load", str(path)], check=False)


def service_status() -> dict:
    system = system_name()
    path = service_artifact_path()
    result = {"platform": system, "installed": path.exists(), "path": str(path)}
    if system == "linux" and path.exists():
        proc = subprocess.run(
            ["systemctl", "--user", "status", f"{SERVICE_NAME}.service", "--no-pager"],
            capture_output=True,
            text=True,
        )
        result["detail"] = proc.stdout or proc.stderr
    elif system == "darwin" and path.exists():
        proc = subprocess.run(["launchctl", "list", f"com.github.yvzhou1111.{SERVICE_NAME}"], capture_output=True, text=True)
        result["detail"] = proc.stdout or proc.stderr
    else:
        result["detail"] = "Startup item written" if path.exists() else "Not installed"
    return result
