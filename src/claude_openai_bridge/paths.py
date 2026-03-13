import os
import platform
from pathlib import Path


APP_NAME = "ClaudeOpenAIBridge"
SERVICE_NAME = "claude-openai-bridge"


def system_name() -> str:
    return platform.system().lower()


def home_dir() -> Path:
    return Path.home()


def config_dir() -> Path:
    system = system_name()
    if system == "windows":
        return Path(os.environ.get("APPDATA", home_dir())) / APP_NAME
    if system == "darwin":
        return home_dir() / "Library" / "Application Support" / APP_NAME
    return home_dir() / ".config" / "claude-openai-bridge"


def config_file() -> Path:
    return config_dir() / "config.json"


def claude_settings_paths() -> list[Path]:
    home = home_dir()
    return [
        home / ".claude" / "settings.json",
        home / ".claude-code" / "settings.json",
    ]


def linux_systemd_user_dir() -> Path:
    return home_dir() / ".config" / "systemd" / "user"


def macos_launch_agents_dir() -> Path:
    return home_dir() / "Library" / "LaunchAgents"


def windows_startup_dir() -> Path:
    appdata = Path(os.environ.get("APPDATA", home_dir()))
    return appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
