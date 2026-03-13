from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path

from .paths import config_file


@dataclass
class BridgeConfig:
    listen_host: str = "127.0.0.1"
    listen_port: int = 18083
    local_auth_token: str = ""
    upstream_base: str = ""
    upstream_api_key: str = ""
    upstream_model: str = "gpt-5.4"
    upstream_api_format: str = "responses"
    upstream_image_support: bool = False
    upstream_timeout: int = 600
    debug_log: bool = False

    @classmethod
    def from_dict(cls, raw: dict) -> "BridgeConfig":
        config = cls(**raw)
        if not config.local_auth_token:
            config.local_auth_token = secrets.token_urlsafe(24)
        return config

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path | None = None) -> BridgeConfig:
    target = path or config_file()
    if not target.exists():
        return BridgeConfig(local_auth_token=secrets.token_urlsafe(24))
    return BridgeConfig.from_dict(json.loads(target.read_text(encoding="utf-8")))


def save_config(config: BridgeConfig, path: Path | None = None) -> Path:
    target = path or config_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target
