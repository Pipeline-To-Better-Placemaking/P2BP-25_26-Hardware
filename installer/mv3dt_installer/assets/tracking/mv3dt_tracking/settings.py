"""Service settings, read from the environment.

The systemd unit loads `installer.conf` and `secrets/tracking.env` as
environment files, so every setting is an environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

DEFAULT_BIND = "127.0.0.1:8090"
DEFAULT_DATA_DIR = "/var/lib/mv3dt-tracking"
_LOOPBACK_HOSTS = {"localhost", "::1"}


def _path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    ticket_secret: str = ""
    project_id: str = ""
    allowed_origins: list[str] = field(default_factory=list)
    ticket_issuer: str | None = None
    location_id: str = ""
    data_dir: Path = Path(DEFAULT_DATA_DIR)
    calib_dir: Path | None = None
    camera_models_dir: Path | None = None
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    mqtt_topic: str = ""
    bind: str = DEFAULT_BIND
    allow_non_local_bind: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        location = env.get("LOCATION_ID", "").strip()
        topic = env.get("MQTT_TOPIC", "").strip()
        if not topic and location:
            topic = f"mv3dt/{location}/sv3d"
        origins = [o.strip().rstrip("/") for o in env.get("ALLOWED_ORIGINS", "").split(",")]
        return cls(
            ticket_secret=env.get("WORKSTATION_TICKET_SECRET", ""),
            project_id=env.get("TRACKING_PROJECT_ID", "").strip(),
            allowed_origins=[o for o in origins if o],
            ticket_issuer=env.get("TICKET_ISSUER", "").strip() or None,
            location_id=location,
            data_dir=Path(env.get("DATA_DIR", "").strip() or DEFAULT_DATA_DIR),
            calib_dir=_path(env.get("CALIB_DIR", "").strip()),
            camera_models_dir=_path(env.get("CAMERA_MODELS_DIR", "").strip()),
            mqtt_host=env.get("MQTT_HOST", "").strip() or "127.0.0.1",
            mqtt_port=int(env.get("MQTT_PORT", "").strip() or 1883),
            mqtt_topic=topic,
            bind=env.get("BIND", "").strip() or DEFAULT_BIND,
            allow_non_local_bind=_truthy(env.get("ALLOW_NON_LOCAL_BIND")),
        )

    def bind_address(self) -> tuple[str, int]:
        """Split BIND into (host, port). IPv6 hosts may be bracketed."""
        host, sep, port = self.bind.rpartition(":")
        if not sep or not host or not port.isdigit():
            raise ValueError(f"BIND must be host:port, got {self.bind!r}")
        return host.strip("[]"), int(port)


def is_loopback(host: str) -> bool:
    return host in _LOOPBACK_HOSTS or host.startswith("127.")
