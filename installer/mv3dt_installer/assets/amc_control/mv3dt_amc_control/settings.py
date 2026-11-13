"""Service settings, read from the environment.

The systemd unit loads `installer.conf` and `secrets/tracking.env` as
environment files and adds the values Step 9 renders (`MV3DT_INSTALL_DIR`,
`AMC_PUBLIC_HOST`, `AMC_AUTOMATION_CONFIG`, `BIND`). The ticket secret, the
web app project id and the allowed origins are shared with the tracking
service (Step 8), because the web app signs both services' tickets with the
same secret.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

DEFAULT_BIND = "127.0.0.1:8091"
DEFAULT_INSTALL_DIR = "/opt/mv3dt"


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    ticket_secret: str = ""
    # The web app project whose tickets are accepted (TRACKING_PROJECT_ID in
    # tracking.env). Not installer.conf's AMC_PROJECT_ID, which is AMC's own id.
    project_id: str = ""
    allowed_origins: list[str] = field(default_factory=list)
    ticket_issuer: str | None = None
    install_dir: Path = Path(DEFAULT_INSTALL_DIR)
    automation_config: Path | None = None
    bind: str = DEFAULT_BIND
    allow_non_local_bind: bool = False

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        origins = [o.strip().rstrip("/") for o in env.get("ALLOWED_ORIGINS", "").split(",")]
        config = env.get("AMC_AUTOMATION_CONFIG", "").strip()
        return cls(
            ticket_secret=env.get("WORKSTATION_TICKET_SECRET", ""),
            project_id=env.get("TRACKING_PROJECT_ID", "").strip(),
            allowed_origins=[o for o in origins if o],
            ticket_issuer=env.get("TICKET_ISSUER", "").strip() or None,
            install_dir=Path(env.get("MV3DT_INSTALL_DIR", "").strip() or DEFAULT_INSTALL_DIR),
            automation_config=Path(config) if config else None,
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
    """True for "localhost" or a loopback IP literal; any other name is not."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
