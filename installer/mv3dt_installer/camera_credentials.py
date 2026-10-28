"""Secure capture and storage for camera RTSP credentials."""

from __future__ import annotations

import getpass
import os
import pathlib
import tempfile
from dataclasses import dataclass
from typing import Optional, Union

from . import privilege, prompting
from .logs import die, log

__all__ = [
    "Credentials",
    "capture_credentials",
    "store_credentials",
    "load_credentials",
]

StrPath = Union[str, "os.PathLike[str]"]
_SECRET_RELATIVE_PATH = pathlib.Path("secrets") / "camera.env"


@dataclass(frozen=True)
class Credentials:
    username: str
    password: str


def _valid(value: str) -> bool:
    return bool(value) and "\n" not in value and "\r" not in value


def capture_credentials(non_interactive: bool) -> Credentials:
    """Capture the required camera username and a no-echo password."""
    if non_interactive:
        die(
            "Camera credentials are required but --non-interactive was passed "
            "without both CAM_USER and CAM_PASSWORD in the environment; re-run "
            "with both exported (sudo -E) or without --non-interactive."
        )
    reason = prompting.prompt_unavailable_reason()
    if reason is not None:
        # A secret has no safe default; fail rather than block on a prompt
        # nobody can see.
        die(
            f"Camera credentials are required but cannot be asked for "
            f"({reason}). Re-run without a pipe, or export CAM_USER and "
            "CAM_PASSWORD and run with sudo -E."
        )

    while True:
        username = input("Camera username (required): ").strip()
        if _valid(username):
            break
        log.warn(
            "Camera username cannot be blank or contain a newline; please try again."
        )

    while True:
        password = getpass.getpass("Camera password (required): ")
        if _valid(password):
            break
        log.warn(
            "Camera password cannot be blank or contain a newline; please try again."
        )

    log.info("Captured camera credentials (CAM_PASSWORD=<redacted>).")
    return Credentials(username=username, password=password)


def store_credentials(creds: Credentials, install_dir: StrPath) -> pathlib.Path:
    """Atomically write ``secrets/camera.env`` with restrictive permissions."""
    if not _valid(creds.username) or not _valid(creds.password):
        raise ValueError(
            "camera username and password are required and cannot contain newlines"
        )

    secrets_dir = pathlib.Path(install_dir) / _SECRET_RELATIVE_PATH.parent
    target = secrets_dir / _SECRET_RELATIVE_PATH.name
    user = privilege.resolve()

    secrets_dir.mkdir(parents=True, exist_ok=True)
    os.chown(secrets_dir, user.uid, user.gid)
    os.chmod(secrets_dir, 0o700)

    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(secrets_dir))
    tmp_path = pathlib.Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"CAM_USER={creds.username}\nCAM_PASSWORD={creds.password}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    os.chown(target, user.uid, user.gid)
    os.chmod(target, 0o600)
    log.info(f"Stored camera credentials at {target} (CAM_PASSWORD=<redacted>).")
    return target


def load_credentials(install_dir: StrPath) -> Optional[Credentials]:
    """Load a complete credential pair, or return ``None`` when unavailable."""
    path = pathlib.Path(install_dir) / _SECRET_RELATIVE_PATH
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    values = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value
    username = values.get("CAM_USER", "")
    password = values.get("CAM_PASSWORD", "")
    if not _valid(username) or not _valid(password):
        return None
    return Credentials(username=username, password=password)
