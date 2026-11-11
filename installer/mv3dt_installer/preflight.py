"""Platform and invoking-user preflight for mv3dt-installer (doc 00 §5.1.1).

Mirrors the *behavior* of installer/bootstrap/bootstrap.sh's
`preflight_os_arch()` and `preflight_root_and_user()`, re-expressed in
Python. Nothing here sources or executes that script — this module is a
self-contained sibling of it, the same discipline logs.py, privilege.py and
report.py follow.

These two gates exist because the installer binary is downloaded and run
directly on a bare workstation: nothing upstream of it validates the
platform, and nothing else notices when there is no real invoking user to
own the secrets, the NGC config, and the AutoMagicCalib clone.

Public API:
    read_os_release(path=Path("/etc/os-release")) -> dict[str, str]
        -- parse the shell-style key=value file into a plain dict.
    require_supported_platform(*, os_release_path=..., machine=platform.machine)
        -- exit (via logs.die) unless the host is Ubuntu 24.04 on x86_64.
    require_invoking_user() -> privilege.InvokingUser
        -- resolve the invoking user, refusing to proceed as bare root.

Why /etc/os-release and not `lsb_release`:
    `lsb_release` is not installed by default on a minimal Ubuntu 24.04
    image, which is exactly why bootstrap.sh had to degrade its distro check
    to a warning ("lsb_release not found; cannot verify Ubuntu 24.04 —
    proceeding"). /etc/os-release is part of systemd and is always present,
    so the check can be a real gate here instead of an advisory one.
    Its `ID` field is lowercase ("ubuntu") where `lsb_release -is` reports
    "Ubuntu", so the distro comparison is case-insensitive.

Escape hatch (development only):
    Setting MV3DT_SKIP_PLATFORM_CHECK=1 downgrades every platform failure —
    including an unreadable /etc/os-release — from a fatal error to a
    warning, so the binary can be exercised on a developer machine that is
    not an Ubuntu 24.04 workstation. It is deliberately undocumented for
    operators: an install performed with it set is unsupported, since every
    pinned version in the step specs assumes Ubuntu 24.04 / x86_64.
"""

from __future__ import annotations

import os
import pathlib
import platform
from typing import Callable

from . import privilege
from .logs import die, log

__all__ = [
    "SUPPORTED_DISTRO",
    "SUPPORTED_RELEASE",
    "SUPPORTED_ARCH",
    "SKIP_ENV",
    "DEFAULT_OS_RELEASE_PATH",
    "read_os_release",
    "require_supported_platform",
    "require_invoking_user",
]

# Pinned platform, matching bootstrap.sh's preflight_os_arch() exactly.
SUPPORTED_DISTRO = "Ubuntu"
SUPPORTED_RELEASE = "24.04"
SUPPORTED_ARCH = "x86_64"

# Development-only escape hatch; see the module docstring.
SKIP_ENV = "MV3DT_SKIP_PLATFORM_CHECK"

DEFAULT_OS_RELEASE_PATH = pathlib.Path("/etc/os-release")

# Values accepted as "on" for SKIP_ENV. Anything else (including "0" and the
# empty string) leaves the gate fatal.
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def _skip_requested() -> bool:
    return os.environ.get(SKIP_ENV, "").strip().lower() in _TRUTHY


def _fail(message: str) -> None:
    """Die on `message`, or warn and continue when SKIP_ENV is set.

    Every platform failure funnels through here so the escape hatch has a
    single, auditable definition: with it set the operator has explicitly
    opted out of the gate, and the transcript records exactly which check
    was skipped.
    """
    if _skip_requested():
        log.warn(f"{message} ({SKIP_ENV} is set — continuing anyway.)")
        return
    die(message)


def read_os_release(
    path: pathlib.Path = DEFAULT_OS_RELEASE_PATH,
) -> dict[str, str]:
    """Parse an os-release file into a dict of its key/value pairs.

    The format is a restricted shell fragment: blank lines and `#` comments
    are ignored, every other line is `KEY=VALUE`, and VALUE may be wrapped in
    single or double quotes. Only the surrounding quotes are stripped — no
    shell expansion is performed, and the file is never sourced.

    Raises `FileNotFoundError` (or another `OSError`) when the file cannot be
    read; `require_supported_platform` turns that into a `logs.die`, so the
    installer never proceeds on a platform it cannot identify.
    """
    values: dict[str, str] = {}
    text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, raw = stripped.partition("=")
        key = key.strip()
        if not key:
            continue
        value = raw.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value

    return values


def require_supported_platform(
    *,
    os_release_path: pathlib.Path = DEFAULT_OS_RELEASE_PATH,
    machine: Callable[[], str] = platform.machine,
) -> None:
    """Exit unless the host is Ubuntu 24.04 on x86_64 (bootstrap.sh preflight_os_arch).

    `os_release_path` and `machine` are injected so tests can drive synthetic
    platforms without touching /etc or the real interpreter's uname.

    With MV3DT_SKIP_PLATFORM_CHECK set, each failure is logged as a warning
    and the function returns normally instead of exiting.
    """
    try:
        os_release = read_os_release(os_release_path)
    except OSError as exc:
        _fail(
            f"Cannot read {os_release_path} ({exc.__class__.__name__}); "
            f"unable to verify {SUPPORTED_DISTRO} {SUPPORTED_RELEASE}."
        )
        os_release = {}

    if os_release:
        distro = os_release.get("ID", "")
        release = os_release.get("VERSION_ID", "")
        # ID is lowercase in os-release ("ubuntu"); SUPPORTED_DISTRO keeps
        # bootstrap.sh's display casing, so compare case-insensitively.
        if (
            distro.lower() != SUPPORTED_DISTRO.lower()
            or release != SUPPORTED_RELEASE
        ):
            _fail(
                f"{SUPPORTED_DISTRO} {SUPPORTED_RELEASE} required "
                f"(found: {distro or 'unknown'} {release or 'unknown'})."
            )
        else:
            log.info(f"OS OK: {SUPPORTED_DISTRO} {release}")

    arch = machine()
    if arch != SUPPORTED_ARCH:
        _fail(f"{SUPPORTED_ARCH} required (found: {arch or 'unknown'}).")
    else:
        log.info(f"Architecture OK: {arch}")


def require_invoking_user() -> privilege.InvokingUser:
    """Resolve the invoking user, refusing a bare root shell (bootstrap.sh
    preflight_root_and_user).

    `privilege.resolve()` falls back to `getpass.getuser()` when `$SUDO_USER`
    is unset or "root". Under `sudo -i` / a bare root login that fallback
    silently returns "root" with /root as the home directory, and every
    artifact the installer writes on the operator's behalf — the secrets
    directory, ~/.ngc/config, the AutoMagicCalib clone — would land in root's
    home, owned by root, invisible to the person who has to use them.

    This reproduces bootstrap.sh's "SUDO_USER is unset or 'root'" gate: a
    resolved name of "root" is fatal. An unresolvable name (a `$SUDO_USER`
    that is not a valid login) is fatal too, mirroring the same function's
    `id -u "$SUDO_USER"` check.

    Root-ness itself is not checked here — that is `privilege.require_root()`,
    which callers run first.
    """
    try:
        user = privilege.resolve()
    except KeyError:
        sudo_user = os.environ.get("SUDO_USER", "")
        die(
            f"SUDO_USER={sudo_user!r} is not a valid login on this system; "
            "cannot resolve an invoking user."
        )

    if user.name == "root":
        die(
            "Could not resolve a non-root invoking user (SUDO_USER is unset "
            "or 'root'). Run the installer with 'sudo -E ./mv3dt-installer' "
            "from a real non-root login — a bare root shell would place the "
            "secrets directory, the NGC config, and the AutoMagicCalib clone "
            "in /root, owned by root."
        )

    log.info(f"Invoking user: {user.name} (home: {user.home})")
    return user
