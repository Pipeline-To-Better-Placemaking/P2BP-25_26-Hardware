"""Web-app connection contract: credential capture, storage, endpoint
normalization, and redaction helpers for mv3dt-installer (doc 00 §14).

Mirrors the shape of §10's NGC key handling (same storage discipline, same
secrecy rules, different consumer). The desktop talks to a cloud web app over
HTTP with a bearer API key; this module owns the credential + endpoint
contract only -- the transport and routes belong to STEP-7, and the separate
MQTT control plane belongs to STEP-6. This module is a self-contained sibling
of laptop/ and of the other framework modules (doc 00 §3.1) -- it does not
import from steps/, app.py, config.py, ngc.py, or reboot.py.

Public API:
    Credentials              -- dataclass(api_key: str | None, endpoint: str | None).
    normalize_endpoint(raw)  -- §14.2 normalization; raises ValueError on an
        empty result.
    join(endpoint, path)     -- §14.2 strict route joining (never urljoin).
    capture_credentials(non_interactive, install_dir=...) -> Credentials --
        the prompt flow.
    store_credentials(creds, install_dir) -> Path -- atomic, chmod 600,
        chowned to the invoking user.
    load_credentials(install_dir=None) -> Credentials | None -- reads back
        and re-normalizes the endpoint on every load.
    enabled(gate_value, install_dir=...) -> bool -- see calling convention
        below.
    redact_url(url) -> str -- §14.4 signed-URL redaction.

Calling convention for `enabled(gate_value, install_dir=...)` (read this
before wiring app.py or any step to this module):

    §14.3 defines `enabled()` as "the §3.4 gate combined with a successful
    `load_credentials()`". The §3.4 gate (`MV3DT_WEBAPP_INTEGRATION`, values
    "off"/"on") lives in `installer.conf`, which is read by `config.py` -- a
    sibling wave-3 module this one must NOT import or otherwise depend on.
    So `enabled()` here takes the already-resolved gate value as a plain
    string parameter instead of resolving it itself:

        enabled(gate_value: str, install_dir=...) -> bool

    The caller (app.py, which has both config.py and webapp.py available) is
    responsible for resolving the gate (e.g. `config.load().webapp_gate` or
    equivalent) and passing the resulting string in. `gate_value` stays
    first and mandatory -- `install_dir` is added alongside it, not instead
    of it. `enabled()` returns True only when `gate_value == "on"` AND
    `load_credentials(install_dir)` succeeds.

Calling convention for `load_credentials()` / `capture_credentials()` /
`enabled()`'s `install_dir`:

    All three take an optional `install_dir`, defaulting to module-level
    `DEFAULT_INSTALL_DIR` (the doc §11.1 default, `/opt/mv3dt`), mirroring
    §10.2's `ngc.load_key()` shape. This closes a real bug: `capture_credentials`
    and `enabled` used to call a bare `load_credentials()` internally, so
    both silently ignored `--install-dir` and read/wrote against the
    default location regardless of the operator's actual choice.
    `load_credentials` itself was always correct. Framework callers (e.g.
    `onboarding.py`) MUST pass the `install_dir` resolved by `config.load()`
    (§11.2) -- the zero-argument call sites this module's docstring
    documents keep working for anything that hasn't been updated yet.

Redaction (§14.4, REQUIRED) is two distinct rules:
    - The API key: never pass the raw key into any logs.py call. Any log
      line this module emits that references the key prints
      "API_KEY=<redacted>" instead.
    - Signed URLs: `redact_url()` below. Signed URLs themselves are a
      STEP-7-owned concept (upload flow) -- this module does not construct
      or consume any; `redact_url` is provided here purely because §14.4
      specifies it as part of this contract, for STEP-7 code to import.
"""

from __future__ import annotations

import contextlib
import getpass
import os
import pathlib
from dataclasses import dataclass
from typing import Optional

from . import privilege, prompting
from .logs import log

__all__ = [
    "Credentials",
    "DEFAULT_INSTALL_DIR",
    "normalize_endpoint",
    "join",
    "capture_credentials",
    "store_credentials",
    "load_credentials",
    "enabled",
    "redact_url",
]

# Doc §11.1 default install directory. Used only as the default location for
# load_credentials()/capture_credentials() -- see the module docstring for
# why this module can't resolve install_dir via config.py itself.
DEFAULT_INSTALL_DIR = pathlib.Path("/opt/mv3dt")

_SECRETS_DIRNAME = "secrets"
_WEBAPP_ENV_FILENAME = "webapp.env"
_SECRETS_DIR_MODE = 0o700
_SECRET_FILE_MODE = 0o600


@dataclass
class Credentials:
    """Web-app credentials (doc 00 §14.1). Either field may be unset."""

    api_key: Optional[str] = None
    endpoint: Optional[str] = None


# ---------------------------------------------------------------------------
# 14.2 -- endpoint normalization + route joining
# ---------------------------------------------------------------------------


def normalize_endpoint(raw: str) -> str:
    """Normalize an operator-supplied base URL (doc 00 §14.2).

    1. Strip surrounding whitespace.
    2. Strip all trailing "/" characters.
    3. If the result ends in "/api" (case-insensitive), remove that suffix,
       then strip trailing "/" again.
    4. Raise ValueError if the result is empty -- an empty endpoint is an
       error, not a valid (empty) value.
    """
    value = raw.strip()
    value = value.rstrip("/")
    if value[-4:].lower() == "/api":
        value = value[: -len("/api")]
        value = value.rstrip("/")
    if not value:
        raise ValueError("endpoint normalizes to an empty string")
    return value


def join(endpoint: str, path: str) -> str:
    """Strict route join (doc 00 §14.2): right-strip "/" from the base,
    left-pad a single "/" onto path, concatenate. Deliberately NOT
    `urllib.parse.urljoin` -- urljoin's relative-reference semantics would
    discard a base path component (e.g. a base ending in a non-root
    segment), which is exactly the bug this contract avoids.
    """
    return endpoint.rstrip("/") + "/" + path.lstrip("/")


# ---------------------------------------------------------------------------
# 14.3 -- capture + handoff API
# ---------------------------------------------------------------------------


def _ask_keep(found_desc: str) -> bool:
    """Ask whether to keep an already-present value; default is keep (Y)."""
    answer = input(f"{found_desc} Keep it? [Y/n]: ").strip().lower()
    return answer in ("", "y", "yes")


def capture_credentials(
    non_interactive: bool,
    install_dir: "pathlib.Path | str | None" = None,
) -> Credentials:
    """The §14.3 prompt flow.

    When a value already exists (per `load_credentials(install_dir)`), ask
    before replacing it, defaulting to keep. The API key prompt suppresses
    echo (`getpass.getpass`); the endpoint prompt does not. Under
    `non_interactive=True`, existing values are kept silently and missing
    ones are left unset -- no prompting occurs at all.

    `install_dir` defaults to `DEFAULT_INSTALL_DIR` -- see the module
    docstring's `install_dir` calling-convention note.
    """
    existing = load_credentials(install_dir)
    existing_key = existing.api_key if existing else None
    existing_endpoint = existing.endpoint if existing else None

    if non_interactive:
        return Credentials(api_key=existing_key, endpoint=existing_endpoint)

    reason = prompting.prompt_unavailable_reason()
    if reason is not None:
        # The documented --non-interactive outcome rather than a hidden
        # prompt that blocks forever.
        log.warn(
            f"{reason}: not asking for web-app credentials; keeping what is "
            "stored and leaving anything missing unset. Run the installer "
            "without a pipe to enter them."
        )
        return Credentials(api_key=existing_key, endpoint=existing_endpoint)

    api_key = existing_key
    if existing_key:
        if not _ask_keep("An API key was found."):
            entered = getpass.getpass("Enter web-app API key: ").strip()
            api_key = entered or None
    else:
        entered = getpass.getpass("Enter web-app API key: ").strip()
        api_key = entered or None

    endpoint = existing_endpoint
    if existing_endpoint:
        if not _ask_keep(f"An endpoint was found ({existing_endpoint})."):
            entered = input("Enter web-app endpoint URL: ").strip()
            endpoint = normalize_endpoint(entered) if entered else None
    else:
        entered = input("Enter web-app endpoint URL: ").strip()
        endpoint = normalize_endpoint(entered) if entered else None

    return Credentials(api_key=api_key, endpoint=endpoint)


def _write_text_atomic_restricted(path: pathlib.Path, content: str, mode: int) -> None:
    """Same tmp-file + fsync + os.replace principle as state.py's
    write_json_atomic (doc 00 §6.3), for a plain KEY=VALUE text file instead
    of JSON, and with a restrictive creation mode from the very first
    os.open() call -- not a separate chmod() after an open() -- so the file
    is never briefly world-readable.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.remove(tmp)
        raise

    os.replace(tmp, path)  # atomic within a filesystem; preserves tmp's mode


def store_credentials(creds: Credentials, install_dir: "pathlib.Path | str") -> pathlib.Path:
    """Write secrets/webapp.env (doc 00 §14.1), owned by the invoking user.

    Requires both creds.api_key and creds.endpoint to be set. The endpoint
    is re-normalized (§14.2) before writing, so a caller that built
    `Credentials` by hand can't accidentally store a non-normalized value.
    """
    if not creds.api_key or not creds.endpoint:
        raise ValueError(
            "store_credentials requires both api_key and endpoint to be set"
        )
    endpoint = normalize_endpoint(creds.endpoint)

    secrets_dir = pathlib.Path(install_dir) / _SECRETS_DIRNAME
    secrets_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(secrets_dir, _SECRETS_DIR_MODE)

    target = secrets_dir / _WEBAPP_ENV_FILENAME
    content = f"API_KEY={creds.api_key}\nENDPOINT={endpoint}\n"
    _write_text_atomic_restricted(target, content, _SECRET_FILE_MODE)

    user = privilege.resolve()
    os.chown(secrets_dir, user.uid, user.gid)
    os.chown(target, user.uid, user.gid)

    # Never the raw key -- see doc 00 §14.4.
    log.info(f"Wrote web-app credentials to {target} (API_KEY=<redacted>)")

    return target


def _parse_env_lines(text: str) -> dict:
    values: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def load_credentials(
    install_dir: "pathlib.Path | str | None" = None,
) -> Optional[Credentials]:
    """Read back secrets/webapp.env (doc 00 §14.3).

    Re-normalizes the endpoint (§14.2) on every load, so a hand-edited file
    is corrected in memory even if never rewritten to disk. Returns None
    when the file is absent, unreadable, or either value is missing --
    callers treat None as "not configured", never as a crash.

    `install_dir` defaults to DEFAULT_INSTALL_DIR -- see the module
    docstring's "load_credentials()" calling-convention note for why.
    """
    base = pathlib.Path(install_dir) if install_dir is not None else DEFAULT_INSTALL_DIR
    path = base / _SECRETS_DIRNAME / _WEBAPP_ENV_FILENAME

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    values = _parse_env_lines(text)
    api_key = values.get("API_KEY") or None
    raw_endpoint = values.get("ENDPOINT") or None
    if api_key is None or raw_endpoint is None:
        return None

    try:
        endpoint = normalize_endpoint(raw_endpoint)
    except ValueError:
        return None

    return Credentials(api_key=api_key, endpoint=endpoint)


def enabled(
    gate_value: str,
    install_dir: "pathlib.Path | str | None" = None,
) -> bool:
    """§3.4 gate combined with a successful load_credentials().

    `gate_value` is the already-resolved MV3DT_WEBAPP_INTEGRATION value
    ("on"/"off") -- see the module docstring's calling-convention note.
    `install_dir` defaults to `DEFAULT_INSTALL_DIR`, same as
    `capture_credentials`/`load_credentials`. True only when
    `gate_value == "on"` AND `load_credentials(install_dir)` succeeds.
    """
    if gate_value != "on":
        return False
    return load_credentials(install_dir) is not None


# ---------------------------------------------------------------------------
# 14.4 -- redaction
# ---------------------------------------------------------------------------


def redact_url(url: str) -> str:
    if not url:
        return ""
    q = url.find("?")
    return url if q < 0 else url[:q] + "?<redacted>"
