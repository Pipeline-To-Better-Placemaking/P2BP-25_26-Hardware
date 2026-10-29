"""Step 8 -- Tracking history recorder and read-only tailnet API.

Implements `installer/plan/STEP-8-TRACKING-HISTORY.md` against the framework
contracts in `installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md` (gates section
3.4, bundled assets section 4.2, reporting strings section 8.3, the Step
protocol section 12, redaction section 14.4).

The step installs the bundled `mv3dt_tracking` package into
`<install_dir>/tracking/`, builds its venv, writes `secrets/tracking.env`,
and runs `mv3dt-tracking.service`. The installer never imports the package;
it only copies it and drives it through the venv's interpreter.

Judgment calls the spec leaves open:

- **Missing `TRACKING_PROJECT_ID` / `ALLOWED_ORIGINS`.** Under
  `--non-interactive` with neither set in the environment, `run` writes the
  secret, skips the unit, and returns `USER_ACTION_REQUIRED`: a unit enabled
  without them would refuse to serve and restart forever.
- **The row-growth check is a warning.** At install time a pipeline is often
  not running, so `verify` reports "no detections arrived" as a warning and
  still completes once the service is healthy. The counters come from the
  recorder stats in `GET /health`, so verify never opens the SQLite files.
- **The ticket secret never leaves the file.** It is generated with
  `secrets.token_urlsafe(48)`, written once with mode 600, and never logged,
  printed, or passed on a command line or in a subprocess environment.

Cleaning files (TRACKING-FUSION-AND-CLEANING.md section 10):

- **`cleaning.json` is seeded once.** When absent, `run` writes the strict
  seed `{"version", "preset", "zones"}` to `<install_dir>/tracking/`, owned
  by the invoking user (who is also the service user) with mode 644. The
  values come from the installed package's own `cleaning.STRICT`, read
  through the venv interpreter, so no default is restated here. The file is
  operator-managed: it is not a managed template, an existing file (or
  symlink) is never touched, and a failed seed is only a warning because a
  missing file already means strict defaults.
- **`offsets.json` is never created here.** Only `learn-offsets --save`
  writes it. The tracking directory is owned by the service user, and the
  `tracking` passthrough runs the package as that user, so the saved file
  lands where the service reads it, owned by the user the service runs as.
- **One environment for the unit and the passthrough.**
  `service_environment()` is the single list of `Environment=` values; the
  unit renders it and `mv3dt-installer tracking ...` passes it to the
  package, run from the unit's working directory. The unit's two
  `EnvironmentFile=` sources are deliberately not forwarded:
  `secrets/tracking.env` because the ticket secret never leaves its file,
  and `installer.conf` because the CLI commands read none of its keys (only
  `serve` uses `LOCATION_ID`, for the MQTT topic).
- **Upgrade.** Re-running the step re-syncs the package (which never
  touches files beside it), re-renders the unit and restarts the service
  when anything changed; both cleaning files stay as they are. A completed
  step re-runs on its own when `pending_update()` finds the installed
  package, requirements or unit differing from what this release would
  install, so an upgrade needs no `--reset-step 8`. That update run never
  prompts and always restarts the unit, and the dispatch loop keeps it
  marked as an update until it completes.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import pathlib
import re
import secrets
import shutil
import sqlite3
import stat as stat_mod
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from mv3dt_installer import app as app_mod
from mv3dt_installer import config as config_mod
from mv3dt_installer import systemd
from mv3dt_installer.logs import log
from mv3dt_installer.steps import StepResult, StepStatus, UserAction, register
from mv3dt_installer.webapp import _write_text_atomic_restricted

__all__ = [
    "STEP_ID",
    "UNIT_NAME",
    "bundled_version",
    "sync_package",
    "update_reason",
    "ensure_venv",
    "ensure_tracking_env",
    "render_tracking_unit",
    "service_environment",
    "seed_cleaning_file",
    "tailscale_state",
    "Step8TrackingHistory",
]

STEP_ID = "step8_tracking_history"
UNIT_NAME = "mv3dt-tracking.service"

PACKAGE_NAME = "mv3dt_tracking"
BIND = "127.0.0.1:8090"
LISTEN_URL = f"http://{BIND}"
HEALTH_URL = f"{LISTEN_URL}/health"
TAILNET_HTTPS_PORT = 8443
TAILSCALE_SERVE_ARGV = (
    "tailscale",
    "serve",
    "--bg",
    f"--https={TAILNET_HTTPS_PORT}",
    LISTEN_URL,
)
TAILSCALE_SERVE_COMMAND = " ".join(("sudo",) + TAILSCALE_SERVE_ARGV)
TAILSCALE_TIMEOUT_S = 30.0

DATA_DIR = pathlib.Path("/var/lib/mv3dt-tracking")
DATA_DIR_MODE = 0o750
SECRETS_DIRNAME = "secrets"
SECRETS_DIR_MODE = 0o700
TRACKING_ENV_NAME = "tracking.env"
TRACKING_ENV_MODE = 0o600
CLEANING_FILE_NAME = "cleaning.json"
CLEANING_FILE_MODE = 0o644
OFFSETS_FILE_NAME = "offsets.json"
REQUIREMENTS_MARKER = ".mv3dt-requirements.sha256"

KEY_SECRET = "WORKSTATION_TICKET_SECRET"
KEY_PROJECT = "TRACKING_PROJECT_ID"
KEY_ORIGINS = "ALLOWED_ORIGINS"
KEY_ISSUER = "TICKET_ISSUER"
REQUIRED_ENV_KEYS = (KEY_SECRET, KEY_PROJECT, KEY_ORIGINS)

CONF_LOCATION_ID_KEY = "LOCATION_ID"
CONF_CALIBRATION_DIR_KEY = "CALIBRATION_DIR"

MIN_PYTHON = (3, 10)

_ORIGIN_RE = re.compile(r"^https?://[^/\s,]+$")
_VERSION_RE = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']""", re.M)

# Injectable seams, replaced by tests.
_PROMPT: Callable[[str], str] = input
_WHICH: Callable[[str], Optional[str]] = shutil.which
_SLEEP: Callable[[float], None] = time.sleep
_HEALTH_ATTEMPTS = 10
_HEALTH_INTERVAL_S = 1.0
_ROW_WINDOW_S = 5.0


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _tracking_dir(ctx: Any) -> pathlib.Path:
    return pathlib.Path(ctx.install_dir) / "tracking"


def _venv_dir(ctx: Any) -> pathlib.Path:
    return _tracking_dir(ctx) / "venv"


def _venv_python(ctx: Any) -> pathlib.Path:
    return _venv_dir(ctx) / "bin" / "python3"


def _tracking_env_path(ctx: Any) -> pathlib.Path:
    return pathlib.Path(ctx.install_dir) / SECRETS_DIRNAME / TRACKING_ENV_NAME


def _cleaning_file(ctx: Any) -> pathlib.Path:
    return _tracking_dir(ctx) / CLEANING_FILE_NAME


def _offsets_file(ctx: Any) -> pathlib.Path:
    return _tracking_dir(ctx) / OFFSETS_FILE_NAME


def _calibration_dir(ctx: Any) -> pathlib.Path:
    persisted = (ctx.conf.get(CONF_CALIBRATION_DIR_KEY) or "").strip()
    if persisted:
        return pathlib.Path(persisted)
    location = (ctx.conf.get(CONF_LOCATION_ID_KEY) or "").strip()
    return pathlib.Path(ctx.install_dir) / "deepstream" / "calibration" / location


def _camera_models_dir(ctx: Any) -> pathlib.Path:
    return pathlib.Path(ctx.install_dir) / "deepstream" / "camera_models"


def _gate_on(ctx: Any) -> bool:
    value = ctx.conf.get(config_mod.GATE_TRACKING_HISTORY, "off")
    return str(value).strip().lower() == "on"


def _runner_from_ctx(ctx: Any):
    return lambda argv, **kwargs: ctx.run_root(*argv, **kwargs)


def _chown(ctx: Any, path: pathlib.Path) -> None:
    try:
        os.chown(path, ctx.user.uid, ctx.user.gid)
    except OSError:
        pass  # best effort, e.g. under a non-root test process


def _chown_tree(ctx: Any, root: pathlib.Path) -> None:
    for entry in (root, *root.rglob("*")):
        _chown(ctx, entry)


# ---------------------------------------------------------------------------
# Package (section 11.2 step 1)
# ---------------------------------------------------------------------------


def bundled_version(ctx: Any) -> str:
    """`__version__` of the bundled package, read without importing it."""
    init = pathlib.Path(ctx.asset_path("tracking", PACKAGE_NAME, "__init__.py"))
    match = _VERSION_RE.search(init.read_text(encoding="utf-8"))
    return match.group(1) if match else "unknown"


def _tree_digest(root: pathlib.Path) -> dict:
    """`{relative path: sha256}` of every file under `root`.

    Content only: file order and mtimes never change it. Byte-code is left
    out (`__pycache__`, `*.pyc`), matching what `sync_package` copies, so a
    compiled bundle or a package the service has already imported compares
    equal to its source.
    """
    digest = {}
    if not root.is_dir():
        return digest
    for path in sorted(root.rglob("*")):
        if (
            "__pycache__" in path.parts
            or path.suffix == ".pyc"
            or not path.is_file()
        ):
            continue
        digest[path.relative_to(root).as_posix()] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return digest


def sync_package(ctx: Any) -> bool:
    """Copy the bundled package and requirements into `<install_dir>/tracking/`.

    Idempotent: returns False when the installed copy already matches the
    bundle file for file, otherwise replaces it and returns True.
    """
    source = pathlib.Path(ctx.asset_path("tracking", PACKAGE_NAME))
    requirements = pathlib.Path(ctx.asset_path("tracking", "requirements.txt"))
    tracking_dir = _tracking_dir(ctx)
    dest = tracking_dir / PACKAGE_NAME
    dest_requirements = tracking_dir / "requirements.txt"

    tracking_dir.mkdir(parents=True, exist_ok=True)
    changed = False

    if _tree_digest(source) != _tree_digest(dest):
        staging = tracking_dir / f".{PACKAGE_NAME}.new"
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(
            source, staging, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        if dest.exists():
            shutil.rmtree(dest)
        staging.rename(dest)
        changed = True

    wanted = requirements.read_bytes()
    if not dest_requirements.exists() or dest_requirements.read_bytes() != wanted:
        dest_requirements.write_bytes(wanted)
        changed = True

    _chown(ctx, tracking_dir)
    _chown_tree(ctx, dest)
    _chown(ctx, dest_requirements)
    return changed


def update_reason(ctx: Any) -> Optional[str]:
    """Why a completed install is out of date with this release, or None.

    Read only. Compares the bundled package and requirements with the
    installed copies, and, when `ctx` carries the invoking user, the unit
    this release would render with the one on disk. `--status` passes no
    user, so it checks the package and requirements only. Never looks at
    `cleaning.json`, `offsets.json`, `tracking.env` or the data directory.
    """
    tracking_dir = _tracking_dir(ctx)
    source = pathlib.Path(ctx.asset_path("tracking", PACKAGE_NAME))
    if _tree_digest(source) != _tree_digest(tracking_dir / PACKAGE_NAME):
        return "bundled tracking package changed"
    requirements = pathlib.Path(ctx.asset_path("tracking", "requirements.txt"))
    try:
        installed = (tracking_dir / "requirements.txt").read_bytes()
    except OSError:
        installed = None
    if installed != requirements.read_bytes():
        return "bundled tracking requirements changed"
    if getattr(ctx, "user", None) is not None:
        unit = systemd.UNIT_DIR / UNIT_NAME
        try:
            current = unit.read_text(encoding="utf-8")
        except OSError:
            current = None
        if current != render_tracking_unit(ctx):
            return f"{UNIT_NAME} differs from this release"
    return None


# ---------------------------------------------------------------------------
# Venv (section 11.2 step 2)
# ---------------------------------------------------------------------------


def _tail(result: Any, limit: int = 1500) -> str:
    text = ((getattr(result, "stderr", "") or "") + (getattr(result, "stdout", "") or "")).strip()
    return text[-limit:]


def ensure_venv(ctx: Any) -> tuple:
    """Build the venv and install the pinned requirements as the invoking user.

    Returns `(changed, error)`. pip runs only when the venv is new or the
    requirements file changed since the last successful install.
    """
    venv_dir = _venv_dir(ctx)
    python = _venv_python(ctx)
    requirements = _tracking_dir(ctx) / "requirements.txt"
    marker = venv_dir / REQUIREMENTS_MARKER
    wanted = hashlib.sha256(requirements.read_bytes()).hexdigest()
    changed = False

    if not python.exists():
        result = ctx.run_as_user(
            "python3", "-m", "venv", str(venv_dir), capture_output=True, text=True
        )
        if getattr(result, "returncode", 1) != 0:
            return changed, f"python3 -m venv {venv_dir} failed: {_tail(result)}"
        changed = True

    try:
        current = marker.read_text(encoding="utf-8").strip()
    except OSError:
        current = ""
    if changed or current != wanted:
        result = ctx.run_as_user(
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "-r",
            str(requirements),
            capture_output=True,
            text=True,
        )
        if getattr(result, "returncode", 1) != 0:
            return changed, f"pip install -r {requirements} failed: {_tail(result)}"
        marker.write_text(wanted + "\n", encoding="utf-8")
        _chown(ctx, marker)
        changed = True
    return changed, None


# ---------------------------------------------------------------------------
# Data directory and secrets (section 11.2 steps 3-4, section 4.3)
# ---------------------------------------------------------------------------


def ensure_data_dir(ctx: Any, data_dir: Optional[pathlib.Path] = None) -> None:
    data_dir = data_dir or DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    data_dir.chmod(DATA_DIR_MODE)
    _chown(ctx, data_dir)


def _read_env_file(path: pathlib.Path) -> dict:
    values: dict = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def normalize_origins(value: str) -> Optional[str]:
    """Comma-separated `http(s)://host[:port]` origins, trailing slashes
    dropped. Returns None for an empty or invalid list."""
    origins = [o.strip().rstrip("/") for o in value.split(",") if o.strip()]
    if not origins or not all(_ORIGIN_RE.match(o) for o in origins):
        return None
    return ",".join(origins)


def _clean_single_line(value: str) -> Optional[str]:
    value = value.strip()
    if not value or any(c.isspace() for c in value):
        return None
    return value


def _ask(question: str, validate: Callable[[str], Optional[str]], *, optional=False):
    while True:
        answer = _PROMPT(question).strip()
        if not answer and optional:
            return ""  # an explicit blank, recorded so it is not asked again
        cleaned = validate(answer) if answer else None
        if cleaned is not None:
            return cleaned
        log.warn("that value is not valid; please answer again")


def _resolve_value(
    ctx: Any,
    key: str,
    existing: dict,
    validate: Callable[[str], Optional[str]],
    question: str,
    *,
    optional: bool = False,
    prompt: bool = True,
) -> Optional[str]:
    """Existing file value, then the environment, then a prompt.

    An optional key already in the file counts as answered even when blank,
    so a blank answer is asked once. With `prompt=False` (an update of a
    completed install) a value found nowhere stays unset instead of being
    asked for; a missing required key is then reported by the caller.
    """
    if existing.get(key) or (optional and key in existing):
        return existing[key]
    env_value = os.environ.get(key, "")
    if env_value:
        cleaned = validate(env_value)
        if cleaned is not None:
            return cleaned
        log.warn(f"ignoring invalid {key} from the environment")
    if ctx.non_interactive or not prompt:
        return None
    return _ask(question, validate, optional=optional)


def ensure_tracking_env(ctx: Any, *, prompt: bool = True) -> tuple:
    """Create or complete `secrets/tracking.env`.

    The secret is generated only when absent and is never overwritten. Keys
    already present are kept as they are. A blank answer to the optional
    `TICKET_ISSUER` prompt is written as `TICKET_ISSUER=`, which the package
    reads as unset, so later runs do not ask again. `prompt=False` never
    asks anything. Returns `(changed, missing_keys)`.
    """
    path = _tracking_env_path(ctx)
    existing = _read_env_file(path)
    values = dict(existing)

    if not values.get(KEY_SECRET):
        values[KEY_SECRET] = secrets.token_urlsafe(48)

    project = _resolve_value(
        ctx, KEY_PROJECT, existing, _clean_single_line,
        "Web app project id whose tickets this workstation accepts: ",
        prompt=prompt,
    )
    origins = _resolve_value(
        ctx, KEY_ORIGINS, existing, normalize_origins,
        "Allowed web app origins, comma separated (https://app.example.org): ",
        prompt=prompt,
    )
    issuer = _resolve_value(
        ctx, KEY_ISSUER, existing, _clean_single_line,
        "Ticket issuer (the web app's JWT_ISSUER), blank to skip: ",
        optional=True,
        prompt=prompt,
    )
    for key, value in ((KEY_PROJECT, project), (KEY_ORIGINS, origins)):
        if value:
            values[key] = value
    if issuer is not None:
        values[KEY_ISSUER] = issuer

    missing = [key for key in REQUIRED_ENV_KEYS if not values.get(key)]

    ordered = [k for k in (KEY_SECRET, KEY_PROJECT, KEY_ORIGINS, KEY_ISSUER) if k in values]
    ordered += sorted(k for k in values if k not in ordered)
    content = (
        "# secrets/tracking.env -- generated by mv3dt-installer (STEP-8 section 4.3).\n"
        "# Read the ticket secret once and copy it into the web app; never commit it.\n"
        + "".join(f"{k}={values[k]}\n" for k in ordered)
    )

    secrets_dir = path.parent
    if not secrets_dir.exists():
        secrets_dir.mkdir(parents=True)
        secrets_dir.chmod(SECRETS_DIR_MODE)
        _chown(ctx, secrets_dir)

    # Rewrite only when a key was added, so a hand-edited file is left alone.
    changed = not path.exists() or values != existing
    if changed:
        _write_text_atomic_restricted(path, content, TRACKING_ENV_MODE)
        log.info(f"wrote {path} ({KEY_SECRET}=<redacted>)")
    path.chmod(TRACKING_ENV_MODE)
    _chown(ctx, path)
    return changed, missing


# ---------------------------------------------------------------------------
# cleaning.json seed (TRACKING-FUSION-AND-CLEANING.md sections 4.1 and 10)
# ---------------------------------------------------------------------------

# Run by the venv interpreter in isolated mode (-I, so neither the caller's
# directory nor PYTHONPATH is on sys.path); argv[1] is the tracking
# directory. It prints the strict seed taken from the package itself and
# checks that the package accepts it.
_SEED_PROGRAM = (
    "import json, sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "from mv3dt_tracking.cleaning import PRESETS, STRICT, parse_config\n"
    "preset = next(name for name, values in PRESETS.items() if values is STRICT)\n"
    "seed = {'version': STRICT['version'], 'preset': preset, 'zones': list(STRICT['zones'])}\n"
    "parse_config(seed)\n"
    "print(json.dumps(seed))\n"
)


def _strict_seed(ctx: Any) -> tuple:
    """`(seed, error)`: the strict seed as the installed package defines it."""
    result = ctx.run_as_user(
        str(_venv_python(ctx)), "-I", "-c", _SEED_PROGRAM, str(_tracking_dir(ctx)),
        capture_output=True, text=True,
    )
    if getattr(result, "returncode", 1) != 0:
        return None, _tail(result, 300) or "the tracking package could not be imported"
    try:
        seed = json.loads(getattr(result, "stdout", "") or "")
    except json.JSONDecodeError:
        seed = None
    if not isinstance(seed, dict):
        return None, "the tracking package printed no seed"
    return seed, None


def seed_cleaning_file(ctx: Any) -> tuple:
    """Write `cleaning.json` once with the strict seed. Returns `(written, error)`.

    An existing path, a dangling symlink included, is left alone. The file
    is created under a temporary name and hard-linked into place, so it
    appears complete or not at all and a file that appeared meanwhile is
    never replaced.
    """
    path = _cleaning_file(ctx)
    if os.path.lexists(path):
        return False, None
    seed, error = _strict_seed(ctx)
    if error:
        return False, error
    content = json.dumps(seed, indent=2) + "\n"
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(tmp), flags, CLEANING_FILE_MODE)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                os.fchmod(f.fileno(), CLEANING_FILE_MODE)  # regardless of umask
                try:
                    os.fchown(f.fileno(), ctx.user.uid, ctx.user.gid)
                except OSError:
                    pass  # best effort, e.g. under a non-root test process
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.link(tmp, path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
    except FileExistsError:
        return False, None
    except OSError as exc:
        return False, f"could not write {path}: {exc}"
    log.info(f"wrote {path} (strict cleaning defaults)")
    return True, None


# ---------------------------------------------------------------------------
# Unit (section 4.4, section 11.2 step 5)
# ---------------------------------------------------------------------------


def service_environment(ctx: Any, *, data_dir: Optional[pathlib.Path] = None) -> dict:
    """The unit's `Environment=` values, in unit order.

    The single source for both the rendered unit and the `tracking`
    passthrough, so a CLI command sees the same data directory,
    calibration, cleaning file and offsets file as `serve`. Holds no secret.
    """
    return {
        "PYTHONPATH": str(_tracking_dir(ctx)),
        "DATA_DIR": str(data_dir or DATA_DIR),
        "CALIB_DIR": str(_calibration_dir(ctx)),
        "CAMERA_MODELS_DIR": str(_camera_models_dir(ctx)),
        "CLEANING_FILE": str(_cleaning_file(ctx)),
        "OFFSETS_FILE": str(_offsets_file(ctx)),
        "BIND": BIND,
        "PYTHONUNBUFFERED": "1",
    }


def render_tracking_unit(ctx: Any, *, data_dir: Optional[pathlib.Path] = None) -> str:
    values = {
        "INSTALL_DIR": str(ctx.install_dir),
        "TRACKING_DIR": str(_tracking_dir(ctx)),
        "VENV_PYTHON": str(_venv_python(ctx)),
    }
    environment = service_environment(ctx, data_dir=data_dir)
    for label, value in (*values.items(), *environment.items()):
        systemd._reject_unquotable(label, value)
    substitutions = {k: systemd._escape_specifiers(v) for k, v in values.items()}
    substitutions["USER"] = ctx.user.name
    substitutions["SERVICE_ENVIRONMENT"] = "\n".join(
        f'Environment="{k}={systemd._escape_specifiers(v)}"' for k, v in environment.items()
    )
    return systemd.render_unit(
        (systemd.TEMPLATE_DIR, "mv3dt-tracking.service.in"), substitutions
    )


def _install_unit(ctx: Any, *, restart: bool) -> bool:
    runner = _runner_from_ctx(ctx)
    changed = systemd.install_unit(
        UNIT_NAME, render_tracking_unit(ctx), unit_dir=systemd.UNIT_DIR, runner=runner
    )
    systemd.daemon_reload(runner=runner)
    systemd.enable_now(UNIT_NAME, runner=runner)
    if changed or restart:
        runner(["systemctl", "restart", UNIT_NAME], check=False)
    return changed


# ---------------------------------------------------------------------------
# Tailscale (section 3, section 11.1)
# ---------------------------------------------------------------------------


def tailscale_state(ctx: Any) -> tuple:
    """`(available, logged_in, dns_name)`. Never raises."""
    if not _WHICH("tailscale"):
        return False, False, None
    try:
        result = ctx.run_root(
            "tailscale",
            "status",
            "--json",
            capture_output=True,
            text=True,
            stream=False,
            timeout=TAILSCALE_TIMEOUT_S,
        )
    except Exception:  # noqa: BLE001 -- a probe, never fatal
        return True, False, None
    if getattr(result, "returncode", 1) != 0:
        return True, False, None
    try:
        status = json.loads(getattr(result, "stdout", "") or "{}")
    except json.JSONDecodeError:
        return True, False, None
    logged_in = status.get("BackendState") == "Running"
    dns = ((status.get("Self") or {}).get("DNSName") or "").rstrip(".") or None
    return True, logged_in, dns


def _expose(ctx: Any) -> bool:
    available, logged_in, _dns = tailscale_state(ctx)
    if not available or not logged_in:
        reason = "not installed" if not available else "not logged in"
        log.warn(
            f"tailscale is {reason}; the tracking API is only reachable on this "
            f"machine. After `sudo tailscale up`, run: {TAILSCALE_SERVE_COMMAND}"
        )
        return False
    try:
        result = ctx.run_root(
            *TAILSCALE_SERVE_ARGV,
            capture_output=True,
            text=True,
            timeout=TAILSCALE_TIMEOUT_S,
        )
        failure = None if getattr(result, "returncode", 1) == 0 else _tail(result, 300)
    except subprocess.TimeoutExpired:
        failure = f"timed out after {TAILSCALE_TIMEOUT_S:.0f}s"
    except Exception as exc:  # noqa: BLE001 -- exposure is never fatal
        failure = str(exc)
    if failure is not None:
        log.warn(
            f"tailscale serve failed ({failure}); run it by hand: "
            f"{TAILSCALE_SERVE_COMMAND}"
        )
        return False
    log.info(f"tailnet HTTPS {TAILNET_HTTPS_PORT} now proxies to {LISTEN_URL}")
    return True


# ---------------------------------------------------------------------------
# Health probe (section 11.3)
# ---------------------------------------------------------------------------


def _http_get(url: str, timeout: float = 3.0) -> tuple:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.getcode(), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except (urllib.error.URLError, OSError):
        return None, b""


_HTTP_GET: Callable[[str], tuple] = _http_get


def _health() -> Optional[dict]:
    status, body = _HTTP_GET(HEALTH_URL)
    if status != 200:
        return None
    try:
        doc = json.loads(body or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError):
        doc = {}
    return doc if isinstance(doc, dict) else {}


def _wait_for_health() -> Optional[dict]:
    for attempt in range(_HEALTH_ATTEMPTS):
        doc = _health()
        if doc is not None:
            return doc
        if attempt + 1 < _HEALTH_ATTEMPTS:
            _SLEEP(_HEALTH_INTERVAL_S)
    return None


def _recorder_counts(doc: dict) -> tuple:
    rec = doc.get("recorder") or {}
    return (
        bool(rec.get("connected")),
        int(rec.get("messages") or 0),
        int(rec.get("rows_flushed") or 0),
    )


# ---------------------------------------------------------------------------
# Today's row count (section 11.4), read only
# ---------------------------------------------------------------------------


def today_row_count(data_dir: Optional[pathlib.Path] = None) -> Optional[int]:
    data_dir = data_dir or DATA_DIR
    day = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    path = data_dir / f"{day}.sqlite"
    if not path.exists():
        return 0
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
        try:
            return int(conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0])
        finally:
            conn.close()
    except sqlite3.Error:
        return None


# ---------------------------------------------------------------------------
# The Step protocol implementation (section 11)
# ---------------------------------------------------------------------------


def _check_python(ctx: Any) -> Optional[str]:
    result = ctx.run_root(
        "python3",
        "-c",
        "import sys, venv, ensurepip; print('%d.%d' % sys.version_info[:2])",
        capture_output=True,
        text=True,
        stream=False,
    )
    if getattr(result, "returncode", 1) != 0:
        return None
    text = (getattr(result, "stdout", "") or "").strip()
    try:
        major, minor = (int(p) for p in text.split(".")[:2])
    except ValueError:
        return None
    return text if (major, minor) >= MIN_PYTHON else None


class Step8TrackingHistory:
    id = STEP_ID
    title = "Tracking history"

    phases = (
        "package and venv",
        "data directory and secrets",
        "tracking unit",
        "tailnet exposure",
    )
    order = 8

    def preflight(self, ctx: Any) -> StepResult:
        if not _gate_on(ctx):
            return StepResult(status=StepStatus.COMPLETE)

        if _check_python(ctx) is None:
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message="python3 3.10 or newer with the venv module is required",
                user_actions=[
                    UserAction(
                        text="Install the Python venv module",
                        command="sudo apt-get install -y python3-venv",
                    )
                ],
            )

        if not (_WHICH("mosquitto") or pathlib.Path("/usr/sbin/mosquitto").exists()):
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message="the Mosquitto broker is not installed",
                user_actions=[UserAction(text="Complete Step 1 (prerequisites) first")],
            )

        location = (ctx.conf.get(CONF_LOCATION_ID_KEY) or "").strip()
        if not location:
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message="LOCATION_ID is not set in installer.conf",
                user_actions=[UserAction(text="Complete Steps 3 and 4 first")],
            )

        calib = _calibration_dir(ctx)
        for name in ("map.png", "transforms.yml"):
            if not (calib / name).is_file():
                log.warn(
                    f"step8: {calib / name} is missing; the floor plan and meta "
                    "routes stay empty until Step 4 calibration output exists"
                )
        models = _camera_models_dir(ctx)
        if not (models.is_dir() and any(models.glob("*.yml"))):
            log.warn(f"step8: no camera models in {models}; camera positions stay empty")

        available, logged_in, _ = tailscale_state(ctx)
        if not available or not logged_in:
            log.warn(
                "step8: tailscale is "
                + ("not installed" if not available else "not logged in")
                + "; the service installs anyway and the expose phase prints "
                "the manual command"
            )
        return StepResult(status=StepStatus.COMPLETE)

    def run(self, ctx: Any) -> StepResult:
        if not _gate_on(ctx):
            return StepResult(status=StepStatus.COMPLETE)

        # An update of a completed install (section 11.5) never prompts and
        # always restarts the unit: the files may have been replaced by an
        # earlier attempt that stopped before its restart.
        updating = bool(getattr(ctx, "update_pending", None))

        ctx.progress.phase(1)
        ctx.progress.task("copying the tracking package")
        package_changed = sync_package(ctx)
        ctx.progress.task("building the tracking venv")
        venv_changed, error = ensure_venv(ctx)
        if error:
            return StepResult(status=StepStatus.FAILED, message=error)

        ctx.progress.phase(2)
        ctx.progress.task("creating the data directory, cleaning.json and tracking.env")
        ensure_data_dir(ctx)
        _, seed_error = seed_cleaning_file(ctx)
        if seed_error:
            log.warn(
                f"step8: {_cleaning_file(ctx)} was not seeded ({seed_error}); the "
                "service uses the strict defaults until the file exists"
            )
        env_changed, missing = ensure_tracking_env(ctx, prompt=not updating)
        if missing:
            path = _tracking_env_path(ctx)
            during = "; the tracking update is not finished" if updating else ""
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message=f"{', '.join(missing)} not set in {path}{during}",
                user_actions=[
                    UserAction(
                        text=f"Add {', '.join(missing)} to tracking.env, then re-run "
                        "the installer",
                        path=str(path),
                    )
                ],
            )

        ctx.progress.phase(3)
        ctx.progress.task(f"installing {UNIT_NAME}")
        unit_changed = _install_unit(
            ctx, restart=package_changed or venv_changed or env_changed or updating
        )

        ctx.progress.phase(4)
        ctx.progress.task("exposing the API on the tailnet")
        _expose(ctx)

        version = bundled_version(ctx)
        if package_changed or unit_changed or venv_changed or updating:
            ctx.report_installed(UNIT_NAME, version)
        else:
            ctx.report_already_installed(UNIT_NAME, version)
        return StepResult(status=StepStatus.COMPLETE)

    def verify(self, ctx: Any) -> StepResult:
        if not _gate_on(ctx):
            return StepResult(status=StepStatus.COMPLETE)

        runner = _runner_from_ctx(ctx)
        if not (
            systemd.is_enabled(UNIT_NAME, runner=runner)
            and systemd.is_active(UNIT_NAME, runner=runner)
        ):
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message=f"{UNIT_NAME} is not enabled and active",
                user_actions=[
                    UserAction(
                        text="Check the service log",
                        command="journalctl -u mv3dt-tracking -n 50",
                    )
                ],
            )

        path = _tracking_env_path(ctx)
        try:
            mode = stat_mod.S_IMODE(path.stat().st_mode)
        except OSError:
            mode = None
        if mode != TRACKING_ENV_MODE:
            return StepResult(
                status=StepStatus.FAILED,
                message=f"{path} must be mode 600 "
                f"(found {oct(mode) if mode is not None else 'missing'})",
            )
        values = _read_env_file(path)
        missing = [k for k in REQUIRED_ENV_KEYS if not values.get(k)]
        if missing:
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message=f"{', '.join(missing)} not set in {path}",
                user_actions=[UserAction(text="Fill in tracking.env", path=str(path))],
            )

        first = _wait_for_health()
        if first is None:
            return StepResult(
                status=StepStatus.FAILED,
                message=f"GET {HEALTH_URL} did not return 200; see "
                "journalctl -u mv3dt-tracking",
            )

        _SLEEP(_ROW_WINDOW_S)
        second = _health() or first
        connected, msgs_a, rows_a = _recorder_counts(first)
        connected_b, msgs_b, rows_b = _recorder_counts(second)
        topic = f"mv3dt/{ctx.conf.get(CONF_LOCATION_ID_KEY, '')}/sv3d"
        if rows_b > rows_a:
            log.info(f"step8 verify: recorded {rows_b - rows_a} detections in {_ROW_WINDOW_S:.0f}s")
        elif not (connected or connected_b):
            log.warn("step8 verify: the recorder is not connected to the local broker")
        elif msgs_b > msgs_a:
            log.warn(
                f"step8 verify: messages arrive on {topic} but no rows were "
                "recorded; check journalctl -u mv3dt-tracking for decode errors"
            )
        else:
            log.warn(
                f"step8 verify: no detections on {topic} in {_ROW_WINDOW_S:.0f}s; "
                "the pipeline is probably not running. Recording starts when it does."
            )
        return StepResult(status=StepStatus.COMPLETE)

    def pending_update(self, ctx: Any) -> Optional[str]:
        """The dispatch loop's upgrade check for a completed step: a reason
        to re-run `run()`, or None to leave the install alone. Never
        prompts; a gated-off install is never updated."""
        if not _gate_on(ctx):
            return None
        return update_reason(ctx)

    def report(self, ctx: Any) -> None:
        if not _gate_on(ctx):
            log.info(f"{STEP_ID}: gate is off; nothing installed")
            return

        runner = _runner_from_ctx(ctx)
        active = "active" if systemd.is_active(UNIT_NAME, runner=runner) else "inactive"
        enabled = "enabled" if systemd.is_enabled(UNIT_NAME, runner=runner) else "disabled"
        rows = today_row_count()
        _available, logged_in, dns = tailscale_state(ctx)
        url = (
            f"https://{dns}:{TAILNET_HTTPS_PORT}"
            if logged_in and dns
            else f"https://<this machine's tailnet name>:{TAILNET_HTTPS_PORT}"
        )
        env_path = _tracking_env_path(ctx)

        log.info(f"tracking history: {UNIT_NAME} {enabled}/{active}")
        log.info(f"  data directory: {DATA_DIR}")
        log.info(f"  rows recorded today: {rows if rows is not None else 'unknown'}")
        log.info("  logs: journalctl -u mv3dt-tracking")
        log.info(
            f"  web app setup: read the ticket secret once with `sudo cat {env_path}`"
            f" and set {KEY_SECRET} in the web app's environment to that value"
        )
        log.info(f"  web app setup: set WORKSTATION_TRACKING_URL={url}")


register(Step8TrackingHistory())


# ---------------------------------------------------------------------------
# `mv3dt-installer tracking stats | wipe --yes [--before DATE]` (section 10)
# and the cleaning commands (TRACKING-FUSION-AND-CLEANING.md section 9)
# ---------------------------------------------------------------------------

# Forwarded to the package unchanged; it owns their options and help.
CLEANING_COMMANDS = {
    "zones-suggest": "list static single-camera hotspots as zone entries",
    "learn-offsets": "learn the per-camera offset correction (--save writes offsets.json)",
    "clean-report": "print the fused summary and drop counters for --from/--to",
}


def _package_argv(ctx: Any, *args: str) -> list:
    """Run the package with the unit's own environment (no secret), as the
    invoking user, so SQLite sidecar files and a saved offsets.json stay
    owned by the service user."""
    return [
        "env",
        *(f"{k}={v}" for k, v in service_environment(ctx).items()),
        str(_venv_python(ctx)),
        "-m",
        PACKAGE_NAME,
        *args,
    ]


def _run_package(ctx: Any, *args: str) -> int:
    # The unit's WorkingDirectory, so `-m` resolves exactly as in `serve`.
    result = ctx.run_as_user(*_package_argv(ctx, *args), cwd=str(_tracking_dir(ctx)))
    return int(getattr(result, "returncode", 1))


def _day(value: str) -> str:
    try:
        return datetime.date.fromisoformat(value).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from None


def _strip_framework_flags(argv: list) -> list:
    """Drop the flags the subcommand bootstrap already consumed, wherever
    they appear, so they never reach the stats/wipe parser."""
    peek = argparse.ArgumentParser(add_help=False)
    for flag in ("--install-dir", "--log-dir"):
        peek.add_argument(flag, default=None)
    for flag in ("--non-interactive", "--verbose"):
        peek.add_argument(flag, action="store_true")
    _known, rest = peek.parse_known_args(argv)
    return rest


def _tracking_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mv3dt-installer tracking")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stats", help="row counts per day file and the data span")
    wipe = sub.add_parser("wipe", help="delete recorded history (local only)")
    wipe.add_argument("--yes", action="store_true", help="confirm the deletion")
    wipe.add_argument("--before", type=_day, help="only delete UTC days before YYYY-MM-DD")
    for name, text in CLEANING_COMMANDS.items():
        sub.add_parser(name, help=text, add_help=False)  # parsed by the package
    return parser


def _not_installed() -> int:
    log.error(
        "tracking history is not installed; enable it with "
        "--tracking-history on and re-run the installer"
    )
    return 1


def _handle_tracking(argv: list, ctx: Any) -> int:
    rest = _strip_framework_flags(argv)
    if rest and rest[0] in CLEANING_COMMANDS:
        if not _gate_on(ctx):
            log.error(
                "tracking history is off in installer.conf; enable it with "
                "--tracking-history on and re-run the installer"
            )
            return 1
        if not _venv_python(ctx).exists():
            return _not_installed()
        return _run_package(ctx, *rest)

    args = _tracking_parser().parse_args(rest)
    if not _venv_python(ctx).exists():
        return _not_installed()

    if args.command == "stats":
        return _run_package(ctx, "stats")

    if not args.yes:
        log.error("refusing to wipe without --yes")
        return 2

    wipe_args = ["wipe", "--yes"] + (["--before", args.before] if args.before else [])
    runner = _runner_from_ctx(ctx)
    runner(["systemctl", "stop", UNIT_NAME], check=False)
    # A missing unit is not active either, so a wipe without the service
    # still works; a unit that refused to stop must not have its files
    # deleted underneath it.
    if systemd.is_active(UNIT_NAME, runner=runner):
        log.error(f"{UNIT_NAME} is still active after stop; refusing to wipe")
        return 1
    try:
        return _run_package(ctx, *wipe_args)
    finally:
        runner(["systemctl", "start", UNIT_NAME], check=False)


app_mod.register_subcommand("tracking", _handle_tracking)
