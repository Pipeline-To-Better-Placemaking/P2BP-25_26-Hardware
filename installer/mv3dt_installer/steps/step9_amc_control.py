"""Step 9 -- AMC control service (start, stop and automate AutoMagicCalib).

Implements `installer/plan/STEP-9-AMC-CONTROL.md` against the framework
contracts in `installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md`, following
Step 8's structure: the step installs the bundled `mv3dt_amc_control` package
into `<install_dir>/amc_control/`, builds its venv, writes a sudoers rule and
runs `mv3dt-amc-control.service`. The installer never imports the package.

Judgment calls:

- **It shares Step 8's `secrets/tracking.env`.** The web app signs AMC and
  tracking tickets with one `WORKSTATION_TICKET_SECRET` (a ticket's scope
  tells them apart), so this step creates no secret of its own and needs
  Step 8's file complete. The unit loads it as an `EnvironmentFile`; the
  secret is never read, logged or passed on a command line here.
- **The service may run exactly two commands as root.** `bin/amc` runs
  installer subcommands, which require root, while the service runs as the
  invoking user. `/etc/sudoers.d/mv3dt-amc-control` allows only
  `bin/amc --keep-up --no-open --non-interactive --host-ip <tailscale IPv4>`
  and `bin/amc --down`. The file is checked with `visudo -cf` before it is
  moved into place, so a bad rule can never break sudo.
- **The Tailscale IPv4 is required.** It is the address a remote browser
  uses to reach AMC (`--host-ip` is documented as an IP) and it is part of
  the sudoers rule, so without a logged-in tailscale the step asks for
  `sudo tailscale up` instead of installing something that cannot work.
  When the address changes, `pending_update()` re-runs the step.
- **`amc_automation.yaml` is seeded once** beside the package (never inside
  it, where an upgrade would replace it), with `project.mode: create_new` so
  a first automated run cannot overwrite the live calibration that Step 4's
  ingest timer would install. The operator switches it once runs are trusted.
"""

from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import json
import os
import pathlib
import re
import secrets
import shutil
import stat as stat_mod
import subprocess
import time
from typing import Any, Callable, Optional

from mv3dt_installer import config as config_mod
from mv3dt_installer import systemd
from mv3dt_installer.logs import log
from mv3dt_installer.steps import StepResult, StepStatus, UserAction, register

# Shared with Step 8 so both services install and probe the same way.
from mv3dt_installer.steps.step8_tracking_history import (
    KEY_ORIGINS,
    KEY_PROJECT,
    KEY_SECRET,
    REQUIRED_ENV_KEYS,
    _check_python,
    _chown,
    _chown_tree,
    _http_get,
    _read_env_file,
    _runner_from_ctx,
    _tail,
    _tracking_env_path,
    _tree_digest,
    tailscale_state,
)

__all__ = [
    "STEP_ID",
    "UNIT_NAME",
    "bundled_version",
    "sync_package",
    "ensure_venv",
    "seed_config_file",
    "tailscale_ipv4",
    "render_sudoers",
    "install_sudoers",
    "service_environment",
    "render_amc_control_unit",
    "update_reason",
    "Step9AmcControl",
]

STEP_ID = "step9_amc_control"
UNIT_NAME = "mv3dt-amc-control.service"

ASSET_DIR = "amc_control"
PACKAGE_NAME = "mv3dt_amc_control"
BIND = "127.0.0.1:8091"
LISTEN_URL = f"http://{BIND}"
HEALTH_URL = f"{LISTEN_URL}/health"
TAILNET_HTTPS_PORT = 8444
TAILSCALE_SERVE_ARGV = (
    "tailscale",
    "serve",
    "--bg",
    f"--https={TAILNET_HTTPS_PORT}",
    LISTEN_URL,
)
TAILSCALE_SERVE_COMMAND = " ".join(("sudo",) + TAILSCALE_SERVE_ARGV)
TAILSCALE_TIMEOUT_S = 30.0

CONFIG_FILE_NAME = "amc_automation.yaml"
CONFIG_FILE_MODE = 0o644
STATE_DIRNAME = "state"
STATE_DIR_MODE = 0o750
REQUIREMENTS_MARKER = ".mv3dt-requirements.sha256"

SUDOERS_DIR = pathlib.Path("/etc/sudoers.d")
# No dot in the name: sudo skips files in sudoers.d whose names contain one.
SUDOERS_NAME = "mv3dt-amc-control"
SUDOERS_MODE = 0o440
# What the service runs (mv3dt_amc_control.controller); the rule must match it.
AMC_START_ARGS = ("--keep-up", "--no-open", "--non-interactive", "--host-ip")
AMC_STOP_ARGS = ("--down",)

_VERSION_RE = re.compile(r"""^__version__\s*=\s*["']([^"']+)["']""", re.M)
# sudoers treats , : = \ and whitespace specially in a command; refuse them.
_SUDOERS_PATH_RE = re.compile(r"^/[A-Za-z0-9_./+-]+$")
_SUDOERS_USER_RE = re.compile(r"^[a-z_][a-z0-9_-]*\$?$")

CONFIG_SEED = """\
# Automation settings for this workstation (STEP-9 section 6).
#
# Add only the keys you want to change. They are merged on top of the base
# file that ships in the package:
#   mv3dt_amc_control/amc_automation.example.yaml   (every setting, documented)
# The installer never replaces this file. Check your edits with:
#   cd {control_dir} && venv/bin/python3 -m mv3dt_amc_control check-config

project:
  # create_new keeps automated runs away from the live calibration: each run
  # makes its own AMC project, which the ingest timer ignores. Change this to
  # reuse_installer_project once automated runs are trusted; a finished run
  # is then installed into DeepStream by the ingest timer.
  mode: create_new

layout:
  # Hosts the floor plan may be downloaded from: the web app's storage host,
  # for example abcdefgh.supabase.co. Empty refuses every download.
  allowed_hosts: []
"""

# Injectable seams, replaced by tests.
_WHICH: Callable[[str], Optional[str]] = shutil.which
_SLEEP: Callable[[float], None] = time.sleep
_HTTP_GET: Callable[[str], tuple] = _http_get
_HEALTH_ATTEMPTS = 10
_HEALTH_INTERVAL_S = 1.0


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _control_dir(ctx: Any) -> pathlib.Path:
    return pathlib.Path(ctx.install_dir) / "amc_control"


def _venv_dir(ctx: Any) -> pathlib.Path:
    return _control_dir(ctx) / "venv"


def _venv_python(ctx: Any) -> pathlib.Path:
    return _venv_dir(ctx) / "bin" / "python3"


def _config_file(ctx: Any) -> pathlib.Path:
    return _control_dir(ctx) / CONFIG_FILE_NAME


def _state_dir(ctx: Any) -> pathlib.Path:
    return _control_dir(ctx) / STATE_DIRNAME


def _amc_bin(ctx: Any) -> pathlib.Path:
    return pathlib.Path(ctx.install_dir) / "bin" / "amc"


def _sudoers_path() -> pathlib.Path:
    return SUDOERS_DIR / SUDOERS_NAME


def _gate_on(ctx: Any) -> bool:
    value = ctx.conf.get(config_mod.GATE_AMC_CONTROL, "off")
    return str(value).strip().lower() == "on"


def _tracking_gate_on(ctx: Any) -> bool:
    value = ctx.conf.get(config_mod.GATE_TRACKING_HISTORY, "off")
    return str(value).strip().lower() == "on"


# ---------------------------------------------------------------------------
# Package and venv (same procedure as Step 8, for this package's directory)
# ---------------------------------------------------------------------------


def bundled_version(ctx: Any) -> str:
    """`__version__` of the bundled package, read without importing it."""
    init = pathlib.Path(ctx.asset_path(ASSET_DIR, PACKAGE_NAME, "__init__.py"))
    match = _VERSION_RE.search(init.read_text(encoding="utf-8"))
    return match.group(1) if match else "unknown"


def sync_package(ctx: Any) -> bool:
    """Copy the bundled package and requirements into `<install_dir>/amc_control/`.

    Idempotent: returns False when the installed copy already matches the
    bundle file for file, otherwise replaces it and returns True. Files
    beside the package (the venv, `amc_automation.yaml`, `state/`) are never
    touched.
    """
    source = pathlib.Path(ctx.asset_path(ASSET_DIR, PACKAGE_NAME))
    requirements = pathlib.Path(ctx.asset_path(ASSET_DIR, "requirements.txt"))
    control_dir = _control_dir(ctx)
    dest = control_dir / PACKAGE_NAME
    dest_requirements = control_dir / "requirements.txt"

    control_dir.mkdir(parents=True, exist_ok=True)
    changed = False

    if _tree_digest(source) != _tree_digest(dest):
        staging = control_dir / f".{PACKAGE_NAME}.new"
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

    _chown(ctx, control_dir)
    _chown_tree(ctx, dest)
    _chown(ctx, dest_requirements)
    return changed


def ensure_venv(ctx: Any) -> tuple:
    """Build the venv and install the pinned requirements as the invoking user.

    Returns `(changed, error)`. pip runs only when the venv is new or the
    requirements file changed since the last successful install.
    """
    venv_dir = _venv_dir(ctx)
    python = _venv_python(ctx)
    requirements = _control_dir(ctx) / "requirements.txt"
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
# Settings file and state directory
# ---------------------------------------------------------------------------


def ensure_state_dir(ctx: Any) -> None:
    """Where the service keeps its launcher log and automation progress."""
    state_dir = _state_dir(ctx)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_dir.chmod(STATE_DIR_MODE)
    _chown(ctx, state_dir)


def seed_config_file(ctx: Any) -> tuple:
    """Write `amc_automation.yaml` once. Returns `(written, error)`.

    An existing path, a dangling symlink included, is left alone: the file
    is operator-managed from then on.
    """
    path = _config_file(ctx)
    if os.path.lexists(path):
        return False, None
    content = CONFIG_SEED.format(control_dir=_control_dir(ctx))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(path), flags, CONFIG_FILE_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            os.fchmod(f.fileno(), CONFIG_FILE_MODE)  # regardless of umask
            f.write(content)
    except FileExistsError:
        return False, None
    except OSError as exc:
        return False, f"could not write {path}: {exc}"
    _chown(ctx, path)
    log.info(f"wrote {path} (automation overrides; project.mode: create_new)")
    return True, None


# ---------------------------------------------------------------------------
# Tailscale address
# ---------------------------------------------------------------------------


def tailscale_ipv4(ctx: Any) -> Optional[str]:
    """This machine's Tailscale IPv4, or None. Never raises."""
    if not _WHICH("tailscale"):
        return None
    try:
        result = ctx.run_root(
            "tailscale",
            "ip",
            "-4",
            capture_output=True,
            text=True,
            stream=False,
            timeout=TAILSCALE_TIMEOUT_S,
        )
    except Exception:  # noqa: BLE001 -- a probe, never fatal
        return None
    if getattr(result, "returncode", 1) != 0:
        return None
    for line in (getattr(result, "stdout", "") or "").splitlines():
        try:
            return str(ipaddress.IPv4Address(line.strip()))
        except ValueError:
            continue
    return None


def _no_tailscale_result() -> StepResult:
    return StepResult(
        status=StepStatus.USER_ACTION_REQUIRED,
        message="tailscale has no IPv4 address; AMC control needs it for the "
        "address remote browsers use and for the sudo rule",
        user_actions=[
            UserAction(text="Log this machine in to the tailnet", command="sudo tailscale up")
        ],
    )


# ---------------------------------------------------------------------------
# sudoers rule
# ---------------------------------------------------------------------------


def render_sudoers(ctx: Any, host_ip: str) -> str:
    """The rule letting the service user start and stop AMC, and nothing else.

    Raises `ValueError` for a user name, install path or address that cannot
    be written into a sudoers command safely.
    """
    user = str(ctx.user.name)
    amc = str(_amc_bin(ctx))
    host = str(ipaddress.IPv4Address(host_ip))  # ValueError if not an IPv4
    if not _SUDOERS_USER_RE.match(user):
        raise ValueError(f"user name {user!r} cannot be written into a sudoers rule")
    if not _SUDOERS_PATH_RE.match(amc):
        raise ValueError(
            f"{amc} contains characters that cannot be written into a sudoers "
            "rule (use an install directory of letters, digits, _ . + and -)"
        )
    start = " ".join((amc, *AMC_START_ARGS, host))
    stop = " ".join((amc, *AMC_STOP_ARGS))
    return (
        f"# {_sudoers_path()} -- generated by mv3dt-installer (STEP-9 section 5).\n"
        "# Lets the AMC control service start and stop AutoMagicCalib, and\n"
        "# nothing else. Do not edit: re-run the installer to change it.\n"
        f"{user} ALL=(root) NOPASSWD: {start}, {stop}\n"
    )


def install_sudoers(ctx: Any, content: str) -> tuple:
    """Install the rule after `visudo -cf` accepts it. Returns `(changed, error)`.

    The candidate is written under a name containing a dot, which sudo
    ignores, checked, and only then renamed into place, so sudo never reads
    an unchecked or half-written rule.
    """
    path = _sudoers_path()
    try:
        current = path.read_text(encoding="utf-8")
        mode = stat_mod.S_IMODE(path.stat().st_mode)
    except OSError:
        current, mode = None, None
    if current == content and mode == SUDOERS_MODE:
        return False, None

    if not _WHICH("visudo"):
        return False, "visudo was not found; install the sudo package"
    tmp = path.with_name(f".{SUDOERS_NAME}.{secrets.token_hex(8)}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, SUDOERS_MODE)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            os.fchmod(f.fileno(), SUDOERS_MODE)
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        result = ctx.run_root(
            "visudo", "-cf", str(tmp), capture_output=True, text=True, stream=False
        )
        if getattr(result, "returncode", 1) != 0:
            return False, f"visudo rejected the sudo rule: {_tail(result, 300)}"
        os.replace(tmp, path)
    except OSError as exc:
        return False, f"could not write {path}: {exc}"
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
    log.info(f"wrote {path} (start and stop AMC only)")
    return True, None


# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------


def service_environment(ctx: Any, host_ip: str) -> dict:
    """The unit's `Environment=` values, in unit order. Holds no secret."""
    return {
        "PYTHONPATH": str(_control_dir(ctx)),
        "MV3DT_INSTALL_DIR": str(ctx.install_dir),
        "AMC_STATE_DIR": str(_state_dir(ctx)),
        "AMC_AUTOMATION_CONFIG": str(_config_file(ctx)),
        "AMC_PUBLIC_HOST": host_ip,
        "BIND": BIND,
        "PYTHONUNBUFFERED": "1",
    }


def render_amc_control_unit(ctx: Any, host_ip: str) -> str:
    values = {
        "INSTALL_DIR": str(ctx.install_dir),
        "CONTROL_DIR": str(_control_dir(ctx)),
        "VENV_PYTHON": str(_venv_python(ctx)),
    }
    environment = service_environment(ctx, host_ip)
    for label, value in (*values.items(), *environment.items()):
        systemd._reject_unquotable(label, value)
    substitutions = {k: systemd._escape_specifiers(v) for k, v in values.items()}
    substitutions["USER"] = ctx.user.name
    substitutions["SERVICE_ENVIRONMENT"] = "\n".join(
        f'Environment="{k}={systemd._escape_specifiers(v)}"' for k, v in environment.items()
    )
    return systemd.render_unit(
        (systemd.TEMPLATE_DIR, "mv3dt-amc-control.service.in"), substitutions
    )


def _install_unit(ctx: Any, host_ip: str, *, restart: bool) -> bool:
    runner = _runner_from_ctx(ctx)
    changed = systemd.install_unit(
        UNIT_NAME,
        render_amc_control_unit(ctx, host_ip),
        unit_dir=systemd.UNIT_DIR,
        runner=runner,
    )
    systemd.daemon_reload(runner=runner)
    systemd.enable_now(UNIT_NAME, runner=runner)
    if changed or restart:
        runner(["systemctl", "restart", UNIT_NAME], check=False)
    return changed


def update_reason(ctx: Any) -> Optional[str]:
    """Why a completed install is out of date with this release, or None.

    Read only. Compares the bundled package and requirements with the
    installed copies and, when `ctx` carries the invoking user and tailscale
    reports an address, the unit and sudo rule this release would write with
    the ones on disk (so a changed Tailscale address is an update too).
    Never looks at `amc_automation.yaml`, `state/` or `tracking.env`.
    """
    control_dir = _control_dir(ctx)
    source = pathlib.Path(ctx.asset_path(ASSET_DIR, PACKAGE_NAME))
    if _tree_digest(source) != _tree_digest(control_dir / PACKAGE_NAME):
        return "bundled AMC control package changed"
    requirements = pathlib.Path(ctx.asset_path(ASSET_DIR, "requirements.txt"))
    try:
        installed = (control_dir / "requirements.txt").read_bytes()
    except OSError:
        installed = None
    if installed != requirements.read_bytes():
        return "bundled AMC control requirements changed"
    if getattr(ctx, "user", None) is None:
        return None
    host_ip = tailscale_ipv4(ctx)
    if host_ip is None:
        return None
    try:
        current = (systemd.UNIT_DIR / UNIT_NAME).read_text(encoding="utf-8")
    except OSError:
        current = None
    if current != render_amc_control_unit(ctx, host_ip):
        return f"{UNIT_NAME} differs from this release"
    try:
        rule = _sudoers_path().read_text(encoding="utf-8")
    except OSError:
        rule = None
    try:
        wanted = render_sudoers(ctx, host_ip)
    except ValueError:
        return None
    if rule != wanted:
        return f"{_sudoers_path()} differs from this release"
    return None


# ---------------------------------------------------------------------------
# Tailnet exposure and health
# ---------------------------------------------------------------------------


def _expose(ctx: Any) -> bool:
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


# ---------------------------------------------------------------------------
# The Step protocol implementation
# ---------------------------------------------------------------------------


def _missing_tracking_env(ctx: Any) -> list:
    values = _read_env_file(_tracking_env_path(ctx))
    return [key for key in REQUIRED_ENV_KEYS if not values.get(key)]


def _needs_step8(ctx: Any, missing: list) -> StepResult:
    """`missing` is empty when the file is complete but the Step 8 gate is off."""
    path = _tracking_env_path(ctx)
    detail = (
        f"{', '.join(missing)} is not set there"
        if missing
        else "tracking history (Step 8) is turned off"
    )
    return StepResult(
        status=StepStatus.USER_ACTION_REQUIRED,
        message=f"AMC control shares {path} with tracking history, and {detail}",
        user_actions=[
            UserAction(
                text="Re-run the installer with --tracking-history on and complete "
                "Step 8 first; it creates the ticket secret both services use",
                path=str(path),
            )
        ],
    )


class Step9AmcControl:
    id = STEP_ID
    title = "AMC control"

    phases = (
        "package and venv",
        "settings and state",
        "sudo rule",
        "AMC control unit",
        "tailnet exposure",
    )
    order = 9

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

        if not _amc_bin(ctx).is_file():
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message=f"{_amc_bin(ctx)} is missing; there is no AMC to control",
                user_actions=[UserAction(text="Complete Step 3 (AMC launcher) first")],
            )

        missing = _missing_tracking_env(ctx)
        if missing or not _tracking_gate_on(ctx):
            return _needs_step8(ctx, missing)

        if not _WHICH("visudo"):
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message="visudo is not installed; the sudo rule cannot be checked",
                user_actions=[
                    UserAction(text="Install sudo", command="sudo apt-get install -y sudo")
                ],
            )

        host_ip = tailscale_ipv4(ctx)
        if host_ip is None:
            return _no_tailscale_result()
        try:
            render_sudoers(ctx, host_ip)
        except ValueError as exc:
            return StepResult(status=StepStatus.FAILED, message=str(exc))
        return StepResult(status=StepStatus.COMPLETE)

    def run(self, ctx: Any) -> StepResult:
        if not _gate_on(ctx):
            return StepResult(status=StepStatus.COMPLETE)

        # An update of a completed install always restarts the unit: the
        # files may have been replaced by an attempt that stopped early.
        updating = bool(getattr(ctx, "update_pending", None))

        missing = _missing_tracking_env(ctx)
        if missing:
            return _needs_step8(ctx, missing)
        host_ip = tailscale_ipv4(ctx)
        if host_ip is None:
            return _no_tailscale_result()

        ctx.progress.phase(1)
        ctx.progress.task("copying the AMC control package")
        package_changed = sync_package(ctx)
        ctx.progress.task("building the AMC control venv")
        venv_changed, error = ensure_venv(ctx)
        if error:
            return StepResult(status=StepStatus.FAILED, message=error)

        ctx.progress.phase(2)
        ctx.progress.task(f"creating the state directory and {CONFIG_FILE_NAME}")
        ensure_state_dir(ctx)
        _, seed_error = seed_config_file(ctx)
        if seed_error:
            log.warn(
                f"step9: {_config_file(ctx)} was not seeded ({seed_error}); the "
                "service uses the packaged defaults until the file exists"
            )

        ctx.progress.phase(3)
        ctx.progress.task(f"installing {_sudoers_path()}")
        try:
            rule = render_sudoers(ctx, host_ip)
        except ValueError as exc:
            return StepResult(status=StepStatus.FAILED, message=str(exc))
        sudoers_changed, error = install_sudoers(ctx, rule)
        if error:
            return StepResult(status=StepStatus.FAILED, message=error)

        ctx.progress.phase(4)
        ctx.progress.task(f"installing {UNIT_NAME}")
        unit_changed = _install_unit(
            ctx, host_ip, restart=package_changed or venv_changed or updating
        )

        ctx.progress.phase(5)
        ctx.progress.task("exposing the API on the tailnet")
        _expose(ctx)

        version = bundled_version(ctx)
        if package_changed or unit_changed or venv_changed or sudoers_changed or updating:
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
                        command="journalctl -u mv3dt-amc-control -n 50",
                    )
                ],
            )

        path = _sudoers_path()
        try:
            mode = stat_mod.S_IMODE(path.stat().st_mode)
            rule = path.read_text(encoding="utf-8")
        except OSError:
            mode, rule = None, None
        if mode != SUDOERS_MODE:
            return StepResult(
                status=StepStatus.FAILED,
                message=f"{path} must be mode 440 "
                f"(found {oct(mode) if mode is not None else 'missing'})",
            )
        host_ip = tailscale_ipv4(ctx)
        if host_ip is not None and rule != render_sudoers(ctx, host_ip):
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message=f"{path} does not match this machine's Tailscale address "
                f"({host_ip}); the service could not start AMC",
                user_actions=[UserAction(text="Re-run the installer to rewrite the rule")],
            )

        doc = _wait_for_health()
        if doc is None:
            return StepResult(
                status=StepStatus.FAILED,
                message=f"GET {HEALTH_URL} did not return 200; see "
                "journalctl -u mv3dt-amc-control",
            )
        if doc.get("ticket_configured") is False:
            return _needs_step8(ctx, [KEY_SECRET])
        return StepResult(status=StepStatus.COMPLETE)

    def pending_update(self, ctx: Any) -> Optional[str]:
        """The dispatch loop's upgrade check for a completed step: a reason
        to re-run `run()`, or None. A gated-off install is never updated."""
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
        _available, logged_in, dns = tailscale_state(ctx)
        url = (
            f"https://{dns}:{TAILNET_HTTPS_PORT}"
            if logged_in and dns
            else f"https://<this machine's tailnet name>:{TAILNET_HTTPS_PORT}"
        )
        control_dir = _control_dir(ctx)

        log.info(f"AMC control: {UNIT_NAME} {enabled}/{active}")
        log.info(f"  automation settings: {_config_file(ctx)}")
        log.info("  logs: journalctl -u mv3dt-amc-control")
        log.info(
            "  AMC's API description: "
            f"cd {control_dir} && venv/bin/python3 -m {PACKAGE_NAME} openapi"
        )
        log.info(f"  web app setup: set WORKSTATION_BASE_URL={url}")
        log.info(
            f"  web app setup: {KEY_SECRET} is the tracking service's value "
            f"(sudo cat {_tracking_env_path(ctx)}); {KEY_PROJECT} and "
            f"{KEY_ORIGINS} there apply to AMC control too"
        )


register(Step9AmcControl())
