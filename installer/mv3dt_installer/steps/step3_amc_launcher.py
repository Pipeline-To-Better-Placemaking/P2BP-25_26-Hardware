"""Step 3 -- AutoMagicCalib (AMC) launcher.

Implements `installer/plan/STEP-3-AMC-LAUNCHER.md` against the framework
contract in `installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md` (step-module
interface section 12.1, `StepResult`/`StepStatus` section 12.2, `Context`
section 12.3, logging/reporting section 8, privilege/USER-ACTION section 9,
NGC key handoff section 10, install-location section 11) and the framework's
subcommand dispatch extension it flags in its own section 6.2, built in
`app.py` as `SUBCOMMAND_REGISTRY`/`register_subcommand`.

Scope: provision Docker and the NVIDIA runtime, bring up the equality-pinned
standalone NVIDIA AutoMagicCalib (AMC) 3.2.1 stack, establish its durable
project identity, open the localhost UI as the invoking user, and hold the
service up until the operator closes the dedicated AMC browser window or
signals done. `run()` always drops the durable `<install_dir>/bin/amc` wrapper
and only offers an immediate launch; declining is not a failure.

`launch_amc(...)` is the one routine `run()`'s optional immediate launch,
the registered `amc` subcommand handler, and (in a later, out-of-scope unit)
Step 5's per-project re-run entry point all call, so behavior is identical
everywhere (section 2's closing paragraph).

Two judgment calls this module makes, documented here since the spec leaves
them open:

- **Root for the standalone `amc` subcommand.** Section 3.1 describes a
  docker-group first-run caveat for an operator running `<install_dir>/bin
  /amc` without root, with a suggested `--sudo` re-exec or a caveat message.
  This module's framework-level bootstrap (`app._bootstrap_subcommand_context`,
  a sibling unit's decision, not this module's) already calls
  `privilege.require_root()` for *every* subcommand, matching doc 00
  section 9.1's blanket "the installer must run as root" -- so by the time
  `handle_amc_subcommand` runs, the process is always root and the
  docker-group caveat's premise (a non-root invocation reaching this far)
  never actually occurs. `--sudo` is still accepted as a recognized,
  currently-inert flag (never raises "unrecognized argument") rather than
  silently dropped, in case a future relaxation of the bootstrap's root
  requirement resurrects the scenario section 3.1 describes.
- **"Step 2 complete" without a state-machine handle on `Context`.**
  Doc 00 section 12.3's `Context` carries no accessor for another step's
  recorded `state.json` status (Step 2 itself does not depend on Step 1's
  status that way either -- it re-verifies Step 1's pins directly, see
  `step2_deepstream_sdk._check_prereq_pins`). This module follows the same
  pattern: `step2_deepstream_sdk.CONF_METHOD_KEY` is the durable signal
  `step2_deepstream_sdk.run()` writes into `installer.conf` the moment it
  resolves an install method, so `preflight()` checks that key's presence
  in `ctx.conf` rather than inventing a new `Context` capability.

Every subprocess call goes through `ctx.run_root`/`ctx.run_as_user` (per doc
00 section 9.2: docker, git, and anything touching the invoking user's home
-- the AMC clone under `$HOME/auto-magic-calib` -- run via
`ctx.run_as_user`; the one root-only operation, `chown 1000:1000` on
`projects/`/`models/`, runs via `ctx.run_root`), so no test here shells out
to a real docker/git/curl or opens a browser.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import pathlib
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

from mv3dt_installer import app as app_mod
from mv3dt_installer import config as config_mod
from mv3dt_installer import progress_exec
from mv3dt_installer.steps import step2_deepstream_sdk as step2_mod
from mv3dt_installer.steps import StepResult, StepStatus, register

if TYPE_CHECKING:  # pragma: no cover -- import-time only, never at runtime.
    from mv3dt_installer.app import Context

__all__ = [
    "AmcConfig",
    "resolve_config",
    "repo_root",
    "check_repo_isolation",
    "locate_compose_dir",
    "ENV_KEYS",
    "check_env_drift",
    "render_env",
    "write_env_atomic",
    "render_amc_wrapper",
    "write_amc_wrapper",
    "decide_hold_strategy",
    "resolved_amc_commit",
    "launch_amc",
    "teardown_amc",
    "handle_amc_subcommand",
    "Step3AmcLauncher",
]

# ---------------------------------------------------------------------------
# STEP-3 section 2 -- standalone AMC 3.2.1 equality pin.
# ---------------------------------------------------------------------------

AMC_REPO_URL = "https://github.com/NVIDIA-AI-IOT/auto-magic-calib.git"
AMC_COMMIT = "0cfd2b790fd77598b0543340a65c2a0e1d192327"
AMC_VERSION = "3.2.1"

DEFAULT_UI_PORT = "5000"
DEFAULT_MS_PORT = "8000"
DEFAULT_PROJECT_NAME = "default"
MIN_COMPOSE_VERSION = (2, 20, 3)
GPU_TEST_IMAGE = "ubuntu:24.04"

# installer.conf keys (section 4.1's table), mirrored so a "run later" via
# the `amc` exe resolves the same config the installer itself would.
CONF_AMC_ROOT_KEY = "AMC_ROOT"
CONF_HOST_IP_KEY = "HOST_IP"
CONF_UI_PORT_KEY = "AUTO_MAGIC_CALIB_UI_PORT"
CONF_MS_PORT_KEY = "AUTO_MAGIC_CALIB_MS_PORT"
CONF_MS_API_URL_KEY = "AUTO_MAGIC_CALIB_MS_API_URL"
CONF_PROJECT_NAME_KEY = "PROJECT_NAME"
CONF_LOCATION_ID_KEY = "LOCATION_ID"
CONF_AMC_PROJECT_ID_KEY = "AMC_PROJECT_ID"

_STEP3_CONF_KEYS: tuple[str, ...] = (
    CONF_AMC_ROOT_KEY,
    CONF_HOST_IP_KEY,
    CONF_UI_PORT_KEY,
    CONF_MS_PORT_KEY,
    CONF_PROJECT_NAME_KEY,
    CONF_LOCATION_ID_KEY,
)

# section 4.2/4.3 -- the compose/.env key set, verbatim (not the same
# spelling as the installer.conf keys above for HOST_IP/PROJECT_DIR, since
# .env additionally derives PROJECT_DIR/MODEL_DIR from AMC_ROOT).
ENV_KEYS: tuple[str, ...] = (
    "HOST_IP",
    "AUTO_MAGIC_CALIB_MS_PORT",
    "AUTO_MAGIC_CALIB_UI_PORT",
    "PROJECT_DIR",
    "MODEL_DIR",
)

AMC_WRAPPER_NAME = "amc"
INSTALLER_BIN_NAME = "mv3dt-installer"

# A first launch downloads several GB of models (geocalib, vggt, moge, ...)
# before the microservice answers /v1/ready, so the backend wait is bounded
# by a generous cap and cut short only when the ms container stops running.
_SERVICE_WAIT_TIMEOUT_S = 1800.0
_UI_WAIT_POLL_S = 1.0
_BACKEND_LOG_INTERVAL_S = 10.0
_MS_SERVICE = "auto-magic-calib-ms"

_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,49}$")


class AmcLaunchError(RuntimeError):
    """Expected provisioning failure with safe, operator-facing evidence."""

# section 5.1 -- browser candidates, in the order the doc lists them.
_CHROMIUM_FAMILY: tuple[str, ...] = (
    "google-chrome",
    "chromium",
    "chromium-browser",
    "microsoft-edge",
)


# ---------------------------------------------------------------------------
# section 4.1 -- config resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AmcConfig:
    amc_root: pathlib.Path
    host_ip: str
    ui_port: str
    ms_port: str
    ms_api_url: str
    project_name: str


_HOST_IP_RE = re.compile(r"\bsrc\s+(\d{1,3}(?:\.\d{1,3}){3})\b")


def _detect_host_ip(ctx: "Context") -> Optional[str]:
    """`ip route get 1.1.1.1`'s `src <ip>` field, or `None` on any failure.

    Best-effort only -- `resolve_config` falls back to `127.0.0.1` (a safe
    default for the localhost-only flow, section 4.1) when this can't
    determine anything.
    """
    result = ctx.run_root(
        "ip", "route", "get", "1.1.1.1", check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    match = _HOST_IP_RE.search(result.stdout or "")
    return match.group(1) if match else None


def resolve_config(
    ctx: "Context",
    *,
    project: Optional[str] = None,
    host_ip_override: Optional[str] = None,
) -> AmcConfig:
    """Resolve Step 3's config (section 4.1): `installer.conf` first, then
    auto-detection/defaults, with explicit per-call overrides (`--project`,
    `--host-ip`) winning over everything.
    """
    conf = ctx.conf

    amc_root_value = conf.get(CONF_AMC_ROOT_KEY) or str(
        pathlib.Path(ctx.user.home) / "auto-magic-calib"
    )
    amc_root = pathlib.Path(amc_root_value).expanduser()

    host_ip = (
        host_ip_override
        or conf.get(CONF_HOST_IP_KEY)
        or _detect_host_ip(ctx)
        or "127.0.0.1"
    )

    return AmcConfig(
        amc_root=amc_root,
        host_ip=host_ip,
        ui_port=conf.get(CONF_UI_PORT_KEY) or DEFAULT_UI_PORT,
        ms_port=conf.get(CONF_MS_PORT_KEY) or DEFAULT_MS_PORT,
        ms_api_url=conf.get(CONF_MS_API_URL_KEY) or "",
        project_name=project or conf.get(CONF_PROJECT_NAME_KEY) or DEFAULT_PROJECT_NAME,
    )


def persist_config(ctx: "Context", cfg: AmcConfig) -> None:
    """Seed the section 4.1 keys into `installer.conf`, once -- so "run
    later" via the `amc` exe always resolves the same config the installer
    itself just did, without clobbering a value an operator hand-edited or
    a later `--host-ip` override chose."""
    values = {
        CONF_AMC_ROOT_KEY: str(cfg.amc_root),
        CONF_HOST_IP_KEY: cfg.host_ip,
        CONF_UI_PORT_KEY: cfg.ui_port,
        CONF_MS_PORT_KEY: cfg.ms_port,
        CONF_PROJECT_NAME_KEY: cfg.project_name,
    }
    for key, value in values.items():
        if key not in ctx.conf:
            config_mod.persist_value(ctx.install_dir, key, value)
            ctx.conf[key] = value


def _persist(ctx: "Context", key: str, value: str) -> None:
    """Persist one resolved Step 3 value and keep the in-memory view current."""
    if ctx.conf.get(key) == value:
        return
    config_mod.persist_value(ctx.install_dir, key, value)
    ctx.conf[key] = value


def _validated_identity(value: str, label: str) -> str:
    value = value.strip()
    if not _IDENTITY_RE.fullmatch(value):
        raise AmcLaunchError(
            f"{label} must be 3-50 characters using letters, numbers, '.', '_', "
            "or '-', and must start with a letter or number"
        )
    return value


def resolve_project_identity(
    ctx: "Context",
    *,
    project: Optional[str] = None,
    location_id: Optional[str] = None,
) -> tuple[str, str]:
    """Resolve and persist the site identity before AMC is launched.

    Interactive installs ask only for missing values. Automation must provide
    both values through persisted configuration or explicit subcommand flags.
    PROJECT_NAME defaults to LOCATION_ID so the names cannot silently drift.
    """
    location = location_id or ctx.conf.get(CONF_LOCATION_ID_KEY)
    if not location:
        if ctx.non_interactive:
            raise AmcLaunchError(
                "LOCATION_ID is required in installer.conf or via --location-id "
                "for a non-interactive AMC launch"
            )
        location = _INPUT("Location ID (3-50 letters, numbers, ._-): ")
    location = _validated_identity(location, CONF_LOCATION_ID_KEY)

    project_name = project or ctx.conf.get(CONF_PROJECT_NAME_KEY)
    if not project_name:
        if ctx.non_interactive:
            project_name = location
        else:
            answer = _INPUT(f"AMC project name [{location}]: ").strip()
            project_name = answer or location
    project_name = _validated_identity(project_name, CONF_PROJECT_NAME_KEY)

    _persist(ctx, CONF_LOCATION_ID_KEY, location)
    _persist(ctx, CONF_PROJECT_NAME_KEY, project_name)
    return location, project_name


# ---------------------------------------------------------------------------
# section 4 step 2 -- repo-isolation guard
# ---------------------------------------------------------------------------


def repo_root() -> Optional[pathlib.Path]:
    """Best-effort locate this checkout's git root by walking up from this
    file. Returns `None` when not inside a git working tree -- the frozen
    binary's own case, since nothing clones this repo onto a workstation
    (doc 00 section 5) -- in which case there is no repo tree to guard
    against and the isolation check is trivially satisfied.
    """
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists():
            return parent
    return None


def check_repo_isolation(amc_root: pathlib.Path) -> Optional[str]:
    """Port of `30_start_amc.sh`'s `REPO_ROOT` guard (section 4 step 2):
    refuse an `AMC_ROOT` that is this repo's working tree, or a child of
    it. Returns an error message when the guard trips, else `None`.
    """
    root = repo_root()
    if root is None:
        return None
    resolved_root = root.resolve()
    resolved_amc = pathlib.Path(amc_root).expanduser()
    try:
        resolved_amc = resolved_amc.resolve()
    except OSError:
        pass
    if resolved_amc == resolved_root or resolved_root in resolved_amc.parents:
        return (
            f"AMC_ROOT ({amc_root}) must not live under this repo ({root}). "
            "Choose a different path (e.g. $HOME/auto-magic-calib)."
        )
    return None


# ---------------------------------------------------------------------------
# section 2 -- pinned standalone checkout
# ---------------------------------------------------------------------------


def _result_detail(result: subprocess.CompletedProcess, limit: int = 1600) -> str:
    detail = ((result.stderr or "") + "\n" + (result.stdout or "")).strip()
    return detail[-limit:] or "no command output"


def _run_git(ctx: "Context", *args: str) -> subprocess.CompletedProcess:
    result = ctx.run_as_user(
        "git", *args, check=False, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise AmcLaunchError(
            f"git {' '.join(args)} failed (exit {result.returncode}): "
            f"{_result_detail(result)}"
        )
    return result


def _normalise_repo_url(value: str) -> str:
    return value.strip().removesuffix("/").removesuffix(".git")


def _expected_amc_worktree_change(line: str) -> bool:
    """Allow installer-managed config and untracked AMC runtime data only."""
    status = line[:2]
    path = line[3:].strip()
    if path == "compose/.env":
        return True
    return status == "??" and path.startswith(("projects/", "models/"))


def clone_amc(ctx: "Context", amc_root: pathlib.Path) -> bool:
    """Install or safely reconcile the standalone AMC checkout at its pin."""
    if not amc_root.exists():
        result = ctx.run_as_user(
            "git",
            "clone",
            "--progress",
            AMC_REPO_URL,
            str(amc_root),
            check=False,
            capture_output=True,
            text=True,
            stream=True,
        )
        if result.returncode != 0:
            raise AmcLaunchError(
                f"AMC clone failed (exit {result.returncode}); any partial path "
                f"was preserved at {amc_root}: {_result_detail(result)}"
            )
        _run_git(ctx, "-C", str(amc_root), "checkout", "--detach", AMC_COMMIT)
        return True

    if not amc_root.is_dir() or not (amc_root / ".git").exists():
        raise AmcLaunchError(
            f"AMC_ROOT exists but is not an AMC git checkout: {amc_root}. "
            "Move it aside or choose a different AMC_ROOT; no files were deleted."
        )

    remote = _run_git(
        ctx, "-C", str(amc_root), "remote", "get-url", "origin"
    ).stdout.strip()
    if _normalise_repo_url(remote) != _normalise_repo_url(AMC_REPO_URL):
        raise AmcLaunchError(
            f"AMC_ROOT origin is {remote!r}, expected {AMC_REPO_URL!r}; "
            "the installer will not overwrite this checkout"
        )

    head = resolved_amc_commit(ctx, amc_root)
    dirty_lines = _run_git(
        ctx, "-C", str(amc_root), "status", "--porcelain"
    ).stdout.splitlines()
    unexpected = [line for line in dirty_lines if not _expected_amc_worktree_change(line)]
    if unexpected:
        detail = ", ".join(line.strip() for line in unexpected[:5])
        raise AmcLaunchError(
            "AMC checkout has local changes outside the installer-managed "
            f"config/runtime paths ({detail}); commit or move those changes "
            "before retrying"
        )
    if head == AMC_COMMIT:
        return False
    _run_git(ctx, "-C", str(amc_root), "fetch", "origin", AMC_COMMIT)
    _run_git(ctx, "-C", str(amc_root), "checkout", "--detach", AMC_COMMIT)
    if resolved_amc_commit(ctx, amc_root) != AMC_COMMIT:
        raise AmcLaunchError("AMC checkout did not resolve to the required commit")
    return False


def resolved_amc_commit(ctx: "Context", amc_root: pathlib.Path) -> Optional[str]:
    """Return the AMC checkout's resolved commit, or ``None`` on failure.

    ``verify()`` compares a returned value with ``AMC_COMMIT``. A missing
    checkout remains valid before the operator accepts the optional launch.
    """
    result = ctx.run_as_user(
        "git",
        "-C",
        str(amc_root),
        "rev-parse",
        "HEAD",
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    commit = (result.stdout or "").strip()
    return commit or None


# ---------------------------------------------------------------------------
# section 4 step 4 -- projects/ + models/, chown 1000:1000
# ---------------------------------------------------------------------------


def ensure_projects_and_models(
    ctx: "Context", amc_root: pathlib.Path
) -> tuple[pathlib.Path, pathlib.Path]:
    """`mkdir -p projects/ models/` + `chown -R 1000:1000` (section 4 step
    4, Notion section 8.3's ownership tweak so the in-container UID 1000
    can write). Root-owned operation -- runs via `ctx.run_root`, unlike the
    clone/compose calls, which run as the invoking user (doc 00 section
    9.2's "as root this is a direct chown" case).
    """
    projects_dir = amc_root / "projects"
    models_dir = amc_root / "models"
    projects_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    ctx.run_root(
        "chown",
        "-R",
        "1000:1000",
        str(projects_dir),
        str(models_dir),
        check=False,
        capture_output=True,
        text=True,
    )
    return projects_dir, models_dir


# ---------------------------------------------------------------------------
# section 5 -- required `docker login nvcr.io`
# ---------------------------------------------------------------------------


def docker_login(ctx: "Context") -> None:
    """Authenticate to NGC without placing the API key in argv or logs."""
    secrets_path = ctx.install_dir / "secrets" / "ngc.env"
    if not secrets_path.is_file():
        raise AmcLaunchError(
            f"NGC credentials are missing at {secrets_path}; AMC images require "
            "an authenticated nvcr.io login"
        )

    script = (
        'set -a; . "$NGC_ENV_FILE"; set +a; '
        "echo \"$NGC_API_KEY\" | docker login nvcr.io "
        "-u '$oauthtoken' --password-stdin"
    )
    result = ctx.run_as_user(
        "env",
        f"NGC_ENV_FILE={secrets_path}",
        "bash",
        "-lc",
        script,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AmcLaunchError(
            f"docker login nvcr.io failed (exit {result.returncode}); child "
            "output omitted because the process handled NGC_API_KEY"
        )


# ---------------------------------------------------------------------------
# section 4 step 6 -- locate the compose dir (search-order fallback)
# ---------------------------------------------------------------------------


def locate_compose_dir(amc_root: pathlib.Path) -> Optional[pathlib.Path]:
    """Return the pinned standalone checkout's compose directory."""
    compose = amc_root / "compose"
    return compose if (compose / "compose.yml").is_file() else None


# ---------------------------------------------------------------------------
# section 4.3 -- upstream drift guard
# ---------------------------------------------------------------------------


def check_env_drift(compose_dir: pathlib.Path) -> list[str]:
    """Compare `ENV_KEYS` against `compose_dir/.env.example`. Returns the
    subset of keys no longer defined upstream (section 4.3); an empty list
    (including when `.env.example` itself is absent -- nothing to diff
    against) means no drift detected.
    """
    text = None
    for name in (".env.example", ".env"):
        try:
            text = (compose_dir / name).read_text(encoding="utf-8")
            break
        except OSError:
            pass
    if text is None:
        return []
    missing = []
    for key in ENV_KEYS:
        if not re.search(rf"(?m)^{re.escape(key)}=", text):
            missing.append(key)
    return missing


# ---------------------------------------------------------------------------
# section 4.2 -- compose/.env contents
# ---------------------------------------------------------------------------


def render_env(cfg: AmcConfig) -> str:
    """Render `compose/.env` exactly as `30_start_amc.sh` does (section
    4.2), including the optional `AUTO_MAGIC_CALIB_MS_API_URL` line only
    when set.
    """
    lines = [
        f"HOST_IP={cfg.host_ip}",
        f"AUTO_MAGIC_CALIB_MS_PORT={cfg.ms_port}",
        f"AUTO_MAGIC_CALIB_UI_PORT={cfg.ui_port}",
        f"PROJECT_DIR={cfg.amc_root / 'projects'}",
        f"MODEL_DIR={cfg.amc_root / 'models'}",
    ]
    if cfg.ms_api_url:
        lines.append(f"AUTO_MAGIC_CALIB_MS_API_URL={cfg.ms_api_url}")
    return "\n".join(lines) + "\n"


def write_env_atomic(compose_dir: pathlib.Path, content: str) -> bool:
    """Write `compose/.env` atomically (tmp + `os.replace`), content-
    idempotent like `systemd.install_unit`. Returns whether the file
    changed.
    """
    dest = compose_dir / ".env"
    if dest.is_file():
        try:
            existing = dest.read_text(encoding="utf-8")
        except OSError:
            existing = None
        if existing == content:
            return False

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, dest)
    return True


# ---------------------------------------------------------------------------
# section 4 step 8 -- pull + up / down
# ---------------------------------------------------------------------------


def _run_compose(ctx: "Context", compose_dir: pathlib.Path, *args: str):
    kwargs: dict[str, Any] = {}
    if args and args[0] in ("pull", "up"):
        kwargs["stream"] = True
    result = ctx.run_as_user(
        "docker",
        "compose",
        *args,
        cwd=str(compose_dir),
        check=False,
        capture_output=True,
        text=True,
        **kwargs,
    )
    if result.returncode != 0:
        raise AmcLaunchError(
            f"docker compose {' '.join(args)} failed (exit {result.returncode}): "
            f"{_result_detail(result)}"
        )
    return result


def compose_validate(ctx: "Context", compose_dir: pathlib.Path) -> None:
    _run_compose(ctx, compose_dir, "config", "--quiet")


def compose_pull(ctx: "Context", compose_dir: pathlib.Path) -> None:
    _run_compose(ctx, compose_dir, "pull")


def compose_up(ctx: "Context", compose_dir: pathlib.Path) -> None:
    _run_compose(ctx, compose_dir, "up", "-d")


def compose_down(ctx: "Context", compose_dir: pathlib.Path) -> None:
    _run_compose(ctx, compose_dir, "down")


def compose_stack_running(ctx: "Context", compose_dir: pathlib.Path) -> bool:
    """Return whether both pinned AMC services are already running."""
    result = ctx.run_as_user(
        "docker", "compose", "ps", "--status", "running", "--services",
        cwd=str(compose_dir), check=False, capture_output=True, text=True,
        stream=False,
    )
    services = set((result.stdout or "").splitlines())
    return result.returncode == 0 and {
        "auto-magic-calib-ms",
        "auto-magic-calib-ui",
    }.issubset(services)


def compose_published_port(
    ctx: "Context", compose_dir: pathlib.Path, service: str, container_port: int
) -> Optional[str]:
    """Resolve a running service's host port from Docker Compose."""
    result = ctx.run_as_user(
        "docker", "compose", "port", service, str(container_port),
        cwd=str(compose_dir), check=False, capture_output=True, text=True,
        stream=False,
    )
    if result.returncode != 0:
        return None
    for line in (result.stdout or "").splitlines():
        match = re.search(r":(\d+)\s*$", line)
        if match:
            return match.group(1)
    return None


# ---------------------------------------------------------------------------
# section 4 step 9 -- readiness poll
# ---------------------------------------------------------------------------


def _ms_service_running(ctx: "Context", compose_dir: pathlib.Path) -> bool:
    """Return whether the AMC microservice container is still running."""
    result = ctx.run_as_user(
        "docker", "compose", "ps", "--status", "running", "--services",
        cwd=str(compose_dir), check=False, capture_output=True, text=True,
        stream=False,
    )
    return result.returncode == 0 and _MS_SERVICE in (result.stdout or "").splitlines()


def _ms_last_log_line(ctx: "Context", compose_dir: pathlib.Path) -> Optional[str]:
    """Return the microservice's most recent non-blank log line, if any."""
    result = ctx.run_as_user(
        "docker", "compose", "logs", "--no-log-prefix", "--tail", "5", _MS_SERVICE,
        cwd=str(compose_dir), check=False, capture_output=True, text=True,
        stream=False,
    )
    if result.returncode != 0:
        return None
    lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
    return lines[-1] if lines else None


def wait_for_backend(
    ctx: "Context",
    url: str,
    *,
    compose_dir: Optional[pathlib.Path] = None,
    timeout_s: float = _SERVICE_WAIT_TIMEOUT_S,
    poll_s: float = _UI_WAIT_POLL_S,
    log_interval_s: float = _BACKEND_LOG_INTERVAL_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Wait until the AMC microservice returns JSON with ``code: 0``.

    With `compose_dir`, every `log_interval_s` the wait also echoes the
    microservice's latest log line to the live window, so first-launch model
    downloads read as progress rather than a hang, and gives up early if the
    ms container is no longer running (nothing left to wait for).
    """
    started = clock()
    last_check = started
    last_line: Optional[str] = None
    while True:
        result = ctx.run_root(
            "curl",
            "-fsS",
            "--max-time",
            "5",
            url,
            check=False,
            capture_output=True,
            text=True,
            stream=False,
        )
        if result.returncode == 0:
            try:
                if json.loads(result.stdout or "{}").get("code") == 0:
                    return True
            except (json.JSONDecodeError, AttributeError):
                pass
        now = clock()
        if now - started >= timeout_s:
            return False
        if compose_dir is not None and now - last_check >= log_interval_s:
            last_check = now
            if not _ms_service_running(ctx, compose_dir):
                ctx.log.warn(f"{_MS_SERVICE} stopped running before it became ready")
                return False
            line = _ms_last_log_line(ctx, compose_dir)
            if line is not None and line != last_line:
                last_line = line
                ctx.progress.line(line)
        ctx.progress.tick()
        sleep(poll_s)


def wait_for_ui(ctx: "Context", url: str) -> bool:
    """Require an exact HTTP 200 response from the UI."""
    result = ctx.run_root(
        "curl",
        "-sS",
        "-o",
        "/dev/null",
        "-w",
        "%{http_code}",
        "--max-time",
        "5",
        url,
        check=False,
        capture_output=True,
        text=True,
        stream=False,
    )
    return result.returncode == 0 and (result.stdout or "").strip() == "200"


def compose_diagnostics(ctx: "Context", compose_dir: pathlib.Path) -> str:
    """Capture bounded status and logs for a fatal readiness failure."""
    chunks = []
    for args in (("ps",), ("logs", "--tail", "80")):
        result = ctx.run_as_user(
            "docker", "compose", *args, cwd=str(compose_dir), check=False,
            capture_output=True, text=True,
        )
        chunks.append(f"docker compose {' '.join(args)}:\n{_result_detail(result, 3000)}")
    return "\n".join(chunks)


# ---------------------------------------------------------------------------
# section 5 -- keep-alive-until-browser-closed
# ---------------------------------------------------------------------------


class HoldStrategy(str, Enum):
    """Which of section 5.1/5.2's mechanisms `execute_hold` should use."""

    DEDICATED_WINDOW = "dedicated_window"
    XDG_OPEN_AND_PROMPT = "xdg_open_and_prompt"
    PRINT_URL_AND_PROMPT = "print_url_and_prompt"
    PRINT_URL_AND_LEAVE_UP = "print_url_and_leave_up"


def decide_hold_strategy(
    *, has_display: bool, browser_available: bool, non_interactive: bool
) -> HoldStrategy:
    """The section 5.1/5.2 decision table, as a pure function of three
    booleans -- the "wait/teardown state machine"'s decision half (the
    execution half, `execute_hold`, is a thin dispatcher over this).

    - Desktop session + a monitorable dedicated-window browser available:
      the primary path (section 5.1).
    - Desktop session, no monitorable browser, interactive: `xdg-open` +
      block on operator input (section 5.2's first bullet).
    - Headless (no `DISPLAY`/`WAYLAND_DISPLAY`), interactive: print the URL
      and block on operator input (section 5.2's second bullet).
    - Any of the above with `non_interactive=True` -- there is no browser
      to monitor and no human to prompt, so the only safe answer is to
      leave AMC running and print the stop command (section 5.2's
      `--non-interactive` bullet). This also covers the one combination
      the doc does not spell out explicitly -- a desktop session present
      but no monitorable browser, under `--non-interactive` -- by the same
      "never blindly tear down, never block on nobody" reasoning.
    """
    if has_display and browser_available:
        return HoldStrategy.DEDICATED_WINDOW
    if non_interactive:
        return HoldStrategy.PRINT_URL_AND_LEAVE_UP
    if has_display:
        return HoldStrategy.XDG_OPEN_AND_PROMPT
    return HoldStrategy.PRINT_URL_AND_PROMPT


def find_browser(*, which: Callable[[str], Optional[str]] = shutil.which) -> Optional[str]:
    """First available browser from the section 5.1 candidate list, in
    order, or `None` if none is on `PATH`. Firefox is checked last, as its
    own fallback within the "dedicated window" family."""
    for name in _CHROMIUM_FAMILY:
        if which(name):
            return name
    if which("firefox"):
        return "firefox"
    return None


def _browser_argv(browser: str, url: str, profile_dir: str) -> list[str]:
    if browser == "firefox":
        return ["firefox", "--new-instance", "--profile", profile_dir, url]
    return [
        browser,
        f"--app={url}",
        f"--user-data-dir={profile_dir}",
        "--no-first-run",
        "--no-default-browser-check",
    ]


def open_dedicated_window(
    ctx: "Context",
    url: str,
    *,
    popen: Optional[Callable[..., Any]] = None,
    which: Callable[[str], Optional[str]] = shutil.which,
    mkdtemp: Callable[..., str] = tempfile.mkdtemp,
) -> Optional[Any]:
    """Launch a brand-new browser *process* in a throwaway profile (section
    5.1) -- the load-bearing bit is the fresh `--user-data-dir`/`--profile`,
    which forces a new OS process the caller can `wait()` on rather than
    handing the URL to an already-running browser and exiting immediately.
    Returns the `Popen` handle, or `None` if no candidate browser is on
    `PATH`.
    """
    popen = popen or subprocess.Popen
    browser = find_browser(which=which)
    if browser is None:
        return None
    profile_dir = mkdtemp(prefix="mv3dt-amc-browser-")
    ownership = ctx.run_root(
        "chown", f"{ctx.user.uid}:{ctx.user.gid}", profile_dir,
        check=False, capture_output=True, text=True,
    )
    if ownership.returncode != 0:
        raise AmcLaunchError(
            f"could not hand browser profile to {ctx.user.name}: "
            f"{_result_detail(ownership)}"
        )
    argv = _browser_argv(browser, url, profile_dir)
    env_args = []
    for key in ("DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "DBUS_SESSION_BUS_ADDRESS"):
        if os.environ.get(key):
            env_args.append(f"{key}={os.environ[key]}")
    return popen(["sudo", "-u", ctx.user.name, "-H", "env", *env_args, *argv])


def _xdg_open(
    ctx: "Context", url: str, *, popen: Optional[Callable[..., Any]] = None
) -> None:
    popen = popen or subprocess.Popen
    try:
        popen(["sudo", "-u", ctx.user.name, "-H", "xdg-open", url])
    except OSError:
        pass


def _install_teardown_guards(teardown: Callable[[], None]) -> Callable[[], None]:
    """Install the section 5.1 step 3 fail-safe: `SIGINT`/`SIGTERM`/normal
    exit all run `teardown` exactly once (`docker compose down`)."""
    state = {"ran": False}

    def _run_once(*_args: Any) -> None:
        if not state["ran"]:
            state["ran"] = True
            teardown()

    atexit.register(_run_once)

    def _signal_handler(signum: int, _frame: Any) -> None:
        _run_once()
        sys.exit(128 + signum)

    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except ValueError:
        # Not the main thread / no signal support here (e.g. under a test
        # runner) -- atexit is still installed, which is the fail-safe that
        # matters most.
        pass

    return _run_once


def execute_hold(
    ctx: "Context",
    strategy: HoldStrategy,
    url: str,
    *,
    teardown: Callable[[], None],
    keep_up: bool = False,
    popen: Optional[Callable[..., Any]] = None,
    which: Callable[[str], Optional[str]] = shutil.which,
    prompt: Callable[[str], str] = input,
) -> None:
    """Execute the strategy `decide_hold_strategy` chose. `--keep-up`
    (section 5.1 step 3) short-circuits every strategy: AMC is left
    running and `teardown` is never called.

    `teardown` is expected to already be the run-once-guarded callable
    `_install_teardown_guards` returns -- section 5.1 step 3 requires the
    `SIGINT`/`SIGTERM`/`atexit` fail-safe installed *before* `docker
    compose up -d` runs, not merely before this function's own wait/prompt,
    so `launch_amc` installs the guard itself, earlier, and this function
    only ever calls the already-guarded `teardown` it was handed. Calling
    it directly here (rather than re-wrapping it) is what keeps a signal
    arriving during `proc.wait()`/the prompt and a normal fall-through both
    routed through the exact same run-once state.
    """
    if strategy is HoldStrategy.DEDICATED_WINDOW:
        proc = open_dedicated_window(ctx, url, popen=popen, which=which)
        if proc is not None:
            ctx.log.info(
                f"AutoMagicCalib is running at {url} -- close the AMC browser "
                "window when you're done; the service stays up until you do."
            )
            proc.wait()
            if keep_up:
                ctx.log.info(f"--keep-up set: leaving AMC running at {url}")
            else:
                teardown()
            return
        # No browser after all (race between decide_hold_strategy's check
        # and here, or a caller passed the strategy in directly) -- fall
        # through to the next-best interactive/headless behavior.
        strategy = HoldStrategy.PRINT_URL_AND_PROMPT

    if strategy is HoldStrategy.XDG_OPEN_AND_PROMPT:
        _xdg_open(ctx, url, popen=popen)
        try:
            prompt("Press Enter (or Ctrl-C) when you have closed AMC to shut it down.")
        except KeyboardInterrupt:
            pass
        if not keep_up:
            teardown()
        return

    if strategy is HoldStrategy.PRINT_URL_AND_PROMPT:
        ctx.log.info(f"Headless session detected; open {url} in your browser.")
        try:
            prompt("Press Enter (or Ctrl-C) when you have closed AMC to shut it down.")
        except KeyboardInterrupt:
            pass
        if not keep_up:
            teardown()
        return

    # PRINT_URL_AND_LEAVE_UP -- non-interactive with nothing to monitor and
    # nobody to prompt: never tear down blindly.
    ctx.log.info(
        f"AMC left running at {url} (non-interactive, nothing to monitor). "
        "Stop it later with `amc --down`."
    )


# ---------------------------------------------------------------------------
# section 6 -- the standalone `amc` executable
# ---------------------------------------------------------------------------


def ensure_installer_binary(ctx: "Context") -> pathlib.Path:
    """`<install_dir>/bin/mv3dt-installer` (section 6.1 step 1): a stable
    copy of the running executable, so the `amc` wrapper survives the
    operator's downloaded copy being moved or deleted. Copies from
    `sys.executable` only when frozen (`sys.frozen`, matching
    `mv3dt_installer/__init__.py`'s own gate for its build stamp) -- a dev
    checkout has no single binary to copy, so a pre-existing destination is
    left alone and a missing one is logged rather than fabricated.
    """
    bin_dir = ctx.install_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    dest = bin_dir / INSTALLER_BIN_NAME

    if getattr(sys, "frozen", False):
        src = pathlib.Path(sys.executable)
        if src.is_file() and (
            not dest.exists() or dest.stat().st_size != src.stat().st_size
        ):
            shutil.copy2(src, dest)
            dest.chmod(0o755)
    elif not dest.exists():
        ctx.log.warn(
            f"not a frozen binary; cannot stage {dest} from a source checkout "
            "(this is expected in dev/test -- a release build always runs frozen)"
        )

    return dest


def render_amc_wrapper(installer_bin: pathlib.Path) -> str:
    """Render the `amc` wrapper script (section 6.1 step 2, verbatim)."""
    return (
        "#!/usr/bin/env bash\n"
        "# Generated by mv3dt-installer Step 3. Brings up AutoMagicCalib, opens\n"
        "# the localhost UI, and holds it open until you close the AMC browser\n"
        "# window.\n"
        f'exec "{installer_bin}" amc "$@"\n'
    )


def write_amc_wrapper(
    ctx: "Context", installer_bin: pathlib.Path
) -> tuple[pathlib.Path, bool]:
    """Write `<install_dir>/bin/amc`, `chmod +x`, chowned to the invoking
    user (section 6.1 step 2). Returns `(path, changed)`."""
    bin_dir = ctx.install_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    dest = bin_dir / AMC_WRAPPER_NAME
    content = render_amc_wrapper(installer_bin)

    changed = True
    if dest.is_file():
        try:
            changed = dest.read_text(encoding="utf-8") != content
        except OSError:
            changed = True

    if changed:
        dest.write_text(content, encoding="utf-8")

    dest.chmod(0o755)
    try:
        os.chown(dest, ctx.user.uid, ctx.user.gid)
    except OSError:
        pass  # best-effort, e.g. under a non-root test process

    return dest, changed


# ---------------------------------------------------------------------------
# section 3 -- Docker + NVIDIA Container Toolkit prerequisite (assert-only)
# ---------------------------------------------------------------------------


def _docker_usable(ctx: "Context") -> bool:
    version = ctx.run_as_user(
        "docker", "--version", check=False, capture_output=True, text=True
    )
    if version.returncode != 0:
        return False
    info = ctx.run_as_user("docker", "info", check=False, capture_output=True, text=True)
    return info.returncode == 0


def _compose_version(ctx: "Context") -> Optional[tuple[int, int, int]]:
    result = ctx.run_as_user(
        "docker", "compose", "version", "--short", check=False,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    match = re.search(r"(?:v)?(\d+)\.(\d+)\.(\d+)", result.stdout or "")
    return tuple(map(int, match.groups())) if match else None


def _compose_available(ctx: "Context") -> bool:
    version = _compose_version(ctx)
    return version is not None and version >= MIN_COMPOSE_VERSION


def _nvidia_runtime_registered(ctx: "Context") -> bool:
    info = ctx.run_as_user("docker", "info", check=False, capture_output=True, text=True)
    return "nvidia" in (info.stdout or "").lower()


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _select_free_port(preferred: str, *, reserved: set[int]) -> str:
    try:
        start = int(preferred)
    except ValueError as exc:
        raise AmcLaunchError(f"invalid TCP port: {preferred!r}") from exc
    if not 1 <= start <= 65535:
        raise AmcLaunchError(f"invalid TCP port: {preferred!r}")
    for port in range(start, min(start + 100, 65536)):
        if port not in reserved and _port_is_free(port):
            reserved.add(port)
            return str(port)
    raise AmcLaunchError(f"no free TCP port found in {start}-{min(start + 99, 65535)}")


def resolve_ports(ctx: "Context", cfg: AmcConfig) -> AmcConfig:
    """Select and persist deterministic free ports, starting at configured values."""
    reserved: set[int] = set()
    ui_port = _select_free_port(cfg.ui_port, reserved=reserved)
    ms_port = _select_free_port(cfg.ms_port, reserved=reserved)
    if ui_port != cfg.ui_port:
        ctx.log.warn(f"AMC UI port {cfg.ui_port} is occupied; using {ui_port}")
    if ms_port != cfg.ms_port:
        ctx.log.warn(f"AMC microservice port {cfg.ms_port} is occupied; using {ms_port}")
    _persist(ctx, CONF_UI_PORT_KEY, ui_port)
    _persist(ctx, CONF_MS_PORT_KEY, ms_port)
    return AmcConfig(
        amc_root=cfg.amc_root, host_ip=cfg.host_ip, ui_port=ui_port,
        ms_port=ms_port, ms_api_url=cfg.ms_api_url,
        project_name=cfg.project_name,
    )


def _json_object(result: subprocess.CompletedProcess, label: str) -> dict[str, Any]:
    if result.returncode != 0:
        raise AmcLaunchError(
            f"{label} failed (exit {result.returncode}): {_result_detail(result)}"
        )
    try:
        payload = json.loads(result.stdout or "")
    except json.JSONDecodeError as exc:
        raise AmcLaunchError(f"{label} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise AmcLaunchError(f"{label} returned an unexpected JSON value")
    if "code" in payload and payload.get("code") != 0:
        message = payload.get("message") or payload.get("detail") or payload
        raise AmcLaunchError(f"{label} returned an error: {message}")
    return payload


def ensure_amc_project(ctx: "Context", cfg: AmcConfig) -> str:
    """Select the persisted AMC project or create it exactly once."""
    api = f"http://localhost:{cfg.ms_port}/v1"
    existing = (ctx.conf.get(CONF_AMC_PROJECT_ID_KEY) or "").strip()
    if existing:
        result = ctx.run_root(
            "curl", "-fsS", "--max-time", "10",
            f"{api}/get_project_info/{existing}", check=False,
            capture_output=True, text=True,
        )
        payload = _json_object(result, f"AMC project {existing} lookup")
        info = payload.get("project_info") or payload
        actual_name = info.get("project_name") if isinstance(info, dict) else None
        if actual_name and actual_name != cfg.project_name:
            raise AmcLaunchError(
                f"persisted AMC_PROJECT_ID {existing} belongs to {actual_name!r}, "
                f"not configured PROJECT_NAME {cfg.project_name!r}"
            )
        return existing

    result = ctx.run_root(
        "curl", "-fsS", "--max-time", "15", "-X", "POST",
        "--data-urlencode", f"project_name={cfg.project_name}",
        f"{api}/create_project", check=False, capture_output=True, text=True,
    )
    payload = _json_object(result, f"AMC project {cfg.project_name!r} creation")
    project_id = str(payload.get("project_id") or "").strip()
    if not project_id:
        raise AmcLaunchError(
            "AMC create_project response did not include project_id; the installer "
            "will not guess among existing projects"
        )
    _persist(ctx, CONF_AMC_PROJECT_ID_KEY, project_id)
    return project_id


def _run_required(ctx: "Context", label: str, *args: str):
    result = ctx.run_root(*args, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise AmcLaunchError(
            f"{label} failed (exit {result.returncode}): {_result_detail(result)}"
        )
    return result


def _package_installed(ctx: "Context", package: str) -> bool:
    result = ctx.run_root(
        "dpkg-query", "-W", "-f=${Status}", package,
        check=False, capture_output=True, text=True,
    )
    return result.returncode == 0 and "install ok installed" in (result.stdout or "")


def _apt_install(ctx: "Context", *packages: str) -> None:
    result = progress_exec.apt(
        ctx, "install", "-y", "--no-install-recommends", *packages,
        check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise AmcLaunchError(
            f"apt-get install {' '.join(packages)} failed (exit "
            f"{result.returncode}): {_result_detail(result)}"
        )


def ensure_container_prerequisites(ctx: "Context") -> None:
    """Idempotently provision Docker, Compose v2 and NVIDIA runtime."""
    ctx.progress.task("Docker and NVIDIA container prerequisites")
    base_packages = ("docker.io", "docker-compose-v2", "git", "curl", "ca-certificates", "gnupg")
    missing = tuple(pkg for pkg in base_packages if not _package_installed(ctx, pkg))
    if missing:
        ctx.progress.task("installing Docker and Compose prerequisites")
        update = progress_exec.apt(
            ctx, "update", check=False, capture_output=True, text=True
        )
        if update.returncode != 0:
            raise AmcLaunchError(
                f"apt-get update failed (exit {update.returncode}): "
                f"{_result_detail(update)}"
            )
        _apt_install(ctx, *missing)

    if not _package_installed(ctx, "nvidia-container-toolkit"):
        ctx.progress.task("installing NVIDIA Container Toolkit")
        key_fd, key_path = tempfile.mkstemp(prefix="mv3dt-nvidia-key-", dir="/tmp")
        os.close(key_fd)
        try:
            _run_required(
                ctx, "NVIDIA toolkit key download", "curl", "-fsSL",
                "-o", key_path, "https://nvidia.github.io/libnvidia-container/gpgkey",
            )
            _run_required(
                ctx, "NVIDIA toolkit key install", "gpg", "--dearmor", "--yes",
                "--output", "/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg",
                key_path,
            )
        finally:
            try:
                os.unlink(key_path)
            except OSError:
                pass
        repo_script = (
            "set -o pipefail; curl -fsSL "
            "https://nvidia.github.io/libnvidia-container/stable/deb/"
            "nvidia-container-toolkit.list | sed "
            "'s#deb https://#deb [signed-by=/usr/share/keyrings/"
            "nvidia-container-toolkit-keyring.gpg] https://#g' > "
            "/etc/apt/sources.list.d/nvidia-container-toolkit.list"
        )
        _run_required(ctx, "NVIDIA toolkit repository setup", "bash", "-c", repo_script)
        update = progress_exec.apt(
            ctx, "update", check=False, capture_output=True, text=True
        )
        if update.returncode != 0:
            raise AmcLaunchError(
                f"apt-get update for NVIDIA toolkit failed (exit "
                f"{update.returncode}): {_result_detail(update)}"
            )
        _apt_install(ctx, "nvidia-container-toolkit")

    _run_required(ctx, "Docker service enable", "systemctl", "enable", "--now", "docker")
    if not _nvidia_runtime_registered(ctx):
        _run_required(
            ctx, "NVIDIA Docker runtime configuration", "nvidia-ctk",
            "runtime", "configure", "--runtime=docker",
        )
        _run_required(ctx, "Docker restart", "systemctl", "restart", "docker")

    _run_required(ctx, "docker group membership", "usermod", "-aG", "docker", ctx.user.name)

    if not _docker_usable(ctx):
        raise AmcLaunchError(
            f"Docker is installed but unavailable to {ctx.user.name}; log out and "
            "back in if group membership was just changed"
        )
    version = _compose_version(ctx)
    if version is None or version < MIN_COMPOSE_VERSION:
        found = ".".join(map(str, version)) if version else "unavailable"
        raise AmcLaunchError(
            f"Docker Compose >=2.20.3 is required for AMC include support; found {found}"
        )
    if not _nvidia_runtime_registered(ctx):
        raise AmcLaunchError("the nvidia runtime was not registered after configuration")

    ctx.progress.task("validating GPU container access")
    gpu = ctx.run_as_user(
        "docker", "run", "--rm", "--gpus", "all", GPU_TEST_IMAGE,
        "nvidia-smi", "-L", check=False, capture_output=True, text=True,
        stream=True,
    )
    if gpu.returncode != 0 or "GPU" not in (gpu.stdout or ""):
        raise AmcLaunchError(
            f"NVIDIA GPU container validation failed (exit {gpu.returncode}): "
            f"{_result_detail(gpu)}"
        )


# ---------------------------------------------------------------------------
# The shared bring-up routine (section 4-5)
# ---------------------------------------------------------------------------


def launch_amc(
    ctx: "Context",
    *,
    project: Optional[str] = None,
    location_id: Optional[str] = None,
    skip_pull: bool = False,
    keep_up: bool = False,
    host_ip_override: Optional[str] = None,
    no_open: bool = False,
    non_interactive: bool = False,
    _prereqs_ready: bool = False,
) -> StepResult:
    """The shared AMC bring-up + hold-until-close routine (sections 4-5).
    Called by `run()`'s optional immediate launch, the registered `amc`
    subcommand, and (a later unit) Step 5's re-run entry point."""
    compose_dir: Optional[pathlib.Path] = None
    up_started = False
    teardown: Optional[Callable[[], None]] = None
    try:
        _, project_name = resolve_project_identity(
            ctx, project=project, location_id=location_id
        )
        if not _prereqs_ready:
            ensure_container_prerequisites(ctx)
        cfg = resolve_config(
            ctx, project=project_name, host_ip_override=host_ip_override
        )

        guard = check_repo_isolation(cfg.amc_root)
        if guard is not None:
            raise AmcLaunchError(guard)

        cloned = clone_amc(ctx, cfg.amc_root)
        label = f"{AMC_VERSION}@{AMC_COMMIT[:12]}"
        if cloned:
            ctx.report_installed("auto-magic-calib", label)
        else:
            ctx.report_already_installed("auto-magic-calib", label)

        ensure_projects_and_models(ctx, cfg.amc_root)

        compose_dir = locate_compose_dir(cfg.amc_root)
        if compose_dir is None:
            raise AmcLaunchError(
                f"cannot locate compose/compose.yml inside {cfg.amc_root}; "
                "the pinned AMC checkout is incomplete"
            )

        stack_running = compose_stack_running(ctx, compose_dir)
        if stack_running:
            ui_port = compose_published_port(
                ctx, compose_dir, "auto-magic-calib-ui", 5000
            )
            ms_port = compose_published_port(
                ctx, compose_dir, "auto-magic-calib-ms", 8000
            )
            if ui_port is None or ms_port is None:
                raise AmcLaunchError(
                    "AMC containers are running, but Docker Compose did not "
                    "report their published UI/API ports"
                )
            if (ui_port, ms_port) != (cfg.ui_port, cfg.ms_port):
                ctx.log.warn(
                    "Recovered AMC published ports from Docker Compose: "
                    f"UI {ui_port}, API {ms_port}"
                )
            cfg = AmcConfig(
                amc_root=cfg.amc_root,
                host_ip=cfg.host_ip,
                ui_port=ui_port,
                ms_port=ms_port,
                ms_api_url=cfg.ms_api_url,
                project_name=cfg.project_name,
            )
            _persist(ctx, CONF_UI_PORT_KEY, ui_port)
            _persist(ctx, CONF_MS_PORT_KEY, ms_port)
            ctx.log.info(
                "AMC containers are already running; reusing the configured ports"
            )
        else:
            cfg = resolve_ports(ctx, cfg)
        persist_config(ctx, cfg)

        for key in check_env_drift(compose_dir):
            ctx.log.warn(f"pinned AMC environment no longer defines {key!r}")

        env_changed = write_env_atomic(compose_dir, render_env(cfg))
        ownership = ctx.run_root(
            "chown", f"{ctx.user.uid}:{ctx.user.gid}", str(compose_dir / ".env"),
            check=False, capture_output=True, text=True,
        )
        if ownership.returncode != 0:
            raise AmcLaunchError(
                f"could not hand AMC .env to {ctx.user.name}: "
                f"{_result_detail(ownership)}"
            )
        if env_changed:
            ctx.report_installed("AMC compose/.env", cfg.project_name)
        else:
            ctx.report_already_installed("AMC compose/.env", cfg.project_name)

        compose_validate(ctx, compose_dir)
        docker_login(ctx)

        def _teardown() -> None:
            assert compose_dir is not None
            compose_down(ctx, compose_dir)

        # Arm cleanup before pull/up/readiness. The two explicit leave-up modes
        # must not register an atexit hook that immediately undoes their work.
        teardown = _teardown if (keep_up or no_open) else _install_teardown_guards(_teardown)

        up_started = True
        if not stack_running:
            if not skip_pull:
                compose_pull(ctx, compose_dir)
            compose_up(ctx, compose_dir)

        api_url = f"http://localhost:{cfg.ms_port}/v1/ready"
        ui_url = f"http://localhost:{cfg.ui_port}"
        ctx.log.info(
            "Waiting for AMC startup; the first launch downloads models and "
            "builds its parser before the API becomes ready; this can take "
            f"up to {int(_SERVICE_WAIT_TIMEOUT_S // 60)} minutes."
        )
        ctx.progress.task("waiting for the AMC microservice")
        if not wait_for_backend(ctx, api_url, compose_dir=compose_dir):
            raise AmcLaunchError(
                f"AMC microservice did not return code 0 at {api_url} "
                f"(waited up to {int(_SERVICE_WAIT_TIMEOUT_S)}s)\n"
                f"{compose_diagnostics(ctx, compose_dir)}"
            )
        if not wait_for_ui(ctx, ui_url):
            raise AmcLaunchError(
                f"AMC UI did not return HTTP 200 at {ui_url}\n"
                f"{compose_diagnostics(ctx, compose_dir)}"
            )

        project_id = ensure_amc_project(ctx, cfg)
        ctx.log.info(
            f"AMC project ready: {cfg.project_name} ({project_id}); "
            f"LOCATION_ID={ctx.conf[CONF_LOCATION_ID_KEY]}"
        )

        if no_open:
            return StepResult(status=StepStatus.COMPLETE)

        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        browser_available = find_browser() is not None
        strategy = decide_hold_strategy(
            has_display=has_display,
            browser_available=browser_available,
            non_interactive=non_interactive,
        )

        assert teardown is not None
        execute_hold(ctx, strategy, ui_url, teardown=teardown, keep_up=keep_up)

        return StepResult(status=StepStatus.COMPLETE)
    except (AmcLaunchError, KeyboardInterrupt) as exc:
        message = (
            str(exc)
            if isinstance(exc, AmcLaunchError)
            else "AMC launch cancelled by the operator"
        )
        if up_started and compose_dir is not None:
            try:
                if teardown is not None:
                    teardown()
                else:  # pragma: no cover - defensive before guard assignment.
                    compose_down(ctx, compose_dir)
            except AmcLaunchError as cleanup:
                return StepResult(
                    status=StepStatus.FAILED,
                    message=f"{message}\nAMC cleanup also failed: {cleanup}",
                )
        return StepResult(status=StepStatus.FAILED, message=message)


def teardown_amc(
    ctx: "Context",
    *,
    project: Optional[str] = None,
    host_ip_override: Optional[str] = None,
) -> StepResult:
    """`amc --down`: tear down without bringing anything up."""
    cfg = resolve_config(ctx, project=project, host_ip_override=host_ip_override)
    compose_dir = locate_compose_dir(cfg.amc_root)
    if compose_dir is None:
        return StepResult(
            status=StepStatus.FAILED,
            message=f"AMC not found at {cfg.amc_root}; nothing to tear down",
        )
    try:
        compose_down(ctx, compose_dir)
    except AmcLaunchError as exc:
        return StepResult(status=StepStatus.FAILED, message=str(exc))
    return StepResult(status=StepStatus.COMPLETE)


# ---------------------------------------------------------------------------
# section 6.2 -- the `amc` subcommand
# ---------------------------------------------------------------------------


def _build_amc_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mv3dt-installer amc", add_help=True)
    parser.add_argument("--project", default=None)
    parser.add_argument("--location-id", default=None)
    parser.add_argument("--skip-pull", action="store_true")
    parser.add_argument("--keep-up", action="store_true")
    parser.add_argument("--down", action="store_true")
    parser.add_argument("--host-ip", default=None)
    # Accepted but currently inert -- see this module's docstring's "Root
    # for the standalone amc subcommand" note: the framework's subcommand
    # bootstrap already requires root unconditionally, so the docker-group
    # re-exec scenario --sudo exists for never actually arises today.
    parser.add_argument("--sudo", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    # doc 00 §3.3-shaped framework flags a subcommand's own argv may carry
    # (systemd ExecStart= lines, `app._bootstrap_subcommand_context`'s own
    # peek parser) -- accepted here too so `amc`'s parser does not reject
    # them, even though this handler does not need their values itself
    # (the bootstrap already resolved them into `ctx`).
    parser.add_argument("--install-dir", default=None)
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--log-dir", default=None)
    return parser


def handle_amc_subcommand(argv: list, ctx: "Context") -> int:
    """`mv3dt-installer amc [...]` (section 6.2's registered handler)."""
    args = _build_amc_arg_parser().parse_args(argv)

    if args.down:
        result = teardown_amc(ctx, project=args.project, host_ip_override=args.host_ip)
    else:
        result = launch_amc(
            ctx,
            project=args.project,
            location_id=args.location_id,
            skip_pull=args.skip_pull,
            keep_up=args.keep_up,
            host_ip_override=args.host_ip,
            no_open=args.no_open,
            non_interactive=ctx.non_interactive,
        )

    if result.status is StepStatus.FAILED:
        ctx.log.error(result.message)
        return 1
    if result.status is StepStatus.USER_ACTION_REQUIRED:
        from mv3dt_installer import privilege

        privilege.show_user_action_block(
            "AutoMagicCalib launcher", result.message, result.user_actions
        )
        return 0
    return 0


# ---------------------------------------------------------------------------
# section 2 -- the "launch now vs. later" prompt
# ---------------------------------------------------------------------------

# Injectable so tests never block on real input().
_INPUT: Callable[[str], str] = input


def _confirm_launch_now(ctx: "Context") -> bool:
    """`Launch AutoMagicCalib now? [y/N]`, default No (section 2).
    `--non-interactive`/`--no-pause` skip the prompt entirely (doc 00
    §3.3's `--no-pause` is a framework-level flag `Context` does not carry,
    so this module honors only `ctx.non_interactive`, consistent with how
    Step 2 makes the same decision)."""
    if ctx.non_interactive:
        return False
    answer = _INPUT("Launch AutoMagicCalib now? [y/N]: ").strip().lower()
    return answer in ("y", "yes")


# ---------------------------------------------------------------------------
# The Step
# ---------------------------------------------------------------------------


class Step3AmcLauncher:
    """STEP-3-AMC-LAUNCHER.md section 1: module identity."""

    id = "step3_amc_launcher"
    title = "AutoMagicCalib launcher"

    # Doc 08 §3.1. The launch phase is conditional (section 7.2's
    # confirm-then-launch), so a run that declines it collapses phase 3 the
    # moment the step completes rather than leaving it drawn.
    phases = (
        "configuration",
        "launcher wrapper",
        "AutoMagicCalib launch",
    )
    order = 3

    # -- preflight (section 7.1) -------------------------------------------

    def preflight(self, ctx: "Context") -> StepResult:
        if not ctx.conf.get(step2_mod.CONF_METHOD_KEY):
            return StepResult(
                status=StepStatus.FAILED,
                message="DeepStream SDK not installed; run Step 2 first",
            )

        return StepResult(status=StepStatus.COMPLETE)

    # -- run (section 7.2) ---------------------------------------------------

    def run(self, ctx: "Context") -> StepResult:
        ctx.progress.phase(1)
        ctx.progress.task("resolving AutoMagicCalib configuration")
        try:
            _, project_name = resolve_project_identity(ctx)
        except AmcLaunchError as exc:
            return StepResult(status=StepStatus.FAILED, message=str(exc))
        cfg = resolve_config(ctx, project=project_name)

        guard = check_repo_isolation(cfg.amc_root)
        if guard is not None:
            return StepResult(status=StepStatus.FAILED, message=guard)

        try:
            ensure_container_prerequisites(ctx)
        except AmcLaunchError as exc:
            return StepResult(status=StepStatus.FAILED, message=str(exc))

        persist_config(ctx, cfg)

        ctx.progress.phase(2)
        ctx.progress.task("installer binary and amc wrapper")
        installer_bin = ensure_installer_binary(ctx)
        wrapper_path, wrapper_changed = write_amc_wrapper(ctx, installer_bin)
        if wrapper_changed:
            ctx.report_installed("amc launcher", str(wrapper_path))
        else:
            ctx.report_already_installed("amc launcher", str(wrapper_path))

        if not _confirm_launch_now(ctx):
            return StepResult(status=StepStatus.COMPLETE)

        ctx.progress.phase(3)
        ctx.progress.task("launching AutoMagicCalib")
        result = launch_amc(
            ctx,
            keep_up=True,
            non_interactive=ctx.non_interactive,
            _prereqs_ready=True,
        )
        if result.status is not StepStatus.COMPLETE:
            return result

        return StepResult(status=StepStatus.COMPLETE)

    # -- verify (section 7.3) -------------------------------------------------

    def verify(self, ctx: "Context") -> StepResult:
        wrapper = ctx.install_dir / "bin" / AMC_WRAPPER_NAME
        installer_bin = ctx.install_dir / "bin" / INSTALLER_BIN_NAME

        if not wrapper.is_file() or not os.access(wrapper, os.X_OK):
            return StepResult(
                status=StepStatus.FAILED,
                message=f"{wrapper} missing or not executable",
            )
        if not installer_bin.is_file():
            return StepResult(
                status=StepStatus.FAILED, message=f"{installer_bin} missing"
            )
        if not _compose_available(ctx):
            return StepResult(
                status=StepStatus.FAILED,
                message="docker compose >=2.20.3 is not available",
            )
        if not _nvidia_runtime_registered(ctx):
            return StepResult(
                status=StepStatus.FAILED,
                message="the nvidia runtime is not registered with docker",
            )
        for key in _STEP3_CONF_KEYS:
            if key not in ctx.conf:
                return StepResult(
                    status=StepStatus.FAILED,
                    message=f"{key} missing from installer.conf",
                )

        # The checkout is lazy, but once present it must match the equality pin.
        cfg = resolve_config(ctx)
        commit = resolved_amc_commit(ctx, cfg.amc_root)
        if commit:
            ctx.log.info(f"AMC resolved commit ({cfg.amc_root}): {commit}")
            if commit != AMC_COMMIT:
                return StepResult(
                    status=StepStatus.FAILED,
                    message=(
                        f"AMC checkout is at {commit}, expected equality pin "
                        f"{AMC_COMMIT}"
                    ),
                )
        else:
            ctx.log.info(
                f"AMC resolved commit: unknown (could not resolve HEAD at "
                f"{cfg.amc_root})"
            )

        return StepResult(status=StepStatus.COMPLETE)

    # -- report (section 7.4) --------------------------------------------------

    def report(self, ctx: "Context") -> None:
        cfg = resolve_config(ctx)
        wrapper = ctx.install_dir / "bin" / AMC_WRAPPER_NAME
        ctx.log.info(
            "AutoMagicCalib launcher installed.\n"
            f"  Run it any time:   {wrapper}\n"
            f"  Web UI:            http://localhost:{cfg.ui_port}\n"
            f"  Microservice API:  http://localhost:{cfg.ms_port}\n"
            f"  AMC project ID:    {ctx.conf.get(CONF_AMC_PROJECT_ID_KEY, 'created on launch')}\n"
            f"  Location ID:       {ctx.conf.get(CONF_LOCATION_ID_KEY, 'not set')}\n"
            f"  AMC clone:         {cfg.amc_root}\n"
            f"  Stop AMC:          {wrapper} --down   (or close the AMC window)\n"
            "\n"
            "The AMC service stays up until you close the AMC browser window "
            "(or run --down).\n"
            "Next: complete the 6-step calibration in the browser (see the DS "
            "9.1 AutoMagicCalib guide), then Step 4 ingests the export."
        )


register(Step3AmcLauncher())
app_mod.register_subcommand("amc", handle_amc_subcommand)
