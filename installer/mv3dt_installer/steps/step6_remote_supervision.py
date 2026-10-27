"""Step 6 -- 24/7 supervised pipelines + remote MQTT control.

Implements `installer/plan/STEP-6-REMOTE-SUPERVISION.md` against the
framework contract in `installer/plan/00-FRAMEWORK-AND-BOOTSTRAP.md`
(step-module interface section 12, `Context` section 12.3, the systemd
helpers in `systemd.py`, install-location config section 11, opt-in gates
section 3.4), Step 5's project registry (`step5_per_project_exes.py`
section 4), and the subcommand dispatch extension `step3_amc_launcher.py`
builds in `app.py` (`SUBCOMMAND_REGISTRY`/`register_subcommand`).

Scope: turn each Step 5 project into a boot-enabled, always-restarting
systemd service (`mv3dt-pipeline@<slug>.service`, section A) and ship a
long-running control agent (`mv3dt-agent`, section B) that lets a cloud
webapp run/stop/restart pipelines over the existing Mosquitto/MQTT stack
using JSON command packages (section C), with a scoped polkit rule
(section B.1.1) authorizing exactly that and nothing else, and hardened
broker config when remote mode is opted in (section D).

**Step 5 handoff (section A.2).** `step5_per_project_exes.py`'s `pipeline`
subcommand now branches at runtime on `_supervision_active()` (defined
there): unsupervised workstations keep the pre-Step-6 foreground/pkill
behavior unchanged; once this step is COMPLETE and its gate is not "off",
`start`/`stop` hand off to `systemctl start/stop mv3dt-pipeline@<slug>`
instead. This module owns installing/enabling that unit and the agent unit;
it does not re-implement Step 5's dispatch.

**Bootstrap caveat (fixed by plan unit U6, a later fix on top of this
step).** `mv3dt-agent.service`
(section B.1) deliberately runs as `User=@USER@` (the whole point of the
scoped polkit rule in section B.1.1 is letting a *non-root* agent drive
`systemctl` on exactly the units it needs), so this module registers
`agent` with `app.register_subcommand(..., requires_root=False)`:
`app._bootstrap_subcommand_context()` skips `privilege.require_root()` for
it and builds its `Context` from a read-only config resolution instead of
`config.load()`'s root-owned side effects. See `app.register_subcommand()`
and `app._bootstrap_subcommand_context()` for the full mechanism.

Every subprocess call goes through `ctx.run_root`/`ctx.run_as_user`, or an
explicitly injected `runner` callable with the same
`Callable[..., subprocess.CompletedProcess]` shape (doc 00 section 9.2), so
no test here shells out to a real systemctl/mosquitto/polkit or opens a
real MQTT connection.
"""

from __future__ import annotations

import pathlib
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional

from mv3dt_installer import __version__
from mv3dt_installer import app as app_mod
from mv3dt_installer import config as config_mod
from mv3dt_installer import managed_files
from mv3dt_installer import systemd as systemd_mod
from mv3dt_installer.steps import StepResult, StepStatus, UserAction, register
from mv3dt_installer.steps import step5_per_project_exes as step5_mod

if TYPE_CHECKING:  # pragma: no cover -- import-time only, never at runtime.
    from mv3dt_installer.app import Context

__all__ = [
    "PIPELINE_UNIT_NAME",
    "AGENT_UNIT_NAME",
    "WATCHDOG_UNIT_TEMPLATE",
    "WATCHDOG_UNIT_NAME",
    "POLKIT_RULE_NAME",
    "resolve_host_id",
    "cmd_topic",
    "result_topic",
    "status_topic",
    "BrokerTopology",
    "BrokerDecision",
    "decide_broker_topology",
    "render_pipeline_unit",
    "render_agent_unit",
    "render_watchdog_unit",
    "render_polkit_rule",
    "validate_command",
    "RequestDedup",
    "resolve_project",
    "handle_command",
    "build_status_payload",
    "status_round_trip_dry_run",
    "status_round_trip_check",
    "check_polkit_authorization",
    "Step6RemoteSupervision",
    "handle_agent_subcommand",
]

Runner = Callable[..., "subprocess.CompletedProcess[Any]"]

# ---------------------------------------------------------------------------
# Unit / rule names (section 1.2, A.1, B.1, B.1.1)
# ---------------------------------------------------------------------------

PIPELINE_UNIT_TEMPLATE = "mv3dt-pipeline@.service"
AGENT_UNIT_NAME = "mv3dt-agent.service"
WATCHDOG_UNIT_TEMPLATE = "mv3dt-watchdog@.service"
POLKIT_RULE_NAME = "49-mv3dt-agent.rules"
POLKIT_RULES_DIR = pathlib.Path("/etc/polkit-1/rules.d")


def PIPELINE_UNIT_NAME(slug: str) -> str:  # noqa: N802 -- reads like a constant at call sites
    """`mv3dt-pipeline@<slug>.service` (mirrors `step5_mod._pipeline_unit_name`,
    duplicated rather than imported private so this module's public surface
    doesn't depend on a step5 underscore-prefixed helper)."""
    return f"mv3dt-pipeline@{slug}.service"


def WATCHDOG_UNIT_NAME(slug: str) -> str:  # noqa: N802
    """`mv3dt-watchdog@<slug>.service`, the hang watchdog of one pipeline."""
    return f"mv3dt-watchdog@{slug}.service"


# Bound on the read-only systemctl/nvidia-smi probes the status path makes
# (section C.5's ActiveState/SubState show, is-active/is-enabled, and the
# GPU snapshot) -- these must never hang the agent's periodic status publish
# or a verify() round trip on a wedged systemd/D-Bus call.
_STATUS_CALL_TIMEOUT_S = 5.0


def _run(
    runner: Runner, *args: str, timeout: Optional[float] = None
) -> "subprocess.CompletedProcess[Any]":
    kwargs: dict[str, Any] = {"check": False, "capture_output": True, "text": True}
    if timeout is not None:
        kwargs["timeout"] = timeout
    try:
        return runner(list(args), **kwargs)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(list(args), 1, "", "timed out")


def _returncode(result: Any) -> int:
    code = getattr(result, "returncode", None)
    return code if isinstance(code, int) else 1


def _stdout(result: Any) -> str:
    return (getattr(result, "stdout", "") or "").strip()


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# section C.1 -- HOST_ID derivation
# ---------------------------------------------------------------------------

CONF_HOST_ID_KEY = "MV3DT_HOST_ID"
MACHINE_ID_PATH = pathlib.Path("/etc/machine-id")


def resolve_host_id(
    conf: dict[str, str],
    *,
    machine_id_path: pathlib.Path = MACHINE_ID_PATH,
    hostname: Callable[[], str] = socket.gethostname,
) -> str:
    """Section C.1's priority order, exactly: `installer.conf`'s
    `MV3DT_HOST_ID` override, else `/etc/machine-id`, else the sanitized
    hostname. Only the hostname fallback is slugified (section C.1: "else
    the hostname ..., sanitized with the same slug rules as Step 5 section
    3.1") -- an operator-chosen `MV3DT_HOST_ID` and the machine id are taken
    verbatim, since both are already stable, opaque identifiers."""
    override = conf.get(CONF_HOST_ID_KEY)
    if override:
        return override

    try:
        machine_id = machine_id_path.read_text(encoding="utf-8").strip()
    except OSError:
        machine_id = ""
    if machine_id:
        return machine_id

    return step5_mod.slugify(hostname())


# ---------------------------------------------------------------------------
# section C.2 -- topic layout
# ---------------------------------------------------------------------------


def cmd_topic(host_id: str) -> str:
    return f"mv3dt/{host_id}/cmd"


def result_topic(host_id: str) -> str:
    return f"mv3dt/{host_id}/cmd/result"


def status_topic(host_id: str) -> str:
    return f"mv3dt/{host_id}/status"


# ---------------------------------------------------------------------------
# section B.2 -- broker topology
# ---------------------------------------------------------------------------

CLOUD_HOST_KEY = "MV3DT_CLOUD_BROKER_HOST"
CLOUD_PORT_KEY = "MV3DT_CLOUD_BROKER_PORT"
CLOUD_USER_KEY = "MV3DT_CLOUD_BROKER_USER"
CLOUD_PASS_KEY = "MV3DT_CLOUD_BROKER_PASS"
_CLOUD_KEYS = (CLOUD_HOST_KEY, CLOUD_PORT_KEY, CLOUD_USER_KEY, CLOUD_PASS_KEY)


class BrokerTopology(str, Enum):
    """Section B.2's two topologies. Topology (2) (agent connects directly
    to the cloud broker, no local bridge) is documented as the fallback for
    a host with no local Mosquitto -- not our case, since Step 1 always
    installs one -- so it is not modeled here."""

    LOCAL_ONLY = "local_only"
    LOCAL_WITH_BRIDGE = "local_with_bridge"


@dataclass(frozen=True)
class BrokerDecision:
    topology: BrokerTopology
    missing_cloud_endpoint: bool


def decide_broker_topology(conf: dict[str, str], *, gate_value: str) -> BrokerDecision:
    """Section B.2's decision, as a pure function of the resolved
    `MV3DT_REMOTE_SUPERVISION` gate and `installer.conf`.

    - `gate_value != "remote"` (i.e. "off" or "local"): always local-only.
      Local mode never reads or requires the `MV3DT_CLOUD_BROKER_*` keys at
      all (section E.2).
    - `gate_value == "remote"` with all four `MV3DT_CLOUD_BROKER_*` keys
      present: local broker + bridge.
    - `gate_value == "remote"` with any of the four missing: stays
      local-only, but `missing_cloud_endpoint` is flagged so the caller can
      surface `USER_ACTION_REQUIRED` (section B.2's closing paragraph) --
      remote mode was explicitly requested and cannot be honored yet, which
      is different from never having asked for it.
    """
    if gate_value != "remote":
        return BrokerDecision(topology=BrokerTopology.LOCAL_ONLY, missing_cloud_endpoint=False)

    if all(conf.get(key) for key in _CLOUD_KEYS):
        return BrokerDecision(topology=BrokerTopology.LOCAL_WITH_BRIDGE, missing_cloud_endpoint=False)

    return BrokerDecision(topology=BrokerTopology.LOCAL_ONLY, missing_cloud_endpoint=True)


def missing_cloud_broker_keys(conf: dict[str, str]) -> list[str]:
    """The subset of `MV3DT_CLOUD_BROKER_*` keys not yet set, in table
    order -- what a `USER_ACTION_REQUIRED` block should list."""
    return [key for key in _CLOUD_KEYS if not conf.get(key)]


# ---------------------------------------------------------------------------
# section A.4 / B.1 -- unit + polkit rule rendering
# ---------------------------------------------------------------------------


def render_pipeline_unit(user: str) -> str:
    """Render `mv3dt-pipeline@.service.in` (section A.1), substituting only
    `@USER@` -- `%i` is systemd's own instance specifier and is left alone
    by `systemd.render_unit()`."""
    return systemd_mod.render_unit(
        ("systemd", "mv3dt-pipeline@.service.in"), {"USER": user}
    )


def render_agent_unit(user: str) -> str:
    """Render `mv3dt-agent.service.in` (section B.1)."""
    return systemd_mod.render_unit(("systemd", "mv3dt-agent.service.in"), {"USER": user})


def render_watchdog_unit(user: str) -> str:
    """Render `mv3dt-watchdog@.service.in`. It runs as root (it restarts the
    pipeline unit and reads its journal); `@USER@` only fills `SUDO_USER`."""
    return systemd_mod.render_unit(
        ("systemd", "mv3dt-watchdog@.service.in"), {"USER": user}
    )


def _install_units(ctx: "Context", units: dict[str, str]) -> bool:
    """Install each unit through the edit protection of `managed_files`: a
    hand-edited or pre-manifest unit is kept and its new version written as
    `<name>.installer-new`. Returns whether any unit file was written."""
    written = False
    for name, content in units.items():
        outcome = systemd_mod.install_managed_unit(name, content)
        if outcome is managed_files.Outcome.WRITTEN:
            written = True
            ctx.report_installed(name, __version__)
        elif outcome is managed_files.Outcome.UNCHANGED:
            ctx.report_already_installed(name, __version__)
        else:
            ctx.log.warn(
                f"kept the existing {name}; review the new version with "
                "`mv3dt-installer configs diff` and take it with "
                "`configs accept`, then `systemctl daemon-reload`"
            )
    return written


def render_polkit_rule(user: str) -> str:
    """Render the LOCKED polkit rule (section B.1.1), substituting only
    `@USER@`. The rule's own JavaScript is reproduced verbatim in the
    bundled `.in` template -- this function does no string manipulation of
    its own beyond `systemd.render_unit()`'s marker substitution, so the
    anchored regex / verb allowlist / `Result.YES` / `NOT_HANDLED` shape is
    exactly what's on disk, not re-derived here."""
    return systemd_mod.render_unit(
        ("polkit", "49-mv3dt-agent.rules.in"), {"USER": user}
    )


# ---------------------------------------------------------------------------
# section A.3 -- disable-before-remove hook into Step 5's reconciliation
# ---------------------------------------------------------------------------


def _supervision_gate_active(ctx: "Context") -> bool:
    gate = ctx.conf.get(config_mod.GATE_REMOTE_SUPERVISION, "off")
    return bool(gate) and gate != "off"


def _disable_pipeline_unit_before_remove(ctx: "Context", entry: "step5_mod.ProjectEntry") -> None:
    """The callback `step5_mod.register_removal_hook()` installs (section
    A.3: "before deleting install-side artifacts, run `systemctl disable
    --now mv3dt-pipeline@<slug>`"). A no-op when Step 6's gate is off, so a
    workstation that never opted in never issues a systemctl call it has no
    business making."""
    if not _supervision_gate_active(ctx):
        return
    # The watchdog first, so it cannot restart the pipeline being removed.
    for unit in (WATCHDOG_UNIT_NAME(entry.slug), PIPELINE_UNIT_NAME(entry.slug)):
        ctx.log.info(f"Step 6: disabling {unit} before removing project artifacts")
        ctx.run_root("systemctl", "disable", "--now", unit, check=False, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# section C.3/C.6 -- command validation
# ---------------------------------------------------------------------------

VALID_ACTIONS = ("run", "stop", "restart", "status", "list")
_LIFECYCLE_VERB = {"run": "start", "stop": "stop", "restart": "restart"}
_REQUIRED_COMMAND_FIELDS = ("action", "request_id", "ts")


def validate_command(payload: Any) -> Optional[str]:
    """Section C.6's malformed-command rule: `None` when `payload` is a
    valid command envelope, else a short `"invalid command: <reason>"`
    string (never raises)."""
    if not isinstance(payload, dict):
        return "invalid command: not a JSON object"
    for key in _REQUIRED_COMMAND_FIELDS:
        if key not in payload:
            return f"invalid command: missing '{key}'"
    if payload["action"] not in VALID_ACTIONS:
        return f"invalid command: unknown action {payload['action']!r}"
    return None


# ---------------------------------------------------------------------------
# section C.6 -- request_id de-dup window
# ---------------------------------------------------------------------------


@dataclass
class RequestDedup:
    """Section C.6: "the agent keys de-dup on `request_id` within a short
    TTL window so a redelivered MQTT message (QoS 1 at-least-once) does not
    double-execute; the cached result is re-published." `clock` is
    injectable so a test can exercise expiry without a real wait."""

    ttl_s: float = 300.0
    clock: Callable[[], float] = time.monotonic
    _cache: dict[str, tuple[float, dict]] = field(default_factory=dict)

    def get(self, request_id: Optional[str]) -> Optional[dict]:
        if not request_id:
            return None
        entry = self._cache.get(request_id)
        if entry is None:
            return None
        stamped_at, result = entry
        if self.clock() - stamped_at > self.ttl_s:
            del self._cache[request_id]
            return None
        return result

    def put(self, request_id: Optional[str], result: dict) -> None:
        if request_id:
            self._cache[request_id] = (self.clock(), result)


# ---------------------------------------------------------------------------
# section C.6 -- settle-and-poll
# ---------------------------------------------------------------------------

_TRANSITIONAL_STATES = ("activating", "deactivating")
SETTLE_TIMEOUT_S = 10.0
SETTLE_POLL_S = 0.5


def _systemctl_state(unit: str, prop: str, *, runner: Runner) -> str:
    result = _run(
        runner,
        "systemctl",
        "is-active" if prop == "active" else "is-enabled",
        unit,
        timeout=_STATUS_CALL_TIMEOUT_S,
    )
    return _stdout(result) or "unknown"


def _settle(
    unit: str,
    *,
    runner: Runner,
    timeout_s: float = SETTLE_TIMEOUT_S,
    poll_s: float = SETTLE_POLL_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Section C.6's REQUIRED settle-and-poll: `systemctl start`/`stop`
    returns once the job is queued, not once the unit has actually settled,
    so poll `is-active` until it leaves `activating`/`deactivating` or
    `timeout_s` elapses. Returns the final observed state -- a unit still
    transitional at the deadline is returned as-is (the caller reports
    `ok: false` / `"timed out waiting for active"`, never an optimistic
    `active`)."""
    started = clock()
    state = _systemctl_state(unit, "active", runner=runner)
    while state in _TRANSITIONAL_STATES:
        if clock() - started >= timeout_s:
            return state
        sleep(poll_s)
        state = _systemctl_state(unit, "active", runner=runner)
    return state


# ---------------------------------------------------------------------------
# section C.4/C.6 -- action -> systemctl mapping, unknown projects, results
# ---------------------------------------------------------------------------


def resolve_project(
    install_dir: pathlib.Path, project_ref: Optional[str]
) -> Optional["step5_mod.ProjectEntry"]:
    """Section C.3: "`project` -- the original `PROJECT_NAME` **or** its
    `slug` (the agent resolves either via the registry)"."""
    if not project_ref:
        return None
    entry = step5_mod.get(install_dir, project_ref)
    if entry is not None:
        return entry
    return step5_mod.get_by_slug(install_dir, project_ref)


def _build_result(
    *,
    request_id: str,
    ok: bool,
    action: Optional[str],
    project: Optional[str],
    state: str,
    enabled: str,
    error: Optional[str],
) -> dict[str, Any]:
    """Section C.4's exact field set."""
    return {
        "request_id": request_id,
        "ok": ok,
        "action": action,
        "project": project,
        "state": state,
        "enabled": enabled,
        "error": error,
        "ts": _now_utc_iso(),
    }


def _handle_lifecycle_action(
    *,
    request_id: str,
    action: str,
    project_ref: Optional[str],
    install_dir: pathlib.Path,
    runner: Runner,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    entry = resolve_project(install_dir, project_ref)
    if entry is None:
        # Section C.6: "if `project` resolves to no registry slug, the
        # agent does not call systemctl". No units are ever created here.
        return _build_result(
            request_id=request_id,
            ok=False,
            action=action,
            project=project_ref,
            state="unknown",
            enabled="unknown",
            error="unknown project",
        )

    unit = PIPELINE_UNIT_NAME(entry.slug)
    verb = _LIFECYCLE_VERB[action]
    call = _run(runner, "systemctl", verb, unit)
    call_ok = _returncode(call) == 0

    state = _settle(unit, runner=runner, clock=clock, sleep=sleep)
    settled = state not in _TRANSITIONAL_STATES
    enabled = _systemctl_state(unit, "enabled", runner=runner)

    error: Optional[str] = None
    if not call_ok:
        error = f"systemctl {verb} failed: {_returncode(call)}"
    elif not settled:
        error = "timed out waiting for active"
    elif action in ("run", "restart") and state != "active":
        error = f"unit not active after {verb} (state={state})"
    elif action == "stop" and state not in ("inactive", "failed"):
        error = f"unit still {state} after stop"

    return _build_result(
        request_id=request_id,
        ok=call_ok and settled and error is None,
        action=action,
        project=entry.slug,
        state=state,
        enabled=enabled,
        error=error,
    )


def _handle_status_action(
    *,
    request_id: str,
    project_ref: Optional[str],
    install_dir: pathlib.Path,
    runner: Runner,
    host_id: str,
) -> dict[str, Any]:
    if not project_ref:
        # Section C.3: "optional for status (omit = whole host)"; section
        # C.3 also says a bare `status` command "returns the current status
        # payload (section C.5) as a result" -- so this returns the real
        # build_status_payload() output (host_id/agent_version/pipelines/
        # gpu), not a per-unit placeholder, with the section C.4 envelope
        # fields layered on top for request correlation.
        payload = build_status_payload(host_id=host_id, install_dir=install_dir, runner=runner)
        payload.update({"request_id": request_id, "ok": True, "action": "status", "error": None})
        return payload
    entry = resolve_project(install_dir, project_ref)
    if entry is None:
        return _build_result(
            request_id=request_id,
            ok=False,
            action="status",
            project=project_ref,
            state="unknown",
            enabled="unknown",
            error="unknown project",
        )
    unit = PIPELINE_UNIT_NAME(entry.slug)
    return _build_result(
        request_id=request_id,
        ok=True,
        action="status",
        project=entry.slug,
        state=_systemctl_state(unit, "active", runner=runner),
        enabled=_systemctl_state(unit, "enabled", runner=runner),
        error=None,
    )


def _handle_list_action(*, request_id: str, install_dir: pathlib.Path, runner: Runner) -> dict[str, Any]:
    projects = [entry.slug for entry in step5_mod.list_projects(install_dir)]
    return {
        "request_id": request_id,
        "ok": True,
        "action": "list",
        "projects": projects,
        "error": None,
        "ts": _now_utc_iso(),
    }


def handle_command(
    payload: Any,
    *,
    install_dir: pathlib.Path,
    runner: Runner,
    dedup: RequestDedup,
    host_id: str = "",
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Section C.3 -> C.4: validate, de-dup, dispatch by `action`
    (section C.6's table), and return the exact result envelope to publish
    to `mv3dt/<HOST_ID>/cmd/result`. Never raises -- every failure path
    (malformed command, unknown project, systemctl failure, settle
    timeout) is represented as `ok: false` with a short `error` string.

    `host_id` is only consumed by the whole-host `status` action (section
    C.3: "omit `project` = whole host"), which embeds it in the returned
    section C.5 payload; every other action ignores it. Defaults to `""`
    so existing per-project callers don't need to pass it."""
    error = validate_command(payload)
    request_id = payload.get("request_id", "") if isinstance(payload, dict) else ""
    action = payload.get("action") if isinstance(payload, dict) else None
    project_ref = payload.get("project") if isinstance(payload, dict) else None

    if error:
        return _build_result(
            request_id=request_id,
            ok=False,
            action=action,
            project=project_ref,
            state="unknown",
            enabled="unknown",
            error=error,
        )

    cached = dedup.get(request_id)
    if cached is not None:
        return cached

    if action == "list":
        result = _handle_list_action(request_id=request_id, install_dir=install_dir, runner=runner)
    elif action == "status":
        result = _handle_status_action(
            request_id=request_id,
            project_ref=project_ref,
            install_dir=install_dir,
            runner=runner,
            host_id=host_id,
        )
    else:
        result = _handle_lifecycle_action(
            request_id=request_id,
            action=action,
            project_ref=project_ref,
            install_dir=install_dir,
            runner=runner,
            clock=clock,
            sleep=sleep,
        )

    dedup.put(request_id, result)
    return result


# ---------------------------------------------------------------------------
# section C.5 -- status / heartbeat payload
# ---------------------------------------------------------------------------


def _gpu_snapshot(runner: Runner) -> Optional[dict[str, Any]]:
    """Best-effort `nvidia-smi` query (section C.5); `None` (omitted from
    the payload) on any failure, matching the Step 5 validation banner's
    own best-effort GPU query."""
    result = _run(
        runner,
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
        "--format=csv,noheader,nounits",
        timeout=_STATUS_CALL_TIMEOUT_S,
    )
    if _returncode(result) != 0:
        return None
    line = _stdout(result).splitlines()[0] if _stdout(result) else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        return None
    try:
        return {
            "utilization_pct": int(float(parts[0])),
            "memory_used_mb": int(float(parts[1])),
            "temperature_c": int(float(parts[2])),
        }
    except ValueError:
        return None


# systemd's `ActiveEnterTimestamp` prints like "Wed 2024-01-17 10:15:23 UTC"
# (day-of-week, date, time, timezone abbreviation) under the installer's own
# C-locale invocation of systemctl; a bare (no-timezone) form is accepted too
# in case a caller's environment strips it.
_SYSTEMD_TIMESTAMP_FORMATS = ("%a %Y-%m-%d %H:%M:%S %Z", "%a %Y-%m-%d %H:%M:%S")
_SYSTEMD_TIMESTAMP_UNSET = ("", "n/a", "0")


def _parse_systemd_timestamp(value: str) -> Optional[datetime]:
    """Best-effort parse of a `systemctl show -p ActiveEnterTimestamp`
    value. Returns `None` for the unset sentinel (never entered active) or
    any value this can't parse -- never raises."""
    text = (value or "").strip()
    if text in _SYSTEMD_TIMESTAMP_UNSET:
        return None
    for fmt in _SYSTEMD_TIMESTAMP_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def _compute_uptime_s(
    active_enter_timestamp: str,
    *,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> int:
    """Section C.5's `uptime_s`: whole seconds since `ActiveEnterTimestamp`,
    or 0 when that timestamp can't be parsed -- matching every other
    section C.5 field's never-raise, degrade-to-a-safe-default contract.
    `now` is injectable so a test can pin "the current time" instead of
    racing a real clock."""
    parsed = _parse_systemd_timestamp(active_enter_timestamp)
    if parsed is None:
        return 0
    return max(0, int((now() - parsed).total_seconds()))


def build_status_payload(
    *,
    host_id: str,
    install_dir: pathlib.Path,
    runner: Runner,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Section C.5's exact schema. `active`/`sub` are section C.5's REQUIRED
    fallback: any failure of `systemctl show ... -p ActiveState -p SubState`
    yields `"unknown"`/`"unknown"` rather than raising, so one unreadable
    unit never blocks the whole payload.

    Note (STEP-7 cross-reference, non-blocking): this module's own
    `active`/`sub` keys are lowercase, matching this doc's (STEP-6 section
    C.5) own worked example verbatim. STEP-7 section D.1's status payload
    uses capitalized `Active`/`Sub` for the same concept -- a pre-existing
    naming inconsistency between the two sibling docs, not introduced here;
    each module matches its own doc's literal spelling.
    """
    pipelines: dict[str, Any] = {}
    for entry in step5_mod.list_projects(install_dir):
        unit = PIPELINE_UNIT_NAME(entry.slug)
        show = _run(
            runner,
            "systemctl",
            "show",
            unit,
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "ActiveEnterTimestamp",
            "-p",
            "ExecMainStatus",
            "-p",
            "NRestarts",
            timeout=_STATUS_CALL_TIMEOUT_S,
        )
        props: dict[str, str] = {}
        if _returncode(show) == 0:
            for line in _stdout(show).splitlines():
                key, _, value = line.partition("=")
                if key:
                    props[key] = value

        active_state = props.get("ActiveState", "unknown")
        # Uptime only means something while the unit is actually active --
        # a failed/inactive unit's ActiveEnterTimestamp is stale history,
        # not "how long it has been up" (matches the doc's own worked
        # example: an active unit gets a real uptime_s, a failed one 0).
        uptime_s = (
            _compute_uptime_s(props.get("ActiveEnterTimestamp", ""), now=now)
            if active_state == "active"
            else 0
        )

        pipelines[entry.slug] = {
            "active": active_state,
            "sub": props.get("SubState", "unknown"),
            "enabled": _systemctl_state(unit, "enabled", runner=runner),
            "uptime_s": uptime_s,
            "last_exit_code": _int_or(props.get("ExecMainStatus"), 0),
            "restarts": _int_or(props.get("NRestarts"), 0),
        }

    payload: dict[str, Any] = {
        "host_id": host_id,
        "agent_version": __version__,
        "ts": _now_utc_iso(),
        "pipelines": pipelines,
    }
    gpu = _gpu_snapshot(runner)
    if gpu is not None:
        payload["gpu"] = gpu
    return payload


def _int_or(value: Optional[str], default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# section E.1 -- REQUIRED round-trip status check
# ---------------------------------------------------------------------------


def status_round_trip_dry_run(
    *, install_dir: pathlib.Path, runner: Runner, host_id: str
) -> dict[str, Any]:
    """Section E.1's dry-run-equivalent round trip: invoke the command
    handler in-process (no real broker) exactly as a real
    `mv3dt/<HOST_ID>/cmd` -> `.../cmd/result` round trip would resolve.
    This is what `verify()` falls back to under `--non-interactive` or when
    no live agent/broker answers -- "verify() never depends on the cloud
    webapp existing"."""
    return handle_command(
        {"action": "status", "request_id": str(uuid.uuid4()), "ts": _now_utc_iso()},
        install_dir=install_dir,
        runner=runner,
        dedup=RequestDedup(),
        host_id=host_id,
    )


def _attempt_live_status_round_trip(
    host_id: str, *, timeout_s: float = 5.0
) -> Optional[dict[str, Any]]:
    """Section E.1's primary path: publish `{"action":"status"}` to
    `mv3dt/<HOST_ID>/cmd` against the local broker and wait for a reply on
    `.../cmd/result` -- the actual end-to-end proof that a running
    `mv3dt-agent.service` is consuming commands over MQTT, not just that the
    unit is `active`. Returns `None` on any connection/timeout failure (no
    local broker reachable, no agent listening yet, `paho-mqtt`
    unavailable) -- this never raises; the caller falls back to
    `status_round_trip_dry_run` in that case."""
    try:
        import json

        import paho.mqtt.client as mqtt
    except ImportError:  # pragma: no cover -- paho-mqtt is a hard dependency
        return None

    received: dict[str, Any] = {}

    def _on_message(_client: Any, _userdata: Any, msg: Any) -> None:
        try:
            received["result"] = json.loads(msg.payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            pass

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = _on_message
    try:
        client.connect("127.0.0.1", 1883, keepalive=int(timeout_s) or 1)
    except OSError:
        return None

    try:
        client.subscribe(result_topic(host_id), qos=1)
        client.loop_start()
        client.publish(
            cmd_topic(host_id),
            json.dumps({"action": "status", "request_id": str(uuid.uuid4()), "ts": _now_utc_iso()}),
            qos=1,
        )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and "result" not in received:
            time.sleep(0.1)
    except OSError:
        return None
    finally:
        client.loop_stop()
        client.disconnect()

    return received.get("result")


def status_round_trip_check(
    *,
    ctx: "Context",
    host_id: str,
    install_dir: pathlib.Path,
    runner: Runner,
    live_round_trip: Callable[[str], Optional[dict[str, Any]]] = _attempt_live_status_round_trip,
) -> tuple[bool, dict[str, Any]]:
    """`verify()`'s REQUIRED round-trip status check (section E.1): "publish
    a `{"action":"status"}` command ... and assert a result returns ...
    within a timeout (the end-to-end remote-control proof). Under
    `--non-interactive` / no reachable cloud, fall back to the dry-run
    equivalent ... so verify never depends on the cloud webapp existing."

    `live_round_trip` is injectable so tests exercise both "the live round
    trip answered" and "it didn't, so this fell back to the in-process
    dry-run equivalent" without ever touching a real broker. Returns
    `(ok, result)` -- `ok` is the result's own `ok` field."""
    if not ctx.non_interactive:
        live = live_round_trip(host_id)
        if live is not None:
            return bool(live.get("ok")), live

    result = status_round_trip_dry_run(install_dir=install_dir, runner=runner, host_id=host_id)
    return bool(result.get("ok")), result


# ---------------------------------------------------------------------------
# section E.1 -- polkit positive/negative authorization check
# ---------------------------------------------------------------------------


def check_polkit_authorization(*, run_as_agent_user: Runner, slug: str) -> dict[str, bool]:
    """`verify()`'s REQUIRED positive-AND-negative check (section E.1 /
    B.1.1's closing paragraph): a positive-only check "passes just as
    happily against a rule that grants everything", so this always probes
    both. `run_as_agent_user` runs a command as the agent's (invoking)
    user -- production callers pass `ctx.run_as_user`; tests inject a fake
    that returns canned exit codes, never touching real polkit.

    Returns three booleans, all of which must be True for the rule to be
    considered correctly scoped:
      - `start_pipeline_permitted` -- `systemctl start mv3dt-pipeline@<slug>`
        succeeds (positive case).
      - `start_mosquitto_denied` -- `systemctl start mosquitto` fails
        (negative case: the rule must not grant anything beyond the
        `mv3dt-pipeline@*` unit pattern).
      - `enable_pipeline_denied` -- `systemctl enable mv3dt-pipeline@<slug>`
        fails (negative case: the verb allowlist excludes `enable`, per
        section B.1.1's "boot-enablement stays the installer's job").
    """
    unit = PIPELINE_UNIT_NAME(slug)
    # --no-ask-password: the agent runs headless, so the check must too. On
    # an operator's TTY systemctl would otherwise spawn a polkit password
    # prompt, and an admin typing their password turns both negative probes
    # into successes.
    systemctl = ("systemctl", "--no-ask-password")
    start_pipeline = _run(run_as_agent_user, *systemctl, "start", unit)
    start_mosquitto = _run(run_as_agent_user, *systemctl, "start", "mosquitto")
    enable_pipeline = _run(run_as_agent_user, *systemctl, "enable", unit)
    return {
        "start_pipeline_permitted": _returncode(start_pipeline) == 0,
        "start_mosquitto_denied": _returncode(start_mosquitto) != 0,
        "enable_pipeline_denied": _returncode(enable_pipeline) != 0,
    }


# ---------------------------------------------------------------------------
# The `agent` subcommand (section B) -- MQTT loop
# ---------------------------------------------------------------------------


def _agent_env_path(install_dir: pathlib.Path) -> pathlib.Path:
    return install_dir / "agent" / "agent.env"


def _read_agent_env(path: pathlib.Path) -> dict[str, str]:
    """Same plain-KEY=VALUE shape as `installer.conf`/`laptop.env.example`
    -- broker creds live here, never in `installer.conf` (section D)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def handle_agent_subcommand(argv: list, ctx: "Context") -> int:
    """`mv3dt-installer agent` (section B): connect outbound to the local
    Mosquitto broker, subscribe to `mv3dt/<HOST_ID>/cmd`, dispatch commands
    through `handle_command`, and publish results/status. This is the
    long-running loop the `mv3dt-agent.service` unit's `ExecStart=` runs --
    see this module's docstring for the `require_root()` bootstrap caveat
    that applies to it in production.
    """
    del argv  # section B: no operator-facing flags; config is env/conf-driven.

    import paho.mqtt.client as mqtt  # deferred: keeps this an optional runtime cost

    host_id = resolve_host_id(ctx.conf)
    agent_env = _read_agent_env(_agent_env_path(ctx.install_dir))
    broker_host = agent_env.get("MV3DT_AGENT_BROKER_HOST", "127.0.0.1")
    broker_port = int(agent_env.get("MV3DT_AGENT_BROKER_PORT", "1883"))
    username = agent_env.get("MV3DT_AGENT_BROKER_USER")
    password = agent_env.get("MV3DT_AGENT_BROKER_PASS")

    dedup = RequestDedup()
    runner: Runner = lambda argv_, **kwargs: ctx.run_root(*argv_, **kwargs)  # noqa: E731

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    if username:
        client.username_pw_set(username, password)

    def _on_message(_client: Any, _userdata: Any, msg: Any) -> None:
        import json

        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            payload = None
        result = handle_command(
            payload, install_dir=ctx.install_dir, runner=runner, dedup=dedup, host_id=host_id
        )
        client.publish(result_topic(host_id), json.dumps(result), qos=1)

    def _on_connect(client_: Any, _userdata: Any, _flags: Any, *_args: Any) -> None:
        client_.subscribe(cmd_topic(host_id), qos=1)
        status = build_status_payload(host_id=host_id, install_dir=ctx.install_dir, runner=runner)
        import json

        client_.publish(status_topic(host_id), json.dumps(status), qos=1, retain=True)

    client.on_connect = _on_connect
    client.on_message = _on_message

    ctx.log.info(f"mv3dt-agent: connecting to {broker_host}:{broker_port} (host_id={host_id})")
    client.connect(broker_host, broker_port, keepalive=60)
    client.loop_forever()
    return 0  # pragma: no cover -- loop_forever blocks until the process is signaled


# ---------------------------------------------------------------------------
# section E -- The Step
# ---------------------------------------------------------------------------


def _bind_run_root(ctx: "Context") -> Runner:
    return lambda argv, **kwargs: ctx.run_root(*argv, **kwargs)


def _bind_run_as_user(ctx: "Context") -> Runner:
    return lambda argv, **kwargs: ctx.run_as_user(*argv, **kwargs)


class Step6RemoteSupervision:
    """STEP-6-REMOTE-SUPERVISION.md section E: module identity."""

    id = "step6_remote_supervision"
    title = "Remote supervision"

    # Doc 08 §3.1. The polkit rule is its own phase because it is written
    # before the agent is ever started against it (section B.1.1), and an
    # operator debugging a permission refusal needs to see that it ran.
    phases = (
        "systemd units",
        "per-project pipeline units",
        "polkit rule and agent",
    )
    order = 6

    # -- preflight ------------------------------------------------------

    def preflight(self, ctx: "Context") -> StepResult:
        registry = step5_mod.list_projects(ctx.install_dir)
        if not registry:
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message="no registered projects to supervise",
                user_actions=[
                    UserAction(text="Complete Step 5 to create/calibrate a project first.")
                ],
            )

        runner = _bind_run_root(ctx)
        active = _run(runner, "systemctl", "is-active", "--quiet", "mosquitto")
        if _returncode(active) != 0:
            return StepResult(
                status=StepStatus.FAILED,
                message=(
                    "mosquitto is not active; re-run the installer so Step 1 "
                    "reinstalls the broker (installer/plan/STEP-1-PREREQUISITES.md "
                    "section 3.2)"
                ),
            )

        gate = ctx.conf.get(config_mod.GATE_REMOTE_SUPERVISION, "off")
        if gate == "remote":
            missing = missing_cloud_broker_keys(ctx.conf)
            if missing:
                return StepResult(
                    status=StepStatus.USER_ACTION_REQUIRED,
                    message="remote mode requested but the cloud broker endpoint is not set",
                    user_actions=[
                        UserAction(text=f"Set {key} in installer.conf.") for key in missing
                    ],
                )

        return StepResult(status=StepStatus.COMPLETE)

    # -- run --------------------------------------------------------------

    def run(self, ctx: "Context") -> StepResult:
        runner = _bind_run_root(ctx)

        ctx.progress.phase(1)
        ctx.progress.task("installing systemd units")
        written = _install_units(
            ctx,
            {
                PIPELINE_UNIT_TEMPLATE: render_pipeline_unit(ctx.user.name),
                WATCHDOG_UNIT_TEMPLATE: render_watchdog_unit(ctx.user.name),
                AGENT_UNIT_NAME: render_agent_unit(ctx.user.name),
            },
        )
        if written:
            systemd_mod.daemon_reload(runner=runner)

        ctx.progress.phase(2)
        ctx.progress.task("enabling per-project pipeline and watchdog units")
        for entry in step5_mod.list_projects(ctx.install_dir):
            for unit in (PIPELINE_UNIT_NAME(entry.slug), WATCHDOG_UNIT_NAME(entry.slug)):
                systemd_mod.enable_now(unit, runner=runner)
                ctx.report_installed(unit, __version__)

        # section B.1.1: the scoped polkit rule, written before the agent is
        # ever started against it.
        ctx.progress.phase(3)
        ctx.progress.task("writing the scoped polkit rule")
        rule_content = render_polkit_rule(ctx.user.name)
        rule_path = POLKIT_RULES_DIR / POLKIT_RULE_NAME
        rule_path.parent.mkdir(parents=True, exist_ok=True)
        existing = rule_path.read_text(encoding="utf-8") if rule_path.is_file() else None
        if existing != rule_content:
            rule_path.write_text(rule_content, encoding="utf-8")
            rule_path.chmod(0o644)
            ctx.report_installed(POLKIT_RULE_NAME, __version__)
        else:
            ctx.report_already_installed(POLKIT_RULE_NAME, __version__)

        # section D: remote mode's hardened broker + bridge drop-ins.
        gate = ctx.conf.get(config_mod.GATE_REMOTE_SUPERVISION, "off")
        decision = decide_broker_topology(ctx.conf, gate_value=gate)
        if decision.missing_cloud_endpoint:
            return StepResult(
                status=StepStatus.USER_ACTION_REQUIRED,
                message="remote mode requested but the cloud broker endpoint is not set",
                user_actions=[
                    UserAction(text=f"Set {key} in installer.conf.")
                    for key in missing_cloud_broker_keys(ctx.conf)
                ],
            )
        # Local-only mode (default) intentionally makes no broker config
        # change (section D: "In local-only mode the broker is left at the
        # simple-testing posture"). Writing the hardened
        # /etc/mosquitto/conf.d/mv3dt-remote.conf + mv3dt-bridge.conf drop-ins
        # and provisioning the agent's broker credentials is human/cloud-side
        # credential provisioning (section D's closing bullet, section 8) --
        # flagged there, not fabricated here.

        systemd_mod.enable_now(AGENT_UNIT_NAME, runner=runner)

        return StepResult(status=StepStatus.COMPLETE)

    # -- verify -------------------------------------------------------------

    def verify(self, ctx: "Context") -> StepResult:
        runner = _bind_run_root(ctx)

        if not systemd_mod.is_enabled(AGENT_UNIT_NAME, runner=runner):
            return StepResult(status=StepStatus.FAILED, message=f"{AGENT_UNIT_NAME} not enabled")
        if not systemd_mod.is_active(AGENT_UNIT_NAME, runner=runner):
            return StepResult(status=StepStatus.FAILED, message=f"{AGENT_UNIT_NAME} not active")

        # section E.1 REQUIRED: "publish a {"action":"status"} command ...
        # and assert a result returns ... the end-to-end remote-control
        # proof" -- falls back to the in-process dry-run equivalent under
        # --non-interactive / no reachable cloud (status_round_trip_check's
        # own contract), so this never depends on the cloud webapp existing.
        host_id = resolve_host_id(ctx.conf)
        round_trip_ok, round_trip_result = status_round_trip_check(
            ctx=ctx, host_id=host_id, install_dir=ctx.install_dir, runner=runner
        )
        if not round_trip_ok:
            return StepResult(
                status=StepStatus.FAILED,
                message=f"status round-trip check failed: {round_trip_result}",
            )

        for entry in step5_mod.list_projects(ctx.install_dir):
            unit = PIPELINE_UNIT_NAME(entry.slug)
            if not systemd_mod.is_enabled(unit, runner=runner):
                return StepResult(status=StepStatus.FAILED, message=f"{unit} not enabled")
            if not (
                systemd_mod.is_active(unit, runner=runner)
                or _systemctl_state(unit, "active", runner=runner) == "activating"
            ):
                return StepResult(
                    status=StepStatus.FAILED,
                    message=f"{unit} not active/activating; check `journalctl -u {unit}`",
                )

            watchdog = WATCHDOG_UNIT_NAME(entry.slug)
            if not systemd_mod.is_enabled(watchdog, runner=runner):
                return StepResult(status=StepStatus.FAILED, message=f"{watchdog} not enabled")
            if not systemd_mod.is_active(watchdog, runner=runner):
                return StepResult(
                    status=StepStatus.FAILED,
                    message=f"{watchdog} not active; check `journalctl -u {watchdog}`",
                )

            auth = check_polkit_authorization(
                run_as_agent_user=_bind_run_as_user(ctx), slug=entry.slug
            )
            if not all(auth.values()):
                return StepResult(
                    status=StepStatus.FAILED,
                    message=f"polkit authorization scope check failed for {unit}: {auth}",
                )

        return StepResult(status=StepStatus.COMPLETE)

    # -- report -----------------------------------------------------------

    def report(self, ctx: "Context") -> None:
        runner = _bind_run_root(ctx)
        host_id = resolve_host_id(ctx.conf)
        gate = ctx.conf.get(config_mod.GATE_REMOTE_SUPERVISION, "off")
        decision = decide_broker_topology(ctx.conf, gate_value=gate)

        lines = [
            "Remote supervision installed.",
            f"  Host ID:        {host_id}",
            f"  Broker mode:    {decision.topology.value}",
            f"  Command topic:  {cmd_topic(host_id)}",
            f"  Result topic:   {result_topic(host_id)}",
            f"  Status topic:   {status_topic(host_id)}",
            f"  Agent unit:     {AGENT_UNIT_NAME} "
            f"(enabled={systemd_mod.is_enabled(AGENT_UNIT_NAME, runner=runner)}, "
            f"active={systemd_mod.is_active(AGENT_UNIT_NAME, runner=runner)})",
            "  Agent logs:     journalctl -u mv3dt-agent",
        ]
        for entry in step5_mod.list_projects(ctx.install_dir):
            unit = PIPELINE_UNIT_NAME(entry.slug)
            lines.append(
                f"  {unit}: enabled={systemd_mod.is_enabled(unit, runner=runner)} "
                f"active={systemd_mod.is_active(unit, runner=runner)} "
                f"(logs: journalctl -u {unit})"
            )
            watchdog = WATCHDOG_UNIT_NAME(entry.slug)
            lines.append(
                f"  {watchdog}: enabled={systemd_mod.is_enabled(watchdog, runner=runner)} "
                f"active={systemd_mod.is_active(watchdog, runner=runner)}"
            )
        ctx.log.info("\n".join(lines))


register(Step6RemoteSupervision())
app_mod.register_subcommand("agent", handle_agent_subcommand, requires_root=False)
step5_mod.register_removal_hook(_disable_pipeline_unit_before_remove)
