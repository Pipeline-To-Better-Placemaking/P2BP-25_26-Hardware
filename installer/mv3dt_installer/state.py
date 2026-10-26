"""Resumable state machine for mv3dt-installer (doc 00 §6).

Owns `state.json` — the single source of truth for which install steps have
completed, the chosen install directory, and any pending-reboot marker. This
module is framework-only: it has no knowledge of what any given step
actually does (that lives in `steps/stepN_*.py`, out of scope here); it only
persists whatever `status`/`finished_utc`/`reboot_pending` the dispatch loop
(`app.py`, a later PR) tells it to.

Public API (doc 00 §6.3):
    write_json_atomic(path, data)  -- standalone, reusable atomic JSON write.
        Shared contract: also used by the Step 5 registry and every
        run-state file STEP-7 writes (doc 00 §6.3 "REQUIRED" note). Callers
        outside this module should import and call this function directly
        rather than reimplementing the flush+fsync+replace sequence.
    State            -- dataclass matching the §6.2 schema.
    StepEntry        -- one `state["steps"][step_id]` record.
    RebootPending     -- the `state["reboot_pending"]` record.
    StateMachine      -- wraps a `state.json` path with the §6.3 operations:
        .load() -> State
        .save(state: State) -> None
        .status(step_id) -> StepStatus
        .set_status(step_id, status) -> None
        .mark_complete(step_id) -> None
        .set_reboot_pending(step_id, boot_id) -> None
        .clear_reboot_pending() -> None
        .all_complete() -> bool

Readers are forgiving by design (doc 00 §6.3): a missing `state.json` yields
the empty/default state, and a `json.JSONDecodeError` (e.g. a partially
written file left by a crash between `open(tmp, "w")` and `os.replace`) is
treated as "absent", never as a fatal error.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any

from mv3dt_installer.steps import StepStatus

__all__ = [
    "write_json_atomic",
    "CANONICAL_STATE_PATH",
    "DEFAULT_INSTALL_DIR",
    "SCHEMA_VERSION",
    "STEP_IDS",
    "StepEntry",
    "RebootPending",
    "State",
    "default_state",
    "StateMachine",
]

# Canonical path (doc 00 §6.1). Root-owned, chmod 0644; its parent directory
# is created chmod 0755. Stays fixed here even if `--install-dir` moves the
# install root elsewhere.
CANONICAL_STATE_PATH = pathlib.Path("/var/lib/mv3dt-installer/state.json")

# Default install directory (doc 00 §11.1) — the `install_dir` value a fresh
# state starts with before the operator picks (or `--install-dir` sets) one.
DEFAULT_INSTALL_DIR = "/opt/mv3dt"

SCHEMA_VERSION = 1

# The full step-id namespace from the doc 00 §6.2 schema example, in
# dispatch order. Used to seed a fresh State's `steps` map (all PENDING) and
# to evaluate all_complete() against the complete set, not just whichever
# ids happen to already have an entry.
STEP_IDS: tuple[str, ...] = (
    "step1_prerequisites",
    "step2_deepstream_sdk",
    "step3_amc_launcher",
    "step4_calib_output_wiring",
    "step5_per_project_exes",
    "step6_remote_supervision",
    "step7_webapp_integration",
    "step8_tracking_history",
)


def _now_utc_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
    )


def write_json_atomic(path: pathlib.Path, data: Any) -> None:
    """Write `data` as JSON to `path` atomically (doc 00 §6.3 REQUIRED).

    Standalone and generic on purpose: other modules (the Step 5 registry,
    STEP-7 run-state fan-in files) import and call this exact function
    rather than reimplementing it. The flush + fsync pair before
    `os.replace` is the part that matters: without it, a power loss mid-
    install can leave a zero-length target file and lose every completed
    step.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)          # atomic within a filesystem


@dataclass
class StepEntry:
    """One `state["steps"][step_id]` record (doc 00 §6.2)."""

    status: StepStatus
    finished_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"status": self.status.value}
        if self.finished_utc is not None:
            d["finished_utc"] = self.finished_utc
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> StepEntry:
        return StepEntry(
            status=StepStatus(d.get("status", StepStatus.PENDING.value)),
            finished_utc=d.get("finished_utc"),
        )


@dataclass
class RebootPending:
    """`state["reboot_pending"]` record (doc 00 §6.2, §7.1)."""

    requested_by: str
    boot_id_at_request: str
    requested_utc: str

    def to_dict(self) -> dict[str, str]:
        return {
            "requested_by": self.requested_by,
            "boot_id_at_request": self.boot_id_at_request,
            "requested_utc": self.requested_utc,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> RebootPending:
        return RebootPending(
            requested_by=d["requested_by"],
            boot_id_at_request=d["boot_id_at_request"],
            requested_utc=d["requested_utc"],
        )


@dataclass
class State:
    """The full `state.json` document (doc 00 §6.2)."""

    schema_version: int = SCHEMA_VERSION
    install_dir: str = DEFAULT_INSTALL_DIR
    created_utc: str = field(default_factory=_now_utc_iso)
    updated_utc: str = field(default_factory=_now_utc_iso)
    steps: dict[str, StepEntry] = field(default_factory=dict)
    reboot_pending: RebootPending | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "install_dir": self.install_dir,
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
            "steps": {
                step_id: entry.to_dict() for step_id, entry in self.steps.items()
            },
            "reboot_pending": (
                self.reboot_pending.to_dict()
                if self.reboot_pending is not None
                else None
            ),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> State:
        steps_raw = d.get("steps") or {}
        reboot_raw = d.get("reboot_pending")
        return State(
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            install_dir=d.get("install_dir", DEFAULT_INSTALL_DIR),
            created_utc=d.get("created_utc", _now_utc_iso()),
            updated_utc=d.get("updated_utc", _now_utc_iso()),
            steps={
                step_id: StepEntry.from_dict(entry)
                for step_id, entry in steps_raw.items()
            },
            reboot_pending=(
                RebootPending.from_dict(reboot_raw) if reboot_raw else None
            ),
        )


def default_state() -> State:
    """The empty/default state: fresh install, no step has run yet.

    `steps` is pre-seeded with every known id (§ `STEP_IDS`) at `PENDING` so
    `all_complete()` is meaningfully `False` on a fresh install rather than
    vacuously `True` over an empty map.
    """
    now = _now_utc_iso()
    return State(
        schema_version=SCHEMA_VERSION,
        install_dir=DEFAULT_INSTALL_DIR,
        created_utc=now,
        updated_utc=now,
        steps={step_id: StepEntry(status=StepStatus.PENDING) for step_id in STEP_IDS},
        reboot_pending=None,
    )


class StateMachine:
    """Wraps a `state.json` path with the doc 00 §6.3 operations.

    `path` defaults to the canonical `/var/lib/mv3dt-installer/state.json`
    (doc 00 §6.1). Tests (and any future `--install-dir`-aware caller that
    still wants an isolated file) should pass an explicit `path` instead —
    this class never assumes the canonical path is writable or even present.
    """

    def __init__(self, path: pathlib.Path | None = None) -> None:
        self.path = pathlib.Path(path) if path is not None else CANONICAL_STATE_PATH

    def load(self) -> State:
        """Read `state.json`, or the empty default if absent/corrupt."""
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return default_state()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # A partially-written file (e.g. a crash between the write and
            # the atomic replace) must never wedge the installer.
            return default_state()
        return State.from_dict(data)

    def save(self, state: State) -> None:
        """Atomically persist `state`, stamping `updated_utc` and fixing
        the permissions required by doc 00 §6.1 (dir 0755, file 0644)."""
        state.updated_utc = _now_utc_iso()
        write_json_atomic(self.path, state.to_dict())
        self.path.parent.chmod(0o755)
        self.path.chmod(0o644)

    def status(self, step_id: str) -> StepStatus:
        """Current status of `step_id`; `PENDING` if it has no entry yet."""
        state = self.load()
        entry = state.steps.get(step_id)
        return entry.status if entry is not None else StepStatus.PENDING

    def set_status(self, step_id: str, status: StepStatus) -> None:
        state = self.load()
        entry = state.steps.get(step_id)
        if entry is None:
            state.steps[step_id] = StepEntry(status=status)
        else:
            entry.status = status
        self.save(state)

    def mark_complete(self, step_id: str) -> None:
        """Set `step_id` to `COMPLETE` and stamp `finished_utc` (doc 00 §6.3)."""
        state = self.load()
        finished = _now_utc_iso()
        entry = state.steps.get(step_id)
        if entry is None:
            state.steps[step_id] = StepEntry(
                status=StepStatus.COMPLETE, finished_utc=finished
            )
        else:
            entry.status = StepStatus.COMPLETE
            entry.finished_utc = finished
        self.save(state)

    def set_reboot_pending(self, step_id: str, boot_id: str) -> None:
        """Record a pending reboot requested by `step_id` (doc 00 §7.1)."""
        state = self.load()
        state.reboot_pending = RebootPending(
            requested_by=step_id,
            boot_id_at_request=boot_id,
            requested_utc=_now_utc_iso(),
        )
        self.save(state)

    def clear_reboot_pending(self) -> None:
        state = self.load()
        state.reboot_pending = None
        self.save(state)

    def all_complete(self) -> bool:
        """True once every known step (`STEP_IDS`) is `COMPLETE`.

        Used to print the final success banner (doc 00 §6.3).
        """
        state = self.load()
        for step_id in STEP_IDS:
            entry = state.steps.get(step_id)
            if entry is None or entry.status is not StepStatus.COMPLETE:
                return False
        return True
