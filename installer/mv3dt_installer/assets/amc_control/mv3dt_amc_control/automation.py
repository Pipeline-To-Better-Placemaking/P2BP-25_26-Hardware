"""Runs AMC 3.3.0's seven-step workflow, pausing only where a person is needed.

    1 Project Setup        automated: reuse the installer's AMC project, create one, or use
                           the web project's bound AMC project (mode web_project)
    2 Video Configuration  automated: installer footage as cam_00.mp4..., floor plan as layout.png
    3 Parameters           automated: AMC's current configuration with the settings file and
                           overrides merged in, the floor plan scale, focal lengths; with no
                           known scale the run waits for the user to set it in AMC
    4 Rectification        automated once the videos are in AMC: AMC's auto estimate committed
                           as its UI does (mode auto), or footage that is already linear staged
                           as is (mode linear); mode ui leaves it to the UI
    5 Manual Alignment     manual the first time; a saved alignment file is re-uploaded after that
    6 Execute Calibration  automated: re-post the run's scale, verify the project is READY,
                           calibrate, wait until AMC's amc_state is COMPLETED
    7 Results              automated: evaluation statistics, optional result ZIP download

Every step that needs a person is marked "waiting". Steps 1-5 all run first,
so everything that can be uploaded is uploaded; the run then pauses before
step 6 until the user presses Continue. Progress is saved to disk after every
change, so a run can be continued after the service restarts.

The floor plan scale (installer/plan/AMC-RECTIFICATION-AND-SCALE.md section 4):
AMC keeps one current configuration, so every config post is the whole object,
built from GET /v1/config, then settings_file, then settings_overrides, then
parameters.scale.settings_key set to the scale. The scale comes from
parameters.scale.source: the web app's floor plan, else (web_project) the
scale remembered in project.json for the same layout.png hash; or
fixed_px_per_m; or none, which neither posts nor asks for one. With no scale
the run posts AMC's factory value and waits for the user to set the scale in
AMC's Parameters step; Continue accepts only a positive value that is not the
factory default, saves it on the job and remembers it with the hash of the
layout this run uploaded. Step 6 reads AMC's configuration first, takes a
correction made during the pause, re-posts the job's scale, then verifies.

Continue never takes a step left to the user on trust (section 5): step 2 needs
get_project_info's video_files_count of at least 1 and layout_file, step 3 a
confirmed scale, step 4 in mode ui a COMPLETED rectification, and step 5
alignment_file. A step AMC does not confirm stays waiting with what AMC
reports, the other steps are still checked, and the run pauses again listing
every unfinished step.

In project.mode web_project a run is locked to the web project's workstation
folder and its AMC project (installer/plan/AMC-WEB-PROJECTS.md section 6):
footage comes from <folder>/footage/, results go to <folder>/calibration/,
and every finished run leaves <folder>/runs/<job id>.json.

A web_project run started with Record (installer/plan/AMC-FOOTAGE-RECORDING.md
section 6) first records new footage in step 2: it writes the request file
<install_dir>/amc_control/state/record-request.json and runs
`[sudo -n] <install_dir>/bin/mv3dt-installer record-footage --pending`, which
records every camera into <folder>/footage/ and reports on stdout (section 5).
Only the run's first pass records; Continue never records again.

Cancelling a web_project run undoes it (installer/plan/AMC-PROJECT-LIFECYCLE.md
section 5): the AMC project is replaced by a new, empty one, footage the run
recorded is deleted and the earlier footage moved back, and the run's
calibration/<UTC> folder is removed. The remembered scale in project.json is
kept: it describes the floor plan, not the run. Each finished undo action is saved, so an
undo cut off by a service restart carries on without repeating any of them.
AMC cannot stop a rectification, so the undo first waits while one is RUNNING
(up to rectification.api.timeout_seconds), then deletes the project either way
(installer/plan/AMC-RECTIFICATION-AND-SCALE.md section 3.3).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from .controller import _read_conf
from .amc_api import AmcApi, AmcApiError, default_base_url
from .automation_config import AutomationConfig, ConfigError, deep_merge, load_config
from .calibration_spec import CalibrationSpec, SpecError, download_floorplan_png
from .projects import STATE_FAILED, STATE_READY, ProjectNotFound, ProjectStore, default_projects_root

STEPS: list[tuple[str, int, str]] = [
    ("project_setup", 1, "Project Setup"),
    ("video_configuration", 2, "Video Configuration"),
    ("parameters", 3, "Parameters"),
    ("rectification", 4, "Rectification"),
    ("manual_alignment", 5, "Manual Alignment"),
    ("execute", 6, "Execute Calibration"),
    ("results", 7, "Results"),
]
# Steps after which nothing may run while an earlier step still needs the user.
GATED = {"execute", "results"}
ACTIVE = {"running"}
RESUMABLE = {"waiting_for_user", "failed", "interrupted"}
TERMINAL = {"completed", "failed", "cancelled"}
WEB_PROJECT = "web_project"

# Recording from step 2 (AMC-FOOTAGE-RECORDING.md sections 4 to 6).
RECORD_MIN_SECONDS = 30
RECORD_MAX_SECONDS = 1800
RECORD_REQUEST_NAME = "record-request.json"
RECORD_SUBCOMMAND = ("record-footage", "--pending")
RECORD_EXIT_FAILED, RECORD_EXIT_INVALID, RECORD_EXIT_CANCELLED = 1, 2, 3
# The service cannot signal the root-owned sudo process, so it cancels the
# recorder through stdin and never kills it (that would orphan its ffmpegs).
RECORD_CANCEL_GRACE_SECONDS = 60.0
# The recorder is cancelled once it runs this much longer than the recording.
RECORD_OVERRUN_SECONDS = 180.0
STDERR_TAIL_LINES = 10
# Step 3 (AMC-RECTIFICATION-AND-SCALE.md section 4.2).
SCALE_WAIT = ("Set the floor plan scale in AMC's Parameters step: measure two points on the floor plan, "
              "or enter its pixels per metre, then Save and press Continue")
# AMC 3.3.0's factory layout_px_per_m, used only when GET /v1/config/defaults lacks the key.
FACTORY_LAYOUT_PX_PER_M = 24.22
# The recorder's manifest in footage/ (AMC-FOOTAGE-RECORDING.md section 5).
FOOTAGE_MANIFEST = ".mv3dt-footage.json"
CANCELLING_MESSAGE = "Cancelling: undoing this run"
# Undo actions, in order, as recorded in Job.undone (AMC-PROJECT-LIFECYCLE.md section 5.2).
AMC_DELETED, AMC_CREATED, AMC_BOUND = "amc_deleted", "amc_created", "amc_bound"
FOOTAGE_CLEARED, FOOTAGE_RESTORED, RESULTS_REMOVED = "footage_cleared", "footage_restored", "results_removed"


def default_record_argv(install_dir: Path) -> list[str]:
    """`[sudo -n] <install_dir>/bin/mv3dt-installer record-footage --pending`.

    Same convention as the controller: `sudo -n` unless AMC_USE_SUDO=0. The
    command line is fixed; everything variable is in the request file.
    """
    use_sudo = os.environ.get("AMC_USE_SUDO", "1") != "0"
    return (["sudo", "-n"] if use_sudo else []) + [str(Path(install_dir) / "bin" / "mv3dt-installer"),
                                                    *RECORD_SUBCOMMAND]


def _clock_text(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _pump(stream, sink: Callable[[str], Any], done: Callable[[], Any] | None) -> None:
    """Feed a child's output to `sink` line by line, then call `done`."""
    try:
        for line in stream:
            sink(line.rstrip("\r\n"))
    except (OSError, ValueError):
        pass
    finally:
        if done is not None:
            done()


def _json_object(line: str) -> dict | None:
    try:
        value = json.loads(line)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


_RECORD_REASONS = {
    RECORD_EXIT_FAILED: "a camera failed or there is not enough free space",
    RECORD_EXIT_INVALID: "the workstation refused the recording request or is not set up for it",
}


def _record_failure(rc: int, final: dict, stderr_tail: list[str]) -> str:
    """Step 2's message for a recorder that exited non-zero with a result line."""
    reason = str(final.get("reason") or "")
    text = f"recording footage failed (exit {rc}): {reason or _RECORD_REASONS.get(rc, 'see the service log')}"
    failed = [f for f in final.get("failed") or [] if isinstance(f, dict)]
    if failed:
        cams = "; ".join(f"{f.get('id', '?')}" + (f" ({f['ip']})" if f.get("ip") else "")
                         + f": {f.get('error') or 'failed'}" for f in failed)
        text += f"; failed cameras: {cams}"
    elif not reason and stderr_tail:
        text += "; last output: " + " | ".join(stderr_tail)
    return text


class Busy(RuntimeError):
    pass


class NotProvisioned(Busy):
    """The web project has no ready workstation folder (a 409 like Busy)."""


class NothingToResume(RuntimeError):
    pass


class _Cancelled(Exception):
    pass


class StepFailed(Exception):
    pass


class _PauseForUser(Exception):
    """Raised by a step that found it cannot go on until the user acts in AMC."""


@dataclass
class Outcome:
    state: str  # done | skipped | waiting
    message: str = ""      # for waiting: what the user has to do
    done: str = ""         # what the automation already finished in this step
    needs_user: bool = True


def _now() -> int:
    return int(time.time())


@dataclass
class Job:
    job_id: str
    project_id: str
    project_title: str
    started_by: str
    floorplan: dict | None
    state: str = "running"
    message: str = ""
    amc_project_id: str = ""
    amc_project_name: str = ""
    amc_project_from_installer: bool = False
    # The web project's workstation folder in mode web_project, else "".
    folder: str = ""
    # True once step 2 uploaded this run's videos, so Continue never sends them twice.
    videos_uploaded: bool = False
    # Seconds of new footage step 2 records first (web_project with Record), else 0.
    record_seconds: int = 0
    # True once step 2 recorded this run's footage. A run records at most once:
    # Continue never records, so a new recording needs a new run.
    footage_recorded: bool = False
    # web_project cancel (AMC-PROJECT-LIFECYCLE.md section 5.1). True once the run is about
    # to change its AMC project (an upload, a config post, verify, calibrate).
    amc_changed: bool = False
    # Step 4 in rectification.mode auto or linear (AMC-RECTIFICATION-AND-SCALE.md section 3.2):
    # True once this run is about to start AMC's rectification, and the phase it is in
    # (estimate, generate or linear), so a resumed run picks up the right wait.
    rectification_started: bool = False
    rectification_phase: str = ""
    # Step 6: True once this run is about to post calibrate, so a Continue after a failure or
    # a restart waits for a calibration AMC is running instead of starting another.
    calibration_started: bool = False
    # Step 6: True once AMC accepted this run's calibrate post or was seen calibrating after it.
    # Only then is a COMPLETED amc_state this run's result rather than an earlier calibration's.
    calibration_posted: bool = False
    # The floor plan scale (AMC-RECTIFICATION-AND-SCALE.md section 4): the SHA-256 of the
    # layout.png step 2 uploaded ("" when this run uploaded none), saved before the upload,
    # and the scale step 3 posted or the user confirmed in AMC (None until there is one).
    layout_sha256: str = ""
    layout_px_per_m: float | None = None
    # The clips the recorder reported (its final line's files), saved with footage_recorded.
    recorded_files: list[str] = field(default_factory=list)
    # The calibration/<UTC> folder name step 7 writes, saved before it is created.
    calibration_dir: str = ""
    # True from cancel until the undo finishes the job cancelled.
    cancelling: bool = False
    # Undo actions finished so far, each saved as it completes.
    undone: list[str] = field(default_factory=list)
    # The replacement AMC project, saved as soon as create_project returns.
    reset_amc_project_id: str = ""
    steps: list[dict] = field(default_factory=list)
    result: dict = field(default_factory=dict)
    started_at: int = field(default_factory=_now)
    updated_at: int = field(default_factory=_now)

    def step(self, key: str) -> dict:
        return next(s for s in self.steps if s["Key"] == key)

    def public(self) -> dict:
        return {
            "JobId": self.job_id,
            "ProjectId": self.project_id,
            "ProjectTitle": self.project_title,
            "State": self.state,
            "Message": self.message,
            "AmcProjectId": self.amc_project_id,
            "AmcProjectName": self.amc_project_name,
            "Steps": [dict(s) for s in self.steps],
            "UserActions": [s["Message"] for s in self.steps
                            if s["State"] == "waiting" and s.get("NeedsUser", True)],
            "Result": dict(self.result),
            "StartedBy": self.started_by,
            "StartedAt": self.started_at,
            "UpdatedAt": self.updated_at,
        }


# Polls after a rectification request during which AMC may still report the earlier state.
LAG_POLLS = 3
RECTIFICATION_MODELS = ("simple_radial", "radial", "simple_divisional")


def _rectification_params(camera: dict) -> tuple[dict, bool]:
    """One camera's /manual entry and whether it had an estimate.

    committed, else the auto estimate, mapped as AMC's UI maps it; a camera
    with neither gets simple_divisional with k1 = k2 = 0, as AMC's UI would.
    """
    t = camera.get("committed") or camera.get("auto_estimate")
    if not isinstance(t, dict):
        return {"model": "simple_divisional", "k1": 0, "k2": 0}, False
    model = t.get("model") if t.get("model") in RECTIFICATION_MODELS else "simple_divisional"
    k1, k2 = t.get("k1"), t.get("k2")
    return {"model": model, "k1": 0 if k1 is None else k1, "k2": 0 if k2 is None else k2}, True


def _num(v: Any) -> str:
    return f"{v:.4g}" if isinstance(v, (int, float)) else str(v)


def _scale_value(config: Any, key: str) -> float | None:
    """The positive, finite number at `key` in an AMC configuration, else None."""
    v = _dig(config, key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        return None
    return float(v)


def _same_scale(a: float, b: float) -> bool:
    return abs(a - b) < 1e-9


def _plain_name(name: Any) -> bool:
    """A single path component: never a directory, a parent or a hidden way out of the folder."""
    return isinstance(name, str) and name not in ("", ".", "..") and "/" not in name and "\\" not in name


def _new_steps() -> list[dict]:
    return [{"Key": k, "Number": n, "Title": t, "State": "pending", "Message": "", "Done": "", "NeedsUser": False}
            for k, n, t in STEPS]


def _dig(data: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        if not isinstance(data, dict) or part not in data:
            return None
        data = data[part]
    return data


def _has_key(data: Any, dotted: str) -> bool:
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not isinstance(data, dict) or part not in data:
            return False
        data = data[part]
    return isinstance(data, dict) and parts[-1] in data


def _set_key(data: dict, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    for part in parts[:-1]:
        data = data[part]
    data[parts[-1]] = value


class CalibrationRunner:
    def __init__(
        self,
        controller,
        *,
        config_loader: Callable[[], AutomationConfig] = load_config,
        http_client_factory: Callable[[], httpx.Client] = httpx.Client,
        download_client_factory: Callable[[], httpx.Client] = httpx.Client,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.time,
        threaded: bool = True,
        store: ProjectStore | None = None,
        record_argv: Callable[[], list[str]] | None = None,
    ):
        self.controller = controller
        self._record_argv = record_argv or (lambda: default_record_argv(Path(self.controller.install_dir)))
        self.record_cancel_grace = RECORD_CANCEL_GRACE_SECONDS
        self.record_overrun = RECORD_OVERRUN_SECONDS
        self._resuming = False
        self.store = store if store is not None else ProjectStore(default_projects_root())
        self._load_config = config_loader
        self._http_factory = http_client_factory
        self._download_factory = download_client_factory
        self._sleep = sleep
        self._clock = clock
        self._threaded = threaded
        self._lock = threading.RLock()
        self._job: Job | None = None
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._loaded = False
        # True while a thread owns the cancel undo, so it never runs twice at once.
        self._undoing = False

    # ---- persistence ------------------------------------------------------
    @property
    def state_path(self) -> Path:
        return Path(self.controller.state_dir) / "amc_automation_job.json"

    def _save(self) -> None:
        job = self._job
        if job is None:
            return
        job.updated_at = _now()
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")
            os.replace(tmp, self.state_path)
        except OSError:
            pass  # status still works from memory

    def _load_once(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            job = Job(**data)
        except (OSError, ValueError, TypeError):
            return
        if job.cancelling:
            # An undo cut off by a restart carries on; it is never continued as a run.
            job.state, job.message = "running", CANCELLING_MESSAGE
            for s in job.steps:
                if s["State"] == "running":
                    s["State"] = "pending"
            self._job = job
            self._save()
            self._start_undo(job)
            return
        if job.state in ACTIVE:
            job.state = "interrupted"
            job.message = "The workstation service restarted during this run. Press Continue to carry on."
            for s in job.steps:
                if s["State"] == "running":
                    s["State"] = "pending"
        self._job = job

    # ---- public API ---------------------------------------------------------
    def status(self) -> dict | None:
        with self._lock:
            self._load_once()
            return self._job.public() if self._job else None

    def is_active(self) -> bool:
        with self._lock:
            self._load_once()
            return bool(self._job and self._job.state in ACTIVE)

    def uses_amc(self) -> bool:
        """True while a run is executing or paused for the user's work inside AMC."""
        with self._lock:
            self._load_once()
            return bool(self._job and self._job.state in ACTIVE | {"waiting_for_user"})

    def start(self, spec: CalibrationSpec, record_seconds: int | None = None) -> dict:
        cfg = self.config()  # fail before creating a job if the config is broken
        if record_seconds is not None:
            if cfg.project.mode != WEB_PROJECT:
                raise SpecError("recording footage needs project.mode web_project in the automation config")
            if isinstance(record_seconds, bool) or not isinstance(record_seconds, int) \
                    or not RECORD_MIN_SECONDS <= record_seconds <= RECORD_MAX_SECONDS:
                raise SpecError(f"Record.Seconds must be a whole number from {RECORD_MIN_SECONDS} "
                                f"to {RECORD_MAX_SECONDS}")
        with self._lock:
            self._load_once()
            if self._job and self._job.state in ACTIVE:
                raise Busy("a calibration run is already in progress")
            if self._job and self._job.state == "waiting_for_user":
                raise Busy("a calibration run is waiting for its manual steps; continue or cancel it first")
            folder = ""
            if cfg.project.mode == WEB_PROJECT:
                binding = self.store.get_by_pid(spec.project_id)
                if binding is None or binding.state != STATE_READY:
                    raise NotProvisioned("set this project up on the workstation first")
                if spec.folder is not None and spec.folder != binding.folder:
                    raise SpecError(f"calibration spec folder {spec.folder!r} does not match this "
                                    f"project's workstation folder {binding.folder!r}")
                folder = binding.folder
            fp = spec.floorplan
            self._job = Job(
                job_id=uuid.uuid4().hex[:12],
                project_id=spec.project_id,
                project_title=spec.title,
                started_by=spec.issued_to,
                floorplan=asdict(fp) if fp else None,
                steps=_new_steps(),
                message="Starting",
                folder=folder,
                record_seconds=record_seconds or 0,
            )
            self._save()
            self._launch()
            return self._job.public()

    def resume(self) -> dict:
        self.config()
        with self._lock:
            self._load_once()
            job = self._job
            if job is None or job.state not in RESUMABLE:
                raise NothingToResume("there is no paused, failed or interrupted run to continue")
            job.state, job.message = "running", "Continuing"
            for s in job.steps:
                if s["State"] == "failed":
                    s["State"], s["Message"] = "pending", ""
            self._save()
            self._launch(resuming=True)
            return job.public()

    def cancel(self) -> dict:
        with self._lock:
            self._load_once()
            job = self._job
            if job is None or job.state not in ACTIVE | RESUMABLE:
                raise NothingToResume("there is no run to cancel")
            if not job.folder:
                self._cancel.set()
                if job.state not in ACTIVE:  # nothing is executing, so finish it here
                    self._finish("cancelled", "Cancelled.")
                return job.public()
            # web_project: cancel undoes the run (AMC-PROJECT-LIFECYCLE.md section 5.2).
            if job.cancelling:
                return job.public()
            executing = job.state in ACTIVE
            job.cancelling = True
            job.state, job.message = "running", CANCELLING_MESSAGE
            self._save()
            self._cancel.set()
            public = job.public()
            if executing:
                return public  # the run's own thread undoes the run once it stops
            if self._threaded:
                self._start_undo(job)
                return public
        self._start_undo(job)  # unthreaded (tests): run it here, without the lock
        with self._lock:
            return job.public()

    def config(self) -> AutomationConfig:
        return self._load_config()

    # ---- execution ------------------------------------------------------------
    def _start_undo(self, job: Job) -> None:
        """Undo on a new thread (or right here when unthreaded), unless a thread already owns it."""
        with self._lock:
            if self._undoing:
                return
            self._undoing = True
        if self._threaded:
            self._thread = threading.Thread(target=self._undo, args=(job,), daemon=True)
            self._thread.start()
        else:
            self._undo(job)

    def _launch(self, resuming: bool = False) -> None:
        self._cancel.clear()
        if self._threaded:
            self._thread = threading.Thread(target=self._run, args=(resuming,), daemon=True)
            self._thread.start()
        else:
            self._run(resuming)

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def _finish(self, state: str, message: str) -> None:
        with self._lock:
            if self._job and self._job.cancelling:
                return  # the undo decides how the run ends
            if self._job:
                self._job.state, self._job.message = state, message
                if self._job.folder and state in TERMINAL:
                    self._close_web_project_run()
                self._save()

    def _folder(self) -> Path:
        return self.store.folder_path(self._job.folder)

    def _close_web_project_run(self) -> None:
        """Copy the alignment back, count the run and leave runs/<job id>.json.

        Best effort: the run's outcome is already decided, so a problem here is
        added to the job's message rather than changing its state.
        """
        job = self._job
        problems = []
        if job.state == "completed":
            saved = self._amc_saved_alignment()
            if saved is not None:
                try:
                    dest = self._folder() / "alignment" / "alignment_data.json"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(saved, dest)
                except OSError as e:
                    problems.append(f"could not copy the alignment into the folder: {e}")
            try:
                current = self.store.get_by_folder(job.folder)
                if current is None:
                    raise ProjectNotFound(f"no project bound to folder {job.folder!r}")
                self.store.update(job.folder, last_run_at=_now(), calib_runs=current.calib_runs + 1)
            except (ProjectNotFound, OSError, ValueError) as e:
                problems.append(f"could not update the project record: {e}")
        if problems:
            job.message = "; ".join([job.message, *problems])
        try:
            runs = self._folder() / "runs"
            runs.mkdir(parents=True, exist_ok=True)
            job.updated_at = _now()
            tmp = runs / f".{job.job_id}.json.tmp"
            tmp.write_text(json.dumps(job.public(), indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, runs / f"{job.job_id}.json")
        except OSError:
            pass  # the service's state file still holds the run

    # ---- cancel: undo a web_project run (AMC-PROJECT-LIFECYCLE.md section 5.2) ----------
    def _undone(self, job: Job, action: str) -> None:
        with self._lock:
            if action not in job.undone:
                job.undone.append(action)
            self._save()

    def _undo(self, job: Job) -> None:
        """Undo the run, then finish it cancelled. Never raises; never holds the lock on AMC.

        Every action already in job.undone is skipped, so a restart part way through
        neither repeats one nor creates a second AMC project.
        """
        undone, problems = [], []
        try:
            if job.amc_changed:
                self._undo_amc(job, undone, problems)
            if job.footage_recorded:
                self._undo_footage(job, undone, problems)
            if job.calibration_dir:
                self._undo_results(job, undone, problems)
        except Exception as e:  # never leave the job stuck cancelling
            problems.append(f"the undo stopped on an unexpected error: {e!r}")
        if problems:
            message = "Cancelled, but " + "; ".join(problems) + "."
            if undone:
                message += " Undone: " + "; ".join(undone) + "."
        elif undone:
            message = "Cancelled. Undone: " + "; ".join(undone) + "."
        else:
            message = "Cancelled. Nothing needed undoing."
        with self._lock:
            job.cancelling = False
            self._undoing = False
            if self._job is job:
                self._finish("cancelled", message)

    def _amc_ready(self, cfg: AutomationConfig, api: AmcApi) -> bool:
        """Start AMC and wait for it, as provisioning does; the undo ignores the cancel flag."""
        if api.ready():
            return True
        self.controller.start()
        deadline = self._clock() + cfg.amc.ready_timeout_seconds
        while not api.ready():
            if self._clock() >= deadline:
                return False
            self._sleep(min(5.0, max(deadline - self._clock(), 0.0)))
        return True

    def _undo_amc(self, job: Job, undone: list[str], problems: list[str]) -> None:
        if AMC_BOUND in job.undone:
            undone.append(f"the AMC project was reset to empty (new AMC project {job.reset_amc_project_id})")
            return
        if AMC_CREATED not in job.undone and not self._replace_amc_project(job, problems):
            return
        new = job.reset_amc_project_id
        try:
            binding = self.store.get_by_folder(job.folder)
            if binding is None or binding.project_id != job.project_id:
                raise ProjectNotFound(f"workstation folder {job.folder} is no longer bound to this project")
            self.store.update(job.folder, amc_project_id=new, amc_project_name=job.folder)
        except (ProjectNotFound, OSError, ValueError) as e:
            problems.append(f"the new AMC project {new} could not be recorded for the folder: {e}")
            return
        self._undone(job, AMC_BOUND)
        undone.append(f"the AMC project was reset to empty (new AMC project {new})")

    def _replace_amc_project(self, job: Job, problems: list[str]) -> bool:
        """Stop, delete and recreate the job's AMC project; True once the new id is saved."""
        old = job.amc_project_id
        failed = "the AMC project could not be reset: "
        try:
            cfg = self.config()
            api = self._api(cfg)
        except (ConfigError, AmcApiError) as e:
            problems.append(failed + str(e))
            return False

        def create_failed(error: str) -> bool:
            # The old project is gone: a failed binding without it makes the next setup create one.
            problems.append(failed + error + "; set the project up on the workstation again")
            try:
                self.store.update(job.folder, state=STATE_FAILED, amc_project_id=None, error=error)
            except (ProjectNotFound, OSError, ValueError):
                pass
            return False

        try:
            if AMC_DELETED not in job.undone:
                try:  # a calibration may still be running; step state cannot say
                    api.stop_calibration(old)
                except Exception:
                    pass
            if not self._amc_ready(cfg, api):
                not_ready = f"AMC did not become ready within {int(cfg.amc.ready_timeout_seconds)} s"
                if AMC_DELETED in job.undone:
                    return create_failed(f"AMC project {old} was deleted but {not_ready}")
                problems.append(failed + not_ready)
                return False
            if AMC_DELETED not in job.undone:
                self._await_rectification(cfg, api, old)
                try:
                    api.delete_project(old)  # False (AMC's 404) counts as deleted
                except (AmcApiError, httpx.HTTPError) as e:
                    problems.append(failed + f"could not delete AMC project {old}: {e}")
                    return False
                self._undone(job, AMC_DELETED)
            try:
                new = api.create_project(job.folder)
            except (AmcApiError, httpx.HTTPError) as e:
                return create_failed(f"AMC project {old} was deleted but a new one could not be created: {e}")
            with self._lock:  # the id and amc_created in one save, before the binding is written
                job.reset_amc_project_id = new
                if AMC_CREATED not in job.undone:
                    job.undone.append(AMC_CREATED)
                self._save()
            return True
        finally:
            api.close()

    def _await_rectification(self, cfg: AutomationConfig, api: AmcApi, pid: str) -> None:
        """Wait while AMC rectifies the project, up to rectification.api.timeout_seconds.

        AMC has no call to stop a rectification (AMC-RECTIFICATION-AND-SCALE.md section 3.3);
        the caller deletes the project either way, and the undo ignores the cancel flag.
        """
        r = cfg.rectification.api
        deadline = self._clock() + r.timeout_seconds
        while True:
            try:
                if api.rectification_state(pid) != "RUNNING":
                    return
            except Exception:  # an AMC without the call, or one that cannot answer: nothing to wait for
                return
            left = deadline - self._clock()
            if left <= 0:
                return
            self._sleep(min(r.poll_seconds, left))

    def _undo_footage(self, job: Job, undone: list[str], problems: list[str]) -> None:
        footage = self._folder() / "footage"
        failed = "the earlier footage could not be restored: "
        if FOOTAGE_CLEARED not in job.undone:
            try:
                for name in [*job.recorded_files, FOOTAGE_MANIFEST]:
                    if _plain_name(name):
                        (footage / name).unlink(missing_ok=True)
            except OSError as e:
                problems.append(failed + f"could not delete this run's footage: {e}")
                return
            self._undone(job, FOOTAGE_CLEARED)
        if FOOTAGE_RESTORED not in job.undone:
            previous = footage / "previous"
            try:
                if previous.is_dir() and not previous.is_symlink():
                    for item in sorted(previous.iterdir()):
                        try:
                            os.replace(item, footage / item.name)
                        except FileNotFoundError:
                            pass  # already moved by an undo that was cut off
                    try:
                        previous.rmdir()
                    except OSError:
                        pass  # not empty: what is left stays where it is
            except OSError as e:
                problems.append(failed + f"could not move it back from {previous}: {e}")
                return
            self._undone(job, FOOTAGE_RESTORED)
        undone.append("this run's footage was deleted and the earlier footage moved back")

    def _undo_results(self, job: Job, undone: list[str], problems: list[str]) -> None:
        if RESULTS_REMOVED not in job.undone:
            if not _plain_name(job.calibration_dir):
                problems.append(f"this run's results were not removed: {job.calibration_dir!r} is not a "
                                "folder name")
                return
            target = self._folder() / "calibration" / job.calibration_dir
            try:
                if target.is_symlink():
                    target.unlink()
                else:
                    shutil.rmtree(target)
            except FileNotFoundError:
                pass
            except OSError as e:
                problems.append(f"this run's results in {target} could not be removed: {e}")
                return
            with self._lock:
                job.result.pop("ResultFile", None)
            self._undone(job, RESULTS_REMOVED)
        undone.append("this run's calibration results were removed")

    def _set_step(self, key: str, state: str, message: str = "", done: str | None = None,
                  needs_user: bool = False) -> None:
        with self._lock:
            s = self._job.step(key)
            s["State"], s["Message"], s["NeedsUser"] = state, message, needs_user
            if done is not None:
                s["Done"] = done
            if self._job.cancelling:  # keep "Cancelling" on show while the run thread stops
                self._save()
                return
            self._job.message = f"Step {s['Number']}: {s['Title']}" + (f" - {message}" if message else "")
            self._save()

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise _Cancelled()

    def _wait(self, seconds: float) -> None:
        end = self._clock() + seconds
        while True:
            self._check_cancel()
            left = end - self._clock()
            if left <= 0:
                return
            self._sleep(min(left, 1.0))

    def _run(self, resuming: bool) -> None:
        job = self._job
        self._run_steps(resuming)
        with self._lock:  # a cancel during the run left the undo to this thread
            undo = self._job is job and job.cancelling and not self._undoing
            if undo:
                self._undoing = True
        if undo:
            self._undo(job)

    def _run_steps(self, resuming: bool) -> None:
        job = self._job
        self._resuming = resuming
        try:
            cfg = self.config()
            api = self._api(cfg)
        except (ConfigError, AmcApiError) as e:
            self._finish("failed", str(e))
            return
        handlers = {
            "project_setup": self._project_setup,
            "video_configuration": self._video_configuration,
            "parameters": self._parameters,
            "rectification": self._rectification,
            "manual_alignment": self._manual_alignment,
            "execute": self._execute,
            "results": self._results,
        }
        try:
            for key, _, _ in STEPS:
                step = job.step(key)
                if step["State"] in ("done", "skipped"):
                    continue
                was_waiting = step["State"] == "waiting"
                if was_waiting and not resuming:
                    continue
                if key in GATED and any(s["State"] == "waiting" for s in job.steps if s["Key"] != key):
                    self._finish("waiting_for_user", self._pause_message())
                    return
                self._check_cancel()
                resumed = was_waiting and resuming
                if key == "manual_alignment" and not step.get("NeedsUser", True):
                    # It waited only for rectification (section 3.1): a first pass, so the user
                    # is asked to align the new footage instead of it counting as finished in AMC.
                    resumed = False
                self._set_step(key, "running")
                outcome = handlers[key](cfg, api, resumed=resumed)
                self._set_step(key, outcome.state, outcome.message, outcome.done,
                               needs_user=outcome.state == "waiting" and outcome.needs_user)
            self._finish("completed", "Calibration finished.")
        except _PauseForUser:
            for s in job.steps:
                if s["State"] == "running":
                    s["State"], s["Message"] = "pending", ""
            self._finish("waiting_for_user", self._pause_message())
        except _Cancelled:
            for s in job.steps:
                if s["State"] == "running":
                    s["State"] = "pending"
            self._finish("cancelled", "Cancelled. A calibration already running inside AMC keeps running there.")
        except (StepFailed, AmcApiError, SpecError, ConfigError, OSError) as e:
            current = next((s for s in job.steps if s["State"] == "running"), None)
            if current:
                current["State"], current["Message"] = "failed", str(e)
            self._finish("failed", str(e))
        except httpx.HTTPError as e:
            current = next((s for s in job.steps if s["State"] == "running"), None)
            if current:
                current["State"], current["Message"] = "failed", f"could not reach AMC: {e}"
            self._finish("failed", f"could not reach AMC: {e}")
        except Exception as e:  # never leave a run stuck as "running", which would block new runs
            current = next((s for s in job.steps if s["State"] == "running"), None)
            if current:
                current["State"], current["Message"] = "failed", f"unexpected error: {e!r}"
            self._finish("failed", f"unexpected error: {e!r}")
        finally:
            api.close()

    # ---- helpers --------------------------------------------------------------
    def _changing_amc(self) -> None:
        """Record, before the call, that this web_project run is about to change its AMC project."""
        job = self._job
        if job.folder and not job.amc_changed:
            with self._lock:
                job.amc_changed = True
                self._save()

    def _finished_in_amc(self, key: str) -> Outcome:
        """Done after Continue; called only once AMC has confirmed the step (section 5)."""
        earlier = self._job.step(key).get("Done", "")
        return Outcome("done", "finished in AMC" + (f"; {earlier}" if earlier else ""), done=earlier)

    def _still_waiting(self, key: str, message: str, done: str = "") -> Outcome:
        """A step AMC did not confirm after Continue: waiting again, keeping what was done."""
        earlier = self._job.step(key).get("Done", "")
        return Outcome("waiting", message, done="; ".join(x for x in (earlier, done) if x))

    def _pause_message(self) -> str:
        waiting = [f"{s['Number']} {s['Title']}" for s in self._job.steps if s["State"] == "waiting"]
        listed = f" (unfinished: step {', step '.join(waiting)})" if waiting else ""
        return f"Waiting for the manual steps in AMC{listed}. Press Continue when they are saved."

    def _conf(self) -> dict[str, str]:
        return _read_conf(Path(self.controller.install_dir) / "installer.conf")

    def _api(self, cfg: AutomationConfig) -> AmcApi:
        base = cfg.amc.api_base_url or default_base_url(self.controller.ports()[0])
        return AmcApi(base, timeout=cfg.amc.request_timeout_seconds,
                      upload_timeout=cfg.amc.upload_timeout_seconds, client=self._http_factory())

    # ---- step 1 ---------------------------------------------------------------
    def _project_setup(self, cfg: AutomationConfig, api: AmcApi, resumed: bool) -> Outcome:
        if not api.ready():
            if not cfg.amc.start_if_down:
                raise StepFailed("AMC is not running and amc.start_if_down is off")
            self._set_step("project_setup", "running", "starting AMC")
            self.controller.start()
            deadline = self._clock() + cfg.amc.ready_timeout_seconds
            while not api.ready():
                if self._clock() >= deadline:
                    raise StepFailed("AMC did not become ready in time (the first launch downloads several GB)")
                self._wait(5)

        job = self._job
        conf = self._conf()
        note = ""
        if job.folder:
            binding = self.store.get_by_folder(job.folder)
            if binding is None or binding.state != STATE_READY or not binding.amc_project_id \
                    or binding.project_id != job.project_id:
                raise StepFailed(f"workstation folder {job.folder} is no longer set up for this project; "
                                 "set it up on the workstation again")
            pid, name = binding.amc_project_id, binding.amc_project_name or job.folder
            try:
                api.get_project_info(pid)
            except AmcApiError as e:
                raise StepFailed(f"the AMC project {pid} for workstation folder {job.folder} is not in AMC "
                                 f"({e}); it is never recreated automatically") from e
            note = f"using workstation folder {job.folder}'s AMC project"
        elif cfg.project.mode == "reuse_installer_project" and conf.get("AMC_PROJECT_ID"):
            pid = conf["AMC_PROJECT_ID"]
            api.get_project_info(pid)  # must exist
            name = conf.get("PROJECT_NAME", "")
            job.amc_project_from_installer = True
            note = "reusing the installer's AMC project"
        else:
            name = cfg.project.name_template.format(
                title=job.project_title, project_id=job.project_id,
                date=datetime.now(timezone.utc).strftime("%Y%m%d"))
            pid = api.create_project(name)
            note = "created a new AMC project"
            if cfg.project.mode == "reuse_installer_project":
                note += " (installer.conf has no AMC_PROJECT_ID, so the installer will not ingest it on its own)"
        if "RUNNING" in api.calibration_states(pid):
            raise StepFailed(f"AMC project {pid} is already calibrating")
        job.amc_project_id, job.amc_project_name = pid, name
        return Outcome("done", f"{note}: {name or pid} ({pid})")

    # ---- step 2 ---------------------------------------------------------------
    def _footage_dir(self, cfg: AutomationConfig) -> Path:
        if self._job.folder:
            return self._folder() / "footage"
        if cfg.videos.source != "installer":
            return cfg.resolve(cfg.videos.source)
        conf = self._conf()
        root = conf.get("CALIB_FOOTAGE_DIR") or str(Path.home() / "Downloads" / "mv3dt-calibration-footage")
        name = conf.get("PROJECT_NAME") or self._job.amc_project_name
        return Path(os.path.expanduser(root)) / name

    def _videos(self, cfg: AutomationConfig) -> list[tuple[str, Path]]:
        folder = self._footage_dir(cfg)
        found = {p.name: p for p in sorted(folder.glob(cfg.videos.pattern)) if p.is_file()}
        if not found and self._job.folder:
            return []  # web_project: the step waits for the user instead
        # videos.order names copied footage; clips this run recorded go up in file-name order.
        if cfg.videos.order and not self._job.footage_recorded:
            missing = [n for n in cfg.videos.order if n not in found]
            if missing:
                raise StepFailed(f"videos.order names files that are not in {folder}: {', '.join(missing)}")
            paths = [found[n] for n in cfg.videos.order]
        else:
            paths = list(found.values())
        if not paths:
            raise StepFailed(f"no footage matching {cfg.videos.pattern} in {folder}; record it with installer step 4")
        return [(f"cam_{i:02d}.mp4", p) for i, p in enumerate(paths)]

    def _layout_png(self, cfg: AutomationConfig) -> bytes | None:
        if cfg.layout.source == "file":
            if not cfg.layout.file:
                raise StepFailed("layout.source is file but layout.file is empty")
            data = cfg.resolve(cfg.layout.file).read_bytes()
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise StepFailed("layout.file must be a PNG")
            return data
        fp = self._job.floorplan
        if not fp:
            if self._job.folder:  # the floor plan saved when the folder was set up
                saved = self._folder() / "floorplan" / "layout.png"
                if saved.is_file():
                    return saved.read_bytes()
            return None
        from .calibration_spec import Floorplan
        client = self._download_factory()
        try:
            return download_floorplan_png(Floorplan(**fp), cfg.layout.allowed_hosts, cfg.layout.max_bytes, client)
        finally:
            client.close()

    def _upload_videos(self, api: AmcApi, pid: str, videos: list[tuple[str, Path]]) -> str:
        self._set_step("video_configuration", "running", f"uploading {len(videos)} videos")
        self._changing_amc()
        api.upload_videos(pid, videos)
        self._job.videos_uploaded = True
        self._save()
        return "uploaded " + ", ".join(f"{p.name} as {n}" for n, p in videos)

    def _video_configuration(self, cfg: AutomationConfig, api: AmcApi, resumed: bool) -> Outcome:
        job = self._job
        web = bool(job.folder)
        pid = job.amc_project_id
        if web and job.record_seconds and not job.footage_recorded and self._resuming:
            # A run records only on its first pass. After a failed, cancelled or interrupted
            # recording step 2 stays failed, so a partial set is never uploaded.
            raise StepFailed("this run's recording did not finish; start a new run to record again")
        if resumed:
            # web_project: footage copied into the folder during the pause is uploaded now,
            # unless this run already uploaded videos. The rest is checked against AMC.
            note = ""
            if web and cfg.videos.enabled and not job.videos_uploaded:
                videos = self._videos(cfg)
                if videos:
                    note = self._upload_videos(api, pid, videos)
            return self._check_video_configuration(cfg, api, note)
        notes, todo = [], []
        if web and job.record_seconds and not job.footage_recorded:  # first pass only (see above)
            notes.append(self._record_footage(job.record_seconds))
        if cfg.videos.enabled and job.videos_uploaded:  # a failed step 2 being retried
            notes.append("videos already uploaded by this run")
        elif cfg.videos.enabled:
            videos = self._videos(cfg)
            if videos:
                notes.append(self._upload_videos(api, pid, videos))
            else:  # web_project only; the other modes fail in _videos
                todo.append(f"Copy the footage ({cfg.videos.pattern}) into {self._footage_dir(cfg)} on the "
                            "workstation, or upload it in AMC's Video Configuration step, then press Continue")
        else:
            notes.append("video upload is off in the config")
        if not cfg.layout.enabled:
            notes.append("layout upload is off in the config")
        else:
            png = self._layout_png(cfg)
            if png is None:
                todo.append("Upload layout.png in AMC's Video Configuration step (this project has no "
                            "floor plan in the web app)")
            else:
                with self._lock:  # saved before the upload it describes
                    job.layout_sha256 = hashlib.sha256(png).hexdigest()
                    self._save()
                if web:
                    saved = self._folder() / "floorplan" / "layout.png"
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    saved.write_bytes(png)
                self._changing_amc()
                api.upload_layout(pid, png)
                notes.append("uploaded the floor plan as layout.png")
        if todo:
            return Outcome("waiting", "; ".join(todo), done="; ".join(notes))
        return Outcome("done", "; ".join(notes))

    def _check_video_configuration(self, cfg: AutomationConfig, api: AmcApi, note: str) -> Outcome:
        """Step 2 after Continue: done only when AMC has at least one video and a layout."""
        info = api.get_project_info(self._job.amc_project_id)
        count = info.get("video_files_count")
        count = count if isinstance(count, int) and not isinstance(count, bool) else 0
        layout = info.get("layout_file") is True
        if count >= 1 and layout:
            if note:
                return Outcome("done", f"{note}; the rest finished in AMC", done=note)
            return self._finished_in_amc("video_configuration")
        todo = []
        if count < 1:
            todo.append(f"copy the footage ({cfg.videos.pattern}) into {self._footage_dir(cfg)} on the "
                        "workstation, or upload it in AMC's Video Configuration step" if self._job.folder
                        else "upload the videos in AMC's Video Configuration step")
        if not layout:
            todo.append("upload layout.png in AMC's Video Configuration step")
        reported = f"AMC reports {count} video{'' if count == 1 else 's'} and {'a' if layout else 'no'} layout.png"
        return self._still_waiting("video_configuration",
                                   f"{reported}: " + "; ".join(todo) + ", then press Continue", note)

    # ---- step 2: recording new footage ------------------------------------------
    @property
    def record_request_path(self) -> Path:
        """Fixed by AMC-FOOTAGE-RECORDING.md section 4; the root command reads exactly this path."""
        return Path(self.controller.install_dir) / "amc_control" / "state" / RECORD_REQUEST_NAME

    def _write_record_request(self, seconds: int) -> Path:
        """Write the request atomically (temp file, fsync, os.replace), mode 600."""
        path = self.record_request_path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps({"schema_version": 1, "folder": self._job.folder, "seconds": seconds,
                           "requested_at": int(self._clock())}).encode("utf-8")
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0), 0o600)
            try:
                os.fchmod(fd, 0o600)  # a leftover temp file keeps its old mode through O_TRUNC
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp, path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
        return path

    def _record_footage(self, seconds: int) -> str:
        """Run the root `record-footage` command and follow its stdout (section 5).

        Exit 0 returns a note for step 2; exit 3 (or a cancel here) ends the run
        cancelled; anything else fails the step with the command's reason and
        per-camera errors. The request file is removed whatever happens. The
        recorder runs as root behind sudo, so it is cancelled by writing
        `cancel` to its stdin and closing it, never by a signal.
        """
        total = _clock_text(seconds)
        hint = "keep people walking through every camera's view"
        shown = f"recording 0:00 of {total}; {hint}"
        self._set_step("video_configuration", "running", shown)
        request = self._write_record_request(seconds)
        try:
            argv = self._record_argv()
            try:
                proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
            except OSError as e:
                raise StepFailed(f"the workstation could not run the recording command ({' '.join(argv)}): "
                                 f"{e}") from e
            lines: queue.Queue = queue.Queue()
            stderr_tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
            readers = [
                threading.Thread(target=_pump, args=(proc.stdout, lines.put, lambda: lines.put(None)),
                                 daemon=True),
                threading.Thread(target=_pump, args=(proc.stderr, stderr_tail.append, None), daemon=True),
            ]
            for t in readers:
                t.start()
            final: dict | None = None
            deadline = time.monotonic() + seconds + self.record_overrun
            eof = False          # the recorder's stdout closed
            exited_at = None     # when proc.poll() first saw the recorder gone
            try:
                # One bounded loop until the recorder has exited: its stdout closing
                # is not the end (it may still be running), and a leftover child can
                # hold stdout open after it exits, so poll() decides. Cancel and the
                # deadline apply the whole time.
                while True:
                    try:
                        line = lines.get(timeout=0.2)
                    except queue.Empty:
                        line = ""
                    if line is None:
                        eof, line = True, ""
                    msg = _json_object(line) if line else None
                    if msg is not None:
                        progress = msg.get("progress")
                        if "ok" in msg:
                            final = msg
                        elif isinstance(progress, (int, float)) and not isinstance(progress, bool):
                            text = (f"recording {_clock_text(min(max(progress, 0.0), 1.0) * seconds)} "
                                    f"of {total}; {hint}")
                            if text != shown:
                                shown = text
                                self._set_step("video_configuration", "running", text)
                    if proc.poll() is not None:
                        if exited_at is None:
                            exited_at = time.monotonic()
                        # Take what is already in the pipe; give the reader a moment
                        # to deliver the final line unless stdout has closed.
                        if eof or (lines.empty() and time.monotonic() - exited_at >= 1.0):
                            break
                        continue
                    if self._cancel.is_set():
                        self._cancel_recorder(proc)
                        raise _Cancelled()
                    if time.monotonic() > deadline:
                        limit = _clock_text(seconds + self.record_overrun)
                        if self._cancel_recorder(proc):
                            raise StepFailed(f"recording footage was still running after {limit} and was "
                                             "cancelled")
                        raise StepFailed(f"recording footage was still running after {limit} and did not "
                                         f"stop within {int(self.record_cancel_grace)} seconds of being "
                                         "cancelled")
                rc = proc.returncode
            finally:
                if proc.poll() is None and not proc.stdin.closed:  # an unexpected error above
                    self._cancel_recorder(proc)
                if proc.poll() is not None:
                    # stdin stays open while the recorder runs (end of file means cancel to it).
                    try:
                        proc.stdin.close()
                    except (OSError, ValueError):
                        pass
                # Close a pipe only once its reader is done: a leftover child may hold it open.
                for t, stream in zip(readers, (proc.stdout, proc.stderr)):
                    t.join(timeout=1)
                    if not t.is_alive():
                        try:
                            stream.close()
                        except (OSError, ValueError):
                            pass
        finally:
            request.unlink(missing_ok=True)

        if rc == RECORD_EXIT_CANCELLED:
            raise _Cancelled()
        if final is None:
            tail = " | ".join(stderr_tail)
            raise StepFailed(f"the workstation refused or could not run the recording command (exit {rc}); "
                             "re-running the installer restores its sudo rule"
                             + (f"; last output: {tail}" if tail else ""))
        if rc != 0:
            raise StepFailed(_record_failure(rc, final, list(stderr_tail)))
        files = [f for f in final.get("files") or [] if isinstance(f, str)]
        with self._lock:
            self._job.recorded_files = files
            self._job.footage_recorded = True
            self._save()
        return f"recorded {len(files)} clips of {total} into {self._folder() / 'footage'}"

    def _cancel_recorder(self, proc: subprocess.Popen) -> bool:
        """Write `cancel` to the recorder's stdin, close it and wait; True once it exited.

        Never a signal: the recorder is root behind sudo, and a kill would orphan
        its ffmpeg children (AMC-FOOTAGE-RECORDING.md section 5).
        """
        if proc.poll() is None:
            try:
                proc.stdin.write("cancel\n")
                proc.stdin.flush()
            except (OSError, ValueError):
                pass  # it is exiting anyway, or stdin is already closed
            try:
                proc.stdin.close()
            except (OSError, ValueError):
                pass
        try:
            proc.wait(timeout=self.record_cancel_grace)
        except subprocess.TimeoutExpired:
            return False
        return True

    # ---- step 3 ---------------------------------------------------------------
    # The scale, AMC-RECTIFICATION-AND-SCALE.md section 4.
    def _scale(self, cfg: AutomationConfig) -> tuple[float | None, str]:
        """The scale parameters.scale.source gives (section 4.1) and where it came from."""
        s = cfg.parameters.scale
        if s.source == "fixed":
            return (float(s.fixed_px_per_m), "fixed_px_per_m") if s.fixed_px_per_m > 0 else (None, "")
        if s.source == "database":
            value = _scale_value(self._job.floorplan or {}, "px_per_m")
            if value is not None:
                return value, "the web app's floor plan"
            value = self._remembered_scale()
            if value is not None:
                return value, "remembered for this floor plan"
        return None, ""

    def _remembered_scale(self) -> float | None:
        """project.json's scale, only for the layout.png this run uploaded (same SHA-256)."""
        job = self._job
        if not job.folder or not job.layout_sha256:
            return None
        binding = self.store.get_by_folder(job.folder)
        if binding is None or binding.project_id != job.project_id or not binding.layout_px_per_m \
                or binding.layout_sha256 != job.layout_sha256:
            return None
        return float(binding.layout_px_per_m)

    def _factory_scale(self, cfg: AutomationConfig, api: AmcApi) -> float:
        value = _scale_value(api.get_default_config(), cfg.parameters.scale.settings_key)
        return FACTORY_LAYOUT_PX_PER_M if value is None else value

    def _set_scale(self, cfg: AutomationConfig, config: dict, value: float) -> None:
        key = cfg.parameters.scale.settings_key
        if not _has_key(config, key):
            raise StepFailed(f"AMC's configuration has no {key}; check parameters.scale.settings_key")
        _set_key(config, key, value)

    def _take_scale(self, value: float) -> str:
        """Save a scale confirmed in AMC on the job, then remember it (section 4.3)."""
        job = self._job
        with self._lock:
            job.layout_px_per_m = value
            self._save()
        if not job.folder:
            return ""
        if not job.layout_sha256:
            return "not remembered: this run uploaded no layout.png"
        try:
            binding = self.store.get_by_folder(job.folder)
            if binding is None or binding.project_id != job.project_id:
                raise ProjectNotFound(f"workstation folder {job.folder} is no longer bound to this project")
            self.store.update(job.folder, layout_px_per_m=value, layout_sha256=job.layout_sha256)
        except (ProjectNotFound, OSError, ValueError) as e:
            return f"could not remember it: {e}"
        return "remembered for this floor plan"

    def _parameters(self, cfg: AutomationConfig, api: AmcApi, resumed: bool) -> Outcome:
        if not cfg.parameters.enabled:
            return Outcome("skipped", "parameters are off in the config")
        p = cfg.parameters
        if resumed:
            if p.scale.source == "none":  # a run paused before the config changed
                return self._finished_in_amc("parameters")
            return self._confirm_scale(cfg, api)
        pid, job = self._job.amc_project_id, self._job
        notes = []
        scale, origin = self._scale(cfg)
        if p.settings_file or p.settings_overrides or p.scale.source != "none":
            # AMC keeps one whole configuration, so every post is built on the current one.
            config = api.get_config()
            if p.settings_file:
                path = cfg.resolve(p.settings_file)
                try:
                    settings = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError) as e:
                    raise StepFailed(f"could not read settings file {path}: {e}") from e
                if not isinstance(settings, dict):
                    raise StepFailed(f"settings file {path} must contain a JSON object")
                config = deep_merge(config, settings)
                notes.append(f"applied {path.name}")
            if p.settings_overrides:
                config = deep_merge(config, p.settings_overrides)
                notes.append("applied settings_overrides")
            if scale is not None:
                scale = round(scale, 4)
                self._set_scale(cfg, config, scale)
                with self._lock:  # saved before the post it describes
                    job.layout_px_per_m = scale
                    self._save()
                notes.append(f"scale {scale:g} px/m from {origin}")
            elif p.scale.source != "none":
                # The factory value, so a scale another project left in AMC cannot pass Continue.
                default = self._factory_scale(cfg, api)
                self._set_scale(cfg, config, default)
                notes.append(f"AMC's factory scale {default:g} px/m posted")
            self._changing_amc()
            api.post_config(pid, config)
        else:
            notes.append("AMC's current settings")
        if p.focal_lengths:
            self._changing_amc()
            api.upload_focal_lengths(pid, p.focal_lengths)
            notes.append(f"{len(p.focal_lengths)} focal lengths")
        if scale is None and p.scale.source != "none":
            return Outcome("waiting", SCALE_WAIT, done="; ".join(notes))
        return Outcome("done", "; ".join(notes))

    def _confirm_scale(self, cfg: AutomationConfig, api: AmcApi) -> Outcome:
        """Step 3 after Continue (section 4.2 item 3): a positive, non-default scale in AMC."""
        key = cfg.parameters.scale.settings_key
        value = _scale_value(api.get_config(), key)
        default = self._factory_scale(cfg, api)
        if value is None or _same_scale(value, default):
            reported = f"{value:g} px/m, the factory default" if value is not None else "no valid value"
            return self._still_waiting("parameters", f"{SCALE_WAIT}. AMC reports {key} {reported}")
        remembered = self._take_scale(value)
        note = f"scale {value:g} px/m set in AMC" + (f" ({remembered})" if remembered else "")
        earlier = self._job.step("parameters").get("Done", "")
        done = "; ".join(x for x in (earlier, note) if x)
        return Outcome("done", done, done=done)

    # ---- step 4 ---------------------------------------------------------------
    def _rectification(self, cfg: AutomationConfig, api: AmcApi, resumed: bool) -> Outcome:
        r = cfg.rectification
        if r.mode == "skip":
            return Outcome("skipped", "rectification.mode is skip")
        if r.mode in ("auto", "linear"):
            # Never the "finished in AMC" shortcut: these always run the flow, also on Continue.
            # A reused AMC project may still hold the previous run's videos while step 2
            # waits for new footage, so the count alone is not enough (section 3.1).
            no_videos = Outcome("waiting", "Rectification runs once the videos are in AMC", needs_user=False)
            if self._job.step("video_configuration")["State"] == "waiting":
                return no_videos
            count = api.get_project_info(self._job.amc_project_id).get("video_files_count")
            if count is not None and not count:
                return no_videos
            return self._rectify(cfg, api, linear=r.mode == "linear")
        if r.mode == "ui":
            todo = "Run Rectification in AMC (Auto, Manual, or Videos Are Rectified) until it shows COMPLETED"
            if resumed:
                state = api.rectification_state(self._job.amc_project_id)
                if state == "COMPLETED":
                    return self._finished_in_amc("rectification")
                return self._still_waiting("rectification", f"{todo}; AMC reports {state}")
            return Outcome("waiting", todo)
        if not r.api.start.path or not r.api.status.path or not r.api.status.state_field:
            raise StepFailed("rectification.mode is api but rectification.api start/status paths are not set "
                             "(take them from AMC's /openapi.yaml)")
        pid = self._job.amc_project_id
        self._changing_amc()
        api.request(r.api.start.method, r.api.start.path.format(project_id=pid),
                    json=r.api.start.json_body or None, label="start rectification")
        deadline = self._clock() + r.api.timeout_seconds
        done = {s.upper() for s in r.api.status.done_states}
        failed = {s.upper() for s in r.api.status.failed_states}
        while True:
            reply = api.request(r.api.status.method, r.api.status.path.format(project_id=pid),
                                label="rectification status")
            state = str(_dig(reply, r.api.status.state_field) or "").upper()
            if state in done:
                return Outcome("done", f"rectification {state}")
            if state in failed:
                raise StepFailed(f"rectification ended in {state}")
            if self._clock() >= deadline:
                raise StepFailed("rectification did not finish in time")
            self._set_step("rectification", "running", f"state {state or 'unknown'}")
            self._wait(r.api.poll_seconds)

    def _rectify(self, cfg: AutomationConfig, api: AmcApi, linear: bool) -> Outcome:
        """Modes auto and linear (AMC-RECTIFICATION-AND-SCALE.md section 3.2).

        Until this run has started a rectification, AMC's state is not trusted: a
        reused AMC project may be COMPLETED from footage the run just replaced.
        """
        job = self._job
        pid = job.amc_project_id
        state = api.rectification_state(pid)
        if not job.rectification_started:
            if state == "RUNNING":
                self._wait_out_rectification(cfg, api)
            action = "start"
        elif state == "COMPLETED":
            return self._rectified(api, linear, None)
        elif state == "RUNNING":
            action = "generate" if job.rectification_phase == "generate" else "estimate"
        elif state == "READY_FOR_REVIEW" and not linear:
            action = "commit"
        else:  # NOT_STARTED or ERROR: start over, so Continue after a failure retries
            action = "start"

        # reply: the state a request just made answered with, which opens a lag window;
        # None when the poll resumes a phase AMC is RUNNING.
        reply: str | None = None
        if linear:
            if action == "start":
                self._begin_rectification("linear")
                reply = api.stage_linear_media(pid)
            self._poll_rectification(cfg, api, "staging the linear videos", "COMPLETED", reply)
            return self._rectified(api, linear, None)

        if action == "start":
            self._begin_rectification("estimate")
            reply = api.start_rectification(pid)
            action = "estimate"
        if action == "estimate":
            # Only READY_FOR_REVIEW ends the estimate, so the run always commits.
            self._poll_rectification(cfg, api, "estimating distortion", "READY_FOR_REVIEW", reply)
            action = "commit"
        reply = None
        committed = None
        if action == "commit":
            committed = {}
            for cam in api.rectification_cameras(pid):
                if cam.get("camera_id") is not None:
                    committed[str(cam["camera_id"])] = _rectification_params(cam)
            if not committed:
                raise StepFailed("AMC lists no cameras to rectify; check the uploaded videos")
            self._begin_rectification("generate")
            reply = api.commit_rectification(pid, {cid: params for cid, (params, _) in committed.items()})
        self._poll_rectification(cfg, api, "generating rectified videos", "COMPLETED", reply)
        return self._rectified(api, linear, committed)

    def _begin_rectification(self, phase: str) -> None:
        """Save, before the AMC call, that this run started rectifying and which phase it is in."""
        self._changing_amc()
        job = self._job
        with self._lock:
            job.rectification_started = True
            job.rectification_phase = phase
            self._save()

    def _rectified(self, api: AmcApi, linear: bool, committed: dict | None) -> Outcome:
        if linear:
            return Outcome("done", "videos staged as already linear")
        if committed is None:  # resumed after the commit: describe what AMC holds
            committed = {str(c["camera_id"]): _rectification_params(c)
                         for c in api.rectification_cameras(self._job.amc_project_id)
                         if c.get("camera_id") is not None}
        if not committed:
            return Outcome("done", "rectified automatically")
        parts = []
        for cid, (p, estimated) in committed.items():
            parts.append(f"camera {cid} {p['model']} (k1 {_num(p['k1'])}, k2 {_num(p['k2'])}"
                         + ("" if estimated else ", no estimate") + ")")
        return Outcome("done", "rectified automatically: " + ", ".join(parts))

    def _poll_rectification(self, cfg: AutomationConfig, api: AmcApi, doing: str, awaited: str,
                            reply: str | None) -> None:
        """Poll until the state is `awaited`; ERROR, another state or a timeout fails the step.

        After a request (reply is its answer's state, possibly "") AMC may still report the
        state from before it, so a lag window lasts until RUNNING is seen or for at most
        LAG_POLLS polls; inside it every state keeps the poll going, ERROR included
        (AMC-RECTIFICATION-AND-SCALE.md section 3.2). Each call has its own
        rectification.api.timeout_seconds; cancel is honoured between polls.
        """
        pid = self._job.amc_project_id
        r = cfg.rectification.api
        deadline = self._clock() + r.timeout_seconds
        lagging = reply is not None and reply != "RUNNING"
        lag_polls = 0
        while True:
            state = api.rectification_state(pid)
            if state == awaited:
                return
            if state == "RUNNING":
                lagging = False
            elif lagging and lag_polls < LAG_POLLS:
                lag_polls += 1
            elif state == "ERROR":
                raise StepFailed(f"AMC rectification ended in ERROR while {doing}. Last log lines:\n"
                                 + api.rectification_log(pid, tail_lines=10))
            else:
                raise StepFailed(f"AMC rectification is {state or 'in no state'} while {doing}, expected "
                                 f"RUNNING or {awaited}. Last log lines:\n"
                                 + api.rectification_log(pid, tail_lines=10))
            if self._clock() >= deadline:
                raise StepFailed(f"rectification still {state} after {r.timeout_seconds:g} s while {doing}. "
                                 "Last log lines:\n" + api.rectification_log(pid, tail_lines=10))
            self._set_step("rectification", "running", f"{doing}: {state}")
            self._wait(r.poll_seconds)

    def _wait_out_rectification(self, cfg: AutomationConfig, api: AmcApi) -> None:
        """Before this run starts its own: poll while an earlier rectification is RUNNING."""
        pid = self._job.amc_project_id
        r = cfg.rectification.api
        deadline = self._clock() + r.timeout_seconds
        doing = "waiting for AMC's earlier rectification to end"
        while (state := api.rectification_state(pid)) == "RUNNING":
            if self._clock() >= deadline:
                raise StepFailed(f"rectification still RUNNING after {r.timeout_seconds:g} s while {doing}. "
                                 "Last log lines:\n" + api.rectification_log(pid, tail_lines=10))
            self._set_step("rectification", "running", f"{doing}: {state}")
            self._wait(r.poll_seconds)

    # ---- step 5 ---------------------------------------------------------------
    def _alignment_file(self, cfg: AutomationConfig) -> Path | None:
        f = cfg.alignment.file
        if not f:
            return None
        if f == "auto":
            if self._job.folder:
                own = self._folder() / "alignment" / "alignment_data.json"
                if own.is_file():
                    return own
            return self._amc_saved_alignment()
        p = cfg.resolve(f)
        if not p.is_file():
            raise StepFailed(f"alignment.file {p} does not exist")
        return p

    def _amc_saved_alignment(self) -> Path | None:
        """The alignment_data.json AMC saved for this run's AMC project, if any."""
        job = self._job
        if not (job.amc_project_from_installer or job.folder) or not job.amc_project_id:
            return None
        root = self._conf().get("AMC_ROOT")
        if not root:
            return None
        p = (Path(os.path.expanduser(root)) / "projects" / f"project_{job.amc_project_id}"
             / "manual_adjustment" / "alignment_data.json")
        return p if p.is_file() else None

    def _manual_alignment(self, cfg: AutomationConfig, api: AmcApi, resumed: bool) -> Outcome:
        path = self._alignment_file(cfg)
        rect_waiting = self._job.step("rectification")["State"] == "waiting"
        if path is None:
            # In auto and linear the run rectifies by itself, so never ask for alignment before that.
            if rect_waiting and cfg.rectification.mode in ("auto", "linear"):
                return Outcome("waiting", "Manual Alignment comes after rectification", needs_user=False)
            todo = ("Do Manual Alignment in AMC: mark at least 4 matching point sets "
                    "(6-8 recommended) on each camera and the layout, then Save")
            if resumed:  # checked against AMC (AMC-RECTIFICATION-AND-SCALE.md section 5)
                if api.get_project_info(self._job.amc_project_id).get("alignment_file") is True:
                    return self._finished_in_amc("manual_alignment")
                return self._still_waiting("manual_alignment", f"{todo}; AMC reports no saved alignment yet")
            return Outcome("waiting", todo)
        if rect_waiting:
            return Outcome("waiting", f"Nothing to do: the saved {path.name} is uploaded after rectification",
                           needs_user=False)
        self._changing_amc()
        api.upload_alignment(self._job.amc_project_id, path, cfg.alignment.coord_space)
        return Outcome("done", f"re-used saved alignment {path} ({cfg.alignment.coord_space} coordinates)")

    # ---- step 6 ---------------------------------------------------------------
    def _repost_scale(self, cfg: AutomationConfig, api: AmcApi) -> str:
        """Section 4.2 item 4: take a correction made in AMC during the pause, post the job's scale.

        Nothing to do when the job has no scale or parameters.scale.source is none.
        """
        job = self._job
        if job.layout_px_per_m is None or cfg.parameters.scale.source == "none":
            return ""
        config = api.get_config()
        value = _scale_value(config, cfg.parameters.scale.settings_key)
        note = ""
        if value is not None and not _same_scale(value, job.layout_px_per_m) \
                and not _same_scale(value, self._factory_scale(cfg, api)):
            old = job.layout_px_per_m
            remembered = self._take_scale(value)
            note = f"scale corrected in AMC from {old:g} to {value:g} px/m" + (f" ({remembered})" if remembered else "")
        self._set_scale(cfg, config, job.layout_px_per_m)
        self._changing_amc()
        api.post_config(job.amc_project_id, config)
        return note or f"scale {job.layout_px_per_m:g} px/m"

    def _execute(self, cfg: AutomationConfig, api: AmcApi, resumed: bool) -> Outcome:
        job = self._job
        pid = job.amc_project_id
        if job.calibration_started:
            # Continue after a failure or a restart: wait for a calibration AMC is running, and
            # take a finished one only when AMC accepted this run's post (else it is an earlier one).
            project_state, amc_state = api.calibration_states(pid)
            if "RUNNING" in (project_state, amc_state):
                self._calibration_posted()
                return self._poll_calibration(cfg, api, None, "")
            if amc_state == "COMPLETED" and job.calibration_posted:
                return Outcome("done", "the calibration had finished in AMC while the run was not watching")
        scale_note = self._repost_scale(cfg, api)
        self._changing_amc()
        state = api.verify_project(pid)
        if state != "READY":
            with self._lock:
                s = job.step("manual_alignment")
                s["State"], s["NeedsUser"] = "waiting", True
                s["Message"] = (f"AMC reports {state or 'no state'} instead of READY. Check the videos, layout, "
                                "rectification and alignment in AMC, save, then press Continue")
            raise _PauseForUser()
        with self._lock:
            job.calibration_started, job.calibration_posted = True, False
            self._save()
        reply = api.calibrate(pid, cfg.calibration.detector_type)
        self._calibration_posted()
        return self._poll_calibration(cfg, api, reply, scale_note)

    def _calibration_posted(self) -> None:
        """Record that AMC took this run's calibration: its post was accepted or AMC runs it."""
        job = self._job
        if not job.calibration_posted:
            with self._lock:
                job.calibration_posted = True
                self._save()

    def _poll_calibration(self, cfg: AutomationConfig, api: AmcApi, reply: str | None,
                          scale_note: str) -> Outcome:
        """Poll get_project_info until AMC's amc_state is COMPLETED with nothing RUNNING.

        AMC 3.3.0 reports a calibration through project_state and amc_state: RUNNING in either
        while it runs, then project_state READY and amc_state COMPLETED (or ERROR). After the
        calibrate post (reply is its answer's project_state, possibly "") a reused project may
        still report its earlier calibration, so a lag window lasts until RUNNING is seen or
        for at most LAG_POLLS polls; inside it every state keeps the poll going. reply is None
        when nothing was posted because AMC was already seen calibrating.
        """
        pid = self._job.amc_project_id
        c = cfg.calibration
        start = self._clock()
        lagging = reply is not None and reply != "RUNNING"
        lag_polls = 0
        while True:
            self._wait(c.poll_seconds)
            project_state, amc_state = api.calibration_states(pid)
            states = (project_state, amc_state)
            shown = f"project_state {project_state or 'missing'}, amc_state {amc_state or 'missing'}"
            minutes = int((self._clock() - start) // 60)
            running = "RUNNING" in states
            if running:
                lagging = False
                self._calibration_posted()
            elif lagging and lag_polls < LAG_POLLS:
                lag_polls += 1
            elif amc_state == "COMPLETED":
                return Outcome("done", f"calibrated with {c.detector_type} in about {minutes} min"
                               + (f"; {scale_note}" if scale_note else ""))
            elif "ERROR" in states or "FAILED" in states:
                raise StepFailed(f"AMC calibration ended in ERROR ({shown}). Last log lines:\n"
                                 + api.calibration_log(pid))
            else:
                raise StepFailed(f"AMC calibration stopped without finishing ({shown}); "
                                 "see AMC's calibration log")
            if self._clock() - start >= c.timeout_seconds:
                raise StepFailed(f"calibration still running after {minutes} min ({shown}); "
                                 "see AMC's calibration log")
            doing = "calibrating" if running else "waiting for AMC to start calibrating"
            self._set_step("execute", "running", f"{doing} for {minutes} min (typically 10-60 min)")

    # ---- step 7 ---------------------------------------------------------------
    def _results(self, cfg: AutomationConfig, api: AmcApi, resumed: bool) -> Outcome:
        job = self._job
        notes = []
        stats = api.evaluation_statistics(job.amc_project_id)
        if stats:
            job.result["EvaluationStatistics"] = stats
            notes.append("evaluation statistics saved")
        if job.folder:
            # Saved before the folder exists, so a cancel always knows what to remove. A retried
            # step 7 reuses it rather than leaving an earlier attempt's folder behind.
            if not job.calibration_dir:
                with self._lock:
                    job.calibration_dir = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                    self._save()
            out_dir = self._folder() / "calibration" / job.calibration_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            if stats:
                (out_dir / "evaluation_statistics.json").write_text(json.dumps(stats, indent=2) + "\n",
                                                                    encoding="utf-8")
            out = out_dir / f"{job.amc_project_id}-{cfg.results.result_type}.zip"
            out.write_bytes(api.mv3dt_result(job.amc_project_id, cfg.results.result_type))
            job.result["ResultFile"] = str(out)
            notes.append(f"result saved to {out_dir}")
        elif cfg.results.download_dir:
            folder = cfg.resolve(cfg.results.download_dir)
            folder.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out = folder / f"{job.amc_project_id}-{cfg.results.result_type}-{stamp}.zip"
            out.write_bytes(api.mv3dt_result(job.amc_project_id, cfg.results.result_type))
            job.result["ResultFile"] = str(out)
            notes.append(f"result saved to {out}")
        if job.amc_project_from_installer:
            notes.append("the installer's ingest timer installs the new result (when auto-ingest is on)")
        return Outcome("done", "; ".join(notes) or "done")

