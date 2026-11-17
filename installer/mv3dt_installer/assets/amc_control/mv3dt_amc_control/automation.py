"""Runs AMC 3.3.0's seven-step workflow, pausing only where a person is needed.

    1 Project Setup        automated: reuse the installer's AMC project, create one, or use
                           the web project's bound AMC project (mode web_project)
    2 Video Configuration  automated: installer footage as cam_00.mp4..., floor plan as layout.png
    3 Parameters           automated: settings file, scale from the floor plan, focal lengths
    4 Rectification        automated when configured (mode api), otherwise done in the UI
    5 Manual Alignment     manual the first time; a saved alignment file is re-uploaded after that
    6 Execute Calibration  automated: verify the project is READY, calibrate, wait
    7 Results              automated: evaluation statistics, optional result ZIP download

Every step that needs a person is marked "waiting". Steps 1-5 all run first,
so everything that can be uploaded is uploaded; the run then pauses before
step 6 until the user presses Continue. Progress is saved to disk after every
change, so a run can be continued after the service restarts.

In project.mode web_project a run is locked to the web project's workstation
folder and its AMC project (installer/plan/AMC-WEB-PROJECTS.md section 6):
footage comes from <folder>/footage/, results go to <folder>/calibration/,
and every finished run leaves <folder>/runs/<job id>.json.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

from .controller import _read_conf
from .amc_api import AmcApi, AmcApiError, default_base_url
from .automation_config import AutomationConfig, ConfigError, load_config
from .calibration_spec import CalibrationSpec, SpecError, download_floorplan_png
from .projects import STATE_READY, ProjectNotFound, ProjectStore, default_projects_root

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
    ):
        self.controller = controller
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

    def start(self, spec: CalibrationSpec) -> dict:
        cfg = self.config()  # fail before creating a job if the config is broken
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
            self._cancel.set()
            if job.state not in ACTIVE:  # nothing is executing, so finish it here
                self._finish("cancelled", "Cancelled.")
            return job.public()

    def config(self) -> AutomationConfig:
        return self._load_config()

    # ---- execution ------------------------------------------------------------
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

    def _set_step(self, key: str, state: str, message: str = "", done: str | None = None,
                  needs_user: bool = False) -> None:
        with self._lock:
            s = self._job.step(key)
            s["State"], s["Message"], s["NeedsUser"] = state, message, needs_user
            if done is not None:
                s["Done"] = done
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
                    self._finish("waiting_for_user",
                                 "Waiting for the manual steps in AMC. Press Continue when they are saved.")
                    return
                self._check_cancel()
                self._set_step(key, "running")
                outcome = handlers[key](cfg, api, resumed=was_waiting and resuming)
                self._set_step(key, outcome.state, outcome.message, outcome.done,
                               needs_user=outcome.state == "waiting" and outcome.needs_user)
            self._finish("completed", "Calibration finished.")
        except _PauseForUser:
            for s in job.steps:
                if s["State"] == "running":
                    s["State"], s["Message"] = "pending", ""
            self._finish("waiting_for_user",
                         "Waiting for the manual steps in AMC. Press Continue when they are saved.")
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
    def _finished_in_amc(self, key: str) -> Outcome:
        earlier = self._job.step(key).get("Done", "")
        return Outcome("done", "finished in AMC" + (f"; {earlier}" if earlier else ""), done=earlier)

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
        if api.project_state(pid) == "RUNNING":
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
        if cfg.videos.order:
            missing = [n for n in cfg.videos.order if n not in found]
            if missing:
                raise StepFailed(f"videos.order names files that are not in {folder}: {', '.join(missing)}")
            paths = [found[n] for n in cfg.videos.order]
        else:
            paths = list(found.values())
        if not paths:
            if self._job.folder:
                return []  # web_project: the step waits for the user instead
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

    def _video_configuration(self, cfg: AutomationConfig, api: AmcApi, resumed: bool) -> Outcome:
        web = bool(self._job.folder)
        if resumed and not (web and cfg.videos.enabled and self._videos(cfg)):
            # Done in AMC's UI, or (web_project) footage copied into the folder since the pause.
            return self._finished_in_amc("video_configuration")
        pid = self._job.amc_project_id
        notes, todo = [], []
        if cfg.videos.enabled:
            videos = self._videos(cfg)
            if videos:
                self._set_step("video_configuration", "running", f"uploading {len(videos)} videos")
                api.upload_videos(pid, videos)
                notes.append("uploaded " + ", ".join(f"{p.name} as {n}" for n, p in videos))
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
                if not resumed:
                    todo.append("Upload layout.png in AMC's Video Configuration step (this project has no "
                                "floor plan in the web app)")
            else:
                if web:
                    saved = self._folder() / "floorplan" / "layout.png"
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    saved.write_bytes(png)
                api.upload_layout(pid, png)
                notes.append("uploaded the floor plan as layout.png")
        if todo:
            return Outcome("waiting", "; ".join(todo), done="; ".join(notes))
        return Outcome("done", "; ".join(notes))

    # ---- step 3 ---------------------------------------------------------------
    def _scale(self, cfg: AutomationConfig) -> float | None:
        src = cfg.parameters.scale.source
        if src == "fixed":
            return cfg.parameters.scale.fixed_px_per_m or None
        if src == "database":
            fp = self._job.floorplan or {}
            return fp.get("px_per_m")
        return None

    def _parameters(self, cfg: AutomationConfig, api: AmcApi, resumed: bool) -> Outcome:
        if resumed:
            return self._finished_in_amc("parameters")
        if not cfg.parameters.enabled:
            return Outcome("skipped", "parameters are off in the config")
        pid, p = self._job.amc_project_id, cfg.parameters
        notes, todo = [], []
        scale = self._scale(cfg)

        if p.settings_overrides and not p.settings_file:
            raise StepFailed("parameters.settings_overrides needs parameters.settings_file")
        if p.settings_file:
            path = cfg.resolve(p.settings_file)
            try:
                settings = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                raise StepFailed(f"could not read settings file {path}: {e}") from e
            if not isinstance(settings, dict):
                raise StepFailed(f"settings file {path} must contain a JSON object")
            from .automation_config import deep_merge
            settings = deep_merge(settings, p.settings_overrides)
            if scale:
                if _has_key(settings, p.scale.settings_key):
                    _set_key(settings, p.scale.settings_key, round(scale, 4))
                    notes.append(f"scale {scale:.2f} px/m")
                else:
                    todo.append(f"Enter {scale:.2f} layout px/m on AMC's Parameters step "
                                f"(the settings file has no {p.scale.settings_key} key)")
            api.post_config(pid, settings)
            notes.insert(0, f"applied {path.name}")
        else:
            notes.append("AMC default settings")
            if scale:
                todo.append(f"Enter {scale:.2f} layout px/m on AMC's Parameters step (Floor-plan to 3D scale)")
        if scale is None and p.scale.source != "none":
            notes.append("no floor plan scale available")
        if p.focal_lengths:
            api.upload_focal_lengths(pid, p.focal_lengths)
            notes.append(f"{len(p.focal_lengths)} focal lengths")
        if todo:
            return Outcome("waiting", "; ".join(todo), done="; ".join(notes))
        return Outcome("done", "; ".join(notes))

    # ---- step 4 ---------------------------------------------------------------
    def _rectification(self, cfg: AutomationConfig, api: AmcApi, resumed: bool) -> Outcome:
        r = cfg.rectification
        if r.mode == "skip":
            return Outcome("skipped", "rectification.mode is skip")
        if r.mode == "ui":
            if resumed:
                return self._finished_in_amc("rectification")
            return Outcome("waiting", "Run Rectification in AMC (Auto, Manual, or Videos Are Rectified) "
                                      "until it shows COMPLETED")
        if not r.api.start.path or not r.api.status.path or not r.api.status.state_field:
            raise StepFailed("rectification.mode is api but rectification.api start/status paths are not set "
                             "(take them from AMC's /openapi.yaml)")
        pid = self._job.amc_project_id
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
        if path is None:
            if resumed:
                return self._finished_in_amc("manual_alignment")
            return Outcome("waiting", "Do Manual Alignment in AMC: mark at least 4 matching point sets "
                                      "(6-8 recommended) on each camera and the layout, then Save")
        if self._job.step("rectification")["State"] == "waiting":
            return Outcome("waiting", f"Nothing to do: the saved {path.name} is uploaded after rectification",
                           needs_user=False)
        api.upload_alignment(self._job.amc_project_id, path, cfg.alignment.coord_space)
        return Outcome("done", f"re-used saved alignment {path} ({cfg.alignment.coord_space} coordinates)")

    # ---- step 6 ---------------------------------------------------------------
    def _execute(self, cfg: AutomationConfig, api: AmcApi, resumed: bool) -> Outcome:
        pid = self._job.amc_project_id
        state = api.verify_project(pid)
        if state != "READY":
            with self._lock:
                s = self._job.step("manual_alignment")
                s["State"], s["NeedsUser"] = "waiting", True
                s["Message"] = (f"AMC reports {state or 'no state'} instead of READY. Check the videos, layout, "
                                "rectification and alignment in AMC, save, then press Continue")
            raise _PauseForUser()
        c = cfg.calibration
        api.calibrate(pid, c.detector_type)
        start = self._clock()
        while True:
            self._wait(c.poll_seconds)
            state = api.project_state(pid)
            minutes = int((self._clock() - start) // 60)
            if state == "COMPLETED":
                return Outcome("done", f"calibrated with {c.detector_type} in about {minutes} min")
            if state == "ERROR":
                raise StepFailed("AMC calibration ended in ERROR. Last log lines:\n" + api.calibration_log(pid))
            if self._clock() - start >= c.timeout_seconds:
                raise StepFailed(f"calibration still {state} after {minutes} min; see AMC's calibration log")
            self._set_step("execute", "running", f"{state} for {minutes} min (typically 10-60 min)")

    # ---- step 7 ---------------------------------------------------------------
    def _results(self, cfg: AutomationConfig, api: AmcApi, resumed: bool) -> Outcome:
        job = self._job
        notes = []
        stats = api.evaluation_statistics(job.amc_project_id)
        if stats:
            job.result["EvaluationStatistics"] = stats
            notes.append("evaluation statistics saved")
        if job.folder:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            out_dir = self._folder() / "calibration" / stamp
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

