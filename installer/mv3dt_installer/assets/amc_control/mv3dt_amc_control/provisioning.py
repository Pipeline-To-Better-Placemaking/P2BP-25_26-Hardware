"""Setting up a web project on the workstation: its folder and its AMC project.

`POST /projects` hands a verified calibration spec to `Provisioner.provision`,
which checks it, applies the idempotency and busy rules, writes the binding in
state `provisioning` and returns at once. A background job then starts AMC
through the controller if it is down, creates an AMC project named after the
folder, saves the floor plan and marks the binding `ready`; any failure marks
it `failed` with the error and keeps the folder, so the web app can retry.

Only one AMC user at a time: while a job runs `busy()` is true, and the API
refuses a second provision, `POST /amc/stop` and an automated run. A run that
is active or waiting for the user likewise blocks provisioning.

`Provisioner.delete` removes a project from the workstation (its AMC project,
then its folder) inside the same exclusive section, so no provisioning job or
run can start meanwhile.

See installer/plan/AMC-WEB-PROJECTS.md section 5 and AMC-PROJECT-LIFECYCLE.md
section 3.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, TypeVar

import httpx

from .amc_api import AmcApi, AmcApiError, default_base_url
from .automation_config import AutomationConfig, load_config
from .calibration_spec import CalibrationSpec, download_floorplan_png
from .projects import (STATE_FAILED, STATE_PROVISIONING, STATE_READY, SUBDIRS, Binding, FolderTaken,
                       InvalidFolder, ProjectAlreadyBound, ProjectNotFound, ProjectStore, validate_folder)

RESTART_ERROR = "interrupted by a service restart; set it up again"
BUSY_MESSAGE = "A project is being set up on the workstation; try again when it finishes"
NO_FOLDER = "This project has no workstation folder"
AMC_STARTING_RETRY_SECONDS = 15
# A run in any other state could still be continued or cancelled, so its project cannot be deleted.
RUN_FINISHED_STATES = frozenset(("completed", "cancelled"))
T = TypeVar("T")
LAYOUT_FILE = Path("floorplan") / "layout.png"


class ProvisionError(Exception):
    """A refused provision request, with the HTTP status the API answers."""

    def __init__(self, status: int, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.retry_after = retry_after  # seconds, for a Retry-After header


def _spawn_thread(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, name="amc-provision", daemon=True).start()


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


class Provisioner:
    def __init__(
        self,
        store: ProjectStore,
        controller,
        *,
        runner_busy: Callable[[], bool] = lambda: False,
        runner_job: Callable[[], dict | None] = lambda: None,
        config_loader: Callable[[], AutomationConfig] = load_config,
        api_factory: Callable[[AutomationConfig], AmcApi] | None = None,
        http_client_factory: Callable[[], httpx.Client] = httpx.Client,
        download_client_factory: Callable[[], httpx.Client] = httpx.Client,
        clock: Callable[[], float] = time.time,
        wait: Callable[[float], None] = time.sleep,
        spawn: Callable[[Callable[[], None]], None] = _spawn_thread,
    ):
        self.store = store
        self.controller = controller
        self._runner_busy = runner_busy
        # The runner's current or last job (its public dict, with ProjectId and State), or None.
        self._runner_job = runner_job
        self._load_config = config_loader
        self._api_factory = api_factory or self._default_api
        self._http_factory = http_client_factory
        self._download_factory = download_client_factory
        self._clock = clock
        self._wait = wait
        self._spawn = spawn
        # Reentrant so a run_exclusive callback may call busy().
        self._lock = threading.RLock()
        self._running: str | None = None  # the folder being provisioned

    def _default_api(self, cfg: AutomationConfig) -> AmcApi:
        base = cfg.amc.api_base_url or default_base_url(self.controller.ports()[0])
        return AmcApi(base, timeout=cfg.amc.request_timeout_seconds,
                      upload_timeout=cfg.amc.upload_timeout_seconds, client=self._http_factory())

    # ---- public API ---------------------------------------------------------
    def busy(self) -> bool:
        """True while a provisioning job runs."""
        with self._lock:
            return self._running is not None

    def run_exclusive(self, fn: Callable[[], T]) -> T:
        """Run `fn` while no provisioning job can start; ProvisionError 409 if one runs.

        Other AMC users (an automated run starting or continuing, AMC being
        stopped) call through here so their busy check and their action are
        one critical section with `provision`'s. Lock order is provisioner
        then runner, the same order `provision` takes through `runner_busy`.
        """
        with self._lock:
            if self._running is not None:
                raise ProvisionError(409, BUSY_MESSAGE)
            return fn()

    def recover(self) -> list[Binding]:
        """At service start: no job survives a restart, so `provisioning` becomes `failed`."""
        out = []
        for b in self.store.list():
            if b.state == STATE_PROVISIONING:
                out.append(self.store.update(b.folder, state=STATE_FAILED, error=RESTART_ERROR))
        return out

    def provision(self, spec: CalibrationSpec, pid: str) -> tuple[Binding, bool]:
        """Start provisioning `pid` into the spec's folder.

        Returns (binding, started): started is False when the binding is
        already ready or provisioning and is returned unchanged.
        Raises ProvisionError (400 or 409) when refused.
        """
        folder = spec.folder
        if not folder:
            raise ProvisionError(400, "the calibration spec has no workstation folder")
        try:
            validate_folder(folder)
        except InvalidFolder as exc:
            raise ProvisionError(400, str(exc)) from None
        if spec.project_id != pid:
            raise ProvisionError(400, "calibration spec is for a different project than the ticket")

        with self._lock:
            existing = self.store.get_by_folder(folder)
            if existing is not None and existing.project_id != pid:
                raise ProvisionError(409, f"workstation folder {folder!r} belongs to another web project")
            bound = self.store.get_by_pid(pid)
            if bound is not None and bound.folder != folder:
                raise ProvisionError(409, f"this project is already set up in workstation folder {bound.folder!r}")
            if existing is not None and existing.state in (STATE_READY, STATE_PROVISIONING):
                return existing, False

            if self._running is not None:
                raise ProvisionError(409, "another project is being set up on the workstation; try again shortly")
            if self._runner_busy():
                raise ProvisionError(409, "a calibration run is using AMC; finish or cancel it first")

            try:
                if existing is None:
                    binding = self.store.create(pid, folder, spec.title)
                else:  # failed: provision it again
                    path = self.store.folder_path(folder)
                    for sub in SUBDIRS:
                        (path / sub).mkdir(parents=True, exist_ok=True)
                    binding = self.store.update(folder, state=STATE_PROVISIONING, error="", title=spec.title)
            except (FolderTaken, ProjectAlreadyBound) as exc:
                raise ProvisionError(409, str(exc)) from None
            except InvalidFolder as exc:
                raise ProvisionError(400, str(exc)) from None
            self._running = folder

        try:
            self._spawn(lambda: self._job(folder, spec))
        except BaseException:
            with self._lock:
                self._running = None
            self.store.update(folder, state=STATE_FAILED, error="could not start the setup job")
            raise
        return binding, True

    def delete(self, pid: str) -> dict:
        """Delete `pid`'s AMC project and folder; the DELETE /projects/current body.

        Raises ProvisionError: 404 without a folder; 409 while a provisioning
        job runs or this project has a run that is not completed or
        cancelled; 503 with a retry delay when AMC is not ready (it is asked
        to start, nothing waits for it and nothing is removed); 502 when AMC
        refuses the delete (nothing is removed then). AMC is left running.
        """
        return self.run_exclusive(lambda: self._delete(pid))

    def _delete(self, pid: str) -> dict:
        binding = self.store.get_by_pid(pid)
        if binding is None:
            raise ProvisionError(404, NO_FOLDER)
        job = self._runner_job()
        if job and job.get("ProjectId") == pid and job.get("State") not in RUN_FINISHED_STATES:
            raise ProvisionError(409, "Cancel this project's run first")
        amc_deleted = False
        if binding.amc_project_id:
            cfg = self._load_config()
            api = self._api_factory(cfg)
            try:
                if not api.ready():
                    raise self._amc_starting()
                try:
                    api.delete_project(binding.amc_project_id)  # its 404 counts as deleted
                except (AmcApiError, httpx.HTTPError) as exc:
                    raise ProvisionError(502, f"AMC did not delete project {binding.amc_project_id}: {exc}") from exc
                amc_deleted = True
            finally:
                api.close()
        try:
            leftover = self.store.remove(binding.folder)
        except ProjectNotFound:
            raise ProvisionError(404, NO_FOLDER) from None
        except OSError as exc:
            raise ProvisionError(500, f"could not remove workstation folder {binding.folder!r}: {exc}") from exc
        return {"Folder": binding.folder, "AmcProjectId": binding.amc_project_id,
                "AmcProjectDeleted": amc_deleted, "Leftover": leftover}

    def _amc_starting(self) -> ProvisionError:
        """Ask AMC to start (returns at once) and build the 503 the client retries after.

        The controller's error is read before and after `start()`: `start()`
        clears it, and an earlier start that exited with an error is only
        recorded when `status()` looks. The later error wins when both exist.
        """
        before = self._controller_error()
        self.controller.start()
        error = self._controller_error() or before
        message = "AMC is starting on the workstation; try again shortly"
        if error:
            message += f" (last start error: {error})"
        return ProvisionError(503, message, retry_after=AMC_STARTING_RETRY_SECONDS)

    def _controller_error(self) -> str:
        try:
            return self.controller.status().get("Error") or ""
        except Exception:
            return ""

    # ---- background job -----------------------------------------------------
    def _job(self, folder: str, spec: CalibrationSpec) -> None:
        try:
            cfg = self._load_config()
            api = self._api_factory(cfg)
            try:
                self._ensure_amc(cfg, api)
                amc_id = self._amc_project(api, folder)
                if spec.floorplan is not None:
                    client = self._download_factory()
                    try:
                        png = download_floorplan_png(spec.floorplan, cfg.layout.allowed_hosts,
                                                     cfg.layout.max_bytes, client)
                    finally:
                        client.close()
                    target = self.store.folder_path(folder) / LAYOUT_FILE
                    target.parent.mkdir(parents=True, exist_ok=True)
                    _write_bytes_atomic(target, png)
            finally:
                api.close()
            self.store.update(folder, amc_project_id=amc_id, amc_project_name=folder,
                              title=spec.title, state=STATE_READY, error="")
        except Exception as exc:  # anything at all leaves a failed binding, never a stuck one
            try:
                self.store.update(folder, state=STATE_FAILED, error=str(exc) or type(exc).__name__)
            except Exception:
                pass
        finally:
            with self._lock:
                self._running = None

    def _ensure_amc(self, cfg: AutomationConfig, api: AmcApi) -> None:
        if api.ready():
            return
        self.controller.start()
        timeout = cfg.amc.ready_timeout_seconds
        deadline = self._clock() + timeout
        while not api.ready():
            if self._clock() >= deadline:
                raise RuntimeError(f"AMC did not become ready within {int(timeout)} s "
                                   "(the first launch downloads several GB)")
            self._wait(5)

    def _amc_project(self, api: AmcApi, folder: str) -> str:
        """The AMC project for the folder: one an earlier attempt created, if AMC still has it."""
        current = self.store.get_by_folder(folder)
        if current is not None and current.amc_project_id:
            try:
                api.get_project_info(current.amc_project_id)
                return current.amc_project_id
            except AmcApiError:
                pass
        amc_id = api.create_project(folder)
        # Record it now so a retry after a later failure reuses it instead of making another.
        self.store.update(folder, amc_project_id=amc_id, amc_project_name=folder)
        return amc_id
