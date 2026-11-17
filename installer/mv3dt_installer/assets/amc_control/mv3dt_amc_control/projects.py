"""The workstation side of a web project: one folder and one AMC project each.

Every web app project that has a workstation folder gets
`<projects root>/<folder>/` with a `project.json` binding it to the web
project id and, once provisioned, to an AMC project. The folder name is user
input used verbatim as a directory and AMC project name, so it is checked
against one fixed rule. Files are served from the folder only after the
requested path is resolved (symlinks followed) and found inside it.

Pure stdlib, no web framework: the API and the automation both read this.
See installer/plan/AMC-WEB-PROJECTS.md sections 3, 4 and 7.

Environment:
  AMC_PROJECTS_DIR   projects root, default <MV3DT_INSTALL_DIR>/amc_projects
  MV3DT_INSTALL_DIR  install root, default /opt/mv3dt
"""
from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import threading
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

FOLDER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,46}[a-z0-9])?$")
# Always fullmatch: with match, "$" also accepts a trailing newline.
FOLDER_MAX = 48
SCHEMA_VERSION = 1
BINDING_FILE = "project.json"
SUBDIRS = ("footage", "floorplan", "alignment", "calibration", "runs")

STATE_PROVISIONING = "provisioning"
STATE_READY = "ready"
STATE_FAILED = "failed"
STATES = (STATE_PROVISIONING, STATE_READY, STATE_FAILED)


class InvalidFolder(ValueError):
    pass


class FolderTaken(Exception):
    """The folder is already bound to another web project."""


class ProjectAlreadyBound(Exception):
    """The web project is already bound to another folder."""


class ProjectNotFound(LookupError):
    pass


def validate_folder(name: str) -> str:
    if not isinstance(name, str) or not FOLDER_RE.fullmatch(name):
        raise InvalidFolder(
            f"invalid workstation folder {name!r}: 1 to {FOLDER_MAX} lowercase "
            "letters, digits and inner hyphens")
    return name


def suggest_folder(title: str) -> str:
    """The web form's prefill rule; "" when the title has nothing usable."""
    s = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")
    return s[:FOLDER_MAX].rstrip("-")


def default_projects_root(env: Mapping[str, str] | None = None) -> Path:
    env = os.environ if env is None else env
    if env.get("AMC_PROJECTS_DIR"):
        return Path(env["AMC_PROJECTS_DIR"])
    return Path(env.get("MV3DT_INSTALL_DIR") or "/opt/mv3dt") / "amc_projects"


# project.json key -> Binding attribute, in on-disk order.
_KEYS = {
    "ProjectId": "project_id",
    "Folder": "folder",
    "Title": "title",
    "AmcProjectId": "amc_project_id",
    "AmcProjectName": "amc_project_name",
    "State": "state",
    "Error": "error",
    "CreatedAt": "created_at",
    "UpdatedAt": "updated_at",
    "LastRunAt": "last_run_at",
    "CalibRuns": "calib_runs",
}
_IMMUTABLE = ("project_id", "folder", "created_at")


@dataclass(frozen=True)
class Binding:
    project_id: str
    folder: str
    title: str
    amc_project_id: str | None
    amc_project_name: str | None
    state: str
    error: str
    created_at: int
    updated_at: int
    last_run_at: int | None
    calib_runs: int

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
        out.update({k: getattr(self, a) for k, a in _KEYS.items()})
        return out

    def to_wire(self, path: Path | str) -> dict[str, Any]:
        out = {k: getattr(self, a) for k, a in _KEYS.items()}
        out["Path"] = str(path)
        return out

    @classmethod
    def from_json(cls, data: Any) -> Binding:
        """Strict parse; ValueError for anything not written by `to_json`."""
        if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("not a version 1 binding")
        try:
            b = cls(**{a: data[k] for k, a in _KEYS.items()})
        except KeyError as e:
            raise ValueError(f"binding is missing {e}") from e
        b._check()
        return b

    def _check(self) -> None:
        def is_int(v: Any) -> bool:
            return isinstance(v, int) and not isinstance(v, bool)

        ok = (isinstance(self.project_id, str) and self.project_id
              and isinstance(self.folder, str) and FOLDER_RE.fullmatch(self.folder)
              and isinstance(self.title, str)
              and isinstance(self.amc_project_id, (str, type(None)))
              and isinstance(self.amc_project_name, (str, type(None)))
              and self.state in STATES
              and isinstance(self.error, str)
              and is_int(self.created_at) and is_int(self.updated_at)
              and (self.last_run_at is None or is_int(self.last_run_at))
              and is_int(self.calib_runs) and self.calib_runs >= 0)
        if not ok:
            raise ValueError("binding has a field of the wrong type")


def _write_atomic(path: Path, data: dict[str, Any]) -> None:
    # Same directory so os.replace is a rename on one filesystem.
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


class ProjectStore:
    """Bindings under one projects root. Safe for one process's threads."""

    def __init__(self, root: Path, clock: Callable[[], float] = time.time):
        self.root = Path(root)
        self._clock = clock
        self._lock = threading.RLock()

    def _now(self) -> int:
        return int(self._clock())

    def folder_path(self, folder: str) -> Path:
        return self.root / validate_folder(folder)

    def _read(self, folder: str) -> Binding | None:
        if not FOLDER_RE.fullmatch(folder):
            return None
        path = self.root / folder / BINDING_FILE
        try:
            if path.is_symlink() or not path.is_file():
                return None
            b = Binding.from_json(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None
        return b if b.folder == folder else None

    def get_by_folder(self, folder: str) -> Binding | None:
        return self._read(folder) if isinstance(folder, str) else None

    def get_by_pid(self, pid: str) -> Binding | None:
        for b in self.list():
            if b.project_id == pid:
                return b
        return None

    def list(self) -> list[Binding]:
        try:
            entries = sorted(self.root.iterdir(), key=lambda p: p.name)
        except OSError:
            return []
        out = []
        for entry in entries:
            if entry.is_symlink() or not entry.is_dir():
                continue
            b = self._read(entry.name)
            if b is not None:
                out.append(b)
        return out

    def create(self, pid: str, folder: str, title: str) -> Binding:
        """Bind `pid` to a new folder in state provisioning.

        Calling it again with the same pid and folder returns the existing
        binding unchanged, whatever its state; re-provisioning a failed one is
        the caller's decision (`update`).
        """
        validate_folder(folder)
        if not isinstance(pid, str) or not pid:
            raise ValueError("web project id is required")
        with self._lock:
            existing = self.get_by_folder(folder)
            if existing is not None and existing.project_id != pid:
                raise FolderTaken(f"folder {folder!r} is bound to another web project")
            bound = self.get_by_pid(pid)
            if bound is not None and bound.folder != folder:
                raise ProjectAlreadyBound(
                    f"web project is already bound to folder {bound.folder!r}")
            if existing is not None:
                return existing
            path = self.folder_path(folder)
            path.mkdir(parents=True, exist_ok=True)
            for sub in SUBDIRS:
                (path / sub).mkdir(exist_ok=True)
            now = self._now()
            b = Binding(project_id=pid, folder=folder, title=title or "",
                        amc_project_id=None, amc_project_name=None,
                        state=STATE_PROVISIONING, error="",
                        created_at=now, updated_at=now,
                        last_run_at=None, calib_runs=0)
            _write_atomic(path / BINDING_FILE, b.to_json())
            return b

    def save(self, binding: Binding) -> Binding:
        b = replace(binding, updated_at=self._now())
        b._check()
        with self._lock:
            _write_atomic(self.folder_path(b.folder) / BINDING_FILE, b.to_json())
        return b

    def update(self, folder: str, /, **changes: Any) -> Binding:
        names = {f.name for f in fields(Binding)}
        unknown = set(changes) - names
        if unknown:
            raise TypeError(f"unknown binding fields: {sorted(unknown)}")
        fixed = set(changes) & set(_IMMUTABLE)
        if fixed:
            raise ValueError(f"binding fields cannot change: {sorted(fixed)}")
        with self._lock:
            current = self.get_by_folder(folder)
            if current is None:
                raise ProjectNotFound(f"no project bound to folder {folder!r}")
            return self.save(replace(current, **changes))

    def list_files(self, folder: str) -> list[dict[str, Any]]:
        base = self.folder_path(folder)
        out: list[dict[str, Any]] = []
        for dirpath, dirnames, filenames in os.walk(base):  # never follows links
            d = Path(dirpath)
            dirnames[:] = [n for n in dirnames if not (d / n).is_symlink()]
            for name in filenames:
                p = d / name
                if d == base and name == BINDING_FILE:
                    continue
                try:
                    st = p.lstat()
                except OSError:
                    continue
                if not stat.S_ISREG(st.st_mode):
                    continue
                out.append({"Path": p.relative_to(base).as_posix(),
                            "Size": st.st_size, "ModifiedAt": int(st.st_mtime)})
        out.sort(key=lambda e: e["Path"])
        return out

    def resolve_file(self, folder: str, rel: str) -> Path:
        """The regular file at `rel` inside the folder; FileNotFoundError otherwise."""
        missing = FileNotFoundError(f"no such file in the project: {rel!r}")
        try:
            base = self.folder_path(folder).resolve(strict=True)
        except (InvalidFolder, OSError):
            raise missing from None
        if not isinstance(rel, str) or not rel or "\x00" in rel or "\\" in rel:
            raise missing
        parts = PurePosixPath(rel)
        if parts.is_absolute() or ".." in parts.parts or parts.parts == (BINDING_FILE,):
            raise missing
        try:
            target = (base / parts).resolve(strict=True)
        except (OSError, RuntimeError):
            raise missing from None
        if target == base / BINDING_FILE or not target.is_relative_to(base) \
                or not target.is_file():
            raise missing
        return target
