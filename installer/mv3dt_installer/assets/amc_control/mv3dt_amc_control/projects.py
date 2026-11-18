"""The workstation side of a web project: one folder and one AMC project each.

Every web app project that has a workstation folder gets
`<projects root>/<folder>/` with a `project.json` binding it to the web
project id and, once provisioned, to an AMC project. The folder name is user
input used verbatim as a directory and AMC project name, so it is checked
against one fixed rule. Files are served from the folder only after the
requested path is resolved (symlinks followed) and found inside it. Video
files are listed but never served: calibration footage stays on the
workstation (`is_video`).

Deleting a project renames its folder to a dot-prefixed name in one
`rename(2)`, so the name is free at once and the leftover can never read as
a project (the folder rule rejects a leading dot), then removes that tree.

Pure stdlib, no web framework: the API and the automation both read this.
See installer/plan/AMC-WEB-PROJECTS.md sections 3, 4 and 7,
AMC-PROJECT-LIFECYCLE.md sections 3 and 4, and AMC-RECTIFICATION-AND-SCALE.md
section 4.3 (the optional LayoutPxPerM and LayoutSha256 keys, null when an
older project.json lacks them).

Environment:
  AMC_PROJECTS_DIR   projects root, default <MV3DT_INSTALL_DIR>/amc_projects
  MV3DT_INSTALL_DIR  install root, default /opt/mv3dt
"""
from __future__ import annotations

import json
import logging
import math
import os
import posixpath
import re
import shutil
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

FOOTAGE_DIR = "footage"
VIDEO_SUFFIXES = frozenset((".mp4", ".mkv", ".mov", ".avi", ".ts", ".h264", ".h265"))
DELETING_PREFIX = ".deleting-"

log = logging.getLogger(__name__)


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


def is_video(rel: str) -> bool:
    """True for a path the service never serves: anything under footage/, or a
    video suffix (case-insensitive). The path is normalized as a POSIX path
    first and decided on the string alone, with no disk access."""
    path = PurePosixPath(posixpath.normpath(rel or "."))
    if path.parts and path.parts[0] == FOOTAGE_DIR:
        return True
    return path.suffix.lower() in VIDEO_SUFFIXES


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
    "LayoutPxPerM": "layout_px_per_m",
    "LayoutSha256": "layout_sha256",
}
# Keys an older project.json may lack; they read as null (AMC-RECTIFICATION-AND-SCALE.md section 4.3).
_OPTIONAL = ("LayoutPxPerM", "LayoutSha256")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
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
    # The remembered floor plan scale and the SHA-256 of the layout.png it was confirmed
    # for, written together (AMC-RECTIFICATION-AND-SCALE.md section 4.3).
    layout_px_per_m: float | None = None
    layout_sha256: str | None = None

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
        version = data.get("schema_version") if isinstance(data, dict) else None
        if type(version) is not int or version != SCHEMA_VERSION:  # True == 1
            raise ValueError("not a version 1 binding")
        try:
            b = cls(**{a: data.get(k) if k in _OPTIONAL else data[k] for k, a in _KEYS.items()})
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
              and is_int(self.calib_runs) and self.calib_runs >= 0
              and (self.layout_px_per_m is None
                   or (isinstance(self.layout_px_per_m, (int, float)) and not isinstance(self.layout_px_per_m, bool)
                       and math.isfinite(self.layout_px_per_m) and self.layout_px_per_m > 0))
              and (self.layout_sha256 is None
                   or (isinstance(self.layout_sha256, str) and SHA256_RE.fullmatch(self.layout_sha256))))
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
        """Where the folder lives (or would). A name check only, no disk access."""
        return self.root / validate_folder(folder)

    def _real_dir(self, folder: Any) -> Path | None:
        """The folder's real path, or None unless it is a plain directory
        directly under the real root (a symlinked entry could point anywhere)."""
        if not isinstance(folder, str) or not FOLDER_RE.fullmatch(folder):
            return None
        entry = self.root / folder
        try:
            if entry.is_symlink() or not entry.is_dir():
                return None
            real = entry.resolve(strict=True)
            if real.parent != self.root.resolve(strict=True) or real.name != folder:
                return None
        except (OSError, RuntimeError):
            return None
        return real

    def _read(self, folder: Any) -> Binding | None:
        base = self._real_dir(folder)
        if base is None:
            return None
        path = base / BINDING_FILE
        try:
            if path.is_symlink() or not path.is_file():
                return None
            b = Binding.from_json(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None
        return b if b.folder == folder else None

    def get_by_folder(self, folder: str) -> Binding | None:
        return self._read(folder)

    def get_by_pid(self, pid: str) -> Binding | None:
        for b in self.list():
            if b.project_id == pid:
                return b
        return None

    def list(self) -> list[Binding]:
        try:
            names = sorted(p.name for p in self.root.iterdir())
        except OSError:
            return []
        return [b for b in map(self._read, names) if b is not None]

    def create(self, pid: str, folder: str, title: str) -> Binding:
        """Bind `pid` to a new folder in state provisioning.

        Calling it again with the same pid and folder returns the existing
        binding unchanged, whatever its state; re-provisioning a failed one is
        the caller's decision (`update`).
        """
        validate_folder(folder)
        if not isinstance(pid, str) or not pid:
            raise ValueError("web project id is required")
        now = self._now()
        b = Binding(project_id=pid, folder=folder, title=title,
                    amc_project_id=None, amc_project_name=None,
                    state=STATE_PROVISIONING, error="",
                    created_at=now, updated_at=now,
                    last_run_at=None, calib_runs=0)
        b._check()  # before anything touches the disk
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
            if path.is_symlink() or (path.exists() and not path.is_dir()):
                raise FolderTaken(f"{path} exists and is not a plain directory")
            path.mkdir(parents=True, exist_ok=True)
            if self._real_dir(folder) is None:
                raise FolderTaken(f"{path} is not a plain directory under the projects root")
            for sub in SUBDIRS:
                (path / sub).mkdir(exist_ok=True)
            _write_atomic(path / BINDING_FILE, b.to_json())
            return b

    def save(self, binding: Binding) -> Binding:
        """Rewrite an existing binding. Only fields `update` may change can
        differ from what is on disk, so a save never moves or steals a binding."""
        b = replace(binding, updated_at=self._now())
        b._check()
        with self._lock:
            current = self._read(b.folder)
            if current is None:
                raise ProjectNotFound(f"no project bound to folder {b.folder!r}")
            for name in _IMMUTABLE:
                if getattr(b, name) != getattr(current, name):
                    raise ValueError(f"binding field cannot change: {name}")
            bound = self.get_by_pid(b.project_id)
            if bound is not None and bound.folder != b.folder:
                raise ProjectAlreadyBound(
                    f"web project is already bound to folder {bound.folder!r}")
            _write_atomic(self.root / b.folder / BINDING_FILE, b.to_json())
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

    def remove(self, folder: str) -> str | None:
        """Delete the folder and its binding. Returns the leftover path, or None when all is gone.

        One rename moves `<root>/<folder>` to `<root>/.deleting-<folder>-<unix seconds>`,
        then the renamed tree is removed without following symlinks. Once the
        rename is done nothing raises: a removal error is logged and the renamed
        path returned. ProjectNotFound unless the folder is a plain directory
        under the root; an OSError from the rename itself propagates.
        """
        with self._lock:
            if self._real_dir(folder) is None:
                raise ProjectNotFound(f"no project folder {folder!r}")
            stamp = self._now()
            target = self.root / f"{DELETING_PREFIX}{folder}-{stamp}"
            n = 1
            while os.path.lexists(target):  # a leftover from an earlier delete this second
                target = self.root / f"{DELETING_PREFIX}{folder}-{stamp}-{n}"
                n += 1
            os.rename(self.root / folder, target)
        try:
            # rmtree never follows a symlink inside the tree: it unlinks the link.
            shutil.rmtree(target)
        except Exception as exc:  # the binding is already gone, so never fail here
            log.warning("could not fully remove %s: %s", target, exc)
            return str(target) if os.path.lexists(target) else None
        return None

    def list_files(self, folder: str) -> list[dict[str, Any]]:
        """FileNotFoundError unless the folder is a plain directory under the root."""
        base = self._real_dir(folder)
        if base is None:
            raise FileNotFoundError(f"no project folder {folder!r}")
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
                rel = p.relative_to(base).as_posix()
                out.append({"Path": rel, "Size": st.st_size, "ModifiedAt": int(st.st_mtime),
                            "Downloadable": not is_video(rel)})
        out.sort(key=lambda e: e["Path"])
        return out

    def resolve_file(self, folder: str, rel: str) -> Path:
        """The regular file at `rel` inside the folder; FileNotFoundError otherwise."""
        missing = FileNotFoundError(f"no such file in the project: {rel!r}")
        base = self._real_dir(folder)
        if base is None:
            raise missing
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
