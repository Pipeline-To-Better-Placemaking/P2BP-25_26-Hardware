"""Edit protection for files the installer renders (dpkg-conffile style).

The installer records, per absolute path, the sha256 of the content it last
wrote. A later write only replaces a file that is still exactly what the
installer left there. A file an operator edited (or one that predates the
manifest, as on an install made before this module existed) is kept, and
the new version lands beside it as ``<path>.installer-new`` for the
operator to review with ``mv3dt-installer configs diff`` and take with
``mv3dt-installer configs accept``.

Manifest shape (``/var/lib/mv3dt-installer/managed-files.json``)::

    {"version": 1, "files": {"/abs/path": "<sha256>" | null}}

``null`` marks a path the installer has offered a new version for but never
wrote itself, so ``configs status`` can still list it.
"""

from __future__ import annotations

import argparse
import difflib
import enum
import hashlib
import json
import os
import pathlib
import stat
import sys
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union

from mv3dt_installer.logs import log
from mv3dt_installer.state import write_json_atomic

if TYPE_CHECKING:  # pragma: no cover
    from mv3dt_installer.app import Context

__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "NEW_SUFFIX",
    "Outcome",
    "FileState",
    "ManagedStatus",
    "Manifest",
    "pending_path",
    "write_managed",
    "status",
    "diff",
    "accept",
    "accept_all",
    "handle_configs_subcommand",
    "configs_requires_root",
]

#: Beside `state.json` (doc 00 §6.1), root-owned and world-readable so
#: `configs status`/`diff` work without sudo. Tests replace it.
DEFAULT_MANIFEST_PATH = pathlib.Path("/var/lib/mv3dt-installer/managed-files.json")
NEW_SUFFIX = ".installer-new"
CORRUPT_SUFFIX = ".corrupt"
DEFAULT_MODE = 0o644
_MANIFEST_VERSION = 1

PathLike = Union[str, os.PathLike]


class Outcome(enum.Enum):
    WRITTEN = "written"
    UNCHANGED = "unchanged"
    KEPT_MODIFIED = "kept-modified"


class FileState(enum.Enum):
    CLEAN = "clean"
    MODIFIED = "modified"
    PENDING_NEW = "pending-new"
    MISSING = "missing"


@dataclass(frozen=True)
class ManagedStatus:
    path: pathlib.Path
    state: FileState


def _key(path: PathLike) -> str:
    return os.path.abspath(os.fspath(path))


def pending_path(path: PathLike) -> pathlib.Path:
    return pathlib.Path(_key(path) + NEW_SUFFIX)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Manifest:
    """The path -> sha256 record. Every mutation is load, change, save, so
    the file on disk is always the single source of truth."""

    def __init__(self, path: Optional[pathlib.Path] = None) -> None:
        self.path = pathlib.Path(path) if path is not None else DEFAULT_MANIFEST_PATH

    def load(self) -> dict[str, Optional[str]]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        try:
            data = json.loads(raw)
            files = data["files"]
            if not isinstance(files, dict) or not all(
                isinstance(k, str) and (v is None or isinstance(v, str))
                for k, v in files.items()
            ):
                raise ValueError("files is not a path -> sha256 mapping")
        except (ValueError, KeyError, TypeError) as exc:
            self._quarantine(exc)
            return {}
        return dict(files)

    def _quarantine(self, exc: Exception) -> None:
        backup = self.path.with_name(self.path.name + CORRUPT_SUFFIX)
        try:
            os.replace(self.path, backup)
            log.warn(
                f"managed-files manifest {self.path} was unreadable ({exc}); "
                f"moved it to {backup} and started an empty one"
            )
        except OSError:
            # Read-only callers (`configs status` without sudo) cannot move
            # it; treating it as empty is still the safe reading.
            log.warn(
                f"managed-files manifest {self.path} is unreadable ({exc}); "
                "treating it as empty"
            )

    def save(self, files: dict[str, Optional[str]]) -> None:
        write_json_atomic(self.path, {"version": _MANIFEST_VERSION, "files": files})
        try:
            self.path.chmod(0o644)
        except OSError:
            pass

    def get(self, path: PathLike) -> Optional[str]:
        return self.load().get(_key(path))

    def record(self, path: PathLike, digest: Optional[str]) -> None:
        files = self.load()
        key = _key(path)
        if key in files and files[key] == digest:
            return
        files[key] = digest
        self.save(files)

    def track(self, path: PathLike) -> None:
        """List `path` without claiming the installer wrote its content."""
        files = self.load()
        key = _key(path)
        if key not in files:
            files[key] = None
            self.save(files)


def _atomic_write(
    path: pathlib.Path,
    data: bytes,
    *,
    mode: Optional[int],
    like: Optional[pathlib.Path] = None,
) -> None:
    """Write `data` to `path` through a temp file in the same directory and
    `os.replace`. The mode (and, where permitted, owner) comes from `mode`,
    else from `like`, else from the file being replaced, else 0644."""
    path.parent.mkdir(parents=True, exist_ok=True)
    reference = None
    for candidate in (like, path):
        if candidate is None:
            continue
        try:
            reference = candidate.stat()
            break
        except OSError:
            continue
    if mode is None:
        mode = stat.S_IMODE(reference.st_mode) if reference else DEFAULT_MODE
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = pathlib.Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, mode)
        if reference is not None:
            try:
                os.chown(tmp, reference.st_uid, reference.st_gid)
            except OSError:
                pass
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write_managed(
    path: PathLike,
    text: str,
    *,
    manifest: Manifest,
    mode: Optional[int] = None,
    label: Optional[str] = None,
) -> Outcome:
    """Write `text` to `path` unless that would discard an operator edit.

    - missing, or still exactly what the installer last wrote: write it
      (WRITTEN);
    - already equal to `text`: record it (UNCHANGED);
    - anything else: leave it, write ``<path>.installer-new`` and warn
      (KEPT_MODIFIED).
    """
    target = pathlib.Path(_key(path))
    data = text.encode("utf-8")
    new_digest = _sha256(data)
    pending = pending_path(target)
    try:
        current = target.read_bytes()
    except FileNotFoundError:
        current = None
    if current is None or _sha256(current) == new_digest:
        if current is None:
            _atomic_write(target, data, mode=mode)
        elif mode is not None and stat.S_IMODE(target.stat().st_mode) != mode:
            target.chmod(mode)
        manifest.record(target, new_digest)
        pending.unlink(missing_ok=True)
        return Outcome.WRITTEN if current is None else Outcome.UNCHANGED
    recorded = manifest.get(target)
    if recorded is not None and recorded == _sha256(current):
        _atomic_write(target, data, mode=mode)
        manifest.record(target, new_digest)
        pending.unlink(missing_ok=True)
        return Outcome.WRITTEN
    try:
        stale = pending.read_bytes() != data
    except FileNotFoundError:
        stale = True
    if stale:
        _atomic_write(pending, data, mode=mode, like=target)
    manifest.track(target)
    log.warn(
        f"kept {label or target.name} at {target} (operator edited), "
        f"new version at {pending}"
    )
    return Outcome.KEPT_MODIFIED


def status(manifest: Manifest) -> list[ManagedStatus]:
    result = []
    for key, recorded in sorted(manifest.load().items()):
        path = pathlib.Path(key)
        if pending_path(path).exists():
            state = FileState.PENDING_NEW
        elif not path.exists():
            state = FileState.MISSING
        else:
            try:
                clean = recorded is not None and _sha256(path.read_bytes()) == recorded
            except OSError:
                clean = False
            state = FileState.CLEAN if clean else FileState.MODIFIED
        result.append(ManagedStatus(path=path, state=state))
    return result


def diff(path: PathLike) -> str:
    """Unified diff from the current file to its ``.installer-new``; empty
    when there is no pending version."""
    target = pathlib.Path(_key(path))
    pending = pending_path(target)
    if not pending.exists():
        return ""
    try:
        current = target.read_text(encoding="utf-8").splitlines(keepends=True)
    except FileNotFoundError:
        current = []
    new = pending.read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(current, new, fromfile=str(target), tofile=str(pending))
    )


def accept(path: PathLike, *, manifest: Manifest) -> None:
    """Replace `path` with its ``.installer-new`` and record it."""
    target = pathlib.Path(_key(path))
    pending = pending_path(target)
    data = pending.read_bytes()  # FileNotFoundError when nothing is pending
    _atomic_write(target, data, mode=None, like=target if target.exists() else pending)
    manifest.record(target, _sha256(data))
    pending.unlink(missing_ok=True)


def accept_all(manifest: Manifest) -> list[pathlib.Path]:
    accepted = []
    for entry in status(manifest):
        if entry.state is FileState.PENDING_NEW:
            accept(entry.path, manifest=manifest)
            accepted.append(entry.path)
    return accepted


# ---------------------------------------------------------------------------
# `mv3dt-installer configs status|diff [PATH]|accept PATH|accept-all`
# ---------------------------------------------------------------------------

_READ_ONLY_ACTIONS = ("status", "diff")
_ACTIONS = _READ_ONLY_ACTIONS + ("accept", "accept-all")


def _build_configs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mv3dt-installer configs", add_help=True)
    parser.add_argument("action", choices=_ACTIONS)
    parser.add_argument("path", nargs="?", default=None)
    parser.add_argument("--install-dir", default=None)
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-dir", default=None)
    return parser


def configs_requires_root(argv: list) -> bool:
    """`status` and `diff` only read; `accept`/`accept-all` rewrite the
    managed files and the root-owned manifest, as Step 4 does. Anything
    unrecognised asks for root, the safe default."""
    action = next((a for a in argv if a in _ACTIONS), None)
    return action not in _READ_ONLY_ACTIONS


def handle_configs_subcommand(
    argv: list,
    ctx: "Context",
    *,
    manifest: Optional[Manifest] = None,
    out=None,
) -> int:
    args = _build_configs_parser().parse_args(argv)
    manifest = manifest or Manifest()
    out = out or sys.stdout
    if args.action == "status":
        entries = status(manifest)
        if not entries:
            print("no installer-managed files recorded", file=out)
        for entry in entries:
            print(f"{entry.state.value:12} {entry.path}", file=out)
        return 0
    if args.action == "diff":
        if args.path:
            targets = [pathlib.Path(_key(args.path))]
        else:
            targets = [
                entry.path
                for entry in status(manifest)
                if entry.state is FileState.PENDING_NEW
            ]
        for target in targets:
            text = diff(target)
            if not text and args.path:
                ctx.log.error(f"configs: no {NEW_SUFFIX} version pending for {target}")
                return 1
            out.write(text)
        return 0
    if args.action == "accept":
        if not args.path:
            ctx.log.error("configs accept needs a PATH")
            return 2
        try:
            accept(args.path, manifest=manifest)
        except FileNotFoundError:
            ctx.log.error(
                f"configs: no {NEW_SUFFIX} version pending for {_key(args.path)}"
            )
            return 1
        ctx.log.info(f"accepted installer version of {_key(args.path)}")
        return 0
    accepted = accept_all(manifest)
    for path in accepted:
        ctx.log.info(f"accepted installer version of {path}")
    if not accepted:
        ctx.log.info("no pending installer versions to accept")
    return 0
