"""`mv3dt-installer record-footage --pending` (AMC-FOOTAGE-RECORDING section 5).

The AMC control service runs as the invoking user and cannot read the
root-only camera credentials, so an automated run records calibration
footage through this one fixed sudo command (section 3). Everything variable
comes from the request file the service writes (section 4), which this
command treats as untrusted input.

Registered with `requires_root=False`, like `watchdog`, so the framework's
root startup (transcript log, `installer.conf` and `state.json` writes)
never runs: the handler checks for root itself, reads `installer.conf`
read-only through the subcommand bootstrap, and logs only to stderr.

1. **Who**: the service user is `SUDO_USER`; it must exist, not be root, and
   own `<install_dir>/amc_projects` (always that path).
2. **Read the request** with `O_RDONLY | O_NOFOLLOW | O_NONBLOCK`, checked
   with `fstat` on the open descriptor: a regular file owned by the user, at
   most 4096 bytes, valid against section 4.
3. **Check the folder**: `amc_projects/<folder>` and its `footage/` are
   directories, not symlinks, owned by the user; `project.json` is a regular
   file.
4. **Load inputs as root**: camera credentials and the `CAMERAS_FILE`
   inventory, then the 1 GiB per camera free-space check.
5. **Drop privileges, irrevocably** to the user, and confirm `setuid(0)`
   fails.
6. **Set old footage aside**: `*.mp4` and the footage marker move into
   `footage/previous/`, which is emptied first.
7. **Record** with `footage.record` and no `user_prefix`.
8. **Restore on failure**: unless every camera succeeded, this run's clips
   are deleted and the set-aside files move back.

Cancel: after the drop a thread reads stdin; a `cancel` line or end of file
interrupts the main thread (`_thread.interrupt_main()`), and `SIGTERM` and
`SIGINT` do the same. The interrupt is raised once only and is caught across
steps 6 to 8, so a cancel during set-aside or restore still finishes the
restore before the exit `3` line.

Output contract (REQUIRED): stdout carries only JSON objects, one per line,
each flushed: `{"progress": <0..1>}` about once per second while recording,
then exactly one final line with every field of
`{"ok", "cancelled", "reason", "files", "failed", "seconds"}`. Exit codes:
`0` recorded, `1` a camera failed or no room, `2` invalid request or setup,
`3` cancelled.

Every os, pwd, stdin and footage call goes through a module-level seam so
the tests run unprivileged.
"""

from __future__ import annotations

import _thread
import errno
import json
import os
import pathlib
import pwd
import re
import shutil
import signal
import stat as stat_mod
import sys
import threading
import time
from typing import Any, Callable, Iterable, Optional, Sequence

from . import camera_credentials
from . import cameras as cameras_mod
from . import config as config_mod
from . import footage as footage_mod
from .logs import log

__all__ = [
    "SUBCOMMAND",
    "PENDING_FLAG",
    "REQUEST_FILE_NAME",
    "MAX_REQUEST_BYTES",
    "MAX_REQUEST_AGE_S",
    "MAX_CLOCK_SKEW_S",
    "MIN_SECONDS",
    "MAX_SECONDS",
    "PREVIOUS_DIRNAME",
    "CANCEL_LINE",
    "EXIT_OK",
    "EXIT_FAILED",
    "EXIT_INVALID",
    "EXIT_CANCELLED",
    "request_path",
    "handle_record_footage_subcommand",
]

SUBCOMMAND = "record-footage"
PENDING_FLAG = "--pending"
CANCEL_LINE = "cancel"

# Section 4: where the service writes the request, and its rules.
CONTROL_DIRNAME = "amc_control"
STATE_DIRNAME = "state"
PROJECTS_DIRNAME = "amc_projects"
REQUEST_FILE_NAME = "record-request.json"
SCHEMA_VERSION = 1
MAX_REQUEST_BYTES = 4096
MAX_REQUEST_AGE_S = 300
MAX_CLOCK_SKEW_S = 60
MIN_SECONDS = 30
MAX_SECONDS = 1800
_REQUEST_KEYS = frozenset({"schema_version", "folder", "seconds", "requested_at"})
_OPEN_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
# AMC-WEB-PROJECTS section 3.
FOLDER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,46}[a-z0-9])?$")

FOOTAGE_DIRNAME = "footage"
PROJECT_FILE_NAME = "project.json"
PREVIOUS_DIRNAME = "previous"
PREVIOUS_DIR_MODE = 0o750

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_INVALID = 2
EXIT_CANCELLED = 3

_CANCELLED = "cancelled"
_REDACTED = "<redacted>"
_RTSP_USERINFO_RE = re.compile(r"(rtsp://[^:/@\s]*:)[^@\s]*@")

# Injectable seams, replaced by tests.
_GETENV: Callable[[str], Optional[str]] = os.environ.get
_ENVIRON: Any = os.environ
_GETEUID: Callable[[], int] = os.geteuid
_GETPWNAM: Callable[[str], Any] = pwd.getpwnam
_OPEN: Callable[..., int] = os.open
_FSTAT: Callable[[int], os.stat_result] = os.fstat
_LSTAT: Callable[[str], os.stat_result] = os.lstat
_INITGROUPS: Callable[[str, int], None] = getattr(os, "initgroups", lambda name, gid: None)
_SETGID: Callable[[int], None] = os.setgid
_SETUID: Callable[[int], None] = os.setuid
_TIME: Callable[[], float] = time.time
_SIGNAL: Callable[..., Any] = signal.signal
_INTERRUPT_MAIN: Callable[[], None] = _thread.interrupt_main
def _stdin_lines() -> Iterable[str]:
    """Lines from file descriptor 0, read unbuffered.

    Not `sys.stdin`: the reading thread is a daemon still blocked here when
    a normal recording ends (the service keeps the pipe open until the
    command exits), and a daemon thread holding `sys.stdin`'s buffer lock
    can abort interpreter shutdown. A raw `os.read` holds no Python lock.
    """
    pending = b""
    while True:
        chunk = os.read(0, 4096)
        if not chunk:
            return
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            yield line.decode("utf-8", "replace")


_READ_STDIN: Callable[[], Iterable[str]] = _stdin_lines
_REPLACE: Callable[[Any, Any], None] = os.replace
_LOAD_CREDENTIALS: Callable[..., Any] = camera_credentials.load_credentials
_HAS_ROOM: Callable[..., tuple] = footage_mod.has_room
_RECORD: Callable[..., Any] = footage_mod.record


class _Refused(Exception):
    """A check failed: the command ends with `code` and this reason."""

    def __init__(self, reason: str, code: int = EXIT_INVALID) -> None:
        super().__init__(reason)
        self.code = code


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------


def _redact(text: str, password: str) -> str:
    """doc 00 section 14.4: no camera password in anything printed."""
    if password:
        text = text.replace(password, _REDACTED)
    return _RTSP_USERINFO_RE.sub(r"\1" + _REDACTED + "@", text)


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _finish(
    code: int,
    *,
    reason: str = "",
    seconds: int = 0,
    files: Sequence[str] = (),
    failed: Sequence[dict] = (),
) -> int:
    """Print the one final line, every field present, and return `code`."""
    if reason:
        log.error(f"{SUBCOMMAND}: {reason}")
    _emit(
        {
            "ok": code == EXIT_OK,
            "cancelled": code == EXIT_CANCELLED,
            "reason": reason,
            "files": list(files),
            "failed": list(failed),
            "seconds": seconds,
        }
    )
    return code


# ---------------------------------------------------------------------------
# Steps 1 to 4: checks as root, before anything is written
# ---------------------------------------------------------------------------


def request_path(install_dir: Any) -> pathlib.Path:
    return pathlib.Path(install_dir) / CONTROL_DIRNAME / STATE_DIRNAME / REQUEST_FILE_NAME


def _service_user() -> Any:
    """Step 1: root, run through sudo by a user that exists and is not root."""
    if _GETEUID() != 0:
        raise _Refused("must be run as root through sudo")
    name = _GETENV("SUDO_USER") or ""
    if not name:
        raise _Refused("not run through sudo (SUDO_USER is unset)")
    try:
        entry = _GETPWNAM(name)
    except KeyError:
        raise _Refused(f"SUDO_USER={name!r} is not a user on this system") from None
    if name == "root" or entry.pw_uid == 0:
        raise _Refused("SUDO_USER is root; the service user must not be root")
    return entry


def _owned_directory(path: pathlib.Path, uid: int) -> None:
    """`path` is a real directory (not a symlink) owned by `uid`."""
    try:
        st = _LSTAT(str(path))
    except OSError:
        raise _Refused(f"{path} is missing") from None
    if stat_mod.S_ISLNK(st.st_mode):
        raise _Refused(f"{path} is a symlink")
    if not stat_mod.S_ISDIR(st.st_mode):
        raise _Refused(f"{path} is not a directory")
    if st.st_uid != uid:
        raise _Refused(f"{path} is owned by uid {st.st_uid}, not the service user")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_request(path: pathlib.Path, uid: int) -> dict:
    """Step 2: open without following a link, check the open descriptor."""
    try:
        fd = _OPEN(str(path), _OPEN_FLAGS)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise _Refused(f"{path} is a symlink") from None
        if exc.errno == errno.ENOENT:
            raise _Refused(f"no pending request: {path} is missing") from None
        raise _Refused(f"cannot open {path}: {exc.strerror or exc}") from None
    try:
        st = _FSTAT(fd)
        if not stat_mod.S_ISREG(st.st_mode):
            raise _Refused(f"{path} is not a regular file")
        if st.st_uid != uid:
            raise _Refused(f"{path} is owned by uid {st.st_uid}, not the service user")
        data = b""
        while len(data) <= MAX_REQUEST_BYTES:
            chunk = os.read(fd, MAX_REQUEST_BYTES + 1 - len(data))
            if not chunk:
                break
            data += chunk
    except OSError as exc:
        raise _Refused(f"cannot read {path}: {exc.strerror or exc}") from None
    finally:
        os.close(fd)
    if len(data) > MAX_REQUEST_BYTES:
        raise _Refused(f"{path} is larger than {MAX_REQUEST_BYTES} bytes")
    try:
        request = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise _Refused(f"{path} is not valid JSON") from None
    return _validate_request(request)


def _validate_request(request: Any) -> dict:
    """The section 4 table."""
    if not isinstance(request, dict):
        raise _Refused("the request is not a JSON object")
    if set(request) != _REQUEST_KEYS:
        raise _Refused(
            "the request must have exactly the keys " + ", ".join(sorted(_REQUEST_KEYS))
        )
    if not _is_int(request["schema_version"]) or request["schema_version"] != SCHEMA_VERSION:
        raise _Refused(f"schema_version must be {SCHEMA_VERSION}")
    folder = request["folder"]
    if not isinstance(folder, str) or not FOLDER_RE.match(folder):
        raise _Refused("folder is not a valid project folder name")
    seconds = request["seconds"]
    if not _is_int(seconds) or not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise _Refused(f"seconds must be an integer from {MIN_SECONDS} to {MAX_SECONDS}")
    requested_at = request["requested_at"]
    if not _is_int(requested_at):
        raise _Refused("requested_at must be integer Unix seconds")
    now = _TIME()
    if now - requested_at > MAX_REQUEST_AGE_S:
        raise _Refused(f"the request is older than {MAX_REQUEST_AGE_S} seconds")
    if requested_at - now > MAX_CLOCK_SKEW_S:
        raise _Refused("the request is dated in the future")
    return request


def _check_folder(projects: pathlib.Path, folder: str, uid: int) -> pathlib.Path:
    """Step 3. Returns the footage directory."""
    project = projects / folder
    _owned_directory(project, uid)
    footage = project / FOOTAGE_DIRNAME
    _owned_directory(footage, uid)
    project_file = project / PROJECT_FILE_NAME
    try:
        st = _LSTAT(str(project_file))
    except OSError:
        raise _Refused(f"{project_file} is missing") from None
    if not stat_mod.S_ISREG(st.st_mode):
        raise _Refused(f"{project_file} is not a regular file")
    return footage


def _load_cameras(ctx: Any) -> list:
    """Enabled cameras of `installer.conf`'s `CAMERAS_FILE`, read as root."""
    value = (ctx.conf.get(config_mod.CAMERAS_FILE_KEY) or "").strip()
    if not value:
        raise _Refused("no camera inventory: CAMERAS_FILE is not set in installer.conf")
    try:
        text = pathlib.Path(value).read_text(encoding="utf-8")
    except OSError as exc:
        raise _Refused(f"cannot read the camera inventory {value}: {exc.strerror or exc}") from None
    cameras = [c for c in cameras_mod.parse_inventory(text) if c.enabled]
    if not cameras:
        raise _Refused(f"the camera inventory {value} lists no enabled camera")
    return cameras


# ---------------------------------------------------------------------------
# Step 5: privilege drop
# ---------------------------------------------------------------------------


def _drop_privileges(entry: Any) -> None:
    """Irrevocably become the service user."""
    try:
        _INITGROUPS(entry.pw_name, entry.pw_gid)
        _SETGID(entry.pw_gid)
        _SETUID(entry.pw_uid)
    except OSError as exc:
        raise _Refused(f"could not drop root privileges: {exc.strerror or exc}") from None
    try:
        _SETUID(0)
    except OSError:
        pass
    else:
        raise _Refused("root privileges were not dropped irrevocably")
    _ENVIRON["HOME"] = entry.pw_dir
    _ENVIRON["USER"] = entry.pw_name
    _ENVIRON["LOGNAME"] = entry.pw_name


# ---------------------------------------------------------------------------
# Cancel channel
# ---------------------------------------------------------------------------


class _Cancel:
    """Turns SIGTERM, SIGINT and the stdin channel into one
    `KeyboardInterrupt`, the path `footage.record` takes for Ctrl-C.

    It is raised once only, and never after `close()`, so a second cancel
    cannot interrupt the restore the first one started, nor the final line.
    `_thread.interrupt_main()` runs the SIGINT handler in the main thread,
    which is this object, so the stdin thread raises through it too.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.raised = False
        self.closed = False

    def __call__(self, signum: int, frame: Any) -> None:
        with self._lock:
            if self.raised or self.closed:
                return
            self.raised = True
        raise KeyboardInterrupt

    def close(self) -> None:
        with self._lock:
            self.closed = True

    def request(self) -> None:
        """From the stdin thread."""
        with self._lock:
            if self.raised or self.closed:
                return
        _INTERRUPT_MAIN()

    def watch_stdin(self) -> None:
        """Start the thread: a `cancel` line or end of file cancels."""

        def read() -> None:
            try:
                # Stops at a `cancel` line or at end of file; both cancel.
                for line in _READ_STDIN():
                    if self.closed:
                        return
                    if line.strip() == CANCEL_LINE:
                        break
            except (OSError, ValueError):
                return  # stdin unreadable: only the signals can cancel
            self.request()

        threading.Thread(target=read, name="record-footage-stdin", daemon=True).start()


# ---------------------------------------------------------------------------
# Steps 6 to 8: set aside, record, restore (as the user)
# ---------------------------------------------------------------------------


class _Footage:
    """The footage directory's set-aside and restore bookkeeping.

    Only files this command moved go back, and this run's clips are deleted
    only once set-aside finished (so every `*.mp4` left in `footage/` is new).
    Both operations are idempotent, so a restore an interrupt cut short is
    simply run again.
    """

    def __init__(self, footage: pathlib.Path) -> None:
        self.footage = footage
        self.previous = footage / PREVIOUS_DIRNAME
        self.moved: list = []
        self.created_previous = False
        self.recording_started = False

    def set_aside(self) -> None:
        """Step 6."""
        previous = self.previous
        if os.path.lexists(previous):
            st = os.lstat(previous)
            if stat_mod.S_ISLNK(st.st_mode) or not stat_mod.S_ISDIR(st.st_mode):
                raise _Refused(f"{previous} is not a directory")
            for child in previous.iterdir():
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        else:
            # Noted before acting, here and below, so an interrupt between
            # the two cannot leave a change the restore does not know about.
            self.created_previous = True
            previous.mkdir(mode=PREVIOUS_DIR_MODE)
        for child in sorted(self.footage.iterdir()):
            if child.name == footage_mod.MARKER_NAME or child.name.endswith(".mp4"):
                self.moved.append(child.name)
                _REPLACE(child, previous / child.name)

    def restore(self) -> None:
        """Step 8: leave `footage/` as it was before step 6."""
        if self.recording_started:
            for child in self.footage.iterdir():
                name = child.name
                if (
                    name == footage_mod.MARKER_NAME
                    or name.endswith(".mp4")
                    or name.endswith(".mp4" + footage_mod.PART_SUFFIX)
                ):
                    child.unlink()
        for name in list(self.moved):
            source = self.previous / name
            if os.path.lexists(source):
                _REPLACE(source, self.footage / name)
            self.moved.remove(name)
        if self.created_previous:
            try:
                self.previous.rmdir()
            except FileNotFoundError:
                pass
            self.created_previous = False


def _restore_through_cancel(state: _Footage) -> tuple:
    """Run the restore to the end even if the one cancel lands inside it.
    Returns `(cancelled, error)`."""
    cancelled = False
    while True:
        try:
            state.restore()
            return cancelled, None
        except KeyboardInterrupt:
            cancelled = True
        except OSError as exc:
            return cancelled, f"could not restore the earlier footage: {exc.strerror or exc}"


def _record_steps(
    cameras: list,
    *,
    creds: Any,
    footage: pathlib.Path,
    folder: str,
    seconds: int,
    cancel: _Cancel,
) -> int:
    """Steps 6 to 8, with the cancel caught anywhere in them."""
    state = _Footage(footage)
    result = None
    cancelled = False
    refused: Optional[_Refused] = None
    try:
        try:
            state.set_aside()
        except OSError as exc:
            raise _Refused(
                f"could not move old footage into {PREVIOUS_DIRNAME}/: {exc.strerror or exc}"
            ) from None
        state.recording_started = True
        log.info(
            f"{SUBCOMMAND}: recording {seconds} s from {len(cameras)} camera(s) into {footage}"
        )
        result = _RECORD(
            cameras,
            user=creds.username,
            password=creds.password,
            project_dir=footage,
            project_name=folder,
            seconds=seconds,
            on_progress=lambda fraction: _emit({"progress": round(float(fraction), 3)}),
        )
        cancelled = result.cancelled
        if not cancelled and not result.failed:
            # Recorded: from here a cancel no longer undoes anything. Closed
            # inside the try so a cancel landing just before still restores.
            cancel.close()
    except KeyboardInterrupt:
        cancelled = True
    except _Refused as exc:
        refused = exc
    except OSError as exc:
        refused = _Refused(f"could not record: {exc.strerror or exc}", EXIT_FAILED)
    except Exception as exc:  # noqa: BLE001 - still restore, still one final line
        refused = _Refused(f"unexpected error: {type(exc).__name__}", EXIT_FAILED)

    success = not cancelled and refused is None and result is not None and not result.failed
    restore_error = None
    if not success:
        restored_cancel, restore_error = _restore_through_cancel(state)
        cancelled = cancelled or restored_cancel
    cancel.close()
    if restore_error:
        log.error(f"{SUBCOMMAND}: {restore_error}")

    if cancelled:
        return _finish(EXIT_CANCELLED, reason=_CANCELLED, seconds=seconds)
    if refused is not None:
        reason = _redact(str(refused), creds.password)
        if restore_error:
            reason = f"{reason}; {restore_error}"
        return _finish(refused.code, reason=reason, seconds=seconds)
    files = [pathlib.Path(p).name for p in result.recorded]
    if success:
        return _finish(EXIT_OK, seconds=seconds, files=files)
    errors = list(getattr(result, "errors", []) or [])
    failed = [
        {
            "id": camera.id,
            "ip": camera.ip,
            "error": _redact(errors[i] if i < len(errors) else "failed", creds.password),
        }
        for i, camera in enumerate(result.failed)
    ]
    reason = f"{len(failed)} of {len(cameras)} camera(s) failed; the earlier footage was kept"
    if restore_error:
        reason = f"{len(failed)} of {len(cameras)} camera(s) failed; {restore_error}"
    return _finish(EXIT_FAILED, reason=reason, seconds=seconds, failed=failed)


def _run(ctx: Any, cancel: _Cancel, run: dict) -> int:
    """Steps 1 to 5, then 6 to 8."""
    entry = _service_user()
    uid = entry.pw_uid
    install_dir = pathlib.Path(ctx.install_dir)
    projects = install_dir / PROJECTS_DIRNAME
    _owned_directory(projects, uid)

    request = _read_request(request_path(install_dir), uid)
    seconds = run["seconds"] = request["seconds"]
    folder = request["folder"]

    footage = _check_folder(projects, folder, uid)

    creds = _LOAD_CREDENTIALS(install_dir)
    if creds is None:
        raise _Refused("no camera credentials (secrets/camera.env)")
    run["password"] = creds.password
    cameras = _load_cameras(ctx)
    ok, free, needed = _HAS_ROOM(footage, len(cameras))
    if not ok:
        raise _Refused(
            f"not enough free space in {footage}: {free} bytes free, {needed} bytes "
            f"needed for {len(cameras)} camera(s)",
            EXIT_FAILED,
        )

    _drop_privileges(entry)
    cancel.watch_stdin()
    return _record_steps(
        cameras, creds=creds, footage=footage, folder=folder, seconds=seconds, cancel=cancel
    )


def handle_record_footage_subcommand(argv: list, ctx: Any) -> int:
    """`mv3dt-installer record-footage --pending`."""
    if list(argv) != [PENDING_FLAG]:
        return _finish(EXIT_INVALID, reason=f"usage: mv3dt-installer {SUBCOMMAND} {PENDING_FLAG}")
    cancel = _Cancel()
    previous = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = _SIGNAL(signum, cancel)
    run: dict = {"seconds": 0, "password": ""}
    try:
        return _run(ctx, cancel, run)
    except _Refused as exc:
        cancel.close()
        return _finish(exc.code, reason=_redact(str(exc), run["password"]), seconds=run["seconds"])
    except KeyboardInterrupt:
        # Before step 6: nothing was written yet.
        cancel.close()
        return _finish(EXIT_CANCELLED, reason=_CANCELLED, seconds=run["seconds"])
    except Exception as exc:  # noqa: BLE001 - the final line is REQUIRED
        cancel.close()
        return _finish(
            EXIT_FAILED,
            reason=f"unexpected error: {type(exc).__name__}",
            seconds=run["seconds"],
        )
    finally:
        cancel.close()
        for signum, handler in previous.items():
            if handler is not None:
                _SIGNAL(signum, handler)
