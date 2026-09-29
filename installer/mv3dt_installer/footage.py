"""Calibration footage capture for mv3dt-installer (STEP-4-CALIBRATION-FOOTAGE).

AMC's browser workflow needs one video per camera before it can calibrate.
This module records a synchronized clip from every camera it is handed into
a per-project directory the operator can reach from the AMC Video Upload
dialog, and deletes that directory once Step 4 no longer needs it.

It is a framework helper like `cameras.py` (§1): it owns directory layout,
recording, completeness, and deletion, and has no knowledge of AMC or of
step state. Step 4 decides when to call it.

Public API (§7):
    footage_root(home, override=None) -> Path
    project_dir(root, project_name) -> Path
    clip_name(camera) -> str
    is_complete(project_dir, cameras, *, project_name, seconds) -> bool
    has_room(root, camera_count) -> (ok, free_bytes, needed_bytes)
    record(cameras, *, user, password, project_dir, project_name, seconds,
        user_prefix=(), spawn=subprocess.Popen, clock=time.monotonic,
        sleep=time.sleep, on_progress=lambda fraction: None) -> RecordResult
    delete_if_owned(project_dir) -> bool

The recording command is ported from the developer-only
laptop/scripts/record_cameras_mp4.sh (not bundled). Every child goes through
an injected `spawn`, and time through injected `clock`/`sleep`, so no test
starts a real `ffmpeg` or waits real time.

REQUIRED (§4): the camera password never reaches a log line or an error
message. It appears only inside the argv handed to `spawn`; any `ffmpeg`
stderr surfaced in a log line is redacted first (doc 00 §14.4).
"""

from __future__ import annotations

import datetime
import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

from . import state
from .cameras import Camera
from .logs import log

__all__ = [
    "DEFAULT_FOOTAGE_SECONDS",
    "FOOTAGE_DIRNAME",
    "MARKER_NAME",
    "MIN_FREE_BYTES_PER_CAMERA",
    "RTSP_TIMEOUT_US",
    "PROCESS_GRACE_SECONDS",
    "PART_SUFFIX",
    "RecordResult",
    "footage_root",
    "project_dir",
    "clip_name",
    "is_complete",
    "has_room",
    "record",
    "delete_if_owned",
]

# ---------------------------------------------------------------------------
# §2 -- pinned values
# ---------------------------------------------------------------------------

DEFAULT_FOOTAGE_SECONDS = 300
FOOTAGE_DIRNAME = "mv3dt-calibration-footage"
MARKER_NAME = ".mv3dt-footage.json"
MIN_FREE_BYTES_PER_CAMERA = 1 << 30
# `-timeout 5000000` before `-i`: the RTSP socket timeout, matching doc 00
# §15.3's probe.
RTSP_TIMEOUT_US = 5_000_000
# Process timeout is the clip length plus this many seconds.
PROCESS_GRACE_SECONDS = 60
PART_SUFFIX = ".part"

_UNLABELED = "unlabeled"
_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]")
# Userinfo of an RTSP URL, for redacting anything ffmpeg echoes back.
_RTSP_USERINFO_RE = re.compile(r"(rtsp://[^:/@\s]*:)[^@\s]*@")
_REDACTED = "<redacted>"
# Seconds a terminated child gets to exit before it is killed (§4 step 6).
_TERMINATE_WAIT_S = 5
# How often the wait loop polls children and reports progress (§4 step 3).
_POLL_INTERVAL_S = 1.0


@dataclass(frozen=True)
class RecordResult:
    """Outcome of one `record()` call (§7)."""

    project_dir: Path
    recorded: list  # list[Path]: final clip paths
    failed: list  # list[Camera]: cameras whose clip did not complete
    cancelled: bool


# ---------------------------------------------------------------------------
# §2 -- directory layout and naming
# ---------------------------------------------------------------------------


def footage_root(home: Path, override: Optional[str] = None) -> Path:
    """§2 footage root: `<home>/Downloads/mv3dt-calibration-footage`, or the
    `CALIB_FOOTAGE_DIR` override when one is set."""
    if override:
        return Path(override)
    return Path(home) / "Downloads" / FOOTAGE_DIRNAME


def project_dir(root: Path, project_name: str) -> Path:
    """§2 project directory: `<footage root>/<PROJECT_NAME>/`."""
    return Path(root) / project_name


def clip_name(camera: Camera) -> str:
    """§2 clip file name: `<id>-<position>.mp4`, `unlabeled` for an empty
    position, every character outside `[A-Za-z0-9._-]` replaced by `-`."""
    position = camera.position or _UNLABELED
    return _UNSAFE_CHARS_RE.sub("-", f"{camera.id}-{position}") + ".mp4"


def _part_path(project: Path, camera: Camera) -> Path:
    return project / (clip_name(camera) + PART_SUFFIX)


# ---------------------------------------------------------------------------
# §3 condition 2 -- completeness
# ---------------------------------------------------------------------------


def _read_marker(project: Path) -> Optional[dict]:
    try:
        data = json.loads((project / MARKER_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_complete(
    project_dir: Path,
    cameras: Sequence[Camera],
    *,
    project_name: str,
    seconds: int,
) -> bool:
    """True only when the §2 marker exists, names this `project_name` and
    clip length, lists exactly this set of camera MACs, and every clip it
    lists is present (§3 condition 2)."""
    project = Path(project_dir)
    marker = _read_marker(project)
    if marker is None:
        return False
    if marker.get("project_name") != project_name or marker.get("seconds") != seconds:
        return False
    entries = marker.get("cameras")
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        return False
    if {e.get("mac") for e in entries} != {c.mac for c in cameras}:
        return False
    for entry in entries:
        name = entry.get("file")
        if not isinstance(name, str) or not name or not (project / name).is_file():
            return False
    return True


# ---------------------------------------------------------------------------
# §3 condition 4 -- free space
# ---------------------------------------------------------------------------


def has_room(root: Path, camera_count: int) -> tuple:
    """§2 minimum free space on the footage root's filesystem. Returns
    `(ok, free_bytes, needed_bytes)`. The root may not exist yet, so the
    nearest existing ancestor is measured."""
    needed = camera_count * MIN_FREE_BYTES_PER_CAMERA
    probe = Path(root)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    return free >= needed, free, needed


# ---------------------------------------------------------------------------
# §4 -- recording
# ---------------------------------------------------------------------------


def _redact(text: str, password: str) -> str:
    """doc 00 §14.4: strip the camera password from anything that may be
    logged, both verbatim and as RTSP URL userinfo."""
    if password:
        text = text.replace(password, _REDACTED)
    return _RTSP_USERINFO_RE.sub(r"\1" + _REDACTED + "@", text)


def _ffmpeg_argv(
    camera: Camera, *, user: str, password: str, seconds: int, part: Path
) -> list:
    """The exact §2 per-camera argv. The URL matches `cameras._rtsp_url`."""
    url = f"rtsp://{user}:{password}@{camera.ip}:554{camera.rtsp_path}"
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-y",
        "-rtsp_transport",
        "tcp",
        "-timeout",
        str(RTSP_TIMEOUT_US),
        "-i",
        url,
        "-t",
        str(seconds),
        "-map",
        "0:v:0",
        "-vf",
        "scale=1920:1080:out_range=tv",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        str(part),
    ]


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        log.warn(f"Could not remove {path}: {exc.strerror or exc}")


def _make_project_dir(project: Path, user_prefix: Sequence[str], spawn) -> None:
    """§4 step 1: create the project directory as the invoking user, so the
    `ffmpeg` children (which run through the same prefix) can write into it
    and the operator owns it. Without a prefix it is created directly."""
    if not user_prefix:
        project.mkdir(parents=True, exist_ok=True)
        return
    proc = spawn(
        [*user_prefix, "mkdir", "-p", str(project)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if proc.wait() != 0 or not project.is_dir():
        raise OSError(f"could not create footage directory {project}")


def _stderr_tail(stream, password: str) -> str:
    """Last non-empty line a child wrote to its stderr file, redacted.

    Each child's stderr goes to its own temporary file rather than a pipe:
    a pipe read only after exit can fill, stall the child, and turn a
    talkative failure into a false timeout."""
    try:
        stream.seek(0)
        text = stream.read() or ""
    except (OSError, ValueError):
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    lines = [line for line in text.strip().splitlines() if line.strip()]
    return _redact(lines[-1], password) if lines else ""


def _stop(proc) -> None:
    """Terminate a child, then kill it if it does not exit in time.

    SIGTERM comes first because in production the child is `sudo -u <user>
    -H ffmpeg ...`: sudo relays SIGTERM to ffmpeg, but a SIGKILL would kill
    only sudo and leave ffmpeg orphaned, still holding the RTSP session and
    its `.part` file."""
    try:
        proc.terminate()
        proc.wait(timeout=_TERMINATE_WAIT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    except OSError:
        pass


def record(
    cameras: Sequence[Camera],
    *,
    user: str,
    password: str,
    project_dir: Path,
    project_name: str,
    seconds: int,
    user_prefix: Sequence[str] = (),
    spawn=subprocess.Popen,
    clock=time.monotonic,
    sleep=time.sleep,
    on_progress: Callable[[float], None] = lambda fraction: None,
) -> RecordResult:
    """§4: record `seconds` of footage from every camera at once.

    Every child is spawned before any is waited on (§4 step 2), progress is
    reported about once per second as elapsed over `seconds` (step 3), each
    camera finishes atomically from its `.part` file (step 4), and the §2
    marker is written only when every camera succeeded (step 5). Ctrl-C
    terminates every child, deletes every `.part` file, writes no marker,
    and returns with `cancelled=True` (step 6).

    Any marker already in `project_dir` is removed before recording starts:
    this run replaces the whole set, so a stale marker must not vouch for a
    mix of old and new clips if this run only partly succeeds.
    """
    project = Path(project_dir)
    _make_project_dir(project, user_prefix, spawn)
    _unlink(project / MARKER_NAME)

    log.info(
        f"Recording {seconds} s of calibration footage from {len(cameras)} "
        f"camera(s) into {project}."
    )
    recorded_utc = _utc_now()
    process_timeout = seconds + PROCESS_GRACE_SECONDS

    running: dict = {}  # index -> proc
    outcome: dict = {}  # index -> True (clip complete) / False (failed)
    errfiles: dict = {}  # index -> temporary stderr file
    try:
        # §4 step 2: spawn every child before waiting on any.
        for i, camera in enumerate(cameras):
            part = _part_path(project, camera)
            argv = _ffmpeg_argv(
                camera, user=user, password=password, seconds=seconds, part=part
            )
            errfiles[i] = tempfile.TemporaryFile()
            try:
                running[i] = spawn(
                    [*user_prefix, *argv],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=errfiles[i],
                )
            except OSError as exc:
                detail = _redact(exc.strerror or str(exc), password)
                log.warn(f"Camera {camera.id} ({camera.ip}): could not start ffmpeg: {detail}")
                outcome[i] = False

        start = clock()
        while running:
            elapsed = clock() - start
            on_progress(min(1.0, elapsed / seconds) if seconds > 0 else 1.0)
            for i in list(running):
                proc = running[i]
                camera = cameras[i]
                part = _part_path(project, camera)
                rc = proc.poll()
                if rc is None:
                    if elapsed <= process_timeout:
                        continue
                    _stop(proc)
                    del running[i]
                    _unlink(part)
                    outcome[i] = False
                    log.warn(
                        f"Camera {camera.id} ({camera.ip}): ffmpeg exceeded "
                        f"{process_timeout} s and was stopped."
                    )
                    continue
                del running[i]
                if rc == 0 and part.is_file() and part.stat().st_size > 0:
                    part.replace(project / clip_name(camera))
                    outcome[i] = True
                    continue
                _unlink(part)
                outcome[i] = False
                reason = f"ffmpeg exited {rc}" if rc != 0 else "ffmpeg produced an empty file"
                tail = _stderr_tail(errfiles[i], password)
                log.warn(
                    f"Camera {camera.id} ({camera.ip}): {reason}"
                    + (f": {tail}" if tail else ".")
                )
            if running:
                sleep(_POLL_INTERVAL_S)
    except KeyboardInterrupt:
        for proc in running.values():
            _stop(proc)
        for camera in cameras:
            _unlink(_part_path(project, camera))
        log.warn("Calibration footage recording cancelled; no marker written.")
        return RecordResult(
            project_dir=project,
            recorded=[project / clip_name(c) for i, c in enumerate(cameras) if outcome.get(i)],
            failed=[c for i, c in enumerate(cameras) if not outcome.get(i)],
            cancelled=True,
        )
    finally:
        for stream in errfiles.values():
            stream.close()

    on_progress(1.0)
    recorded = [project / clip_name(c) for i, c in enumerate(cameras) if outcome.get(i)]
    failed = [c for i, c in enumerate(cameras) if not outcome.get(i)]
    if not failed:
        # §4 step 5: marker only on full success.
        state.write_json_atomic(
            project / MARKER_NAME,
            {
                "project_name": project_name,
                "seconds": seconds,
                "recorded_utc": recorded_utc,
                "cameras": [
                    {"id": c.id, "mac": c.mac, "file": clip_name(c)} for c in cameras
                ],
            },
        )
        log.info(f"Recorded {len(recorded)} calibration clip(s) in {project}.")
    return RecordResult(project_dir=project, recorded=recorded, failed=failed, cancelled=False)


# ---------------------------------------------------------------------------
# §6 -- deletion after a successful export
# ---------------------------------------------------------------------------


def delete_if_owned(project_dir: Path) -> bool:
    """§6: delete `project_dir` only if it carries the §2 ownership marker,
    then remove the footage root too if that left it empty. Returns whether
    anything was deleted. A failure is a `warn` line, never an exception:
    the calibration it served is already installed."""
    project = Path(project_dir)
    if not (project / MARKER_NAME).is_file():
        return False
    try:
        shutil.rmtree(project)
    except OSError as exc:
        log.warn(f"Could not delete calibration footage {project}: {exc.strerror or exc}")
        return False
    log.info(f"Deleted calibration footage {project}.")
    root = project.parent
    try:
        root.rmdir()  # only succeeds when empty
    except OSError:
        pass
    else:
        log.info(f"Deleted empty footage root {root}.")
    return True
