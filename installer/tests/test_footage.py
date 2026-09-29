"""Tests for mv3dt_installer.footage (STEP-4-CALIBRATION-FOOTAGE §1, §2, §4,
§6, §7).

Run from installer/: `python3 -m pytest tests/test_footage.py -v`

No test starts a real ffmpeg or waits real time: `spawn`, `clock`, and
`sleep` are fakes, and a fake child "writes" its `.part` file when it exits.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
from collections import namedtuple

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from mv3dt_installer import footage, logs  # noqa: E402
from mv3dt_installer.cameras import Camera  # noqa: E402

PASSWORD = "s3cr3t-Pa55"
USER = "admin"

CAM1 = Camera(id="c1", mac="d0:3b:f4:02:44:e2", ip="169.254.1.10", position="top-left")
CAM2 = Camera(id="c2", mac="d0:3b:f4:02:44:e3", ip="169.254.1.11", position="top-right")


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeProc:
    """A child that exits at `exit_at` (clock time) with `rc`, writing
    `size` bytes to its output path first. `exit_at=None` never exits."""

    def __init__(self, harness, argv, *, exit_at, rc, size, stderr_text, stderr=None):
        self.h = harness
        self.argv = argv
        self.stderr_target = stderr
        self.stderr_text = stderr_text
        self.signals = []
        self.exit_at = exit_at
        self.rc = rc
        self.size = size
        self.returncode = None
        self.terminated = False
        self.killed = False

    def _finish(self, rc):
        if self.returncode is None:
            self.returncode = rc
            if self.stderr_target is not None and self.stderr_text:
                self.stderr_target.write(self.stderr_text.encode())

    def poll(self):
        self.h.events.append(("poll", self))
        if self.returncode is None and self.exit_at is not None and self.h.clock.now >= self.exit_at:
            out = pathlib.Path(self.argv[-1])
            if self.size is not None:
                out.write_bytes(b"x" * self.size)
            self._finish(self.rc)
        return self.returncode

    def wait(self, timeout=None):
        self.h.events.append(("wait", self))
        if self.returncode is None:
            if self.terminated and self.h.ignore_terminate and timeout is not None:
                raise subprocess.TimeoutExpired(self.argv, timeout)
            self._finish(-15 if self.terminated else -9)
        return self.returncode

    def terminate(self):
        self.signals.append("term")
        self.terminated = True

    def kill(self):
        self.signals.append("kill")
        self.killed = True
        self._finish(-9)


Behaviour = namedtuple("Behaviour", "exit_at rc size stderr", defaults=(None, 0, 1024, ""))


class Harness:
    """Owns the fake clock and the per-camera behaviour of spawned
    children. `behaviour` maps camera IP to a `Behaviour`."""

    def __init__(self, behaviour=None, *, ignore_terminate=False, interrupt_at=None):
        self.clock = FakeClock()
        self.behaviour = behaviour or {}
        self.events = []
        self.procs = []
        self.mkdirs = []
        self.ignore_terminate = ignore_terminate
        self.interrupt_at = interrupt_at
        self.progress = []

    def spawn(self, argv, **kwargs):
        if "mkdir" in argv:
            self.mkdirs.append(list(argv))
            pathlib.Path(argv[-1]).mkdir(parents=True, exist_ok=True)
            proc = FakeProc(self, argv, exit_at=0, rc=0, size=None, stderr_text="")
            proc.returncode = 0
            return proc
        url = argv[argv.index("-i") + 1]
        ip = re.search(r"@([\d.]+):554", url).group(1)
        b = self.behaviour.get(ip, Behaviour(exit_at=self.clock.now + 300))
        proc = FakeProc(
            self, argv, exit_at=b.exit_at, rc=b.rc, size=b.size,
            stderr_text=b.stderr, stderr=kwargs.get("stderr"),
        )
        self.spawn_kwargs = getattr(self, "spawn_kwargs", []) + [kwargs]
        self.events.append(("spawn", proc))
        self.procs.append(proc)
        return proc

    def sleep(self, seconds):
        if self.interrupt_at is not None and self.clock.now >= self.interrupt_at:
            raise KeyboardInterrupt
        self.clock.sleep(seconds)

    def record(self, cameras, project, **kw):
        kw.setdefault("seconds", 300)
        return footage.record(
            cameras,
            user=USER,
            password=PASSWORD,
            project_dir=project,
            project_name="Valencia-West",
            spawn=self.spawn,
            clock=self.clock,
            sleep=self.sleep,
            on_progress=self.progress.append,
            **kw,
        )


@pytest.fixture
def captured_logs(monkeypatch):
    lines = []
    for level in ("info", "warn", "error"):
        monkeypatch.setattr(
            logs.log, level, staticmethod(lambda msg, _l=level: lines.append((_l, msg)))
        )
    return lines


def _exits(at, rc=0, size=1024, stderr=""):
    return Behaviour(exit_at=at, rc=rc, size=size, stderr=stderr)


# ---------------------------------------------------------------------------
# §2 naming and layout
# ---------------------------------------------------------------------------


def test_pinned_constants():
    assert footage.DEFAULT_FOOTAGE_SECONDS == 300
    assert footage.FOOTAGE_DIRNAME == "mv3dt-calibration-footage"
    assert footage.MARKER_NAME == ".mv3dt-footage.json"
    assert footage.MIN_FREE_BYTES_PER_CAMERA == 1 << 30


def test_clip_name_uses_id_and_position():
    assert footage.clip_name(CAM1) == "c1-top-left.mp4"


def test_clip_name_empty_position_is_unlabeled():
    cam = Camera(id="c3", mac="m", ip="1.2.3.4", position="")
    assert footage.clip_name(cam) == "c3-unlabeled.mp4"


def test_clip_name_sanitizes_unsafe_characters():
    cam = Camera(id="cam 1/a", mac="m", ip="1.2.3.4", position="top:left*x")
    assert footage.clip_name(cam) == "cam-1-a-top-left-x.mp4"


def test_footage_root_default_and_override(tmp_path):
    assert footage.footage_root(tmp_path) == tmp_path / "Downloads" / "mv3dt-calibration-footage"
    assert footage.footage_root(tmp_path, "/data/footage") == pathlib.Path("/data/footage")
    assert footage.footage_root(tmp_path, "") == tmp_path / "Downloads" / "mv3dt-calibration-footage"


def test_project_dir(tmp_path):
    assert footage.project_dir(tmp_path, "Valencia-West") == tmp_path / "Valencia-West"


# ---------------------------------------------------------------------------
# §4 recording
# ---------------------------------------------------------------------------


def test_exact_ffmpeg_argv(tmp_path):
    h = Harness({CAM1.ip: _exits(1300)})
    project = tmp_path / "proj"
    h.record([CAM1], project)
    assert h.procs[0].argv == [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostats", "-y",
        "-rtsp_transport", "tcp", "-timeout", "5000000",
        "-i", f"rtsp://{USER}:{PASSWORD}@169.254.1.10:554/Streaming/Channels/101",
        "-t", "300", "-map", "0:v:0",
        "-vf", "scale=1920:1080:out_range=tv",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-an",
        "-movflags", "+faststart",
        "-f", "mp4", str(project / "c1-top-left.mp4.part"),
    ]


def test_user_prefix_is_prepended_to_every_ffmpeg_argv(tmp_path):
    h = Harness({CAM1.ip: _exits(1300), CAM2.ip: _exits(1300)})
    prefix = ("sudo", "-u", "p2bp-admin", "-H")
    h.record([CAM1, CAM2], tmp_path / "proj", user_prefix=prefix)
    assert len(h.procs) == 2
    for proc in h.procs:
        assert proc.argv[:5] == ["sudo", "-u", "p2bp-admin", "-H", "ffmpeg"]
    # The project directory is created through the same prefix (§4 step 1).
    assert h.mkdirs == [["sudo", "-u", "p2bp-admin", "-H", "mkdir", "-p", str(tmp_path / "proj")]]


def test_record_creates_project_dir(tmp_path):
    h = Harness({CAM1.ip: _exits(1300)})
    project = tmp_path / "a" / "b" / "proj"
    h.record([CAM1], project)
    assert project.is_dir()


def test_all_children_spawned_before_any_is_waited_on(tmp_path):
    cams = [CAM1, CAM2, Camera(id="c3", mac="d0:3b:f4:00:00:03", ip="169.254.1.12", position="x")]
    h = Harness({c.ip: _exits(1300) for c in cams})
    h.record(cams, tmp_path / "proj")
    kinds = [kind for kind, _ in h.events]
    first_non_spawn = next(i for i, k in enumerate(kinds) if k != "spawn")
    assert kinds[:first_non_spawn] == ["spawn"] * 3
    assert "spawn" not in kinds[first_non_spawn:]


def test_success_renames_part_and_writes_marker(tmp_path):
    h = Harness({CAM1.ip: _exits(1300), CAM2.ip: _exits(1301)})
    project = tmp_path / "proj"
    result = h.record([CAM1, CAM2], project)

    assert result == footage.RecordResult(
        project_dir=project,
        recorded=[project / "c1-top-left.mp4", project / "c2-top-right.mp4"],
        failed=[],
        cancelled=False,
    )
    assert (project / "c1-top-left.mp4").stat().st_size == 1024
    assert not list(project.glob("*.part"))
    marker = json.loads((project / ".mv3dt-footage.json").read_text())
    assert marker["project_name"] == "Valencia-West"
    assert marker["seconds"] == 300
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", marker["recorded_utc"])
    assert marker["cameras"] == [
        {"id": "c1", "mac": "d0:3b:f4:02:44:e2", "file": "c1-top-left.mp4"},
        {"id": "c2", "mac": "d0:3b:f4:02:44:e3", "file": "c2-top-right.mp4"},
    ]
    assert footage.is_complete(project, [CAM1, CAM2], project_name="Valencia-West", seconds=300)


def test_progress_reported_about_once_per_second(tmp_path):
    h = Harness({CAM1.ip: _exits(1010)})
    h.record([CAM1], tmp_path / "proj", seconds=10)
    assert all(s == 1.0 for s in h.clock.sleeps)
    assert h.progress[0] == 0.0
    assert h.progress[-1] == 1.0
    assert h.progress == sorted(h.progress)
    assert pytest.approx(h.progress[5]) == 0.5
    assert len(h.progress) >= 10


def test_partial_failure_keeps_good_clip_and_writes_no_marker(tmp_path, captured_logs):
    h = Harness({
        CAM1.ip: _exits(1300),
        CAM2.ip: _exits(1002, rc=1, stderr=f"rtsp://{USER}:{PASSWORD}@169.254.1.11:554/x: 401 Unauthorized\n"),
    })
    project = tmp_path / "proj"
    result = h.record([CAM1, CAM2], project)

    assert result.failed == [CAM2]
    assert result.recorded == [project / "c1-top-left.mp4"]
    assert not result.cancelled
    assert (project / "c1-top-left.mp4").is_file()
    assert not (project / "c2-top-right.mp4.part").exists()
    assert not (project / "c2-top-right.mp4").exists()
    assert not (project / ".mv3dt-footage.json").exists()
    warn = [m for lvl, m in captured_logs if lvl == "warn"]
    assert any("c2" in m and "401 Unauthorized" in m for m in warn)
    assert all(PASSWORD not in m for _, m in captured_logs)


def test_empty_part_file_is_a_failure(tmp_path):
    h = Harness({CAM1.ip: _exits(1300, rc=0, size=0)})
    project = tmp_path / "proj"
    result = h.record([CAM1], project)
    assert result.failed == [CAM1]
    assert not (project / "c1-top-left.mp4.part").exists()
    assert not (project / "c1-top-left.mp4").exists()
    assert not (project / ".mv3dt-footage.json").exists()


def test_child_exceeding_process_timeout_is_killed(tmp_path):
    h = Harness({CAM1.ip: _exits(None), CAM2.ip: _exits(1300)})
    project = tmp_path / "proj"
    (project).mkdir()
    (project / "c1-top-left.mp4.part").write_bytes(b"partial")
    result = h.record([CAM1, CAM2], project, seconds=300)

    hung = h.procs[0]
    # SIGTERM first: in production the child is sudo, which relays SIGTERM
    # to ffmpeg but cannot relay SIGKILL.
    assert hung.signals and hung.signals[0] == "term"
    assert not h.procs[1].signals
    assert h.clock.now - 1000.0 > 360  # killed only after seconds + 60
    assert h.clock.now - 1000.0 <= 362
    assert result.failed == [CAM1]
    assert result.recorded == [project / "c2-top-right.mp4"]
    assert not (project / "c1-top-left.mp4.part").exists()
    assert not (project / ".mv3dt-footage.json").exists()


def test_keyboard_interrupt_terminates_children_and_cleans_up(tmp_path):
    h = Harness({CAM1.ip: _exits(1300), CAM2.ip: _exits(1300)}, interrupt_at=1005)
    project = tmp_path / "proj"
    project.mkdir()
    for name in ("c1-top-left.mp4.part", "c2-top-right.mp4.part"):
        (project / name).write_bytes(b"partial")

    result = h.record([CAM1, CAM2], project)

    assert result.cancelled
    assert result.failed == [CAM1, CAM2]
    assert result.recorded == []
    assert all(p.terminated for p in h.procs)
    assert not list(project.glob("*.part"))
    assert not (project / ".mv3dt-footage.json").exists()


def test_timeout_kills_child_only_after_terminate_is_ignored(tmp_path):
    h = Harness({CAM1.ip: _exits(None)}, ignore_terminate=True)
    result = h.record([CAM1], tmp_path / "proj", seconds=5)
    assert h.procs[0].signals == ["term", "kill"]
    assert result.failed == [CAM1]


def test_stderr_goes_to_a_file_not_an_unread_pipe(tmp_path, captured_logs):
    noisy = "".join(f"noise line {n}\n" for n in range(20000)) + "Connection refused\n"
    h = Harness({CAM1.ip: _exits(1001, rc=1, stderr=noisy)})
    h.record([CAM1], tmp_path / "proj")
    target = h.spawn_kwargs[0]["stderr"]
    assert target is not subprocess.PIPE
    assert hasattr(target, "fileno")
    assert target.closed  # cleaned up after recording
    warn = [m for lvl, m in captured_logs if lvl == "warn"]
    assert any(m.endswith(": Connection refused") for m in warn)


def test_keyboard_interrupt_kills_child_that_ignores_terminate(tmp_path):
    h = Harness({CAM1.ip: _exits(1300)}, interrupt_at=1002, ignore_terminate=True)
    result = h.record([CAM1], tmp_path / "proj")
    assert result.cancelled
    assert h.procs[0].terminated and h.procs[0].killed


def test_rerecord_removes_stale_marker_before_starting(tmp_path):
    project = tmp_path / "proj"
    Harness({CAM1.ip: _exits(1300), CAM2.ip: _exits(1300)}).record([CAM1, CAM2], project)
    assert (project / ".mv3dt-footage.json").exists()
    # A second run where one camera fails must not leave the old marker
    # vouching for a mix of old and new clips.
    Harness({CAM1.ip: _exits(1300), CAM2.ip: _exits(1001, rc=1)}).record([CAM1, CAM2], project)
    assert not (project / ".mv3dt-footage.json").exists()


def test_spawn_oserror_marks_camera_failed(tmp_path):
    h = Harness({CAM2.ip: _exits(1300)})
    real_spawn = h.spawn

    def spawn(argv, **kw):
        if "169.254.1.10" in " ".join(argv):
            raise FileNotFoundError(2, "No such file or directory")
        return real_spawn(argv, **kw)

    h.spawn = spawn
    result = h.record([CAM1, CAM2], tmp_path / "proj")
    assert result.failed == [CAM1]
    assert not (tmp_path / "proj" / ".mv3dt-footage.json").exists()


def test_password_never_in_logs_or_result(tmp_path, captured_logs):
    h = Harness({
        CAM1.ip: _exits(None),
        CAM2.ip: _exits(1001, rc=1, stderr=f"error {PASSWORD} rtsp://{USER}:{PASSWORD}@x:554/y\n"),
    })
    result = h.record([CAM1, CAM2], tmp_path / "proj", seconds=5)
    assert captured_logs
    for _, msg in captured_logs:
        assert PASSWORD not in msg
    assert PASSWORD not in repr(result)


# ---------------------------------------------------------------------------
# §3 condition 2 -- is_complete
# ---------------------------------------------------------------------------


def _complete_set(tmp_path):
    project = tmp_path / "proj"
    Harness({CAM1.ip: _exits(1300), CAM2.ip: _exits(1300)}).record([CAM1, CAM2], project)
    return project


def test_is_complete_true_for_matching_set(tmp_path):
    project = _complete_set(tmp_path)
    assert footage.is_complete(project, [CAM2, CAM1], project_name="Valencia-West", seconds=300)


def test_is_complete_false_without_marker(tmp_path):
    assert not footage.is_complete(tmp_path / "nope", [CAM1], project_name="Valencia-West", seconds=300)


def test_is_complete_false_for_unreadable_marker(tmp_path):
    project = _complete_set(tmp_path)
    (project / ".mv3dt-footage.json").write_text("{not json")
    assert not footage.is_complete(project, [CAM1, CAM2], project_name="Valencia-West", seconds=300)


def test_is_complete_false_for_other_project_or_length(tmp_path):
    project = _complete_set(tmp_path)
    assert not footage.is_complete(project, [CAM1, CAM2], project_name="Other", seconds=300)
    assert not footage.is_complete(project, [CAM1, CAM2], project_name="Valencia-West", seconds=120)


def test_is_complete_false_for_different_camera_set(tmp_path):
    project = _complete_set(tmp_path)
    assert not footage.is_complete(project, [CAM1], project_name="Valencia-West", seconds=300)
    cam3 = Camera(id="c3", mac="d0:3b:f4:00:00:03", ip="169.254.1.12", position="x")
    assert not footage.is_complete(project, [CAM1, CAM2, cam3], project_name="Valencia-West", seconds=300)


def test_is_complete_false_when_a_clip_is_missing(tmp_path):
    project = _complete_set(tmp_path)
    (project / "c2-top-right.mp4").unlink()
    assert not footage.is_complete(project, [CAM1, CAM2], project_name="Valencia-West", seconds=300)


# ---------------------------------------------------------------------------
# §3 condition 4 -- has_room
# ---------------------------------------------------------------------------

_Usage = namedtuple("_Usage", "total used free")


def test_has_room_checks_nearest_existing_ancestor(tmp_path, monkeypatch):
    seen = []

    def disk_usage(path):
        seen.append(pathlib.Path(path))
        return _Usage(0, 0, 5 << 30)

    monkeypatch.setattr(footage.shutil, "disk_usage", disk_usage)
    root = tmp_path / "Downloads" / "mv3dt-calibration-footage"
    assert footage.has_room(root, 2) == (True, 5 << 30, 2 << 30)
    assert seen == [tmp_path]


def test_has_room_reports_shortfall(tmp_path, monkeypatch):
    monkeypatch.setattr(footage.shutil, "disk_usage", lambda p: _Usage(0, 0, 1 << 30))
    assert footage.has_room(tmp_path, 2) == (False, 1 << 30, 2 << 30)


# ---------------------------------------------------------------------------
# §6 -- delete_if_owned
# ---------------------------------------------------------------------------


def test_delete_if_owned_removes_marked_dir_and_empty_root(tmp_path):
    root = tmp_path / "mv3dt-calibration-footage"
    project = root / "Valencia-West"
    Harness({CAM1.ip: _exits(1300)}).record([CAM1], project)
    assert footage.delete_if_owned(project) is True
    assert not project.exists()
    assert not root.exists()


def test_delete_if_owned_keeps_nonempty_root(tmp_path):
    root = tmp_path / "mv3dt-calibration-footage"
    project = root / "Valencia-West"
    Harness({CAM1.ip: _exits(1300)}).record([CAM1], project)
    (root / "Other").mkdir()
    assert footage.delete_if_owned(project) is True
    assert not project.exists()
    assert (root / "Other").is_dir()


def test_delete_if_owned_never_touches_unmarked_dir(tmp_path):
    project = tmp_path / "root" / "proj"
    project.mkdir(parents=True)
    (project / "c1-top-left.mp4").write_bytes(b"x")
    assert footage.delete_if_owned(project) is False
    assert (project / "c1-top-left.mp4").is_file()


def test_delete_if_owned_missing_dir(tmp_path):
    assert footage.delete_if_owned(tmp_path / "absent") is False


def test_delete_if_owned_failure_is_a_warning(tmp_path, monkeypatch, captured_logs):
    project = tmp_path / "root" / "proj"
    Harness({CAM1.ip: _exits(1300)}).record([CAM1], project)

    def boom(path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(footage.shutil, "rmtree", boom)
    assert footage.delete_if_owned(project) is False
    assert any(lvl == "warn" for lvl, _ in captured_logs)
