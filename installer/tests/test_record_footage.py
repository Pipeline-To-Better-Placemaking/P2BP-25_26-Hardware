"""Tests for mv3dt_installer.record_footage (AMC-FOOTAGE-RECORDING section 5).

Run from installer/: `python3 -m pytest tests/test_record_footage.py -v`

Nothing here needs root or starts ffmpeg. The service user is the test
process's own uid, so the real `lstat`/`fstat` ownership checks pass on files
the test creates; root, `pwd`, the privilege drop and the free-space check go
through the module's seams, and recording runs the real `footage.record` with
a fake `spawn`, `clock` and `sleep`.
"""

from __future__ import annotations

import functools
import json
import os
import pathlib
import queue
import signal
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from mv3dt_installer import camera_credentials, footage, logs  # noqa: E402
from mv3dt_installer import record_footage as rf  # noqa: E402

UID = os.getuid()
GID = os.getgid()
USER = "operator"
PASSWORD = "pw-that-must-never-print"
NOW = 1_794_886_679
FOLDER = "automation-test"

INVENTORY = """\
cameras:
  - id: c1
    mac: d0:3b:f4:00:00:01
    ip: 169.254.1.1
    position: left
  - id: c2
    mac: d0:3b:f4:00:00:02
    ip: 169.254.1.2
    position: right
  - id: c3
    mac: d0:3b:f4:00:00:03
    ip: 169.254.1.3
    position: back
    enabled: false
"""


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeProc:
    """An ffmpeg child: writes its `.part` file at once and exits `rc`."""

    def __init__(self, argv, rc, stderr_text, kwargs):
        self.argv = argv
        self.rc = rc
        self.part = pathlib.Path(argv[-1])
        self.part.write_bytes(b"\x00" * 16)
        if stderr_text:
            kwargs["stderr"].write(stderr_text.encode())
            kwargs["stderr"].flush()
        self.terminated = False

    def poll(self):
        return self.rc

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return self.rc if self.rc is not None else -15

    def kill(self):
        pass


class Spawner:
    def __init__(self, rcs=None, stderr=None):
        self.rcs = rcs or {}
        self.stderr = stderr or {}
        self.procs = []

    def __call__(self, argv, **kwargs):
        url = next(a for a in argv if a.startswith("rtsp://"))
        ip = url.split("@", 1)[1].split(":", 1)[0]
        proc = FakeProc(argv, self.rcs.get(ip, 0), self.stderr.get(ip, ""), kwargs)
        self.procs.append(proc)
        return proc


class Calls:
    """Records the privilege calls in order."""

    def __init__(self, setuid_zero_works=False):
        self.calls = []
        self.dropped = False
        self.setuid_zero_works = setuid_zero_works

    def initgroups(self, name, gid):
        self.calls.append(("initgroups", name, gid))

    def setgid(self, gid):
        self.calls.append(("setgid", gid))

    def setuid(self, uid):
        self.calls.append(("setuid", uid))
        if uid == 0 and self.dropped and not self.setuid_zero_works:
            raise PermissionError(1, "Operation not permitted")
        self.dropped = True


@pytest.fixture
def env(monkeypatch, tmp_path):
    logs._transcript_path = None
    install = tmp_path / "mv3dt"
    projects = install / "amc_projects"
    footage_dir = projects / FOLDER / "footage"
    footage_dir.mkdir(parents=True)
    (projects / FOLDER / "project.json").write_text("{}", encoding="utf-8")
    (install / "amc_control" / "state").mkdir(parents=True)
    inventory = install / "cameras.yml"
    inventory.write_text(INVENTORY, encoding="utf-8")

    calls = Calls()
    environ = {}
    spawner = Spawner()
    state = SimpleNamespace(
        install=install,
        projects=projects,
        footage=footage_dir,
        ctx=SimpleNamespace(install_dir=install, conf={"CAMERAS_FILE": str(inventory)}),
        calls=calls,
        environ=environ,
        spawner=spawner,
        sudo_user=USER,
        on_sleep=None,
        stdin=Stdin(),
        sleeping=threading.Event(),
    )

    def sleep(seconds):
        state.sleeping.set()
        if state.on_sleep is not None:
            state.on_sleep()

    monkeypatch.setattr(rf, "_READ_STDIN", state.stdin)

    monkeypatch.setattr(rf, "_GETEUID", lambda: 0)
    monkeypatch.setattr(rf, "_GETENV", lambda key: state.sudo_user if key == "SUDO_USER" else None)
    monkeypatch.setattr(rf, "_GETPWNAM", _getpwnam)
    monkeypatch.setattr(rf, "_INITGROUPS", calls.initgroups)
    monkeypatch.setattr(rf, "_SETGID", calls.setgid)
    monkeypatch.setattr(rf, "_SETUID", calls.setuid)
    monkeypatch.setattr(rf, "_ENVIRON", environ)
    monkeypatch.setattr(rf, "_TIME", lambda: float(NOW))
    monkeypatch.setattr(
        rf,
        "_LOAD_CREDENTIALS",
        lambda d: camera_credentials.Credentials(username="admin", password=PASSWORD),
    )
    monkeypatch.setattr(rf, "_HAS_ROOM", lambda root, n: (True, 10 << 30, n << 30))
    clock = iter(range(0, 10_000))
    monkeypatch.setattr(
        rf,
        "_RECORD",
        functools.partial(
            footage.record, spawn=spawner, clock=lambda: next(clock), sleep=sleep
        ),
    )
    write_request(state)
    yield state
    state.stdin.release()
    logs._transcript_path = None


class Stdin:
    """The service's pipe: stays open (blocks) until the test sends a line,
    closes it, or tears down. Teardown ends it without a cancel being
    possible, since the command has closed its cancel channel by then."""

    def __init__(self):
        self.lines = queue.Queue()

    def send(self, line):
        self.lines.put(line)

    def close(self):
        self.lines.put(None)

    def release(self):
        self.lines.put(None)

    def __call__(self):
        while True:
            line = self.lines.get()
            if line is None:
                return
            yield line


def _getpwnam(name):
    if name == USER:
        return SimpleNamespace(pw_name=USER, pw_uid=UID, pw_gid=GID, pw_dir="/home/operator")
    if name == "root":
        return SimpleNamespace(pw_name="root", pw_uid=0, pw_gid=0, pw_dir="/root")
    raise KeyError(name)


def write_request(state, **overrides):
    body = {"schema_version": 1, "folder": FOLDER, "seconds": 300, "requested_at": NOW}
    body.update(overrides)
    path = rf.request_path(state.install)
    if os.path.lexists(path):
        path.unlink()
    path.write_text(json.dumps(body), encoding="utf-8")
    path.chmod(0o600)
    return path


def run(state, capsys, argv=("--pending",)):
    code = rf.handle_record_footage_subcommand(list(argv), state.ctx)
    out = capsys.readouterr()
    lines = [json.loads(line) for line in out.out.splitlines()]
    return code, lines, out.err


FIELDS = {"ok", "cancelled", "reason", "files", "failed", "seconds"}


def final(lines):
    *progress, last = lines
    assert set(last) == FIELDS
    assert all(set(p) == {"progress"} for p in progress)
    return last


def assert_refused(state, capsys, needle, code=rf.EXIT_INVALID):
    exit_code, lines, err = run(state, capsys)
    assert exit_code == code
    assert len(lines) == 1  # no progress: nothing was recorded
    last = final(lines)
    assert last["ok"] is False and last["cancelled"] is False
    assert last["failed"] == [] and last["files"] == []
    assert needle in last["reason"]
    assert needle in err
    assert state.calls.calls == []  # privileges never dropped
    assert list(state.footage.iterdir()) == []
    return last


def seed_old_footage(state):
    """An earlier recording and an unrelated file in `footage/`."""
    (state.footage / "c1-left.mp4").write_text("old-c1")
    (state.footage / "c2-right.mp4").write_text("old-c2")
    (state.footage / footage.MARKER_NAME).write_text('{"old": true}')
    (state.footage / "notes.txt").write_text("keep")
    return snapshot(state.footage)


def snapshot(directory):
    return {
        str(p.relative_to(directory)): (p.read_text() if p.is_file() else "<dir>")
        for p in sorted(directory.rglob("*"))
    }


def never_finishing(state):
    for ip in ("169.254.1.1", "169.254.1.2"):
        state.spawner.rcs[ip] = None


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


def test_success_records_every_enabled_camera(env, capsys):
    code, lines, err = run(env, capsys)

    assert code == rf.EXIT_OK
    *progress, last = lines
    assert progress and progress[-1] == {"progress": 1.0}
    assert last == {
        "ok": True,
        "cancelled": False,
        "reason": "",
        "files": ["c1-left.mp4", "c2-right.mp4"],
        "failed": [],
        "seconds": 300,
    }
    assert sorted(p.name for p in env.footage.iterdir()) == [
        footage.MARKER_NAME, "c1-left.mp4", "c2-right.mp4", "previous",
    ]
    assert not list(env.footage.glob("*.part"))
    assert PASSWORD not in err


def test_recording_has_no_user_prefix_and_runs_after_the_drop(env, capsys, monkeypatch):
    seen = {}
    real = rf._RECORD

    def record(cameras, **kwargs):
        seen.update(kwargs, dropped=env.calls.dropped, ids=[c.id for c in cameras])
        return real(cameras, **kwargs)

    monkeypatch.setattr(rf, "_RECORD", record)
    code, _lines, _err = run(env, capsys)

    assert code == rf.EXIT_OK
    assert "user_prefix" not in seen
    assert seen["dropped"] is True
    assert seen["ids"] == ["c1", "c2"]
    assert seen["project_dir"] == env.footage and seen["project_name"] == FOLDER
    assert seen["seconds"] == 300 and seen["user"] == "admin"
    assert env.spawner.procs[0].argv[0] == "ffmpeg"


def test_privileges_are_dropped_in_order_then_checked(env, capsys):
    code, _lines, _err = run(env, capsys)

    assert code == rf.EXIT_OK
    assert env.calls.calls == [
        ("initgroups", USER, GID),
        ("setgid", GID),
        ("setuid", UID),
        ("setuid", 0),
    ]
    assert env.environ == {"HOME": "/home/operator", "USER": USER, "LOGNAME": USER}


def test_a_revocable_drop_stops_before_recording(env, capsys):
    env.calls.setuid_zero_works = True
    code, lines, _err = run(env, capsys)

    assert code == rf.EXIT_INVALID
    assert "not dropped" in final(lines)["reason"]
    assert env.spawner.procs == []


def test_old_footage_moves_into_previous(env, capsys):
    seed_old_footage(env)
    previous = env.footage / "previous"
    previous.mkdir()
    (previous / "ancient.mp4").write_text("gone")
    (previous / "sub").mkdir()

    code, _lines, _err = run(env, capsys)

    assert code == rf.EXIT_OK
    assert sorted(p.name for p in previous.iterdir()) == [
        footage.MARKER_NAME, "c1-left.mp4", "c2-right.mp4",
    ]
    assert (previous / "c1-left.mp4").read_text() == "old-c1"
    assert (env.footage / "notes.txt").read_text() == "keep"
    assert (env.footage / "c1-left.mp4").read_bytes() == b"\x00" * 16


def test_a_symlinked_previous_is_refused(env, capsys, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "precious.mp4").write_text("x")
    (env.footage / "previous").symlink_to(elsewhere)

    code, lines, _err = run(env, capsys)

    assert code == rf.EXIT_INVALID
    assert "previous" in final(lines)["reason"]
    assert (elsewhere / "precious.mp4").exists()
    assert env.spawner.procs == []


# ---------------------------------------------------------------------------
# Failure, restore and cancel
# ---------------------------------------------------------------------------


def test_a_failed_camera_is_exit_1_and_restores_the_folder(env, capsys):
    before = seed_old_footage(env)
    env.spawner.rcs = {"169.254.1.2": 1}
    env.spawner.stderr = {
        "169.254.1.2": f"rtsp://admin:{PASSWORD}@169.254.1.2:554/x: 401 Unauthorized ({PASSWORD})\n"
    }
    code, lines, err = run(env, capsys)

    assert code == rf.EXIT_FAILED
    last = final(lines)
    assert last["ok"] is False and last["cancelled"] is False
    assert "1 of 2 camera(s) failed" in last["reason"]
    assert last["files"] == []  # this run's clips were deleted
    (failed,) = last["failed"]
    assert failed["id"] == "c2" and failed["ip"] == "169.254.1.2"
    assert "401 Unauthorized" in failed["error"] and "ffmpeg exited 1" in failed["error"]
    assert "<redacted>" in failed["error"]
    for text in (json.dumps(lines), err):
        assert PASSWORD not in text
    assert snapshot(env.footage) == before  # exactly as it was


def test_a_failure_with_no_earlier_footage_leaves_footage_empty(env, capsys):
    env.spawner.rcs = {"169.254.1.1": 1, "169.254.1.2": 1}
    code, lines, _err = run(env, capsys)

    assert code == rf.EXIT_FAILED and len(final(lines)["failed"]) == 2
    assert list(env.footage.iterdir()) == []


def test_sigterm_cancels_and_restores(env, capsys):
    before = seed_old_footage(env)
    never_finishing(env)
    env.on_sleep = lambda: os.kill(os.getpid(), signal.SIGTERM)
    handler_before = signal.getsignal(signal.SIGTERM)

    code, lines, _err = run(env, capsys)

    assert code == rf.EXIT_CANCELLED
    assert final(lines) == {
        "ok": False, "cancelled": True, "reason": "cancelled",
        "files": [], "failed": [], "seconds": 300,
    }
    assert all(p.terminated for p in env.spawner.procs)
    assert snapshot(env.footage) == before  # no .part, no new clip, no marker
    assert signal.getsignal(signal.SIGTERM) is handler_before


def test_sigint_cancels_the_same_way(env, capsys):
    never_finishing(env)
    env.on_sleep = lambda: os.kill(os.getpid(), signal.SIGINT)

    code, lines, _err = run(env, capsys)

    assert code == rf.EXIT_CANCELLED and final(lines)["reason"] == "cancelled"
    assert list(env.footage.iterdir()) == []


def _slow_sleep(env):
    """Recording loop that gives the stdin thread time to interrupt."""
    env.on_sleep = lambda: time.sleep(0.01)


def test_a_cancel_line_on_stdin_cancels(env, capsys):
    before = seed_old_footage(env)
    never_finishing(env)
    _slow_sleep(env)
    threading.Thread(
        target=lambda: env.sleeping.wait(5) and env.stdin.send("cancel\n"), daemon=True
    ).start()

    code, lines, _err = run(env, capsys)

    assert code == rf.EXIT_CANCELLED and final(lines)["cancelled"] is True
    assert env.spawner.procs and all(p.terminated for p in env.spawner.procs)
    assert snapshot(env.footage) == before


def test_other_stdin_lines_are_ignored(env, capsys):
    env.stdin.send("hello\n")
    code, _lines, _err = run(env, capsys)
    assert code == rf.EXIT_OK


def test_end_of_stdin_cancels(env, capsys):
    never_finishing(env)
    _slow_sleep(env)
    threading.Thread(
        target=lambda: env.sleeping.wait(5) and env.stdin.close(), daemon=True
    ).start()

    code, lines, _err = run(env, capsys)

    assert code == rf.EXIT_CANCELLED and final(lines)["reason"] == "cancelled"
    assert list(env.footage.iterdir()) == []


def test_stdin_is_only_read_after_the_drop(env, capsys, monkeypatch):
    seen = []

    def read():
        seen.append(env.calls.dropped)
        return iter(())  # immediate end of file would cancel

    monkeypatch.setattr(rf, "_READ_STDIN", read)
    env.calls.setuid_zero_works = True  # refused at the drop check
    code, _lines, _err = run(env, capsys)
    assert code == rf.EXIT_INVALID and seen == []


def test_a_cancel_during_set_aside_restores_then_exits_3(env, capsys, monkeypatch):
    before = seed_old_footage(env)
    real_replace = os.replace
    moves = []

    def replace(src, dst):
        real_replace(src, dst)
        moves.append((str(src), str(dst)))
        if len(moves) == 2:  # midway through setting aside three files
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(rf, "_REPLACE", replace)
    code, lines, _err = run(env, capsys)

    assert code == rf.EXIT_CANCELLED and final(lines)["reason"] == "cancelled"
    assert env.spawner.procs == []  # never recorded
    assert snapshot(env.footage) == before


def test_a_cancel_during_restore_finishes_the_restore_then_exits_3(env, capsys, monkeypatch):
    before = seed_old_footage(env)
    env.spawner.rcs = {"169.254.1.2": 1}
    real_replace = os.replace
    restoring = []

    def replace(src, dst):
        if pathlib.Path(src).parent.name == "previous":  # moving back
            restoring.append(src)
            if len(restoring) == 1:
                os.kill(os.getpid(), signal.SIGTERM)
        real_replace(src, dst)

    monkeypatch.setattr(rf, "_REPLACE", replace)
    code, lines, _err = run(env, capsys)

    assert code == rf.EXIT_CANCELLED
    assert final(lines) == {
        "ok": False, "cancelled": True, "reason": "cancelled",
        "files": [], "failed": [], "seconds": 300,
    }
    assert snapshot(env.footage) == before


def test_no_room_is_exit_1_with_free_and_needed_bytes(env, capsys, monkeypatch):
    monkeypatch.setattr(rf, "_HAS_ROOM", lambda root, n: (False, 1 << 30, n << 30))
    last = assert_refused(env, capsys, "not enough free space", code=rf.EXIT_FAILED)
    assert f"{1 << 30} bytes free" in last["reason"]
    assert f"{2 << 30} bytes needed" in last["reason"]


# ---------------------------------------------------------------------------
# Validation: every refusal is exit 2 with one final line
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [(), ("--pending", "x"), ("--now",), ("--folder=a",)])
def test_anything_but_pending_is_a_usage_error(env, capsys, argv):
    code, lines, _err = run(env, capsys, argv)
    assert code == rf.EXIT_INVALID
    assert lines == [final(lines)] and "usage" in lines[0]["reason"]


def test_not_run_as_root(env, capsys, monkeypatch):
    monkeypatch.setattr(rf, "_GETEUID", lambda: 1000)
    assert_refused(env, capsys, "must be run as root")


def test_not_run_under_sudo(env, capsys):
    env.sudo_user = None
    assert_refused(env, capsys, "SUDO_USER is unset")


def test_sudo_user_root_is_refused(env, capsys):
    env.sudo_user = "root"
    assert_refused(env, capsys, "SUDO_USER is root")


def test_unknown_sudo_user_is_refused(env, capsys):
    env.sudo_user = "ghost"
    assert_refused(env, capsys, "not a user")


def test_projects_root_owned_by_someone_else(env, capsys, monkeypatch):
    _owner_override(monkeypatch, env.projects, uid=4242)
    assert_refused(env, capsys, "owned by uid 4242")


def test_missing_request_file(env, capsys):
    rf.request_path(env.install).unlink()
    assert_refused(env, capsys, "no pending request")


def test_symlinked_request_file(env, capsys, tmp_path):
    real = tmp_path / "real.json"
    real.write_text(rf.request_path(env.install).read_text())
    path = rf.request_path(env.install)
    path.unlink()
    path.symlink_to(real)
    assert_refused(env, capsys, "is a symlink")


def test_request_is_opened_with_o_nofollow(env, capsys, monkeypatch):
    flags_seen = []

    def spy(path, flags, *args):
        flags_seen.append(flags)
        return os.open(path, flags, *args)

    monkeypatch.setattr(rf, "_OPEN", spy)
    run(env, capsys)
    assert flags_seen and flags_seen[0] & os.O_NOFOLLOW and not flags_seen[0] & os.O_WRONLY


def test_request_owned_by_someone_else(env, capsys, monkeypatch):
    def fstat(fd):
        st = os.fstat(fd)
        return SimpleNamespace(st_mode=st.st_mode, st_uid=4242, st_size=st.st_size)

    monkeypatch.setattr(rf, "_FSTAT", fstat)
    assert_refused(env, capsys, "owned by uid 4242")


def test_request_that_is_a_directory(env, capsys):
    path = rf.request_path(env.install)
    path.unlink()
    path.mkdir()
    assert_refused(env, capsys, "not a regular file")


def test_oversize_request(env, capsys):
    path = rf.request_path(env.install)
    body = json.loads(path.read_text())
    path.write_text(json.dumps(body) + " " * rf.MAX_REQUEST_BYTES)
    assert_refused(env, capsys, "larger than 4096 bytes")


def test_stale_request(env, capsys):
    write_request(env, requested_at=NOW - rf.MAX_REQUEST_AGE_S - 1)
    assert_refused(env, capsys, "older than 300 seconds")


def test_request_just_inside_the_window_is_accepted(env, capsys):
    write_request(env, requested_at=NOW - rf.MAX_REQUEST_AGE_S)
    code, _lines, _err = run(env, capsys)
    assert code == rf.EXIT_OK


def test_request_from_the_future(env, capsys):
    write_request(env, requested_at=NOW + rf.MAX_CLOCK_SKEW_S + 1)
    assert_refused(env, capsys, "in the future")


def test_not_json(env, capsys):
    rf.request_path(env.install).write_text("{not json")
    assert_refused(env, capsys, "not valid JSON")


def test_not_an_object(env, capsys):
    rf.request_path(env.install).write_text("[1, 2]")
    assert_refused(env, capsys, "not a JSON object")


@pytest.mark.parametrize(
    "overrides, needle",
    [
        ({"schema_version": 2}, "schema_version"),
        ({"schema_version": True}, "schema_version"),
        ({"schema_version": "1"}, "schema_version"),
        ({"extra": 1}, "exactly the keys"),
        ({"folder": "../etc"}, "folder"),
        ({"folder": "Upper"}, "folder"),
        ({"folder": "-dash"}, "folder"),
        ({"folder": "a" * 49}, "folder"),
        ({"folder": 7}, "folder"),
        ({"seconds": 29}, "seconds"),
        ({"seconds": 1801}, "seconds"),
        ({"seconds": 300.0}, "seconds"),
        ({"seconds": True}, "seconds"),
        ({"seconds": "300"}, "seconds"),
        ({"requested_at": float(NOW)}, "requested_at"),
        ({"requested_at": None}, "requested_at"),
    ],
)
def test_bad_schema(env, capsys, overrides, needle):
    write_request(env, **overrides)
    assert_refused(env, capsys, needle)


def test_missing_key(env, capsys):
    path = rf.request_path(env.install)
    body = json.loads(path.read_text())
    del body["seconds"]
    path.write_text(json.dumps(body))
    assert_refused(env, capsys, "exactly the keys")


def test_unknown_folder(env, capsys):
    write_request(env, folder="no-such-project")
    assert_refused(env, capsys, "is missing")


def test_symlinked_folder(env, capsys, tmp_path):
    target = tmp_path / "target"
    (target / "footage").mkdir(parents=True)
    (target / "project.json").write_text("{}")
    (env.projects / "linked").symlink_to(target)
    write_request(env, folder="linked")
    assert_refused(env, capsys, "is a symlink")


def test_symlinked_footage_dir(env, capsys, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    env.footage.rmdir()
    env.footage.symlink_to(target)
    exit_code, lines, _err = run(env, capsys)
    assert exit_code == rf.EXIT_INVALID and "is a symlink" in final(lines)["reason"]
    assert list(target.iterdir()) == []


def test_footage_that_is_a_file(env, capsys):
    env.footage.rmdir()
    env.footage.write_text("")
    exit_code, lines, _err = run(env, capsys)
    assert exit_code == rf.EXIT_INVALID and "not a directory" in final(lines)["reason"]


def test_folder_owned_by_someone_else(env, capsys, monkeypatch):
    _owner_override(monkeypatch, env.projects / FOLDER, uid=0)
    assert_refused(env, capsys, "owned by uid 0")


def test_footage_dir_owned_by_someone_else(env, capsys, monkeypatch):
    _owner_override(monkeypatch, env.footage, uid=0)
    assert_refused(env, capsys, "owned by uid 0")


def test_missing_project_json(env, capsys):
    (env.projects / FOLDER / "project.json").unlink()
    assert_refused(env, capsys, "project.json is missing")


def test_project_json_that_is_a_symlink(env, capsys, tmp_path):
    real = tmp_path / "p.json"
    real.write_text("{}")
    project_file = env.projects / FOLDER / "project.json"
    project_file.unlink()
    project_file.symlink_to(real)
    assert_refused(env, capsys, "not a regular file")


def test_no_credentials(env, capsys, monkeypatch):
    monkeypatch.setattr(rf, "_LOAD_CREDENTIALS", lambda d: None)
    assert_refused(env, capsys, "no camera credentials")


def test_no_inventory_configured(env, capsys):
    env.ctx.conf = {}
    assert_refused(env, capsys, "no camera inventory")


def test_inventory_file_missing(env, capsys):
    env.ctx.conf = {"CAMERAS_FILE": str(env.install / "missing.yml")}
    assert_refused(env, capsys, "cannot read the camera inventory")


def test_empty_inventory(env, capsys):
    pathlib.Path(env.ctx.conf["CAMERAS_FILE"]).write_text("cameras:\n")
    assert_refused(env, capsys, "no enabled camera")


def test_errors_never_carry_the_password(env, capsys, monkeypatch):
    def boom(root, n):
        raise OSError(f"rtsp://admin:{PASSWORD}@x failed")

    monkeypatch.setattr(rf, "_HAS_ROOM", boom)
    code, lines, err = run(env, capsys)
    assert code == rf.EXIT_FAILED and len(lines) == 1
    assert PASSWORD not in json.dumps(lines) and PASSWORD not in err


def _owner_override(monkeypatch, path, *, uid):
    """`_LSTAT` reporting `uid` as the owner of `path` only."""

    def lstat(p):
        st = os.lstat(p)
        if p == str(path):
            return SimpleNamespace(st_mode=st.st_mode, st_uid=uid)
        return st

    monkeypatch.setattr(rf, "_LSTAT", lstat)


def test_request_file_mode_is_not_relaxed_by_the_command(env, capsys):
    path = rf.request_path(env.install)
    run(env, capsys)
    assert path.exists() and stat.S_IMODE(path.stat().st_mode) == 0o600


# ---------------------------------------------------------------------------
# Process exit with the service's stdin still open
# ---------------------------------------------------------------------------

_CHILD = r"""
import os, pathlib, sys
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
from mv3dt_installer import camera_credentials, footage, record_footage as rf

install = pathlib.Path(sys.argv[2])
entry = SimpleNamespace(pw_name="operator", pw_uid=os.getuid(), pw_gid=os.getgid(), pw_dir="/tmp")
dropped = []

def setuid(uid):
    if uid == 0 and dropped:
        raise PermissionError(1, "Operation not permitted")
    dropped.append(uid)

def record(cameras, *, project_dir, on_progress, **kw):
    on_progress(1.0)
    clips = []
    for c in cameras:
        clip = pathlib.Path(project_dir) / footage.clip_name(c)
        clip.write_bytes(b"x")
        clips.append(clip)
    return footage.RecordResult(project_dir=project_dir, recorded=clips, failed=[], cancelled=False)

rf._GETEUID = lambda: 0
rf._GETENV = lambda key: "operator" if key == "SUDO_USER" else None
rf._GETPWNAM = lambda name: entry
rf._INITGROUPS = lambda name, gid: None
rf._SETGID = lambda gid: None
rf._SETUID = setuid
rf._ENVIRON = {}
rf._TIME = lambda: float(int(sys.argv[3]))
rf._LOAD_CREDENTIALS = lambda d: camera_credentials.Credentials("admin", "pw")
rf._HAS_ROOM = lambda root, n: (True, 1, 0)
rf._RECORD = record
ctx = SimpleNamespace(install_dir=install, conf={"CAMERAS_FILE": str(install / "cameras.yml")})
sys.exit(rf.handle_record_footage_subcommand(["--pending"], ctx))
"""


def test_exits_after_a_successful_recording_with_stdin_left_open(env, tmp_path):
    """The service closes stdin only to cancel or after the command exits,
    so the real stdin reader (a daemon thread) must never hold the process
    open or break interpreter shutdown."""
    script = tmp_path / "child.py"
    script.write_text(_CHILD, encoding="utf-8")
    installer_dir = str(pathlib.Path(__file__).resolve().parent.parent)
    proc = subprocess.Popen(
        [sys.executable, "-I", str(script), installer_dir, str(env.install), str(NOW)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        code = proc.wait(timeout=30)  # stdin is still open here
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.stdin.close()
    out = proc.stdout.read().decode()
    err = proc.stderr.read().decode()
    proc.stdout.close()
    proc.stderr.close()

    assert code == rf.EXIT_OK, err
    assert "Fatal Python error" not in err
    lines = [json.loads(line) for line in out.splitlines()]
    assert final(lines)["ok"] is True
    assert final(lines)["files"] == ["c1-left.mp4", "c2-right.mp4"]
