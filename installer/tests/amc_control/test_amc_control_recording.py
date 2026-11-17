"""Recording footage from step 2 of a web_project run (AMC-FOOTAGE-RECORDING.md section 6).

A small script stands in for `mv3dt-installer record-footage --pending`,
following section 5's stdout, stdin-cancel and exit-code contract; it logs
what it saw of the request file on every call. Run from installer/:
`python3 -m pytest tests/amc_control -v`
"""
import json
import os
import signal
import sys
import time

import pytest

pytest.importorskip("jwt")
pytest.importorskip("httpx")
pytest.importorskip("yaml")
pytest.importorskip("fastapi")
pytest.importorskip("multipart")
pytest.importorskip("PIL")

from mv3dt_amc_control import automation as automation_mod  # noqa: E402
from mv3dt_amc_control.automation import CalibrationRunner  # noqa: E402
from mv3dt_amc_control.calibration_spec import SpecError  # noqa: E402
from mv3dt_amc_control.projects import ProjectStore  # noqa: E402
from test_amc_control_automation import (  # noqa: E402,F401  (ws, web and api are fixtures)
    FULLY_AUTOMATED, PNG, WEB, WEB_AMC_ID, FakeController, amc_saved_alignment, api, calls, make_client,
    make_runner, make_spec, spec, web, web_runner, ws,
)

HINT = "keep people walking through every camera's view"

FAKE_RECORDER = r'''
import json, os, shutil, signal, stat, sys, threading, time
mode, request, footage, log = sys.argv[1:5]
st = os.stat(request)
with open(request) as fh:
    req = json.load(fh)
with open(log, "a") as fh:
    fh.write(json.dumps({"mode": stat.S_IMODE(st.st_mode), "request": req, "pid": os.getpid()}) + "\n")

def out(obj):
    print(json.dumps(obj), flush=True)

def final(ok=False, cancelled=False, reason="", files=(), failed=()):
    out({"ok": ok, "cancelled": cancelled, "reason": reason, "files": list(files), "failed": list(failed),
         "seconds": req["seconds"]})

if mode == "ok":
    import select
    if select.select([sys.stdin], [], [], 0.2)[0]:   # end of file on stdin would mean cancel
        print("stdin closed before the recorder finished", file=sys.stderr, flush=True)
        sys.exit(4)
    os.makedirs(os.path.join(footage, "previous"), exist_ok=True)
    for name in os.listdir(footage):
        if name.endswith(".mp4"):
            shutil.move(os.path.join(footage, name), os.path.join(footage, "previous", name))
    print("not json", flush=True)
    for f in (0.0, 0.5, 1.0):
        out({"progress": f})
    for name in ("c2-right.mp4", "c1-left.mp4"):
        with open(os.path.join(footage, name), "wb") as fh:
            fh.write(b"recorded")
    final(ok=True, files=["c1-left.mp4", "c2-right.mp4"])
    sys.exit(0)
if mode == "camfail":
    out({"progress": 1.0})
    final(reason="1 of 2 cameras failed", files=["c1-left.mp4"],
          failed=[{"id": "c2-right", "ip": "169.254.1.2", "error": "connection refused"}])
    sys.exit(1)
if mode == "invalid":
    print("record-footage: diagnostics", file=sys.stderr, flush=True)
    final(reason="the request is more than 300 seconds old")
    sys.exit(2)
if mode == "nofinal":
    print("sudo: a password is required", file=sys.stderr, flush=True)
    sys.exit(1)
if mode == "selfcancel":
    final(cancelled=True, reason="cancelled")
    sys.exit(3)
if mode == "hang":
    # Cancelled through stdin: a line "cancel" or end of file.
    signal.signal(signal.SIGTERM, signal.SIG_IGN)   # signals are not how the service cancels
    out({"progress": 0.25})
    got = sys.stdin.readline()
    with open(log + ".cancel", "w") as fh:
        fh.write(got)
    final(cancelled=True, reason="cancelled")
    sys.exit(3)
if mode == "closedout":
    # Still running with stdout closed: only stdin (cancel) ends it.
    out({"progress": 0.25})
    sys.stdout.close()
    got = sys.stdin.readline()
    with open(log + ".cancel", "w") as fh:
        fh.write(got)
    sys.exit(3)
if mode == "leftover":
    # Exits while a grandchild still holds stdout and stderr open.
    import subprocess
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    with open(log, "a") as fh:
        fh.write(json.dumps({"pid": child.pid}) + "\n")
    final(reason="1 of 2 cameras failed", files=["c1-left.mp4"],
          failed=[{"id": "c2-right", "ip": "169.254.1.2", "error": "connection refused"}])
    os._exit(1)
if mode == "stubborn":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    out({"progress": 0.1})
    while True:
        time.sleep(0.05)
'''


@pytest.fixture
def recorder(ws, web, tmp_path):
    """make(mode) builds a web_project runner whose recorder is the fake script in that mode."""
    _, install = ws
    _, folder, _ = web
    script = tmp_path / "fake_recorder.py"
    script.write_text(FAKE_RECORDER)
    log = tmp_path / "recorder.log"
    amc_saved_alignment(ws)

    def make(mode, overrides=WEB):
        runner = web_runner(ws, web, overrides)
        runner._record_argv = lambda: [sys.executable, str(script), mode, str(runner.record_request_path),
                                       str(folder / "footage"), str(log)]
        messages = []
        real = runner._set_step

        def spy(key, state, message="", *a, **kw):
            messages.append(message)
            return real(key, state, message, *a, **kw)

        runner._set_step = spy
        runner.messages = messages
        return runner

    def seen():
        if not log.exists():
            return []
        return [c for c in (json.loads(line) for line in log.read_text().splitlines()) if "request" in c]

    def pids():
        if not log.exists():
            return []
        return [json.loads(line)["pid"] for line in log.read_text().splitlines()]

    make.seen = seen
    make.cancel_line = log.parent / (log.name + ".cancel")
    make.request = install / "amc_control" / "state" / "record-request.json"
    yield make
    for pid in pids():  # never leave a stubborn fake or a leftover child behind
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def wait_for_message(runner, prefix):
    deadline = time.time() + 20
    while not any(m.startswith(prefix) for m in runner.messages):
        assert time.time() < deadline, runner.messages
        time.sleep(0.05)


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# ---------------------------------------------------------------- success
def test_recording_from_step_2_uploads_the_new_clips(recorder, web):
    _, folder, fake = web
    runner = recorder("ok")
    before = int(time.time())
    job = runner.start(spec(), record_seconds=300)
    assert job["State"] == "completed", job

    # The request file as the root command saw it, then removed.
    (call,) = recorder.seen()
    assert call["mode"] == 0o600
    req = call["request"]
    assert set(req) == {"schema_version", "folder", "seconds", "requested_at"}
    assert req["schema_version"] == 1 and req["folder"] == "living-room" and req["seconds"] == 300
    assert before <= req["requested_at"] <= int(time.time())
    assert not recorder.request.exists()
    assert not list(recorder.request.parent.glob(".record-request*"))

    # Progress in step 2's message, then the recorded clips (not the old ones) uploaded.
    progress = [m for m in runner.messages if m.startswith("recording")]
    assert progress == [f"recording 0:00 of 5:00; {HINT}", f"recording 2:30 of 5:00; {HINT}",
                        f"recording 5:00 of 5:00; {HINT}"]
    assert calls(fake, "upload_video_files") == [{"names": ["cam_00.mp4", "cam_01.mp4"]}]
    msg = job["Steps"][1]["Message"]
    assert "recorded 2 clips of 5:00" in msg and "c1-left.mp4 as cam_00.mp4" in msg
    assert sorted(p.name for p in (folder / "footage" / "previous").iterdir()) == ["a.mp4", "b.mp4"]
    state = json.loads(runner.state_path.read_text())
    assert state["record_seconds"] == 300 and state["footage_recorded"] is True


def test_recorded_clips_ignore_videos_order(recorder):
    runner = recorder("ok", {**WEB, "videos": {"order": ["b.mp4", "a.mp4"]}})
    job = runner.start(spec(), record_seconds=60)
    assert job["State"] == "completed", job
    msg = job["Steps"][1]["Message"]
    assert "c1-left.mp4 as cam_00.mp4" in msg and "c2-right.mp4 as cam_01.mp4" in msg


def test_a_run_without_record_never_runs_the_recorder(recorder, web):
    _, _, fake = web
    assert recorder("ok").start(spec())["State"] == "completed"
    assert recorder.seen() == []
    assert calls(fake, "upload_video_files") == [{"names": ["cam_00.mp4", "cam_01.mp4"]}]


# ---------------------------------------------------------------- failures
def test_recording_camera_failure_fails_step_2_with_each_cameras_error(recorder, web):
    _, folder, fake = web
    job = recorder("camfail").start(spec(), record_seconds=60)
    assert job["State"] == "failed"
    assert job["Steps"][1]["State"] == "failed"
    msg = job["Steps"][1]["Message"]
    assert "exit 1" in msg and "1 of 2 cameras failed" in msg
    assert "c2-right (169.254.1.2): connection refused" in msg
    assert calls(fake, "upload_video_files") == []
    assert not recorder.request.exists()
    assert json.loads((folder / "runs" / f"{job['JobId']}.json").read_text())["State"] == "failed"


def test_recording_refused_request_reports_the_reason(recorder, web):
    _, _, fake = web
    job = recorder("invalid").start(spec(), record_seconds=60)
    assert job["State"] == "failed"
    msg = job["Steps"][1]["Message"]
    assert "exit 2" in msg and "the request is more than 300 seconds old" in msg
    assert calls(fake, "upload_video_files") == []
    assert not recorder.request.exists()


def test_recording_without_a_result_line_says_the_command_was_refused(recorder):
    job = recorder("nofinal").start(spec(), record_seconds=60)
    assert job["State"] == "failed"
    msg = job["Steps"][1]["Message"]
    assert "refused or could not run the recording command (exit 1)" in msg
    assert "re-running the installer" in msg and "sudo: a password is required" in msg
    assert not recorder.request.exists()


def test_recorder_that_cannot_start_fails_step_2(recorder, tmp_path):
    runner = recorder("ok")
    runner._record_argv = lambda: [str(tmp_path / "no-such-binary"), "record-footage", "--pending"]
    job = runner.start(spec(), record_seconds=60)
    assert job["State"] == "failed"
    assert "could not run the recording command" in job["Steps"][1]["Message"]
    assert not recorder.request.exists()


# ---------------------------------------------------------------- cancel and timeout
def test_recorder_exit_3_ends_the_run_cancelled(recorder, web):
    _, _, fake = web
    job = recorder("selfcancel").start(spec(), record_seconds=60)
    assert job["State"] == "cancelled"
    assert calls(fake, "upload_video_files") == []
    assert not recorder.request.exists()


def test_cancel_during_recording_writes_cancel_to_stdin(recorder, web):
    _, folder, fake = web
    runner = recorder("hang")
    runner._threaded = True
    runner.start(spec(), record_seconds=600)
    wait_for_message(runner, "recording 2:30 of 10:00")
    assert recorder.request.exists()
    runner.cancel()
    runner.join(20)
    job = runner.status()
    assert job["State"] == "cancelled", job
    assert recorder.cancel_line.read_text() == "cancel\n"
    assert not recorder.request.exists()
    assert calls(fake, "upload_video_files") == []
    assert json.loads((folder / "runs" / f"{job['JobId']}.json").read_text())["State"] == "cancelled"


def test_a_recorder_that_does_not_stop_is_never_killed(recorder):
    runner = recorder("stubborn")
    runner.record_cancel_grace = 0.3
    runner._threaded = True
    runner.start(spec(), record_seconds=60)
    wait_for_message(runner, "recording 0:06 of 1:00")
    runner.cancel()
    runner.join(20)
    assert runner.status()["State"] == "cancelled"
    assert not recorder.request.exists()
    assert alive(recorder.seen()[0]["pid"])   # only stdin and a bounded wait, no SIGKILL


def test_a_recording_that_overruns_is_cancelled_and_fails_the_step(recorder):
    runner = recorder("hang")
    runner.record_overrun = 0.5 - 30           # the deadline is half a second after the start
    job = runner.start(spec(), record_seconds=30)
    assert job["State"] == "failed", job
    msg = job["Steps"][1]["Message"]
    assert "still running after" in msg and msg.endswith("was cancelled")
    assert recorder.cancel_line.read_text() == "cancel\n"
    assert not recorder.request.exists()


def test_an_overrun_recording_that_ignores_cancel_fails_the_step(recorder):
    runner = recorder("stubborn")
    runner.record_overrun = 0.5 - 30
    runner.record_cancel_grace = 0.3
    job = runner.start(spec(), record_seconds=30)
    assert job["State"] == "failed", job
    assert "did not stop within 0 seconds of being cancelled" in job["Steps"][1]["Message"]
    assert alive(recorder.seen()[0]["pid"])
    assert not recorder.request.exists()


def test_cancel_still_works_after_the_recorder_closes_stdout(recorder):
    runner = recorder("closedout")
    runner._threaded = True
    runner.start(spec(), record_seconds=600)
    wait_for_message(runner, "recording 2:30 of 10:00")
    time.sleep(0.5)                            # stdout has closed; the recorder is still running
    assert runner.status()["State"] == "running"
    runner.cancel()
    runner.join(20)
    assert runner.status()["State"] == "cancelled"
    assert recorder.cancel_line.read_text() == "cancel\n"
    assert not recorder.request.exists()


def test_the_deadline_still_applies_after_the_recorder_closes_stdout(recorder):
    runner = recorder("closedout")
    runner.record_overrun = 1.0 - 30
    job = runner.start(spec(), record_seconds=30)
    assert job["State"] == "failed", job
    assert job["Steps"][1]["Message"].endswith("was cancelled")
    assert recorder.cancel_line.read_text() == "cancel\n"


def test_a_recorder_that_exits_while_a_child_holds_stdout_is_detected(recorder, web):
    _, _, fake = web
    started = time.monotonic()
    job = recorder("leftover").start(spec(), record_seconds=60)
    assert time.monotonic() - started < 15     # not the grandchild's 60 s
    assert job["State"] == "failed", job
    msg = job["Steps"][1]["Message"]
    assert "exit 1" in msg and "c2-right (169.254.1.2): connection refused" in msg
    assert calls(fake, "upload_video_files") == []
    assert not recorder.request.exists()


# ---------------------------------------------------------------- resume
def test_continue_after_a_failed_recording_stays_failed(recorder, web):
    _, _, fake = web
    runner = recorder("camfail")
    assert runner.start(spec(), record_seconds=60)["State"] == "failed"
    job = runner.resume()
    assert job["State"] == "failed", job
    assert job["Steps"][1]["State"] == "failed"
    assert "start a new run to record again" in job["Steps"][1]["Message"]
    assert len(recorder.seen()) == 1
    assert calls(fake, "upload_video_files") == []      # never a partial or an old set


def test_continue_after_a_restart_does_not_record_again(recorder, web):
    _, _, fake = web
    job = recorder("ok").start(spec(floorplan=False), record_seconds=60)  # recorded, then waits for the layout
    assert job["State"] == "waiting_for_user"
    assert len(recorder.seen()) == 1

    runner2 = recorder("ok")                                             # the service restarted
    assert runner2.status()["JobId"] == job["JobId"]
    assert runner2._job.record_seconds == 60 and runner2._job.footage_recorded is True
    fake.projects[WEB_AMC_ID]["layout"] = PNG
    assert runner2.resume()["State"] == "completed"
    assert len(recorder.seen()) == 1
    assert len(calls(fake, "upload_video_files")) == 1


def test_interrupted_recording_stays_failed_on_continue(recorder, web):
    _, _, fake = web
    runner = recorder("ok")
    runner._launch = lambda resuming=False: None        # the service dies during the recording
    runner.start(spec(), record_seconds=60)
    state = json.loads(runner.state_path.read_text())
    state["steps"][1]["State"] = "running"
    runner.state_path.write_text(json.dumps(state))

    runner2 = recorder("ok")
    assert runner2.status()["State"] == "interrupted"
    job = runner2.resume()
    assert job["State"] == "failed" and "start a new run" in job["Steps"][1]["Message"]
    assert recorder.seen() == []
    assert calls(fake, "upload_video_files") == []


def test_state_file_from_before_recording_still_loads(recorder, web):
    _, _, fake = web
    runner = recorder("ok")
    job = runner.start(spec(floorplan=False))
    assert job["State"] == "waiting_for_user"
    state = json.loads(runner.state_path.read_text())
    del state["record_seconds"], state["footage_recorded"]
    runner.state_path.write_text(json.dumps(state))

    runner2 = recorder("ok")
    assert runner2.status()["JobId"] == job["JobId"]
    assert runner2._job.record_seconds == 0 and runner2._job.footage_recorded is False
    fake.projects[WEB_AMC_ID]["layout"] = PNG
    assert runner2.resume()["State"] == "completed"
    assert recorder.seen() == []


# ---------------------------------------------------------------- start validation
def test_record_outside_web_project_mode_is_refused(ws):
    runner, _ = make_runner(ws, FULLY_AUTOMATED)
    with pytest.raises(SpecError, match="web_project"):
        runner.start(spec(), record_seconds=60)
    assert runner.status() is None


@pytest.mark.parametrize("seconds", [29, 1801, True, 60.0])
def test_record_seconds_out_of_range_is_refused_by_the_runner(recorder, seconds):
    runner = recorder("ok")
    with pytest.raises(SpecError, match="30 to 1800"):
        runner.start(spec(), record_seconds=seconds)
    assert runner.status() is None


def test_record_through_the_api(recorder):
    client = make_client(recorder("ok"))
    r = client.post("/amc/automation/start", json={"Spec": make_spec(), "Record": {"Seconds": 30}})
    assert r.status_code == 202 and r.json()["State"] == "completed", r.json()
    assert recorder.seen()[0]["request"]["seconds"] == 30
    assert "recorded 2 clips of 0:30" in r.json()["Steps"][1]["Message"]


def test_record_outside_web_project_mode_is_a_400_through_the_api(api):
    client, _ = api
    r = client.post("/amc/automation/start", json={"Spec": make_spec(), "Record": {"Seconds": 300}})
    assert r.status_code == 400 and "web_project" in r.json()["detail"]
    assert client.get("/amc/automation/status").json() is None


@pytest.mark.parametrize("record", [{"Seconds": 29}, {"Seconds": 1801}, {"Seconds": "300"}, {"Seconds": 300.5},
                                    {"Seconds": True}, {}, {"Seconds": None}, 300])
def test_invalid_record_is_a_400_through_the_api(recorder, record):
    runner = recorder("ok")
    client = make_client(runner)
    r = client.post("/amc/automation/start", json={"Spec": make_spec(), "Record": record})
    assert r.status_code == 400, r.json()
    assert r.json()["detail"].startswith("invalid start request: Record")
    assert runner.status() is None and recorder.seen() == []


def test_record_null_is_the_same_as_absent(recorder):
    client = make_client(recorder("ok"))
    r = client.post("/amc/automation/start", json={"Spec": make_spec(), "Record": None})
    assert r.status_code == 202 and r.json()["State"] == "completed"
    assert recorder.seen() == []


def test_a_start_body_without_spec_is_a_400_too(api):
    client, _ = api
    r = client.post("/amc/automation/start", json={})
    assert r.status_code == 400 and r.json()["detail"].startswith("invalid start request: Spec")


# ---------------------------------------------------------------- the command line
def test_record_argv_uses_sudo_unless_amc_use_sudo_is_0(monkeypatch, tmp_path):
    monkeypatch.delenv("AMC_USE_SUDO", raising=False)
    binary = str(tmp_path / "bin" / "mv3dt-installer")
    assert automation_mod.default_record_argv(tmp_path) == ["sudo", "-n", binary, "record-footage", "--pending"]
    monkeypatch.setenv("AMC_USE_SUDO", "0")
    assert automation_mod.default_record_argv(tmp_path) == [binary, "record-footage", "--pending"]
    runner = CalibrationRunner(FakeController(tmp_path), store=ProjectStore(tmp_path / "p"))
    assert runner._record_argv() == [binary, "record-footage", "--pending"]
    assert runner.record_request_path == tmp_path / "amc_control" / "state" / "record-request.json"
