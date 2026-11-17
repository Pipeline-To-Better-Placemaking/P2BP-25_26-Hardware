"""Recording footage from step 2 of a web_project run (AMC-FOOTAGE-RECORDING.md section 6).

A small script stands in for `mv3dt-installer record-footage --pending`,
following section 5's stdout and exit-code contract; it logs what it saw of
the request file on every call. Run from installer/:
`python3 -m pytest tests/amc_control -v`
"""
import json
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
import json, os, shutil, signal, stat, sys, time
mode, request, footage, log = sys.argv[1:5]
st = os.stat(request)
with open(request) as fh:
    req = json.load(fh)
with open(log, "a") as fh:
    fh.write(json.dumps({"mode": stat.S_IMODE(st.st_mode), "request": req}) + "\n")

def out(obj):
    print(json.dumps(obj), flush=True)

seconds = req["seconds"]
if mode == "ok":
    os.makedirs(os.path.join(footage, "previous"), exist_ok=True)
    for name in os.listdir(footage):
        if name.endswith(".mp4"):
            shutil.move(os.path.join(footage, name), os.path.join(footage, "previous", name))
    print("not json", flush=True)
    for f in (0.0, 0.5, 1.0):
        out({"progress": f})
    for name in ("c1-left.mp4", "c2-right.mp4"):
        with open(os.path.join(footage, name), "wb") as fh:
            fh.write(b"recorded")
    out({"ok": True, "cancelled": False, "files": ["c1-left.mp4", "c2-right.mp4"], "failed": [],
         "seconds": seconds})
    sys.exit(0)
if mode == "camfail":
    out({"progress": 1.0})
    out({"ok": False, "cancelled": False, "files": ["c1-left.mp4"],
         "failed": [{"id": "c2-right", "ip": "169.254.1.2", "error": "connection refused"}], "seconds": seconds})
    sys.exit(1)
if mode == "invalid":
    print("record-footage: the request is more than 300 seconds old", file=sys.stderr, flush=True)
    out({"ok": False, "cancelled": False, "files": [], "failed": [], "seconds": seconds})
    sys.exit(2)
if mode == "nofinal":
    print("sudo: a password is required", file=sys.stderr, flush=True)
    sys.exit(1)
if mode == "selfcancel":
    out({"ok": False, "cancelled": True, "files": [], "failed": [], "seconds": seconds})
    sys.exit(3)
if mode == "hang":
    def stop(signum, frame):
        with open(log + ".sigterm", "w") as fh:
            fh.write(str(signum))
        out({"ok": False, "cancelled": True, "files": [], "failed": [], "seconds": seconds})
        sys.exit(3)
    signal.signal(signal.SIGTERM, stop)
    out({"progress": 0.25})
    while True:
        time.sleep(0.05)
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
        return [json.loads(line) for line in log.read_text().splitlines()]

    make.seen = seen
    make.sigterm = log.parent / (log.name + ".sigterm")
    make.request = install / "amc_control" / "state" / "record-request.json"
    return make


def wait_for_message(runner, prefix):
    deadline = time.time() + 20
    while not any(m.startswith(prefix) for m in runner.messages):
        assert time.time() < deadline, runner.messages
        time.sleep(0.05)


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
    assert "exit 1" in msg and "c2-right (169.254.1.2): connection refused" in msg
    assert calls(fake, "upload_video_files") == []
    assert not recorder.request.exists()
    assert json.loads((folder / "runs" / f"{job['JobId']}.json").read_text())["State"] == "failed"


def test_recording_refused_request_reports_the_recorders_diagnostics(recorder, web):
    _, _, fake = web
    job = recorder("invalid").start(spec(), record_seconds=60)
    assert job["State"] == "failed"
    msg = job["Steps"][1]["Message"]
    assert "exit 2" in msg and "more than 300 seconds old" in msg
    assert calls(fake, "upload_video_files") == []
    assert not recorder.request.exists()


def test_recording_without_a_result_line_reports_the_exit_code_and_stderr(recorder):
    job = recorder("nofinal").start(spec(), record_seconds=60)
    assert job["State"] == "failed"
    msg = job["Steps"][1]["Message"]
    assert "exited 1 without a result" in msg and "sudo: a password is required" in msg
    assert not recorder.request.exists()


def test_recorder_that_cannot_start_fails_step_2(recorder, tmp_path):
    runner = recorder("ok")
    runner._record_argv = lambda: [str(tmp_path / "no-such-binary"), "record-footage", "--pending"]
    job = runner.start(spec(), record_seconds=60)
    assert job["State"] == "failed" and "could not run the footage recorder" in job["Steps"][1]["Message"]
    assert not recorder.request.exists()


# ---------------------------------------------------------------- cancel
def test_recorder_exit_3_ends_the_run_cancelled(recorder, web):
    _, _, fake = web
    job = recorder("selfcancel").start(spec(), record_seconds=60)
    assert job["State"] == "cancelled"
    assert calls(fake, "upload_video_files") == []
    assert not recorder.request.exists()


def test_cancel_during_recording_sends_sigterm_and_cancels_the_run(recorder, web):
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
    assert recorder.sigterm.read_text() == str(int(signal.SIGTERM))
    assert not recorder.request.exists()
    assert calls(fake, "upload_video_files") == []
    assert json.loads((folder / "runs" / f"{job['JobId']}.json").read_text())["State"] == "cancelled"


def test_recorder_ignoring_sigterm_is_killed_after_the_grace_period(recorder):
    runner = recorder("stubborn")
    runner.record_term_grace = 0.2
    runner._threaded = True
    runner.start(spec(), record_seconds=60)
    wait_for_message(runner, "recording 0:06 of 1:00")
    runner.cancel()
    runner.join(20)
    assert runner.status()["State"] == "cancelled"
    assert not recorder.request.exists()


# ---------------------------------------------------------------- resume
def test_continue_after_a_failed_recording_does_not_record_again(recorder, web):
    _, _, fake = web
    runner = recorder("camfail")
    assert runner.start(spec(), record_seconds=60)["State"] == "failed"
    job = runner.resume()
    assert job["State"] == "completed", job
    assert len(recorder.seen()) == 1
    # The footage already in the folder is uploaded instead.
    assert calls(fake, "upload_video_files") == [{"names": ["cam_00.mp4", "cam_01.mp4"]}]


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


def test_interrupted_run_does_not_record_on_continue(recorder, web):
    _, _, fake = web
    runner = recorder("ok")
    runner._launch = lambda resuming=False: None        # the service dies right after the start
    runner.start(spec(), record_seconds=60)
    state = json.loads(runner.state_path.read_text())
    state["steps"][1]["State"] = "running"
    runner.state_path.write_text(json.dumps(state))

    runner2 = recorder("ok")
    assert runner2.status()["State"] == "interrupted"
    assert runner2.resume()["State"] == "completed"
    assert recorder.seen() == []
    assert calls(fake, "upload_video_files") == [{"names": ["cam_00.mp4", "cam_01.mp4"]}]


def test_state_file_from_before_recording_still_loads(recorder, web):
    _, _, fake = web
    job = recorder("ok").start(spec(floorplan=False))
    assert job["State"] == "waiting_for_user"
    state_path = recorder("ok").state_path
    state = json.loads(state_path.read_text())
    del state["record_seconds"], state["footage_recorded"]
    state_path.write_text(json.dumps(state))

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
