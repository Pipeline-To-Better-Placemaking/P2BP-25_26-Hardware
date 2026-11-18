"""mv3dt_amc_control automation, run against the fake AMC API (no GPU, Docker or network).

Run from installer/: `python3 -m pytest tests/amc_control -v`
"""
import hashlib
import io
import json
import sys
import time

import pytest

jwt = pytest.importorskip("jwt")
httpx = pytest.importorskip("httpx")
yaml = pytest.importorskip("yaml")
pytest.importorskip("fastapi")
pytest.importorskip("multipart")  # python-multipart, used by the fake AMC
pytest.importorskip("PIL")

from fastapi.testclient import TestClient  # noqa: E402

from amc_control_fake_api import DEFAULT_CONFIG, FakeAmc  # noqa: E402
from mv3dt_amc_control import automation as automation_mod  # noqa: E402
from mv3dt_amc_control.api import create_app  # noqa: E402
from mv3dt_amc_control.automation import CalibrationRunner  # noqa: E402
from mv3dt_amc_control.automation_config import ConfigError, load_config  # noqa: E402
from mv3dt_amc_control.calibration_spec import SPEC_SCOPE, SpecError, verify_spec  # noqa: E402
from mv3dt_amc_control.projects import STATE_FAILED, STATE_READY, ProjectStore  # noqa: E402
from mv3dt_amc_control.settings import Settings  # noqa: E402

TICKET_SECRET = "ticket-secret-ticket-secret-ticket-secret"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
FLOORPLAN_URL = "https://abc.supabase.co/storage/v1/object/sign/media/fp/new.png?token=t"


def make_spec(pid="proj-1", floorplan=True, px_per_m=100.0, url=FLOORPLAN_URL, scope=SPEC_SCOPE,
              secret=TICKET_SECRET, exp_delta=900, folder=None):
    spec = {"title": "Living Room", "floorplan": None}
    if folder is not None:
        spec["folder"] = folder
    if floorplan:
        spec["floorplan"] = {"id": "fp-1", "url": url, "contentType": "image/png", "width": 741, "height": 502,
                             "pxPerM": px_per_m}
    claims = {"sub": "user-1", "pid": pid, "scope": scope, "aud": "p2bp-workstation", "spec": spec,
              "exp": int(time.time()) + exp_delta}
    return jwt.encode(claims, secret, algorithm="HS256")


def make_ticket(pid="proj-1"):
    claims = {"sub": "user-1", "pid": pid, "scope": "amc", "aud": "p2bp-workstation",
              "exp": int(time.time()) + 900}
    return jwt.encode(claims, TICKET_SECRET, algorithm="HS256")


class FakeController:
    def __init__(self, install_dir):
        self.install_dir = install_dir
        self.state_dir = install_dir / "amc_control" / "state"
        self.started = 0
        self.stopped = 0

    def stop(self):
        self.stopped += 1

    def status(self):
        return {"State": "down", "UiUrl": "", "Error": "", "UpdatedAt": 0}

    def ports(self):
        return 18000, 15000

    def start(self):
        self.started += 1


@pytest.fixture
def ws(tmp_path):
    """An install tree with installer.conf, recorded footage and a saved alignment."""
    install = tmp_path / "opt-mv3dt"
    amc_root = tmp_path / "auto-magic-calib"
    footage = tmp_path / "footage" / "Living-Room"
    footage.mkdir(parents=True)
    for name in ("cam1-right.mp4", "cam0-left.mp4", "cam2-door.mp4.part"):
        (footage / name).write_bytes(b"video")
    install.mkdir()
    (install / "installer.conf").write_text(
        "AUTO_MAGIC_CALIB_MS_PORT=18000\nAUTO_MAGIC_CALIB_UI_PORT=15000\n"
        f"AMC_PROJECT_ID=20261019_151133_2885\nPROJECT_NAME=Living-Room\nAMC_ROOT={amc_root}\n"
        f"CALIB_FOOTAGE_DIR={tmp_path / 'footage'}\n"
    )
    align_dir = amc_root / "projects" / "project_20261019_151133_2885" / "manual_adjustment"
    align_dir.mkdir(parents=True)
    (align_dir / "alignment_data.json").write_text(json.dumps([[[1, 2], [3, 4], [5, 6]]]))
    return tmp_path, install


def write_config(tmp_path, overrides):
    path = tmp_path / "amc_automation.yaml"
    path.write_text(yaml.safe_dump(overrides))
    return path


FULLY_AUTOMATED = {
    "layout": {"allowed_hosts": ["abc.supabase.co"]},
    "parameters": {"settings_file": "settings.json", "settings_overrides": {"detector": "transformer"},
                   "focal_lengths": [1269.0, 1099.5]},
    "rectification": {"mode": "skip"},
    "calibration": {"poll_seconds": 0},
    "results": {"download_dir": "results"},
}


def make_runner(ws, overrides, fake=None, floorplan_bytes=PNG, download_status=200, store=None):
    tmp_path, install = ws
    fake = fake or FakeAmc()
    if "20261019_151133_2885" not in fake.projects:
        fake.add_project("20261019_151133_2885", "Living-Room")
    (tmp_path / "settings.json").write_text(json.dumps({"layout_px_per_m": 1.0, "detector": "resnet",
                                                        "bundle_adjustment": {"iterations": 10}}))
    cfg_path = write_config(tmp_path, overrides)
    downloads = []

    def serve_floorplan(request):
        downloads.append(str(request.url))
        return httpx.Response(download_status, content=floorplan_bytes)

    # Sleeping moves the runner's clock on instead of waiting, so polls with the default
    # poll_seconds (rectification's is 10) finish at once while real time still counts.
    skew = [0.0]

    def sleep(seconds):
        skew[0] += seconds

    runner = CalibrationRunner(
        FakeController(install),
        config_loader=lambda: load_config(cfg_path),
        http_client_factory=lambda: TestClient(fake.app()),
        download_client_factory=lambda: httpx.Client(transport=httpx.MockTransport(serve_floorplan)),
        sleep=sleep,
        clock=lambda: time.time() + skew[0],
        threaded=False,
        store=store if store is not None else ProjectStore(tmp_path / "amc_projects"),
    )
    runner.downloads = downloads
    return runner, fake


def calls(fake, name):
    return [kw for n, kw in fake.calls if n == name]


def spec(**kw):
    return verify_spec(make_spec(**kw), TICKET_SECRET, kw.get("pid", "proj-1"))


# ---------------------------------------------------------------- full automatic run
def test_fully_configured_run_needs_no_user(ws):
    runner, fake = make_runner(ws, FULLY_AUTOMATED)
    job = runner.start(spec())

    assert job["State"] == "completed", job
    assert [s["State"] for s in job["Steps"]] == ["done", "done", "done", "skipped", "done", "done", "done"]
    assert job["UserActions"] == []
    assert job["AmcProjectId"] == "20261019_151133_2885"   # recalibrates the installer's project

    # Step 2: sorted footage, renamed for AMC, .part files ignored; floor plan uploaded as layout.png.
    assert calls(fake, "upload_video_files") == [{"names": ["cam_00.mp4", "cam_01.mp4"]}]
    assert "cam0-left.mp4 as cam_00.mp4" in job["Steps"][1]["Message"]
    assert calls(fake, "upload_layout") == [{"filename": "layout.png", "png": True}]
    assert runner.downloads == [FLOORPLAN_URL]
    # Step 3: settings file + override + scale from the database, focal lengths.
    settings = calls(fake, "config")[0]["settings"]
    assert settings["layout_px_per_m"] == 100.0 and settings["detector"] == "transformer"
    assert settings["bundle_adjustment"] == {"iterations": 10}
    assert calls(fake, "upload_focal_length") == [{"values": [1269.0, 1099.5]}]
    # Step 5: saved alignment re-uploaded in rectified coordinates. Step 6: detector from the config.
    assert calls(fake, "upload_alignment") == [{"coord_space": "rectified"}]
    assert calls(fake, "calibrate") == [{"detector_type": "resnet"}]
    # Step 7: result ZIP saved.
    assert job["Result"]["ResultFile"].endswith(".zip")
    assert (ws[0] / "results").exists()


def test_order_of_calls_matches_amcs_workflow(ws):
    runner, fake = make_runner(ws, FULLY_AUTOMATED)
    runner.start(spec())
    order = [n for n, _ in fake.calls]
    assert order.index("upload_video_files") < order.index("upload_layout") < order.index("config")
    assert order.index("config") < order.index("upload_alignment") < order.index("verify_project")
    assert order.index("verify_project") < order.index("calibrate") < order.index("mv3dt_result")


# ---------------------------------------------------------------- default config pauses for the UI
def test_default_config_pauses_once_with_a_todo_list_then_finishes(ws):
    tmp_path, _ = ws
    (tmp_path / "auto-magic-calib").rename(tmp_path / "moved")  # no saved alignment
    runner, fake = make_runner(ws, {"layout": {"allowed_hosts": ["abc.supabase.co"]},
                                    "calibration": {"poll_seconds": 0}})
    job = runner.start(spec())

    assert job["State"] == "waiting_for_user"
    states = {s["Key"]: s["State"] for s in job["Steps"]}
    assert states == {"project_setup": "done", "video_configuration": "done", "parameters": "done",
                      "rectification": "done", "manual_alignment": "waiting", "execute": "pending",
                      "results": "pending"}
    todo = " | ".join(job["UserActions"])
    assert "px/m" not in todo and "Rectification" not in todo and "Manual Alignment" in todo
    # The known scale is posted on AMC's current config, even without a settings_file.
    assert calls(fake, "config")[0]["settings"] == {**DEFAULT_CONFIG, "layout_px_per_m": 100.0}
    assert len(calls(fake, "start_rectification")) == 1   # rectification is automatic by default
    assert calls(fake, "calibrate") == []          # nothing calibrates before the user is done
    assert calls(fake, "upload_video_files")       # but everything that could be uploaded was

    fake.ui_alignment_done = True                  # the user finishes steps 3-5 in AMC
    job = runner.resume()
    assert job["State"] == "completed"
    assert calls(fake, "calibrate") == [{"detector_type": "resnet"}]


def test_continue_before_the_ui_work_is_saved_pauses_again(ws):
    tmp_path, _ = ws
    (tmp_path / "auto-magic-calib").rename(tmp_path / "moved")
    runner, fake = make_runner(ws, {"layout": {"enabled": False}, "rectification": {"mode": "ui"},
                                    "parameters": {"scale": {"source": "none"}}})
    assert runner.start(spec(floorplan=False))["State"] == "waiting_for_user"
    job = runner.resume()                          # user pressed Continue too early
    assert job["State"] == "waiting_for_user"
    # Every step left to the user is checked against AMC and listed, not marked done.
    states = {s["Key"]: s["State"] for s in job["Steps"]}
    assert states["rectification"] == "waiting" and states["manual_alignment"] == "waiting"
    assert len(job["UserActions"]) == 2
    assert job["UserActions"][0].endswith("until it shows COMPLETED; AMC reports NOT_STARTED")
    assert job["UserActions"][1].endswith("AMC reports no saved alignment yet")
    assert "step 4 Rectification, step 5 Manual Alignment" in job["Message"]
    assert calls(fake, "verify_project") == [] and calls(fake, "calibrate") == []
    assert calls(fake, "config") == [] and calls(fake, "get_config") == []   # scale.source none

    fake.projects[INSTALLER_AMC_ID]["rect_state"] = "COMPLETED"   # the user finishes both in AMC
    fake.ui_alignment_done = True
    job = runner.resume()
    assert job["State"] == "completed", job
    assert job["Steps"][3]["Message"].startswith("finished in AMC")
    assert job["Steps"][4]["Message"].startswith("finished in AMC")


def test_video_configuration_stays_waiting_until_amc_has_the_layout(ws):
    runner, fake = make_runner(ws, FULLY_AUTOMATED)
    assert runner.start(spec(floorplan=False))["State"] == "waiting_for_user"
    fake.config["layout_px_per_m"] = 50.0                 # step 3 is confirmed; step 2 is not
    job = runner.resume()
    assert job["State"] == "waiting_for_user"
    assert job["Steps"][1]["State"] == "waiting" and job["Steps"][2]["State"] == "done"
    assert job["UserActions"] == ["AMC reports 2 videos and no layout.png: upload layout.png in AMC's Video "
                                  "Configuration step, then press Continue"]
    assert "cam_00.mp4" in job["Steps"][1]["Done"]        # what was done is kept
    fake.projects[INSTALLER_AMC_ID]["layout"] = PNG
    assert runner.resume()["State"] == "completed"


def test_saved_alignment_waits_for_ui_rectification_before_upload(ws):
    runner, fake = make_runner(ws, {**FULLY_AUTOMATED, "rectification": {"mode": "ui"}})
    job = runner.start(spec())
    assert job["State"] == "waiting_for_user"
    assert calls(fake, "upload_alignment") == []
    assert len(job["UserActions"]) == 1 and "Rectification" in job["UserActions"][0]  # alignment needs nothing
    fake.projects[INSTALLER_AMC_ID]["rect_state"] = "COMPLETED"   # the user rectified in AMC's UI
    job = runner.resume()
    assert job["State"] == "completed"
    assert calls(fake, "upload_alignment") == [{"coord_space": "rectified"}]


# ---------------------------------------------------------------- data from the database
def test_project_without_floor_plan_asks_for_the_layout_in_amc(ws):
    runner, fake = make_runner(ws, {**FULLY_AUTOMATED})
    job = runner.start(spec(floorplan=False))
    assert job["State"] == "waiting_for_user"
    assert job["UserActions"][0].startswith("Upload layout.png")
    assert "cam_00.mp4" in job["Steps"][1]["Done"]
    assert calls(fake, "upload_layout") == []


def test_jpeg_floor_plan_is_converted_to_png(ws):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 4), "white").save(buf, format="JPEG")
    runner, fake = make_runner(ws, FULLY_AUTOMATED, floorplan_bytes=buf.getvalue())
    assert runner.start(spec())["State"] == "completed"
    assert calls(fake, "upload_layout")[0]["png"] is True


def test_floor_plan_is_only_downloaded_from_allowed_hosts(ws):
    runner, _ = make_runner(ws, FULLY_AUTOMATED)
    job = runner.start(spec(url="https://evil.example/fp.png"))
    assert job["State"] == "failed"
    assert "not in layout.allowed_hosts" in job["Steps"][1]["Message"]
    assert runner.downloads == []


def test_scale_key_missing_from_amcs_config_fails_instead_of_being_invented(ws):
    tmp_path, _ = ws
    fake = FakeAmc(config={"detector": "resnet"})
    runner, _ = make_runner(ws, FULLY_AUTOMATED, fake=fake)
    (tmp_path / "settings.json").write_text(json.dumps({"detector": "resnet"}))
    job = runner.start(spec())
    assert job["State"] == "failed"
    assert "AMC's configuration has no layout_px_per_m" in job["Steps"][2]["Message"]
    assert calls(fake, "config") == []


def test_fixed_scale_overrides_the_database(ws):
    runner, fake = make_runner(ws, {**FULLY_AUTOMATED,
                                    "parameters": {**FULLY_AUTOMATED["parameters"],
                                                   "scale": {"source": "fixed", "fixed_px_per_m": 42}}})
    runner.start(spec())
    assert calls(fake, "config")[0]["settings"]["layout_px_per_m"] == 42


# ---------------------------------------------------------------- other configured behaviour
def test_create_new_mode_names_the_project_from_the_template(ws):
    runner, fake = make_runner(ws, {**FULLY_AUTOMATED, "project": {"mode": "create_new",
                                                                    "name_template": "{title}-{project_id}"},
                                    "alignment": {"file": ""}})
    job = runner.start(spec())
    assert calls(fake, "create_project") == [{"name": "Living Room-proj-1"}]
    assert job["AmcProjectId"] != "20261019_151133_2885"


def test_rectification_api_mode_follows_the_configured_requests(ws):
    rect = {"mode": "api", "api": {
        "start": {"method": "POST", "path": "/test_rectify/{project_id}", "json": {"mode": "auto"}},
        "status": {"method": "GET", "path": "/test_rectify/{project_id}/status", "state_field": "rectification.state"},
        "poll_seconds": 0}}
    runner, fake = make_runner(ws, {**FULLY_AUTOMATED, "rectification": rect})
    job = runner.start(spec())
    assert job["State"] == "completed"
    assert calls(fake, "rectify") == [{"body": {"mode": "auto"}}]
    assert job["Steps"][3]["Message"] == "rectification COMPLETED"


def test_rectification_api_mode_without_paths_fails_clearly(ws):
    runner, _ = make_runner(ws, {**FULLY_AUTOMATED, "rectification": {"mode": "api"}})
    job = runner.start(spec())
    assert job["State"] == "failed" and "openapi.yaml" in job["Steps"][3]["Message"]


def test_amc_calibration_error_reports_the_log(ws):
    fake = FakeAmc(fail_calibration=True)
    runner, _ = make_runner(ws, FULLY_AUTOMATED, fake=fake)
    job = runner.start(spec())
    assert job["State"] == "failed"
    assert "not enough tracklets" in job["Steps"][5]["Message"]


# ---------------------------------------------------------------- step 6: AMC's calibration states
INSTALLER_AMC_ID = "20261019_151133_2885"
SLOW_POLLS = {**FULLY_AUTOMATED, "calibration": {"poll_seconds": 1, "timeout_seconds": 3}}


def calibration_reads(fake):
    return [(kw["project_state"], kw["amc_state"]) for kw in calls(fake, "get_project_info")]


def test_calibration_finishes_on_amc_state_completed_with_project_state_back_to_ready(ws):
    # AMC 3.3.0: project_state goes RUNNING then back to READY; only amc_state says COMPLETED.
    runner, fake = make_runner(ws, FULLY_AUTOMATED)
    job = runner.start(spec())
    assert job["State"] == "completed", job
    assert calibration_reads(fake)[-1] == ("READY", "COMPLETED")
    assert ("RUNNING", "RUNNING") in calibration_reads(fake)
    assert job["Steps"][5]["Message"].startswith("calibrated with resnet in about 0 min; scale 100 px/m")
    assert job_file(runner)["calibration_started"] is True


def test_stale_completed_from_an_earlier_calibration_does_not_finish_step_6(ws):
    # The reused project still reports its earlier COMPLETED calibration for a few reads.
    fake = FakeAmc(calib_stale_reads=3)
    runner, _ = make_runner(ws, FULLY_AUTOMATED, fake=fake)
    job = runner.start(spec())
    assert job["State"] == "completed", job
    reads = calibration_reads(fake)
    first = reads.index(("RUNNING", "RUNNING"))
    assert reads[first - 3:first] == [("READY", "COMPLETED")] * 3   # all polled through
    assert calls(fake, "calibrate") == [{"detector_type": "resnet"}]


def test_stale_completed_beyond_the_lag_window_finishes_the_step(ws):
    # Without RUNNING ever seen, COMPLETED counts once the lag window is over.
    fake = FakeAmc(calib_stale_reads=10)
    runner, _ = make_runner(ws, FULLY_AUTOMATED, fake=fake)
    job = runner.start(spec())
    assert job["State"] == "completed", job
    assert ("RUNNING", "RUNNING") not in calibration_reads(fake)


def test_calibration_error_in_amc_state_fails_with_the_log(ws):
    runner, fake = make_runner(ws, FULLY_AUTOMATED, fake=FakeAmc(fail_calibration=True))
    job = runner.start(spec())
    assert job["State"] == "failed"
    message = job["Steps"][5]["Message"]
    assert "ended in ERROR (project_state READY, amc_state ERROR)" in message
    assert "not enough tracklets" in message


def test_calibration_stopped_without_finishing_names_both_states(ws):
    runner, _ = make_runner(ws, FULLY_AUTOMATED, fake=FakeAmc(calibration_end="READY"))
    job = runner.start(spec())
    assert job["State"] == "failed"
    message = job["Steps"][5]["Message"]
    assert "stopped without finishing (project_state READY, amc_state READY)" in message
    assert "calibration log" in message


def test_calibration_timeout_fails_and_shows_it_was_still_running(ws):
    runner, fake = make_runner(ws, SLOW_POLLS, fake=FakeAmc(running_polls=10 ** 9))
    job = runner.start(spec())
    assert job["State"] == "failed"
    assert "calibration still running after 0 min (project_state RUNNING, amc_state RUNNING)" \
        in job["Steps"][5]["Message"]


def test_running_step_message_reads_calibrating_not_ready(ws):
    seen = []
    runner, fake = make_runner(ws, {**FULLY_AUTOMATED, "calibration": {"poll_seconds": 1}})
    set_step = runner._set_step

    def spy(key, state, message="", *a, **kw):
        if key == "execute" and state == "running" and message:
            seen.append(message)
        return set_step(key, state, message, *a, **kw)

    runner._set_step = spy
    assert runner.start(spec())["State"] == "completed"
    assert seen and all(m.startswith("calibrating for 0 min (typically 10-60 min)") for m in seen)


def interrupt_at_step_6(ws, fake):
    """Turn the saved job into one cut off by a restart during step 6, and reload it."""
    _, install = ws
    state = install / "amc_control" / "state" / "amc_automation_job.json"
    data = json.loads(state.read_text())
    data["state"] = "running"
    data["steps"][5]["State"] = "running"
    state.write_text(json.dumps(data))
    runner2, _ = make_runner(ws, FULLY_AUTOMATED, fake=fake)
    assert runner2.status()["State"] == "interrupted"
    return runner2


def test_continue_after_a_restart_while_amc_still_calibrates_polls_without_a_second_post(ws):
    fake = FakeAmc(running_polls=10 ** 9)
    runner, _ = make_runner(ws, SLOW_POLLS, fake=fake)
    assert runner.start(spec())["State"] == "failed"     # timed out; AMC keeps calibrating
    runner2 = interrupt_at_step_6(ws, fake)
    fake.running_polls = fake.projects[INSTALLER_AMC_ID]["polls"] + 2
    job = runner2.resume()
    assert job["State"] == "completed", job
    assert len(calls(fake, "calibrate")) == 1
    assert len(calls(fake, "verify_project")) == 1
    assert job["Steps"][5]["Message"].startswith("calibrated with resnet")


def test_continue_after_amc_finished_unwatched_is_done_without_a_post(ws):
    fake = FakeAmc(running_polls=10 ** 9)
    runner, _ = make_runner(ws, SLOW_POLLS, fake=fake)
    assert runner.start(spec())["State"] == "failed"
    p = fake.projects[INSTALLER_AMC_ID]
    p["state"], p["amc_state"] = "READY", "COMPLETED"   # AMC finished while nobody watched
    job = interrupt_at_step_6(ws, fake).resume()
    assert job["State"] == "completed", job
    assert len(calls(fake, "calibrate")) == 1 and len(calls(fake, "verify_project")) == 1
    assert job["Steps"][5]["Message"] == "the calibration had finished in AMC while the run was not watching"


def test_continue_after_a_failed_calibration_calibrates_again(ws):
    fake = FakeAmc(fail_calibration=True)
    runner, _ = make_runner(ws, FULLY_AUTOMATED, fake=fake)
    assert runner.start(spec())["State"] == "failed"
    fake.fail_calibration = False
    job = runner.resume()
    assert job["State"] == "completed", job
    assert len(calls(fake, "calibrate")) == 2 and len(calls(fake, "verify_project")) == 2


def test_a_job_file_from_before_calibration_started_still_loads(ws):
    _, install = ws
    runner, fake = make_runner(ws, FULLY_AUTOMATED)
    runner.start(spec())
    state = install / "amc_control" / "state" / "amc_automation_job.json"
    data = json.loads(state.read_text())
    del data["calibration_started"]
    state.write_text(json.dumps(data))
    runner2, _ = make_runner(ws, FULLY_AUTOMATED, fake=fake)
    assert runner2.status()["State"] == "completed"
    assert runner2._job.calibration_started is False


def test_project_calibrating_in_amc_state_only_is_refused(ws):
    fake = FakeAmc(running_polls=100)
    fake.add_project(INSTALLER_AMC_ID, "Living-Room")
    p = fake.projects[INSTALLER_AMC_ID]
    p["state"], p["amc_state"] = "READY", "RUNNING"
    runner, _ = make_runner(ws, FULLY_AUTOMATED, fake=fake)
    job = runner.start(spec())
    assert job["State"] == "failed" and "already calibrating" in job["Steps"][0]["Message"]
    assert calls(fake, "calibrate") == []


def test_no_footage_fails_with_a_hint(ws):
    tmp_path, _ = ws
    for f in (tmp_path / "footage" / "Living-Room").iterdir():
        f.unlink()
    runner, _ = make_runner(ws, FULLY_AUTOMATED)
    job = runner.start(spec())
    assert job["State"] == "failed" and "record it with installer step 4" in job["Steps"][1]["Message"]


def test_amc_down_is_started_first(ws):
    fake = FakeAmc(ready=False)
    runner, _ = make_runner(ws, {**FULLY_AUTOMATED, "amc": {"ready_timeout_seconds": 0}}, fake=fake)
    job = runner.start(spec())
    assert runner.controller.started == 1
    assert job["State"] == "failed" and "did not become ready" in job["Message"]


def test_evaluation_statistics_are_kept_when_ground_truth_exists(ws):
    runner, _ = make_runner(ws, FULLY_AUTOMATED, fake=FakeAmc(with_gt=True))
    job = runner.start(spec())
    assert job["Result"]["EvaluationStatistics"]["Average L2 distance(m)"] == 0.4


# ---------------------------------------------------------------- lifecycle
def test_one_run_at_a_time_and_a_paused_run_blocks_a_new_one(ws):
    tmp_path, _ = ws
    (tmp_path / "auto-magic-calib").rename(tmp_path / "moved")
    runner, _ = make_runner(ws, {"layout": {"allowed_hosts": ["abc.supabase.co"]}})
    runner.start(spec())
    with pytest.raises(automation_mod.Busy):
        runner.start(spec())
    assert runner.cancel()["State"] == "cancelled"
    assert runner.start(spec())["State"] == "waiting_for_user"  # allowed again after cancelling


def test_progress_survives_a_restart(ws):
    tmp_path, install = ws
    (tmp_path / "auto-magic-calib").rename(tmp_path / "moved")
    runner, fake = make_runner(ws, {"layout": {"allowed_hosts": ["abc.supabase.co"]},
                                    "calibration": {"poll_seconds": 0}})
    first = runner.start(spec())
    assert (install / "amc_control" / "state" / "amc_automation_job.json").exists()

    runner2, _ = make_runner(ws, {"layout": {"allowed_hosts": ["abc.supabase.co"]},
                                  "calibration": {"poll_seconds": 0}}, fake=fake)
    assert runner2.status()["JobId"] == first["JobId"]
    assert runner2.status()["State"] == "waiting_for_user"
    fake.ui_alignment_done = True
    assert runner2.resume()["State"] == "completed"


def test_a_run_cut_off_mid_step_is_marked_interrupted(ws):
    _, install = ws
    runner, fake = make_runner(ws, FULLY_AUTOMATED)
    runner.start(spec())
    state = install / "amc_control" / "state" / "amc_automation_job.json"
    data = json.loads(state.read_text())
    data["state"] = "running"
    data["steps"][5]["State"] = "running"
    state.write_text(json.dumps(data))
    runner2, _ = make_runner(ws, FULLY_AUTOMATED, fake=fake)
    job = runner2.status()
    assert job["State"] == "interrupted" and job["Steps"][5]["State"] == "pending"


# ---------------------------------------------------------------- config file
def test_override_file_only_changes_what_it_names(ws):
    cfg = load_config(write_config(ws[0], {"calibration": {"detector_type": "transformer"}}))
    assert cfg.calibration.detector_type == "transformer"
    assert cfg.calibration.timeout_seconds == 5400          # still from the base file
    assert cfg.project.mode == "reuse_installer_project"


def test_misspelled_setting_is_rejected(ws):
    with pytest.raises(ConfigError):
        load_config(write_config(ws[0], {"calibration": {"detecter_type": "resnet"}}))
    with pytest.raises(ConfigError):
        load_config(write_config(ws[0], {"calibration": {"detector_type": "yolo"}}))


def test_overrides_without_a_settings_file_merge_into_amcs_current_config(ws):
    fake = FakeAmc()
    fake.config["detector"] = "custom"
    runner, _ = make_runner(ws, {**FULLY_AUTOMATED, "parameters": {"settings_overrides": {"a": 1}}}, fake=fake)
    job = runner.start(spec())
    assert job["State"] == "completed", job
    assert calls(fake, "config")[0]["settings"] == {**DEFAULT_CONFIG, "detector": "custom", "a": 1,
                                                    "layout_px_per_m": 100.0}


def test_base_config_loads_on_its_own():
    cfg = load_config()
    assert cfg.rectification.mode == "auto" and cfg.alignment.file == "auto"


# ---------------------------------------------------------------- signed spec
@pytest.mark.parametrize("kw", [{"secret": "x" * 40}, {"scope": "amc"}, {"exp_delta": -5}])
def test_bad_specs_are_rejected(kw):
    with pytest.raises(SpecError):
        verify_spec(make_spec(**kw), TICKET_SECRET, "proj-1")


def test_spec_must_match_the_ticket_project():
    with pytest.raises(SpecError):
        verify_spec(make_spec(pid="proj-2"), TICKET_SECRET, "proj-1")


# ---------------------------------------------------------------- HTTP routes
SETTINGS = Settings(ticket_secret=TICKET_SECRET, allowed_origins=["https://app.example.org"])


def make_client(runner):
    client = TestClient(create_app(SETTINGS, controller=runner.controller, runner=runner))
    client.headers["Authorization"] = f"Bearer {make_ticket()}"
    return client


@pytest.fixture
def api(ws):
    runner, fake = make_runner(ws, FULLY_AUTOMATED)
    return make_client(runner), fake


def test_routes_run_a_calibration(api):
    client, _ = api
    assert client.get("/amc/automation/status").json() is None
    r = client.post("/amc/automation/start", json={"Spec": make_spec()})
    assert r.status_code == 202 and r.json()["State"] == "completed"
    assert client.get("/amc/automation/status").json()["State"] == "completed"
    assert client.get("/amc/automation/config").json()["calibration"]["detector_type"] == "resnet"


def test_routes_need_a_ticket(api):
    client, _ = api
    assert client.post("/amc/automation/start", json={"Spec": make_spec()},
                       headers={"Authorization": "Bearer junk"}).status_code == 401


def test_spec_for_another_project_is_a_400(api):
    client, _ = api
    assert client.post("/amc/automation/start", json={"Spec": make_spec(pid="proj-2")}).status_code == 400


def test_another_projects_run_is_hidden_and_cannot_be_touched(api):
    client, _ = api
    client.post("/amc/automation/start", json={"Spec": make_spec()})
    other = {"Authorization": f"Bearer {make_ticket('proj-2')}"}
    assert client.get("/amc/automation/status", headers=other).json() is None
    assert client.post("/amc/automation/cancel", headers=other).status_code == 403
    assert client.post("/amc/automation/continue", headers=other).status_code == 403


def test_continue_with_nothing_paused_is_a_409(api):
    client, _ = api
    client.post("/amc/automation/start", json={"Spec": make_spec()})
    assert client.post("/amc/automation/continue").status_code == 409


def test_stop_amc_is_refused_while_a_run_is_active(api, monkeypatch):
    client, _ = api
    runner = client.app.state.runner
    monkeypatch.setattr(runner, "uses_amc", lambda: True)
    assert client.post("/amc/stop").status_code == 409
    assert runner.controller.stopped == 0


def test_stop_amc_is_refused_while_a_run_waits_for_the_users_ui_work(ws, monkeypatch):
    tmp_path, _ = ws
    (tmp_path / "auto-magic-calib").rename(tmp_path / "moved")
    runner, _ = make_runner(ws, {"layout": {"allowed_hosts": ["abc.supabase.co"]}})
    client = make_client(runner)
    assert client.post("/amc/automation/start", json={"Spec": make_spec()}).json()["State"] == "waiting_for_user"
    assert client.post("/amc/stop").status_code == 409
    client.post("/amc/automation/cancel")
    assert client.post("/amc/stop").status_code == 202
    assert runner.controller.stopped == 1


def test_an_unexpected_error_fails_the_run_instead_of_leaving_it_stuck(ws, monkeypatch):
    runner, _ = make_runner(ws, FULLY_AUTOMATED)

    def boom(*a, **kw):
        raise KeyError("surprise")

    monkeypatch.setattr(runner, "_parameters", boom)
    job = runner.start(spec())
    assert job["State"] == "failed" and "surprise" in job["Steps"][2]["Message"]
    assert not runner.is_active()
    monkeypatch.undo()
    assert runner.start(spec())["State"] == "completed"   # a new run is allowed


# ---------------------------------------------------------------- project.mode web_project
WEB_AMC_ID = "20261117_101500_1234"
WEB = {**FULLY_AUTOMATED, "project": {"mode": "web_project"}}


@pytest.fixture
def web(ws):
    """A ready binding for proj-1 in folder living-room, with footage in its folder."""
    tmp_path, _ = ws
    store = ProjectStore(tmp_path / "amc_projects")
    store.create("proj-1", "living-room", "Living Room")
    store.update("living-room", amc_project_id=WEB_AMC_ID, amc_project_name="living-room", state=STATE_READY)
    folder = store.folder_path("living-room")
    for name in ("b.mp4", "a.mp4", "notes.txt"):
        (folder / "footage" / name).write_bytes(b"video")
    fake = FakeAmc()
    fake.add_project(WEB_AMC_ID, "living-room")
    return store, folder, fake


def amc_saved_alignment(ws, amc_id=WEB_AMC_ID):
    path = ws[0] / "auto-magic-calib" / "projects" / f"project_{amc_id}" / "manual_adjustment" / "alignment_data.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([[[9, 9], [9, 9], [9, 9]]]))
    return path


def web_runner(ws, web, overrides=WEB, **kw):
    store, _, fake = web
    return make_runner(ws, overrides, fake=fake, store=store, **kw)[0]


def test_web_project_mode_is_accepted_and_the_base_default_is_unchanged(ws):
    assert load_config(write_config(ws[0], {"project": {"mode": "web_project"}})).project.mode == "web_project"
    assert load_config().project.mode == "reuse_installer_project"


def test_web_project_run_uses_the_bound_amc_project_and_folder(ws, web):
    store, folder, fake = web
    amc_saved_alignment(ws)
    runner = web_runner(ws, web)
    job = runner.start(spec(folder="living-room"))

    assert job["State"] == "completed", job
    assert job["AmcProjectId"] == WEB_AMC_ID and job["AmcProjectName"] == "living-room"
    assert calls(fake, "create_project") == []
    assert "living-room" in job["Steps"][0]["Message"]
    # Step 2: footage from <folder>/footage/ with videos.pattern, floor plan kept in the folder.
    assert calls(fake, "upload_video_files") == [{"names": ["cam_00.mp4", "cam_01.mp4"]}]
    assert "a.mp4 as cam_00.mp4" in job["Steps"][1]["Message"]
    assert (folder / "floorplan" / "layout.png").read_bytes() == PNG
    assert fake.projects[WEB_AMC_ID]["layout"] == PNG


def test_web_project_follows_videos_order(ws, web):
    amc_saved_alignment(ws)
    runner = web_runner(ws, web, {**WEB, "videos": {"order": ["b.mp4", "a.mp4"]}})
    job = runner.start(spec())
    assert job["State"] == "completed"
    assert "b.mp4 as cam_00.mp4" in job["Steps"][1]["Message"]


@pytest.mark.parametrize("bind", [False, True])
def test_web_project_not_set_up_is_refused(ws, web, bind):
    store, _, _ = web
    runner = web_runner(ws, web)
    if bind:
        store.update("living-room", state=STATE_FAILED, error="boom")
        pid = "proj-1"
    else:
        pid = "proj-9"
    with pytest.raises(automation_mod.NotProvisioned, match="set this project up on the workstation first"):
        runner.start(spec(pid=pid))
    assert issubclass(automation_mod.NotProvisioned, automation_mod.Busy)
    assert runner.status() is None


def test_web_project_spec_folder_must_match_the_binding(ws, web):
    runner = web_runner(ws, web)
    with pytest.raises(SpecError, match="other-room"):
        runner.start(spec(folder="other-room"))
    assert runner.status() is None


def test_web_project_routes_map_not_set_up_to_409_and_wrong_folder_to_400(ws, web):
    store, _, _ = web
    client = make_client(web_runner(ws, web))
    assert client.post("/amc/automation/start",
                       json={"Spec": make_spec(folder="other-room")}).status_code == 400
    store.update("living-room", state=STATE_FAILED)
    r = client.post("/amc/automation/start", json={"Spec": make_spec()})
    assert r.status_code == 409 and "set this project up" in r.json()["detail"]


def test_web_project_deleted_amc_project_fails_naming_the_folder(ws, web):
    store, folder, fake = web
    del fake.projects[WEB_AMC_ID]
    runner = web_runner(ws, web)
    job = runner.start(spec())
    assert job["State"] == "failed"
    assert "living-room" in job["Steps"][0]["Message"]
    assert calls(fake, "create_project") == []
    assert json.loads((folder / "runs" / f"{job['JobId']}.json").read_text())["State"] == "failed"
    assert store.get_by_folder("living-room").calib_runs == 0


def test_web_project_already_calibrating_is_refused(ws, web):
    _, _, fake = web
    fake.projects[WEB_AMC_ID]["state"] = "RUNNING"
    fake.running_polls = 100
    job = web_runner(ws, web).start(spec())
    assert job["State"] == "failed" and "already calibrating" in job["Steps"][0]["Message"]


def test_web_project_without_footage_waits_with_a_todo_then_uploads_it(ws, web):
    store, folder, fake = web
    for f in (folder / "footage").iterdir():
        f.unlink()
    amc_saved_alignment(ws)
    runner = web_runner(ws, web)
    job = runner.start(spec())
    assert job["State"] == "waiting_for_user"
    assert job["Steps"][1]["State"] == "waiting"
    todo = job["UserActions"][0]
    assert str(folder / "footage") in todo and "AMC" in todo
    assert calls(fake, "upload_video_files") == []
    assert calls(fake, "upload_layout")            # the floor plan still went up
    assert calls(fake, "calibrate") == []

    (folder / "footage" / "cam.mp4").write_bytes(b"video")  # the user copies footage in
    job = runner.resume()
    assert job["State"] == "completed", job
    assert calls(fake, "upload_video_files") == [{"names": ["cam_00.mp4"]}]


def test_web_project_footage_uploaded_in_amc_counts_on_continue(ws, web):
    _, folder, fake = web
    for f in (folder / "footage").iterdir():
        f.unlink()
    amc_saved_alignment(ws)
    runner = web_runner(ws, web)
    assert runner.start(spec())["State"] == "waiting_for_user"
    fake.projects[WEB_AMC_ID]["videos"] = [("cam_00.mp4", 5)]   # uploaded in AMC's UI
    job = runner.resume()
    assert job["State"] == "completed"
    assert job["Steps"][1]["Message"].startswith("finished in AMC")


def test_web_project_uses_the_floor_plan_saved_at_setup(ws, web):
    _, folder, fake = web
    (folder / "floorplan" / "layout.png").write_bytes(PNG + b"saved")
    job = web_runner(ws, web).start(spec(floorplan=False))
    assert job["Steps"][1]["State"] == "done"
    assert fake.projects[WEB_AMC_ID]["layout"] == PNG + b"saved"


def test_web_project_prefers_the_folders_alignment(ws, web):
    _, folder, fake = web
    amc_saved_alignment(ws)
    (folder / "alignment" / "alignment_data.json").write_text(json.dumps([[[1, 1], [2, 2], [3, 3]]]))
    job = web_runner(ws, web).start(spec())
    assert job["State"] == "completed"
    assert str(folder / "alignment" / "alignment_data.json") in job["Steps"][4]["Message"]
    assert fake.projects[WEB_AMC_ID]["alignment"] == [[[1, 1], [2, 2], [3, 3]]]


def test_web_project_falls_back_to_amcs_saved_alignment(ws, web):
    _, _, fake = web
    saved = amc_saved_alignment(ws)
    job = web_runner(ws, web).start(spec())
    assert job["State"] == "completed"
    assert str(saved) in job["Steps"][4]["Message"]
    assert fake.projects[WEB_AMC_ID]["alignment"] == [[[9, 9], [9, 9], [9, 9]]]


def test_web_project_without_any_alignment_waits_for_the_manual_step(ws, web):
    job = web_runner(ws, web).start(spec())
    assert job["State"] == "waiting_for_user"
    assert job["Steps"][4]["State"] == "waiting"


def test_web_project_results_and_records_go_into_the_folder(ws, web):
    store, folder, fake = web
    fake.with_gt = True
    saved = amc_saved_alignment(ws)
    runner = web_runner(ws, web, {**WEB, "results": {"download_dir": ""}})
    job = runner.start(spec())
    assert job["State"] == "completed", job

    # Step 7: one UTC-stamped directory with the statistics and the result ZIP.
    (run_dir,) = list((folder / "calibration").iterdir())
    assert len(run_dir.name) == 16 and run_dir.name.endswith("Z")
    assert json.loads((run_dir / "evaluation_statistics.json").read_text())["Average L2 distance(m)"] == 0.4
    (zip_file,) = list(run_dir.glob("*.zip"))
    assert job["Result"]["ResultFile"] == str(zip_file)
    assert not (ws[0] / "results").exists()
    # End of run: the record, the counters and the alignment copied back.
    record = json.loads((folder / "runs" / f"{job['JobId']}.json").read_text())
    assert record["State"] == "completed" and record["JobId"] == job["JobId"]
    b = store.get_by_folder("living-room")
    assert b.calib_runs == 1 and b.last_run_at is not None
    assert (folder / "alignment" / "alignment_data.json").read_text() == saved.read_text()

    time.sleep(1.1)  # a distinct UTC stamp for the second run
    runner.start(spec())
    assert store.get_by_folder("living-room").calib_runs == 2
    assert len(list((folder / "runs").glob("*.json"))) == 2
    assert len(list((folder / "calibration").iterdir())) == 2


def test_web_project_results_ignore_download_dir(ws, web):
    _, folder, _ = web
    amc_saved_alignment(ws)
    job = web_runner(ws, web).start(spec())     # WEB keeps results.download_dir: results
    assert job["Result"]["ResultFile"].startswith(str(folder / "calibration"))
    assert not (ws[0] / "results").exists()


def test_web_project_failed_and_cancelled_runs_are_recorded(ws, web):
    store, folder, fake = web
    amc_saved_alignment(ws)
    fake.fail_calibration = True
    runner = web_runner(ws, web)
    failed = runner.start(spec())
    assert failed["State"] == "failed"
    assert json.loads((folder / "runs" / f"{failed['JobId']}.json").read_text())["State"] == "failed"
    assert runner.cancel()["State"] == "cancelled"
    assert json.loads((folder / "runs" / f"{failed['JobId']}.json").read_text())["State"] == "cancelled"
    b = store.get_by_folder("living-room")
    assert b.calib_runs == 0 and b.last_run_at is None
    assert not (folder / "alignment" / "alignment_data.json").exists()


def test_web_project_run_survives_a_restart(ws, web):
    store, folder, fake = web
    for f in (folder / "footage").iterdir():
        f.unlink()
    first = web_runner(ws, web).start(spec())
    assert first["State"] == "waiting_for_user"

    # A new runner (the service restarted) whose config no longer says web_project still
    # finishes the run in the folder it started in.
    (folder / "footage" / "cam.mp4").write_bytes(b"video")
    amc_saved_alignment(ws)
    runner2 = make_runner(ws, FULLY_AUTOMATED, fake=fake, store=store)[0]
    assert runner2.status()["JobId"] == first["JobId"]
    job = runner2.resume()
    assert job["State"] == "completed", job
    assert job["AmcProjectId"] == WEB_AMC_ID
    assert job["Result"]["ResultFile"].startswith(str(folder / "calibration"))
    assert (folder / "runs" / f"{first['JobId']}.json").exists()
    assert store.get_by_folder("living-room").calib_runs == 1


def test_other_modes_write_nothing_into_project_folders(ws, web):
    store, folder, _ = web
    runner, _ = make_runner(ws, FULLY_AUTOMATED, store=store)
    assert runner.start(spec())["State"] == "completed"
    assert list((folder / "runs").iterdir()) == []
    assert list((folder / "calibration").iterdir()) == []


def test_runner_builds_its_store_from_the_environment(ws, monkeypatch):
    monkeypatch.setenv("AMC_PROJECTS_DIR", str(ws[0] / "elsewhere"))
    runner = CalibrationRunner(FakeController(ws[1]))
    assert runner.store.root == ws[0] / "elsewhere"


ORDERED = {**WEB, "videos": {"order": ["b.mp4", "a.mp4"]}}


def test_web_project_empty_footage_with_videos_order_waits(ws, web):
    _, folder, fake = web
    for f in (folder / "footage").iterdir():
        f.unlink()
    amc_saved_alignment(ws)
    runner = web_runner(ws, web, ORDERED)
    job = runner.start(spec())
    assert job["State"] == "waiting_for_user"
    assert job["Steps"][1]["State"] == "waiting"
    assert str(folder / "footage") in job["UserActions"][0]

    fake.projects[WEB_AMC_ID]["videos"] = [("cam_00.mp4", 5)]   # uploaded in AMC's UI
    job = runner.resume()
    assert job["State"] == "completed", job
    assert job["Steps"][1]["Message"].startswith("finished in AMC")
    assert calls(fake, "upload_video_files") == []


def test_web_project_partial_footage_with_videos_order_still_fails(ws, web):
    _, folder, _ = web
    (folder / "footage" / "b.mp4").unlink()
    job = web_runner(ws, web, ORDERED).start(spec())
    assert job["State"] == "failed" and "b.mp4" in job["Steps"][1]["Message"]


def test_web_project_layout_only_wait_does_not_upload_videos_twice(ws, web):
    _, _, fake = web
    amc_saved_alignment(ws)
    runner = web_runner(ws, web)
    job = runner.start(spec(floorplan=False))
    assert job["State"] == "waiting_for_user"
    assert job["UserActions"][0].startswith("Upload layout.png")
    assert len(calls(fake, "upload_video_files")) == 1

    fake.projects[WEB_AMC_ID]["layout"] = PNG              # the user adds the layout in AMC
    fake.config["layout_px_per_m"] = 50.0                  # and sets its scale
    job = runner.resume()
    assert job["State"] == "completed", job
    assert len(calls(fake, "upload_video_files")) == 1


def test_web_project_videos_uploaded_flag_survives_a_restart(ws, web):
    _, _, fake = web
    amc_saved_alignment(ws)
    web_runner(ws, web).start(spec(floorplan=False))
    runner2 = web_runner(ws, web)
    fake.projects[WEB_AMC_ID]["layout"] = PNG
    fake.config["layout_px_per_m"] = 50.0
    assert runner2.resume()["State"] == "completed"
    assert len(calls(fake, "upload_video_files")) == 1


# ---------------------------------------------------------------- web_project cancel undoes the run
# installer/plan/AMC-PROJECT-LIFECYCLE.md section 5.
CANCELLING = "Cancelling: undoing this run"
NEW_AMC_ID = "20260001"   # the fake AMC's first create_project id

# Stands in for `record-footage --pending` (AMC-FOOTAGE-RECORDING.md section 5): sets the
# old clips and manifest aside in previous/, then records a.mp4 (the same name as an old
# clip) and c.mp4 with a new manifest.
UNDO_RECORDER = r'''
import json, os, sys
footage = sys.argv[1]
prev = os.path.join(footage, "previous")
os.makedirs(prev, exist_ok=True)
for name in os.listdir(footage):
    if name.endswith(".mp4") or name == ".mv3dt-footage.json":
        os.replace(os.path.join(footage, name), os.path.join(prev, name))
for name in ("a.mp4", "c.mp4", ".mv3dt-footage.json"):
    with open(os.path.join(footage, name), "wb") as fh:
        fh.write(b"recorded")
print(json.dumps({"ok": True, "cancelled": False, "reason": "", "files": ["a.mp4", "c.mp4"],
                  "failed": [], "seconds": 60}), flush=True)
'''


def recording_runner(ws, web, overrides=WEB):
    _, folder, _ = web
    script = ws[0] / "undo_recorder.py"
    script.write_text(UNDO_RECORDER)
    runner = web_runner(ws, web, overrides)
    runner._record_argv = lambda: [sys.executable, str(script), str(folder / "footage")]
    return runner


def paused_recorded_run(ws, web):
    """A web_project run that recorded footage, uploaded it and paused for the layout."""
    _, folder, fake = web
    (folder / "footage" / ".mv3dt-footage.json").write_bytes(b"old manifest")
    runner = recording_runner(ws, web)
    job = runner.start(spec(floorplan=False), record_seconds=60)
    assert job["State"] == "waiting_for_user", job
    assert calls(fake, "upload_video_files")
    return runner


def job_file(runner):
    return json.loads(runner.state_path.read_text())


def assert_old_footage_back(folder):
    footage = folder / "footage"
    assert sorted(p.name for p in footage.iterdir()) == [".mv3dt-footage.json", "a.mp4", "b.mp4", "notes.txt"]
    assert (footage / "a.mp4").read_bytes() == b"video"      # the old clip, not the recorded one
    assert (footage / ".mv3dt-footage.json").read_bytes() == b"old manifest"


def test_cancel_of_a_paused_run_resets_amc_and_restores_footage(ws, web):
    store, folder, fake = web
    runner = paused_recorded_run(ws, web)
    saved = job_file(runner)
    assert saved["amc_changed"] is True and saved["recorded_files"] == ["a.mp4", "c.mp4"]
    (folder / "floorplan" / "layout.png").write_bytes(PNG)
    (folder / "alignment" / "alignment_data.json").write_text("[]")
    (folder / "calibration" / "20260101T000000Z").mkdir()

    job = runner.cancel()
    assert job["State"] == "cancelled", job
    assert "AMC project was reset to empty" in job["Message"] and NEW_AMC_ID in job["Message"]
    assert "earlier footage moved back" in job["Message"]
    # AMC: the run's project stopped, deleted, replaced by an empty one named after the folder.
    assert calls(fake, "stop_calibration") == [{"pid": WEB_AMC_ID}]
    assert calls(fake, "delete_project") == [{"pid": WEB_AMC_ID}]
    assert calls(fake, "create_project") == [{"name": "living-room"}]
    assert WEB_AMC_ID not in fake.projects and fake.projects[NEW_AMC_ID]["state"] == "INIT"
    b = store.get_by_folder("living-room")
    assert b.amc_project_id == NEW_AMC_ID and b.amc_project_name == "living-room" and b.state == STATE_READY
    assert b.calib_runs == 0
    # Footage: the recorded clips gone, the earlier ones back, previous/ removed.
    assert_old_footage_back(folder)
    # Kept: project inputs, earlier results, the run record.
    assert (folder / "floorplan" / "layout.png").exists()
    assert (folder / "alignment" / "alignment_data.json").exists()
    assert (folder / "calibration" / "20260101T000000Z").is_dir()
    record = json.loads((folder / "runs" / f"{job['JobId']}.json").read_text())
    assert record["State"] == "cancelled" and record["Message"] == job["Message"]
    saved = job_file(runner)
    assert saved["cancelling"] is False and saved["reset_amc_project_id"] == NEW_AMC_ID
    assert saved["undone"] == ["amc_deleted", "amc_created", "amc_bound", "footage_cleared", "footage_restored"]
    # A new run uses the new AMC project.
    assert runner.start(spec(floorplan=False))["AmcProjectId"] == NEW_AMC_ID


def test_cancel_of_a_running_run_undoes_it_on_the_run_thread(ws, web):
    store, folder, fake = web
    amc_saved_alignment(ws)
    fake.running_polls = 10 ** 9            # AMC calibrates until the run is cancelled
    runner = web_runner(ws, web)
    runner._threaded = True
    runner.start(spec())
    deadline = time.time() + 20
    while not calls(fake, "calibrate"):
        assert time.time() < deadline
        time.sleep(0.02)

    job = runner.cancel()
    assert job["State"] == "running" and job["Message"] == CANCELLING
    assert runner.cancel()["Message"] == CANCELLING          # a second cancel changes nothing
    with pytest.raises(automation_mod.NothingToResume):
        runner.resume()
    runner.join(20)
    job = runner.status()
    assert job["State"] == "cancelled", job
    assert calls(fake, "stop_calibration") == [{"pid": WEB_AMC_ID}]
    assert len(calls(fake, "create_project")) == 1
    assert store.get_by_folder("living-room").amc_project_id == NEW_AMC_ID
    assert job["Steps"][5]["State"] == "pending"
    assert json.loads((folder / "runs" / f"{job['JobId']}.json").read_text())["State"] == "cancelled"


def test_cancel_before_amc_was_changed_leaves_amc_alone(ws, web):
    store, folder, fake = web
    for f in (folder / "footage").iterdir():
        f.unlink()
    runner = web_runner(ws, web, {**WEB, "parameters": {"enabled": False}})
    job = runner.start(spec(floorplan=False))            # nothing to upload: the run only waits
    assert job["State"] == "waiting_for_user"
    assert job_file(runner)["amc_changed"] is False
    job = runner.cancel()
    assert job["State"] == "cancelled" and job["Message"] == "Cancelled. Nothing needed undoing."
    assert calls(fake, "delete_project") == [] and calls(fake, "stop_calibration") == []
    assert store.get_by_folder("living-room").amc_project_id == WEB_AMC_ID


def test_cancel_removes_this_runs_results_only(ws, web, monkeypatch):
    store, folder, fake = web
    amc_saved_alignment(ws)
    fake.with_gt = True
    (folder / "calibration" / "20260101T000000Z").mkdir()

    def broken(*a, **kw):
        raise automation_mod.AmcApiError("result download failed")

    monkeypatch.setattr(automation_mod.AmcApi, "mv3dt_result", broken)
    runner = web_runner(ws, web)
    job = runner.start(spec())
    assert job["State"] == "failed" and job["Steps"][6]["State"] == "failed"
    stamp = job_file(runner)["calibration_dir"]
    assert (folder / "calibration" / stamp / "evaluation_statistics.json").exists()

    job = runner.cancel()
    assert job["State"] == "cancelled", job
    assert "calibration results were removed" in job["Message"]
    assert [p.name for p in (folder / "calibration").iterdir()] == ["20260101T000000Z"]
    assert "results_removed" in job_file(runner)["undone"]
    assert json.loads((folder / "runs" / f"{job['JobId']}.json").read_text())["State"] == "cancelled"
    assert not (folder / "calibration" / stamp).exists()   # the run record did not recreate it


def test_retried_results_step_reuses_its_folder(ws, web, monkeypatch):
    _, folder, _ = web
    amc_saved_alignment(ws)
    real = automation_mod.AmcApi.mv3dt_result

    def broken(*a, **kw):
        raise automation_mod.AmcApiError("down")

    monkeypatch.setattr(automation_mod.AmcApi, "mv3dt_result", broken)
    runner = web_runner(ws, web)
    assert runner.start(spec())["State"] == "failed"
    stamp = job_file(runner)["calibration_dir"]
    monkeypatch.setattr(automation_mod.AmcApi, "mv3dt_result", real)
    time.sleep(1.1)                                      # a new attempt would get a new stamp
    assert runner.resume()["State"] == "completed"
    assert [p.name for p in (folder / "calibration").iterdir()] == [stamp]


def test_create_failure_after_delete_leaves_a_failed_binding_without_an_amc_project(ws, web, monkeypatch):
    store, folder, fake = web
    runner = paused_recorded_run(ws, web)

    def refused(*a, **kw):
        raise automation_mod.AmcApiError("create_project failed: disk full")

    monkeypatch.setattr(automation_mod.AmcApi, "create_project", refused)
    job = runner.cancel()
    assert job["State"] == "cancelled"
    assert job["Message"].startswith("Cancelled, but the AMC project could not be reset: ")
    assert "disk full" in job["Message"] and "earlier footage moved back" in job["Message"]
    b = store.get_by_folder("living-room")
    assert b.state == STATE_FAILED and b.amc_project_id is None and "disk full" in b.error
    assert WEB_AMC_ID not in fake.projects
    assert_old_footage_back(folder)


def test_amc_not_ready_leaves_the_amc_project_as_it_is(ws, web):
    store, folder, fake = web
    runner = paused_recorded_run(ws, web)
    fake.ready = False
    write_config(ws[0], {**WEB, "amc": {"ready_timeout_seconds": 0}})
    job = runner.cancel()
    assert job["State"] == "cancelled"
    assert "the AMC project could not be reset: AMC did not become ready within 0 s" in job["Message"]
    assert runner.controller.started == 1
    assert calls(fake, "delete_project") == [] and calls(fake, "create_project") == []
    b = store.get_by_folder("living-room")
    assert b.amc_project_id == WEB_AMC_ID and b.state == STATE_READY
    assert_old_footage_back(folder)                      # the rest is still undone


def test_delete_failure_leaves_the_amc_project_as_it_is(ws, web):
    store, _, fake = web
    runner = paused_recorded_run(ws, web)
    fake.delete_status = 500
    job = runner.cancel()
    assert "could not delete AMC project" in job["Message"]
    assert calls(fake, "create_project") == []
    assert store.get_by_folder("living-room").amc_project_id == WEB_AMC_ID
    assert "amc_deleted" not in job_file(runner)["undone"]


def test_amc_404_on_delete_counts_as_deleted(ws, web):
    store, _, fake = web
    runner = paused_recorded_run(ws, web)
    del fake.projects[WEB_AMC_ID]
    assert runner.cancel()["State"] == "cancelled"
    assert store.get_by_folder("living-room").amc_project_id == NEW_AMC_ID


def cut_off(runner, undone, **fields):
    """Rewrite the job file as if the service died during the undo after `undone`."""
    state = job_file(runner)
    state.update(state="running", message=CANCELLING, cancelling=True, undone=list(undone), **fields)
    runner.state_path.write_text(json.dumps(state))


def test_restart_after_amc_deleted_creates_one_project(ws, web):
    store, folder, fake = web
    runner = paused_recorded_run(ws, web)
    del fake.projects[WEB_AMC_ID]
    cut_off(runner, ["amc_deleted"])
    fake.calls.clear()

    job = web_runner(ws, web).status()                   # the restarted service carries on
    assert job["State"] == "cancelled", job
    assert calls(fake, "stop_calibration") == [] and calls(fake, "delete_project") == []
    assert calls(fake, "create_project") == [{"name": "living-room"}]
    assert store.get_by_folder("living-room").amc_project_id == NEW_AMC_ID
    assert_old_footage_back(folder)


def test_restart_after_amc_created_reuses_the_saved_project(ws, web):
    store, folder, fake = web
    runner = paused_recorded_run(ws, web)
    del fake.projects[WEB_AMC_ID]
    fake.add_project("20269999", "living-room")
    cut_off(runner, ["amc_deleted", "amc_created"], reset_amc_project_id="20269999")
    fake.calls.clear()

    job = web_runner(ws, web).status()
    assert job["State"] == "cancelled", job
    assert fake.calls == []                              # AMC is not touched at all
    assert store.get_by_folder("living-room").amc_project_id == "20269999"
    assert "20269999" in job["Message"]
    assert_old_footage_back(folder)


def test_restart_after_footage_cleared_never_deletes_the_older_footage(ws, web):
    store, folder, fake = web
    runner = paused_recorded_run(ws, web)
    footage = folder / "footage"
    # The undo got as far as deleting the recorded clips and moving a.mp4 back.
    for name in ("a.mp4", "c.mp4", ".mv3dt-footage.json"):
        (footage / name).unlink()
    (footage / "previous" / "a.mp4").rename(footage / "a.mp4")
    del fake.projects[WEB_AMC_ID]
    fake.add_project(NEW_AMC_ID, "living-room")
    store.update("living-room", amc_project_id=NEW_AMC_ID)
    cut_off(runner, ["amc_deleted", "amc_created", "amc_bound", "footage_cleared"],
            reset_amc_project_id=NEW_AMC_ID)
    fake.calls.clear()

    job = web_runner(ws, web).status()
    assert job["State"] == "cancelled", job
    assert fake.calls == []
    assert_old_footage_back(folder)                      # a.mp4 (a recorded clip's name) kept
    assert job_file(runner)["undone"][-1] == "footage_restored"


def test_a_cancelling_job_cannot_be_continued(ws, web, monkeypatch):
    runner = paused_recorded_run(ws, web)
    cut_off(runner, [])
    runner2 = web_runner(ws, web)
    started = []
    monkeypatch.setattr(runner2, "_start_undo", started.append)   # the undo is still running
    assert runner2.status()["State"] == "running" and runner2.status()["Message"] == CANCELLING
    assert len(started) == 1
    with pytest.raises(automation_mod.NothingToResume):
        runner2.resume()
    with pytest.raises(automation_mod.Busy):
        runner2.start(spec())
    assert runner2.uses_amc()


def test_job_file_without_the_cancel_fields_still_loads(ws, web):
    _, _, fake = web
    runner = web_runner(ws, web)
    job = runner.start(spec(floorplan=False))
    state = job_file(runner)
    for key in ("amc_changed", "recorded_files", "calibration_dir", "cancelling", "undone",
                "reset_amc_project_id"):
        del state[key]
    runner.state_path.write_text(json.dumps(state))
    runner2 = web_runner(ws, web)
    assert runner2.status()["JobId"] == job["JobId"]
    assert runner2._job.undone == [] and runner2._job.cancelling is False
    assert runner2.cancel()["State"] == "cancelled"
    assert calls(fake, "delete_project") == []           # amc_changed defaults to false


def test_other_modes_keep_the_plain_cancel(ws):
    tmp_path, _ = ws
    (tmp_path / "auto-magic-calib").rename(tmp_path / "moved")
    runner, fake = make_runner(ws, {"layout": {"allowed_hosts": ["abc.supabase.co"]}})
    assert runner.start(spec())["State"] == "waiting_for_user"
    job = runner.cancel()
    assert job["State"] == "cancelled" and job["Message"] == "Cancelled."
    assert calls(fake, "delete_project") == [] and calls(fake, "stop_calibration") == []
    saved = job_file(runner)
    assert saved["amc_changed"] is False and saved["undone"] == []


# ---------------------------------------------------------------- step 4: rectification.mode auto and linear
AUTO = {**FULLY_AUTOMATED, "rectification": {"mode": "auto"}}
INSTALLER_AMC_ID = "20261019_151133_2885"


def amc_project(fake, pid=INSTALLER_AMC_ID):
    return fake.projects[pid]


def rect_states(fake):
    return [kw["state"] for kw in calls(fake, "rectification_state")]


def test_auto_rectification_estimates_commits_and_generates(ws):
    runner, fake = make_runner(ws, AUTO)
    changing = runner._changing_amc

    def spy():
        fake.calls.append(("changing_amc", {}))
        changing()
    runner._changing_amc = spy

    job = runner.start(spec())
    assert job["State"] == "completed", job
    assert len(calls(fake, "start_rectification")) == 1
    assert calls(fake, "commit_rectification") == [{"body": {"cameras": {
        "0": {"model": "simple_radial", "k1": -0.12, "k2": 0},
        "1": {"model": "simple_divisional", "k1": -0.3, "k2": 0.01}}}}]
    order = [n for n, _ in fake.calls]
    assert order[order.index("start_rectification") - 1] == "changing_amc"
    assert order[order.index("commit_rectification") - 1] == "changing_amc"
    assert order.index("commit_rectification") < order.index("upload_alignment")
    message = job["Steps"][3]["Message"]
    assert job["Steps"][3]["State"] == "done"
    assert "camera 0 simple_radial (k1 -0.12, k2 0)" in message
    assert "camera 1 simple_divisional (k1 -0.3, k2 0.01)" in message
    saved = job_file(runner)
    assert saved["rectification_started"] is True and saved["rectification_phase"] == "generate"


def test_auto_rectification_never_trusts_an_earlier_completed_state(ws):
    fake = FakeAmc()
    fake.add_project(INSTALLER_AMC_ID, "Living-Room")
    amc_project(fake)["rect_state"] = "COMPLETED"   # from footage this run replaces
    runner, _ = make_runner(ws, AUTO, fake=fake)
    assert runner.start(spec())["State"] == "completed"
    assert len(calls(fake, "start_rectification")) == 1 and len(calls(fake, "commit_rectification")) == 1


def test_auto_rectification_waits_for_an_earlier_running_one_then_starts(ws):
    fake = FakeAmc(rect_running_polls=2)
    fake.add_project(INSTALLER_AMC_ID, "Living-Room")
    p = amc_project(fake)
    p["rect_state"], p["rect_phase"] = "RUNNING", "estimate"
    runner, _ = make_runner(ws, AUTO, fake=fake)
    assert runner.start(spec())["State"] == "completed"
    assert rect_states(fake)[:3] == ["RUNNING", "RUNNING", "READY_FOR_REVIEW"]   # waited, then started anyway
    names = [n for n, _ in fake.calls]
    assert names[:names.index("start_rectification")].count("rectification_state") == 3
    assert len(calls(fake, "start_rectification")) == 1


def failed_rectification(ws, phase):
    """A run whose rectification failed in `phase`, ready to be continued."""
    fake = FakeAmc(rect_fail=phase)
    runner, _ = make_runner(ws, AUTO, fake=fake)
    job = runner.start(spec())
    assert job["State"] == "failed" and job["Steps"][3]["State"] == "failed", job
    fake.rect_fail = ""
    return runner, fake


def test_auto_rectification_error_fails_with_the_log_tail_and_continue_starts_over(ws):
    runner, fake = failed_rectification(ws, "estimate")
    message = runner.status()["Steps"][3]["Message"]
    assert "ERROR while estimating distortion" in message
    assert "no grid lines found" in message and "rect line 6" in message and "rect line 5" not in message
    assert calls(fake, "commit_rectification") == []
    job = runner.resume()
    assert job["State"] == "completed", job
    assert len(calls(fake, "start_rectification")) == 2 and len(calls(fake, "commit_rectification")) == 1


def test_auto_rectification_resumes_at_ready_for_review_by_committing(ws):
    runner, fake = failed_rectification(ws, "estimate")
    amc_project(fake)["rect_state"] = "READY_FOR_REVIEW"
    assert runner.resume()["State"] == "completed"
    assert len(calls(fake, "start_rectification")) == 1 and len(calls(fake, "commit_rectification")) == 1


def test_auto_rectification_resumes_a_running_estimate(ws):
    runner, fake = failed_rectification(ws, "estimate")
    p = amc_project(fake)
    p["rect_state"], p["rect_phase"], p["rect_n"] = "RUNNING", "estimate", 0
    assert runner.resume()["State"] == "completed"
    assert len(calls(fake, "start_rectification")) == 1 and len(calls(fake, "commit_rectification")) == 1


def test_auto_rectification_resumes_a_running_generate_without_committing_again(ws):
    runner, fake = failed_rectification(ws, "generate")
    assert "ERROR while generating rectified videos" in runner.status()["Steps"][3]["Message"]
    assert job_file(runner)["rectification_phase"] == "generate"
    p = amc_project(fake)
    p["rect_state"], p["rect_phase"], p["rect_n"] = "RUNNING", "generate", 0
    job = runner.resume()
    assert job["State"] == "completed", job
    assert len(calls(fake, "start_rectification")) == 1 and len(calls(fake, "commit_rectification")) == 1
    assert "camera 0 simple_radial" in job["Steps"][3]["Message"]


def test_auto_rectification_once_started_takes_completed_as_done(ws):
    runner, fake = failed_rectification(ws, "generate")
    amc_project(fake)["rect_state"] = "COMPLETED"
    job = runner.resume()
    assert job["State"] == "completed" and job["Steps"][3]["State"] == "done"
    assert len(calls(fake, "start_rectification")) == 1 and len(calls(fake, "commit_rectification")) == 1


def test_auto_rectification_times_out_with_the_log_tail(ws):
    fake = FakeAmc(rect_running_polls=10 ** 9)
    rect = {"mode": "auto", "api": {"poll_seconds": 10, "timeout_seconds": 30}}
    runner, _ = make_runner(ws, {**AUTO, "rectification": rect}, fake=fake)
    job = runner.start(spec())
    assert job["State"] == "failed"
    message = job["Steps"][3]["Message"]
    assert "still RUNNING after 30 s while estimating distortion" in message and "no grid lines found" in message
    assert 3 <= len(rect_states(fake)) <= 6


def test_each_rectification_phase_has_its_own_timeout(ws):
    fake = FakeAmc(rect_running_polls=2)
    rect = {"mode": "auto", "api": {"poll_seconds": 10, "timeout_seconds": 25}}   # 3 reads fit in each phase
    runner, _ = make_runner(ws, {**AUTO, "rectification": rect}, fake=fake)
    assert runner.start(spec())["State"] == "completed"


def test_rectification_parameters_follow_amcs_ui():
    params = automation_mod._rectification_params
    est = {"model": "radial", "k1": -0.2, "k2": 0.05}
    assert params({"committed": {"model": "simple_radial", "k1": 0.1, "k2": 0.2}, "auto_estimate": est}) == (
        {"model": "simple_radial", "k1": 0.1, "k2": 0.2}, True)
    assert params({"committed": None, "auto_estimate": est}) == ({"model": "radial", "k1": -0.2, "k2": 0.05}, True)
    assert params({"committed": None, "auto_estimate": {"model": "fisheye", "k1": 0.3}}) == (
        {"model": "simple_divisional", "k1": 0.3, "k2": 0}, True)
    assert params({"committed": None, "auto_estimate": None}) == (
        {"model": "simple_divisional", "k1": 0, "k2": 0}, False)


def test_a_camera_without_an_estimate_is_committed_flat_and_named(ws):
    fake = FakeAmc()
    fake.rect_cameras = [
        {"camera_id": 0, "committed": None, "auto_estimate": {"model": "radial", "k1": -0.1, "k2": 0.02}},
        {"camera_id": 1, "committed": None, "auto_estimate": None},
    ]
    runner, _ = make_runner(ws, AUTO, fake=fake)
    job = runner.start(spec())
    assert job["State"] == "completed"
    assert calls(fake, "commit_rectification")[0]["body"]["cameras"]["1"] == {
        "model": "simple_divisional", "k1": 0, "k2": 0}
    assert "camera 1 simple_divisional (k1 0, k2 0, no estimate)" in job["Steps"][3]["Message"]


def test_linear_mode_stages_the_videos_unchanged(ws):
    runner, fake = make_runner(ws, {**AUTO, "rectification": {"mode": "linear"}})
    job = runner.start(spec())
    assert job["State"] == "completed", job
    assert calls(fake, "stage_linear_media") == [{"pid": INSTALLER_AMC_ID}]
    assert calls(fake, "start_rectification") == [] and calls(fake, "commit_rectification") == []
    assert job["Steps"][3]["Message"] == "videos staged as already linear"
    assert job_file(runner)["rectification_phase"] == "linear"


def test_linear_mode_error_fails_with_the_log_tail(ws):
    fake = FakeAmc(rect_fail="linear")
    runner, _ = make_runner(ws, {**AUTO, "rectification": {"mode": "linear"}}, fake=fake)
    job = runner.start(spec())
    assert job["State"] == "failed" and "no grid lines found" in job["Steps"][3]["Message"]


def test_cancel_while_rectification_polls(ws):
    fake = FakeAmc(rect_running_polls=10 ** 9)
    runner, _ = make_runner(ws, AUTO, fake=fake)
    runner._sleep = lambda s: runner.cancel()
    job = runner.start(spec())
    assert job["State"] == "cancelled", job
    assert job["Steps"][3]["State"] == "pending"
    assert calls(fake, "commit_rectification") == [] and calls(fake, "calibrate") == []


def test_rectification_waits_while_step_2_waits(ws):
    runner, fake = make_runner(ws, AUTO)
    job = runner.start(spec(floorplan=False))        # step 2 waits for the layout; the videos are in AMC
    assert job["State"] == "waiting_for_user"
    rect = job["Steps"][3]
    assert rect["State"] == "waiting" and rect["Message"] == "Rectification runs once the videos are in AMC"
    assert rect["NeedsUser"] is False and not [a for a in job["UserActions"] if "Rectification" in a]
    assert calls(fake, "start_rectification") == [] and calls(fake, "rectification_state") == []


def test_rectification_waits_while_amc_has_no_videos(ws):
    runner, fake = make_runner(ws, {**AUTO, "videos": {"enabled": False}})
    job = runner.start(spec())
    assert job["State"] == "waiting_for_user"
    assert job["Steps"][1]["State"] == "done"
    assert job["Steps"][3]["State"] == "waiting" and job["Steps"][3]["NeedsUser"] is False
    assert job["UserActions"] == []
    assert calls(fake, "start_rectification") == []


def test_skip_and_ui_modes_never_call_the_rectification_api(ws):
    runner, fake = make_runner(ws, FULLY_AUTOMATED)
    assert runner.start(spec())["Steps"][3]["State"] == "skipped"
    runner, fake2 = make_runner(ws, {**FULLY_AUTOMATED, "rectification": {"mode": "ui"}})
    job = runner.start(spec())
    assert job["Steps"][3]["State"] == "waiting"
    fake2.projects[INSTALLER_AMC_ID]["rect_state"] = "COMPLETED"
    assert runner.resume()["Steps"][3]["Message"].startswith("finished in AMC")
    assert not [n for n, _ in fake.calls if "rectification" in n or n == "stage_linear_media"]
    # ui mode only reads the state, to check the user's work on Continue (section 5).
    assert {n for n, _ in fake2.calls if "rectification" in n or n == "stage_linear_media"} == {
        "rectification_state"}


def paused_web_rectified_run(ws, web):
    """A web_project run that rectified automatically, then paused for Manual Alignment."""
    rect = {"mode": "auto", "api": {"poll_seconds": 10, "timeout_seconds": 30}}
    runner = web_runner(ws, web, {**WEB, "rectification": rect})
    job = runner.start(spec())
    assert job["State"] == "waiting_for_user" and job["Steps"][3]["State"] == "done", job
    assert job_file(runner)["amc_changed"] is True
    return runner


def test_cancel_undo_waits_for_a_running_rectification_before_deleting(ws, web):
    _, _, fake = web
    runner = paused_web_rectified_run(ws, web)
    p = fake.projects[WEB_AMC_ID]
    p["rect_state"], p["rect_phase"], p["rect_n"] = "RUNNING", "estimate", 0
    fake.rect_running_polls = 2
    before = len(fake.calls)
    assert runner.cancel()["State"] == "cancelled"
    after = [(n, kw) for n, kw in fake.calls[before:] if n in ("rectification_state", "delete_project")]
    assert [n for n, _ in after] == ["rectification_state"] * 3 + ["delete_project"]
    assert [kw["state"] for _, kw in after[:3]] == ["RUNNING", "RUNNING", "READY_FOR_REVIEW"]


def test_cancel_undo_deletes_after_the_rectification_timeout(ws, web):
    _, _, fake = web
    runner = paused_web_rectified_run(ws, web)
    p = fake.projects[WEB_AMC_ID]
    p["rect_state"], p["rect_phase"], p["rect_n"] = "RUNNING", "estimate", 0
    fake.rect_running_polls = 10 ** 9
    before = len(fake.calls)
    assert runner.cancel()["State"] == "cancelled"
    names = [n for n, _ in fake.calls[before:]]
    assert 3 <= names.count("rectification_state") <= 6
    assert names.count("delete_project") == 1 and names.index("delete_project") > names.index("rectification_state")


# ---------------------------------------------------------------- step 3: the floor plan scale
# installer/plan/AMC-RECTIFICATION-AND-SCALE.md section 4.
SCALE_WAIT = automation_mod.SCALE_WAIT
LAYOUT_SHA = hashlib.sha256(PNG).hexdigest()
NO_SCALE = {**FULLY_AUTOMATED, "parameters": {}}           # no settings_file, scale.source database


def scale_posts(fake):
    return [kw["settings"]["layout_px_per_m"] for kw in calls(fake, "config")]


def test_merge_order_is_amc_config_then_settings_file_then_overrides_then_scale(ws):
    fake = FakeAmc()
    fake.config = {**DEFAULT_CONFIG, "amc_only": 7, "bundle_adjustment": {"iterations": 5, "tolerance": 1}}
    overrides = {"detector": "transformer", "bundle_adjustment": {"iterations": 20}, "layout_px_per_m": 3.0}
    runner, _ = make_runner(ws, {**FULLY_AUTOMATED, "parameters": {"settings_file": "settings.json",
                                                                   "settings_overrides": overrides}}, fake=fake)
    assert runner.start(spec())["State"] == "completed"
    assert calls(fake, "config")[0]["settings"] == {
        "layout_px_per_m": 100.0, "detector": "transformer", "amc_only": 7,
        "bundle_adjustment": {"iterations": 20, "tolerance": 1}}


@pytest.mark.parametrize("source,fixed,expected,origin", [
    ("database", 0, 100.0, "the web app's floor plan"),
    ("fixed", 42, 42.0, "fixed_px_per_m"),
])
def test_known_scale_is_posted_and_saved_on_the_job(ws, source, fixed, expected, origin):
    cfg = {**NO_SCALE, "parameters": {"scale": {"source": source, "fixed_px_per_m": fixed}}}
    runner, fake = make_runner(ws, cfg)
    job = runner.start(spec(px_per_m=100.0))
    assert job["State"] == "completed", job
    assert f"scale {expected:g} px/m from {origin}" in job["Steps"][2]["Message"]
    assert scale_posts(fake) == [expected, expected]       # step 3, then step 6 before verifying
    assert job_file(runner)["layout_px_per_m"] == expected
    assert "Enter" not in " ".join(s["Message"] for s in job["Steps"])


@pytest.mark.parametrize("cfg", [
    {"scale": {"source": "database"}},
    {"scale": {"source": "fixed", "fixed_px_per_m": 0}},
])
def test_no_scale_posts_the_factory_default_and_waits(ws, cfg):
    fake = FakeAmc()
    fake.config["layout_px_per_m"] = 55.0                  # left by another project
    runner, _ = make_runner(ws, {**NO_SCALE, "parameters": cfg}, fake=fake)
    job = runner.start(spec(px_per_m=None))
    assert job["State"] == "waiting_for_user"
    step = job["Steps"][2]
    assert step["State"] == "waiting" and step["NeedsUser"] is True and step["Message"] == SCALE_WAIT
    assert job["UserActions"] == [SCALE_WAIT]
    assert scale_posts(fake) == [24.22] and fake.config["layout_px_per_m"] == 24.22
    assert job_file(runner)["layout_px_per_m"] is None
    assert calls(fake, "calibrate") == []


def test_scale_source_none_neither_posts_nor_asks(ws):
    runner, fake = make_runner(ws, {**NO_SCALE, "parameters": {"scale": {"source": "none"}}})
    job = runner.start(spec(px_per_m=None))
    assert job["State"] == "completed", job
    assert calls(fake, "config") == [] and calls(fake, "get_config") == []   # step 6 re-posts nothing either
    assert job["Steps"][2]["Message"] == "AMC's current settings"


def test_continue_accepts_a_scale_set_in_amc(ws):
    runner, fake = make_runner(ws, NO_SCALE)
    assert runner.start(spec(px_per_m=None))["State"] == "waiting_for_user"
    fake.config["layout_px_per_m"] = 24.2201             # the flagged way to confirm a true 24.22
    job = runner.resume()
    assert job["State"] == "completed", job
    assert "scale 24.2201 px/m set in AMC" in job["Steps"][2]["Message"]
    assert job_file(runner)["layout_px_per_m"] == 24.2201
    assert scale_posts(fake) == [24.22, 24.2201]          # step 6 re-posts the confirmed value


@pytest.mark.parametrize("value,reported", [(24.22, "24.22 px/m, the factory default"),
                                             (0, "no valid value"), (-3, "no valid value")])
def test_continue_rejects_the_default_or_a_bad_scale(ws, value, reported):
    runner, fake = make_runner(ws, NO_SCALE)
    runner.start(spec(px_per_m=None))
    fake.config["layout_px_per_m"] = value
    job = runner.resume()
    assert job["State"] == "waiting_for_user"
    step = job["Steps"][2]
    assert step["State"] == "waiting" and step["NeedsUser"] is True
    assert step["Message"] == f"{SCALE_WAIT}. AMC reports layout_px_per_m {reported}"
    assert "factory scale 24.22" in step["Done"]
    assert job_file(runner)["layout_px_per_m"] is None
    assert calls(fake, "verify_project") == [] and calls(fake, "calibrate") == []


def test_step_six_takes_a_correction_made_during_the_pause_then_posts_before_verifying(ws):
    tmp_path, _ = ws
    (tmp_path / "auto-magic-calib").rename(tmp_path / "moved")    # pause for Manual Alignment
    runner, fake = make_runner(ws, NO_SCALE)
    assert runner.start(spec())["State"] == "waiting_for_user"
    fake.config["layout_px_per_m"] = 80.0                 # corrected in AMC during the pause
    fake.ui_alignment_done = True
    fake.calls.clear()
    job = runner.resume()
    assert job["State"] == "completed", job
    names = [n for n, _ in fake.calls if n in ("get_config", "config", "verify_project", "calibrate")]
    assert names == ["get_config", "config", "verify_project", "calibrate"]
    assert scale_posts(fake) == [80.0]
    assert job_file(runner)["layout_px_per_m"] == 80.0
    assert "scale corrected in AMC from 100 to 80 px/m" in job["Steps"][5]["Message"]


def test_step_six_posts_the_jobs_scale_over_the_default(ws):
    tmp_path, _ = ws
    (tmp_path / "auto-magic-calib").rename(tmp_path / "moved")
    runner, fake = make_runner(ws, NO_SCALE)
    runner.start(spec())
    fake.config["layout_px_per_m"] = 24.22                # reset to the default: not a correction
    fake.ui_alignment_done = True
    assert runner.resume()["State"] == "completed"
    assert scale_posts(fake) == [100.0, 100.0]
    assert job_file(runner)["layout_px_per_m"] == 100.0


# ---------------------------------------------------------------- remembered scale (section 4.3)
def test_confirmed_scale_is_remembered_with_the_layout_hash_and_reused(ws, web):
    store, folder, fake = web
    amc_saved_alignment(ws)
    runner = web_runner(ws, web)
    job = runner.start(spec(px_per_m=None))
    assert job["State"] == "waiting_for_user" and job["UserActions"] == [SCALE_WAIT]
    assert job_file(runner)["layout_sha256"] == LAYOUT_SHA
    fake.config["layout_px_per_m"] = 61.5
    job = runner.resume()
    assert job["State"] == "completed", job
    assert "remembered for this floor plan" in job["Steps"][2]["Message"]
    b = store.get_by_folder("living-room")
    assert (b.layout_px_per_m, b.layout_sha256) == (61.5, LAYOUT_SHA)

    fake.config["layout_px_per_m"] = 24.22
    job = runner.start(spec(px_per_m=None))                # the same floor plan: no question
    assert job["State"] == "completed", job
    assert "scale 61.5 px/m from remembered for this floor plan" in job["Steps"][2]["Message"]
    assert fake.config["layout_px_per_m"] == 61.5


def test_remembered_scale_is_not_used_for_another_floor_plan(ws, web):
    store, _, fake = web
    amc_saved_alignment(ws)
    store.update("living-room", layout_px_per_m=61.5, layout_sha256="0" * 64)
    job = web_runner(ws, web).start(spec(px_per_m=None))
    assert job["State"] == "waiting_for_user" and job["UserActions"] == [SCALE_WAIT]
    assert fake.config["layout_px_per_m"] == 24.22


def test_database_scale_wins_over_the_remembered_one(ws, web):
    store, _, fake = web
    amc_saved_alignment(ws)
    store.update("living-room", layout_px_per_m=61.5, layout_sha256=LAYOUT_SHA)
    job = web_runner(ws, web).start(spec(px_per_m=90.0))
    assert job["State"] == "completed"
    assert scale_posts(fake)[0] == 90.0
    assert store.get_by_folder("living-room").layout_px_per_m == 61.5   # a database scale is not written


def test_no_uploaded_layout_means_no_remembered_scale_is_used_or_written(ws, web):
    store, _, fake = web
    amc_saved_alignment(ws)
    store.update("living-room", layout_px_per_m=61.5, layout_sha256=LAYOUT_SHA)
    runner = web_runner(ws, web, {**WEB, "layout": {"enabled": False}})
    job = runner.start(spec(px_per_m=None))
    assert job["State"] == "waiting_for_user" and SCALE_WAIT in job["UserActions"]
    assert job_file(runner)["layout_sha256"] == ""
    store.update("living-room", layout_px_per_m=None, layout_sha256=None)
    fake.projects[WEB_AMC_ID]["layout"] = PNG              # the user uploads it in AMC
    fake.config["layout_px_per_m"] = 70.0
    job = runner.resume()
    assert job["State"] == "completed", job
    assert "not remembered: this run uploaded no layout.png" in job["Steps"][2]["Message"]
    b = store.get_by_folder("living-room")
    assert b.layout_px_per_m is None and b.layout_sha256 is None
    assert job_file(runner)["layout_px_per_m"] == 70.0


def test_step_six_correction_is_remembered(ws, web):
    store, _, fake = web
    runner = web_runner(ws, web)                           # no alignment: pauses for step 5
    assert runner.start(spec())["State"] == "waiting_for_user"
    fake.config["layout_px_per_m"] = 88.0
    fake.ui_alignment_done = True
    assert runner.resume()["State"] == "completed"
    b = store.get_by_folder("living-room")
    assert (b.layout_px_per_m, b.layout_sha256) == (88.0, LAYOUT_SHA)


def test_cancel_undo_keeps_the_remembered_scale(ws, web):
    store, _, fake = web
    runner = web_runner(ws, web)
    assert runner.start(spec(px_per_m=None))["State"] == "waiting_for_user"
    fake.config["layout_px_per_m"] = 61.5
    job = runner.resume()                                  # scale confirmed; still waits for step 5
    assert job["State"] == "waiting_for_user" and job["Steps"][2]["State"] == "done"
    assert runner.cancel()["State"] == "cancelled"
    b = store.get_by_folder("living-room")
    assert b.amc_project_id == NEW_AMC_ID
    assert (b.layout_px_per_m, b.layout_sha256) == (61.5, LAYOUT_SHA)


# ---------------------------------------------------------------- checked resume (section 5)
def test_ui_rectification_stays_waiting_until_amc_reports_completed(ws):
    runner, fake = make_runner(ws, {**FULLY_AUTOMATED, "rectification": {"mode": "ui"}})
    runner.start(spec())
    amc_project(fake)["rect_state"] = "READY_FOR_REVIEW"
    job = runner.resume()
    assert job["State"] == "waiting_for_user"
    assert job["Steps"][3]["State"] == "waiting"
    assert job["Steps"][3]["Message"].endswith("AMC reports READY_FOR_REVIEW")
    assert calls(fake, "upload_alignment") == []


def test_manual_alignment_stays_waiting_until_amc_has_the_file(ws):
    tmp_path, _ = ws
    (tmp_path / "auto-magic-calib").rename(tmp_path / "moved")
    runner, fake = make_runner(ws, FULLY_AUTOMATED)
    assert runner.start(spec())["State"] == "waiting_for_user"
    job = runner.resume()
    assert job["State"] == "waiting_for_user" and job["Steps"][4]["State"] == "waiting"
    assert job["UserActions"] == [job["Steps"][4]["Message"]]
    assert job["Steps"][4]["Message"].endswith("AMC reports no saved alignment yet")
    assert "(unfinished: step 5 Manual Alignment)" in job["Message"]
    assert calls(fake, "verify_project") == []
    amc_project(fake)["alignment"] = [[[1, 1]]]            # saved in AMC's UI
    assert runner.resume()["State"] == "completed"


def test_a_step_that_waited_for_the_run_is_not_checked_on_continue(ws):
    tmp_path, _ = ws
    (tmp_path / "auto-magic-calib").rename(tmp_path / "moved")
    runner, fake = make_runner(ws, FULLY_AUTOMATED)
    runner.start(spec())
    state = job_file(runner)
    state["steps"][4]["NeedsUser"] = False                 # step 5 waited only for rectification
    runner.state_path.write_text(json.dumps(state))
    runner2, _ = make_runner(ws, FULLY_AUTOMATED, fake=fake)
    job = runner2.resume()
    step = job["Steps"][4]
    assert step["State"] == "waiting" and step["NeedsUser"] is True
    assert "AMC reports" not in step["Message"]            # a first pass asks the user to align


def test_job_file_without_the_scale_fields_still_loads(ws):
    runner, fake = make_runner(ws, NO_SCALE)
    job = runner.start(spec(px_per_m=None))
    state = job_file(runner)
    del state["layout_sha256"], state["layout_px_per_m"]
    runner.state_path.write_text(json.dumps(state))
    runner2, _ = make_runner(ws, NO_SCALE, fake=fake)
    assert runner2.status()["JobId"] == job["JobId"]
    assert runner2._job.layout_sha256 == "" and runner2._job.layout_px_per_m is None
    fake.config["layout_px_per_m"] = 50.0
    assert runner2.resume()["State"] == "completed"


@pytest.mark.parametrize("mode", ["auto", "linear"])
def test_auto_and_linear_run_on_continue_instead_of_counting_as_finished_in_amc(ws, mode):
    runner, fake = make_runner(ws, {**AUTO, "rectification": {"mode": mode}})
    job = runner.start(spec(floorplan=False))        # step 2 waits for the layout, so step 4 waits too
    assert job["Steps"][3]["State"] == "waiting"
    fake.ui_alignment_done = True                    # the user uploads the layout in AMC
    amc_project(fake)["layout"] = PNG
    fake.config["layout_px_per_m"] = 50.0            # and sets the floor plan scale
    job = runner.resume()
    assert job["State"] == "completed", job
    if mode == "auto":
        assert len(calls(fake, "start_rectification")) == 1 and len(calls(fake, "commit_rectification")) == 1
        assert job["Steps"][3]["Message"].startswith("rectified automatically: camera 0")
    else:
        assert len(calls(fake, "stage_linear_media")) == 1
        assert job["Steps"][3]["Message"] == "videos staged as already linear"


def test_linear_mode_marks_amc_changed_before_staging(ws, web):
    _, _, fake = web
    amc_saved_alignment(ws)
    runner = web_runner(ws, web, {**WEB, "videos": {"enabled": False}, "layout": {"enabled": False},
                                  "parameters": {"enabled": False}, "alignment": {"file": ""},
                                  "rectification": {"mode": "linear"}})
    fake.projects[WEB_AMC_ID]["videos"] = [("cam_00.mp4", 5)]   # already in AMC: step 4 is the first change
    changing = runner._changing_amc

    def spy():
        fake.calls.append(("changing_amc", {"before": job_file(runner)["amc_changed"]}))
        changing()
    runner._changing_amc = spy
    job = runner.start(spec(floorplan=False))
    assert job["Steps"][3]["State"] == "done", job
    order = [n for n, _ in fake.calls]
    assert order[order.index("stage_linear_media") - 1] == "changing_amc"
    assert calls(fake, "changing_amc")[0] == {"before": False}
    assert job_file(runner)["amc_changed"] is True


def test_alignment_waits_for_rectification_then_asks_for_the_new_footage(ws):
    tmp_path, _ = ws
    (tmp_path / "auto-magic-calib").rename(tmp_path / "moved")      # no saved alignment
    runner, fake = make_runner(ws, AUTO)
    job = runner.start(spec(floorplan=False))
    align = job["Steps"][4]
    assert job["Steps"][3]["State"] == "waiting"
    assert align["State"] == "waiting" and align["NeedsUser"] is False
    assert align["Message"] == "Manual Alignment comes after rectification"
    assert job["UserActions"][0].startswith("Upload layout.png") and job["UserActions"][1] == SCALE_WAIT
    assert len(job["UserActions"]) == 2

    amc_project(fake)["alignment"] = [[[0, 0]]]   # stale: from footage before this run
    fake.ui_alignment_done = True                  # the layout is uploaded in AMC
    amc_project(fake)["layout"] = PNG
    fake.config["layout_px_per_m"] = 50.0
    job = runner.resume()
    assert job["State"] == "waiting_for_user"
    assert job["Steps"][3]["State"] == "done"
    align = job["Steps"][4]
    assert align["State"] == "waiting" and align["NeedsUser"] is True
    assert align["Message"].startswith("Do Manual Alignment in AMC")
    assert calls(fake, "calibrate") == []
    job = runner.resume()                          # the user aligned the new footage
    assert job["State"] == "completed"
    assert job["Steps"][4]["Message"].startswith("finished in AMC")


def test_lag_window_tolerates_the_state_from_before_each_request(ws):
    fake = FakeAmc(rect_stale_reads=2)
    fake.add_project(INSTALLER_AMC_ID, "Living-Room")
    amc_project(fake)["rect_state"] = "COMPLETED"          # earlier footage's rectification
    runner, _ = make_runner(ws, AUTO, fake=fake)
    job = runner.start(spec())
    assert job["State"] == "completed", job
    assert len(calls(fake, "commit_rectification")) == 1   # a lagging COMPLETED never skips the commit
    states = rect_states(fake)
    assert states[1:3] == ["COMPLETED", "COMPLETED"]       # after the start
    after_commit = states[states.index("READY_FOR_REVIEW", 3) + 1:]
    assert after_commit[:2] == ["READY_FOR_REVIEW", "READY_FOR_REVIEW"]   # after the commit


def test_lag_window_keeps_polling_through_an_earlier_error(ws):
    fake = FakeAmc(rect_stale_reads=3)
    fake.add_project(INSTALLER_AMC_ID, "Living-Room")
    amc_project(fake)["rect_state"] = "ERROR"
    runner, _ = make_runner(ws, AUTO, fake=fake)
    assert runner.start(spec())["State"] == "completed"
    assert rect_states(fake)[1:4] == ["ERROR"] * 3


def test_lag_window_ends_after_three_polls(ws):
    fake = FakeAmc(rect_stale_reads=4)
    fake.add_project(INSTALLER_AMC_ID, "Living-Room")
    amc_project(fake)["rect_state"] = "COMPLETED"
    runner, _ = make_runner(ws, AUTO, fake=fake)
    job = runner.start(spec())
    assert job["State"] == "failed"
    message = job["Steps"][3]["Message"]
    assert "is COMPLETED while estimating distortion, expected RUNNING or READY_FOR_REVIEW" in message
    assert "no grid lines found" in message
    assert calls(fake, "commit_rectification") == []


def test_running_in_the_reply_closes_the_lag_window(ws):
    fake = FakeAmc(rect_stale_reads=1, rect_reply="RUNNING")
    fake.add_project(INSTALLER_AMC_ID, "Living-Room")
    amc_project(fake)["rect_state"] = "COMPLETED"
    runner, _ = make_runner(ws, AUTO, fake=fake)
    job = runner.start(spec())
    assert job["State"] == "failed" and "is COMPLETED while estimating distortion" in job["Steps"][3]["Message"]


def test_running_seen_in_a_poll_closes_the_lag_window(ws):
    fake = FakeAmc(rect_fail="estimate", rect_reply="")    # no state in the reply
    runner, _ = make_runner(ws, AUTO, fake=fake)
    job = runner.start(spec())
    assert job["State"] == "failed" and "ERROR while estimating distortion" in job["Steps"][3]["Message"]
    assert rect_states(fake)[1:3] == ["RUNNING", "ERROR"]
