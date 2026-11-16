"""mv3dt_amc_control automation, run against the fake AMC API (no GPU, Docker or network).

Run from installer/: `python3 -m pytest tests/amc_control -v`
"""
import io
import json
import time

import pytest

jwt = pytest.importorskip("jwt")
httpx = pytest.importorskip("httpx")
yaml = pytest.importorskip("yaml")
pytest.importorskip("fastapi")
pytest.importorskip("multipart")  # python-multipart, used by the fake AMC
pytest.importorskip("PIL")

from fastapi.testclient import TestClient  # noqa: E402

from amc_control_fake_api import FakeAmc  # noqa: E402
from mv3dt_amc_control import automation as automation_mod  # noqa: E402
from mv3dt_amc_control.api import create_app  # noqa: E402
from mv3dt_amc_control.automation import CalibrationRunner  # noqa: E402
from mv3dt_amc_control.automation_config import ConfigError, load_config  # noqa: E402
from mv3dt_amc_control.calibration_spec import SPEC_SCOPE, SpecError, verify_spec  # noqa: E402
from mv3dt_amc_control.settings import Settings  # noqa: E402

TICKET_SECRET = "ticket-secret-ticket-secret-ticket-secret"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
FLOORPLAN_URL = "https://abc.supabase.co/storage/v1/object/sign/media/fp/new.png?token=t"


def make_spec(pid="proj-1", floorplan=True, px_per_m=100.0, url=FLOORPLAN_URL, scope=SPEC_SCOPE,
              secret=TICKET_SECRET, exp_delta=900):
    spec = {"title": "Living Room", "floorplan": None}
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


def make_runner(ws, overrides, fake=None, floorplan_bytes=PNG, download_status=200):
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

    runner = CalibrationRunner(
        FakeController(install),
        config_loader=lambda: load_config(cfg_path),
        http_client_factory=lambda: TestClient(fake.app()),
        download_client_factory=lambda: httpx.Client(transport=httpx.MockTransport(serve_floorplan)),
        sleep=lambda s: None,
        threaded=False,
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
    assert states == {"project_setup": "done", "video_configuration": "done", "parameters": "waiting",
                      "rectification": "waiting", "manual_alignment": "waiting", "execute": "pending",
                      "results": "pending"}
    todo = " | ".join(job["UserActions"])
    assert "100.00 layout px/m" in todo and "Rectification" in todo and "Manual Alignment" in todo
    assert calls(fake, "calibrate") == []          # nothing calibrates before the user is done
    assert calls(fake, "upload_video_files")       # but everything that could be uploaded was

    fake.ui_alignment_done = True                  # the user finishes steps 3-5 in AMC
    job = runner.resume()
    assert job["State"] == "completed"
    assert calls(fake, "calibrate") == [{"detector_type": "resnet"}]


def test_continue_before_the_ui_work_is_saved_pauses_again(ws):
    tmp_path, _ = ws
    (tmp_path / "auto-magic-calib").rename(tmp_path / "moved")
    runner, fake = make_runner(ws, {"layout": {"enabled": False}, "rectification": {"mode": "skip"},
                                    "parameters": {"scale": {"source": "none"}}})
    assert runner.start(spec(floorplan=False))["State"] == "waiting_for_user"
    job = runner.resume()                          # user pressed Continue too early
    assert job["State"] == "waiting_for_user"
    assert "instead of READY" in job["UserActions"][0]
    assert calls(fake, "calibrate") == []


def test_saved_alignment_waits_for_ui_rectification_before_upload(ws):
    runner, fake = make_runner(ws, {**FULLY_AUTOMATED, "rectification": {"mode": "ui"}})
    job = runner.start(spec())
    assert job["State"] == "waiting_for_user"
    assert calls(fake, "upload_alignment") == []
    assert len(job["UserActions"]) == 1 and "Rectification" in job["UserActions"][0]  # alignment needs nothing
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


def test_scale_key_missing_from_settings_file_is_reported_not_invented(ws):
    tmp_path, _ = ws
    runner, fake = make_runner(ws, FULLY_AUTOMATED)
    (tmp_path / "settings.json").write_text(json.dumps({"detector": "resnet"}))
    job = runner.start(spec())
    assert job["State"] == "waiting_for_user"
    assert "layout_px_per_m" not in calls(fake, "config")[0]["settings"]
    assert "no layout_px_per_m key" in job["Steps"][2]["Message"]
    assert job["Steps"][2]["Done"].startswith("applied settings.json")


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


def test_overrides_without_a_settings_file_fail_instead_of_posting_a_partial_file(ws):
    runner, fake = make_runner(ws, {**FULLY_AUTOMATED, "parameters": {"settings_overrides": {"a": 1}}})
    job = runner.start(spec())
    assert job["State"] == "failed" and calls(fake, "config") == []


def test_base_config_loads_on_its_own():
    cfg = load_config()
    assert cfg.rectification.mode == "ui" and cfg.alignment.file == "auto"


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
