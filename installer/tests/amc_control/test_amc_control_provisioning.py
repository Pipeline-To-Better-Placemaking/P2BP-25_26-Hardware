"""mv3dt_amc_control provisioning and the /projects routes (AMC-WEB-PROJECTS.md sections 5, 7 and 8).

Run against the fake AMC API and a fake controller, so no GPU, Docker or network.

Run from installer/: `python3 -m pytest tests/amc_control -v`
"""
from __future__ import annotations

import os
import threading
import time

import pytest

jwt = pytest.importorskip("jwt")
httpx = pytest.importorskip("httpx")
yaml = pytest.importorskip("yaml")
pytest.importorskip("fastapi")
pytest.importorskip("multipart")  # python-multipart, used by the fake AMC

from fastapi.testclient import TestClient  # noqa: E402

from amc_control_fake_api import FakeAmc  # noqa: E402
from mv3dt_amc_control import cli  # noqa: E402
from mv3dt_amc_control.api import create_app  # noqa: E402
from mv3dt_amc_control.automation_config import load_config  # noqa: E402
from mv3dt_amc_control.calibration_spec import SPEC_SCOPE, verify_spec  # noqa: E402
from mv3dt_amc_control.projects import ProjectStore  # noqa: E402
from mv3dt_amc_control.provisioning import RESTART_ERROR, Provisioner, ProvisionError  # noqa: E402
from mv3dt_amc_control.settings import Settings  # noqa: E402

SECRET = "ticket-secret-ticket-secret-ticket-secret"
ORIGIN = "https://app.example.org"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
FLOORPLAN_URL = "https://abc.supabase.co/storage/v1/object/sign/media/fp/new.png?token=t"
SETTINGS = Settings(ticket_secret=SECRET, allowed_origins=[ORIGIN])


def make_ticket(pid="proj-1", scope="amc"):
    claims = {"sub": "user-1", "pid": pid, "scope": scope, "aud": "p2bp-workstation",
              "exp": int(time.time()) + 900}
    return jwt.encode(claims, SECRET, algorithm="HS256")


def make_spec(pid="proj-1", folder="living-room", floorplan=False, title="Living Room"):
    spec = {"title": title, "floorplan": None, "folder": folder}
    if floorplan:
        spec["floorplan"] = {"id": "fp-1", "url": FLOORPLAN_URL, "contentType": "image/png",
                             "width": 741, "height": 502, "pxPerM": 100.0}
    claims = {"sub": "user-1", "pid": pid, "scope": SPEC_SCOPE, "aud": "p2bp-workstation", "spec": spec,
              "exp": int(time.time()) + 900}
    return jwt.encode(claims, SECRET, algorithm="HS256")


def auth(pid="proj-1", scope="amc"):
    return {"Authorization": f"Bearer {make_ticket(pid, scope)}"}


PROVISION = {"scope": "amc-provision"}


class FakeController:
    def __init__(self, fake: FakeAmc, install_dir):
        self.fake = fake
        self.install_dir = install_dir
        self.state_dir = install_dir / "state"
        self.started = 0
        self.stopped = 0
        self.comes_up = True

    def start(self):
        self.started += 1
        if self.comes_up:
            self.fake.ready = True

    def stop(self):
        self.stopped += 1

    def status(self):
        return {"State": "ready" if self.fake.ready else "down", "UiUrl": "", "Error": "", "UpdatedAt": 0}

    def ports(self):
        return 18000, 15000


class FakeRunner:
    def __init__(self):
        self.active = False
        self.started = []
        self.resumed = 0
        self.inside_start = None  # a test hook run inside start()

    def uses_amc(self):
        return self.active

    def status(self):
        return None

    def _job(self, pid="proj-1", title="Living Room", by="user-1"):
        now = int(time.time())
        return {"JobId": "j1", "ProjectId": pid, "ProjectTitle": title, "State": "running",
                "Steps": [], "StartedBy": by, "StartedAt": now, "UpdatedAt": now}

    def start(self, spec):
        if self.inside_start:
            self.inside_start()
        self.started.append(spec)
        self.active = True
        return self._job(spec.project_id, spec.title, spec.issued_to)

    def resume(self):
        self.resumed += 1
        self.active = True
        return self._job()


class Env:
    """A projects root, a fake AMC, a fake controller and runner, and a provisioner wired to them."""

    def __init__(self, tmp_path, deferred=False):
        self.root = tmp_path / "amc_projects"
        self.store = ProjectStore(self.root)
        self.fake = FakeAmc()
        self.controller = FakeController(self.fake, tmp_path)
        self.runner = FakeRunner()
        self.pending = []
        self.downloads = []
        self.now = [1000.0]
        cfg = tmp_path / "amc_automation.yaml"
        cfg.write_text(yaml.safe_dump({"amc": {"ready_timeout_seconds": 60},
                                       "layout": {"allowed_hosts": ["abc.supabase.co"]}}))

        def serve_floorplan(request):
            self.downloads.append(str(request.url))
            return httpx.Response(200, content=PNG)

        def wait(seconds):
            self.now[0] += seconds

        self.provisioner = Provisioner(
            self.store, self.controller,
            runner_busy=lambda: self.runner.uses_amc(),
            config_loader=lambda: load_config(cfg),
            http_client_factory=lambda: TestClient(self.fake.app()),
            download_client_factory=lambda: httpx.Client(transport=httpx.MockTransport(serve_floorplan)),
            clock=lambda: self.now[0],
            wait=wait,
            spawn=self.pending.append if deferred else (lambda fn: fn()),
        )

    def run_pending(self):
        while self.pending:
            self.pending.pop(0)()

    def client(self, **headers):
        app = create_app(SETTINGS, controller=self.controller, runner=self.runner, provisioner=self.provisioner)
        c = TestClient(app)
        c.headers.update(headers)
        return c

    def spec(self, **kw):
        return verify_spec(make_spec(**kw), SECRET, kw.get("pid", "proj-1"))


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


@pytest.fixture
def slow(tmp_path):
    """Jobs do not run until `run_pending()`, so the provisioner stays busy."""
    return Env(tmp_path, deferred=True)


def provision(client, pid="proj-1", **kw):
    return client.post("/projects", json={"Spec": make_spec(pid=pid, **kw)}, headers=auth(pid, "amc-provision"))


def calls(fake, name):
    return [kw for n, kw in fake.calls if n == name]


# ---------------------------------------------------------------- provisioning
def test_provision_returns_202_then_ready(slow):
    c = slow.client()
    r = provision(c)
    assert r.status_code == 202
    body = r.json()
    assert body["State"] == "provisioning" and body["Folder"] == "living-room"
    assert body["ProjectId"] == "proj-1" and body["Path"] == str(slow.root / "living-room")
    assert "schema_version" not in body
    for sub in ("footage", "floorplan", "alignment", "calibration", "runs"):
        assert (slow.root / "living-room" / sub).is_dir()

    slow.run_pending()
    b = c.get("/projects/current", headers=auth()).json()
    assert b["State"] == "ready" and b["Error"] == ""
    assert b["AmcProjectName"] == "living-room" and b["AmcProjectId"] in slow.fake.projects
    assert b["Title"] == "Living Room"
    assert calls(slow.fake, "create_project") == [{"name": "living-room"}]
    assert slow.controller.started == 0  # AMC was already up


def test_amc_is_started_when_down(env):
    env.fake.ready = False
    provision(env.client())
    assert env.controller.started == 1
    assert env.store.get_by_pid("proj-1").state == "ready"


def test_amc_never_ready_fails_the_binding(env):
    env.fake.ready = False
    env.controller.comes_up = False
    provision(env.client())
    b = env.store.get_by_pid("proj-1")
    assert b.state == "failed" and "did not become ready within 60 s" in b.error
    assert (env.root / "living-room").is_dir()  # kept for a retry


def test_floor_plan_is_saved(env):
    provision(env.client(), floorplan=True)
    assert (env.root / "living-room" / "floorplan" / "layout.png").read_bytes() == PNG
    assert env.downloads == [FLOORPLAN_URL]
    assert env.store.get_by_pid("proj-1").state == "ready"


def test_same_request_again_is_idempotent(env):
    c = env.client()
    assert provision(c).status_code == 202
    r = provision(c)
    assert r.status_code == 200 and r.json()["State"] == "ready"
    assert len(calls(env.fake, "create_project")) == 1


def test_idempotent_while_still_provisioning(slow):
    c = slow.client()
    provision(c)
    r = provision(c)
    assert r.status_code == 200 and r.json()["State"] == "provisioning"


def test_failed_binding_is_provisioned_again(env, monkeypatch):
    c = env.client()
    env.fake.ready = False
    env.controller.comes_up = False
    provision(c)
    assert env.store.get_by_pid("proj-1").state == "failed"
    env.controller.comes_up = True
    r = provision(c)
    assert r.status_code == 202
    b = env.store.get_by_pid("proj-1")
    assert b.state == "ready" and b.error == ""


def test_retry_reuses_the_amc_project_an_earlier_attempt_made(env):
    c = env.client()
    provision(c, floorplan=True)
    first = env.store.get_by_pid("proj-1").amc_project_id
    env.store.update("living-room", state="failed", error="floor plan download failed")
    assert provision(c, floorplan=True).status_code == 202
    assert env.store.get_by_pid("proj-1").amc_project_id == first
    assert len(calls(env.fake, "create_project")) == 1


def test_folder_of_another_project_is_409(env):
    c = env.client()
    provision(c)
    r = provision(c, pid="proj-2")
    assert r.status_code == 409 and "another web project" in r.json()["detail"]


def test_project_bound_to_another_folder_is_409(env):
    c = env.client()
    provision(c)
    r = provision(c, folder="kitchen")
    assert r.status_code == 409 and "living-room" in r.json()["detail"]
    assert not (env.root / "kitchen").exists()


def test_spec_without_folder_is_400(env):
    r = provision(env.client(), folder=None)
    assert r.status_code == 400 and "folder" in r.json()["detail"]


def test_spec_with_invalid_folder_is_400(env):
    assert provision(env.client(), folder="../etc").status_code == 400
    assert not env.root.exists()


def test_spec_for_another_project_is_400(env):
    r = env.client().post("/projects", json={"Spec": make_spec(pid="proj-2")}, headers=auth("proj-1", "amc-provision"))
    assert r.status_code == 400


def test_provisioner_refuses_spec_pid_mismatch(env):
    with pytest.raises(ProvisionError) as exc:
        env.provisioner.provision(env.spec(pid="proj-2"), "proj-1")
    assert exc.value.status == 400


# ---------------------------------------------------------------- busy rules
def test_second_provision_while_busy_is_409(slow):
    c = slow.client()
    assert provision(c).status_code == 202
    assert slow.provisioner.busy()
    r = provision(c, pid="proj-2", folder="kitchen")
    assert r.status_code == 409 and "being set up" in r.json()["detail"]
    slow.run_pending()
    assert not slow.provisioner.busy()
    assert provision(c, pid="proj-2", folder="kitchen").status_code == 202


def test_provision_while_run_active_is_409(env):
    env.runner.active = True
    r = provision(env.client())
    assert r.status_code == 409 and "calibration run" in r.json()["detail"]
    assert not env.root.exists()


def test_amc_stop_and_automation_start_and_continue_are_409_while_provisioning(slow):
    c = slow.client()
    provision(c)
    assert c.post("/amc/stop", headers=auth()).status_code == 409
    r = c.post("/amc/automation/start", json={"Spec": make_spec()}, headers=auth())
    assert r.status_code == 409 and "being set up" in r.json()["detail"]
    r = c.post("/amc/automation/continue", headers=auth())
    assert r.status_code == 409 and "being set up" in r.json()["detail"]
    assert slow.controller.stopped == 0 and slow.runner.started == [] and slow.runner.resumed == 0
    slow.run_pending()
    assert c.post("/amc/stop", headers=auth()).status_code == 202
    assert c.post("/amc/automation/continue", headers=auth()).status_code == 202
    slow.runner.active = False
    assert c.post("/amc/automation/start", json={"Spec": make_spec()}, headers=auth()).status_code == 202


def test_provision_racing_a_run_start_waits_and_is_refused(slow):
    """A provision arriving while a run is starting blocks on the check-and-start, then sees the run."""
    c = slow.client()
    outcome = {}

    def other_request():
        try:
            outcome["result"] = slow.provisioner.provision(slow.spec(pid="proj-2", folder="kitchen"), "proj-2")
        except ProvisionError as exc:
            outcome["error"] = exc

    def inside_start():
        t = threading.Thread(target=other_request)
        t.start()
        t.join(0.3)
        outcome["blocked"] = t.is_alive()
        outcome["thread"] = t

    slow.runner.inside_start = inside_start
    r = c.post("/amc/automation/start", json={"Spec": make_spec()}, headers=auth())
    assert r.status_code == 202
    outcome["thread"].join(5)
    assert outcome["blocked"] is True
    assert "result" not in outcome
    assert outcome["error"].status == 409 and "calibration run" in outcome["error"].message
    assert not (slow.root / "kitchen").exists() and slow.pending == []


def test_run_start_racing_a_provision_is_refused(slow):
    """The reverse order: a provision that won the lock makes the run start 409."""
    c = slow.client()
    assert provision(c).status_code == 202
    with pytest.raises(ProvisionError) as exc:
        slow.provisioner.run_exclusive(lambda: slow.runner.start(slow.spec()))
    assert exc.value.status == 409 and slow.runner.started == []


def test_amc_stop_is_refused_while_a_run_uses_amc(env):
    env.runner.active = True
    r = env.client().post("/amc/stop", headers=auth())
    assert r.status_code == 409 and "calibration run" in r.json()["detail"]
    assert env.controller.stopped == 0


def test_job_failure_clears_busy(env, monkeypatch):
    def boom(cfg):
        raise RuntimeError("no AMC here")

    monkeypatch.setattr(env.provisioner, "_api_factory", boom)
    provision(env.client())
    assert not env.provisioner.busy()
    b = env.store.get_by_pid("proj-1")
    assert b.state == "failed" and b.error == "no AMC here"


# ---------------------------------------------------------------- restart recovery
def test_recover_on_startup_fails_interrupted_bindings(env):
    env.store.create("proj-1", "living-room", "Living Room")
    env.store.create("proj-2", "kitchen", "Kitchen")
    env.store.update("kitchen", state="ready")
    env.client()  # create_app runs recover()
    assert env.store.get_by_folder("living-room").state == "failed"
    assert env.store.get_by_folder("living-room").error == RESTART_ERROR
    assert env.store.get_by_folder("kitchen").state == "ready"


def test_default_app_builds_its_store_from_the_environment(tmp_path, monkeypatch):
    root = tmp_path / "amc_projects"
    monkeypatch.setenv("AMC_PROJECTS_DIR", str(root))
    ProjectStore(root).create("proj-1", "living-room", "Living Room")
    app = create_app(SETTINGS, controller=FakeController(FakeAmc(), tmp_path), runner=FakeRunner())
    assert app.state.store.root == root
    b = TestClient(app).get("/projects/current", headers=auth()).json()
    assert b["State"] == "failed" and b["Error"] == RESTART_ERROR


# ---------------------------------------------------------------- scopes and missing folders
def test_provision_ticket_is_accepted_on_current(env):
    c = env.client()
    provision(c)
    assert c.get("/projects/current", headers=auth(scope="amc-provision")).status_code == 200
    assert c.get("/projects/current", headers=auth(scope="amc")).status_code == 200
    assert c.get("/projects/current", headers=auth(scope="tracking")).status_code == 403


def test_provision_ticket_is_refused_on_file_routes(env):
    c = env.client()
    provision(c)
    (env.root / "living-room" / "footage" / "cam_00.mp4").write_bytes(b"v")
    h = auth(scope="amc-provision")
    assert c.get("/projects/current/files", headers=h).status_code == 403
    assert c.get("/projects/current/files/footage/cam_00.mp4", headers=h).status_code == 403
    q = make_ticket(scope="amc-provision")
    assert c.get(f"/projects/current/files/footage/cam_00.mp4?ticket={q}").status_code == 403


def test_amc_ticket_is_refused_on_provision(env):
    r = env.client().post("/projects", json={"Spec": make_spec()}, headers=auth(scope="amc"))
    assert r.status_code == 403
    assert not env.root.exists()


@pytest.mark.parametrize("method,path", [
    ("post", "/projects"), ("get", "/projects/current"), ("get", "/projects/current/files"),
    ("get", "/projects/current/files/footage/cam_00.mp4"),
])
def test_project_routes_need_a_ticket(env, method, path):
    c = env.client()
    kw = {"json": {"Spec": make_spec()}} if method == "post" else {}
    assert getattr(c, method)(path, **kw).status_code == 401
    assert getattr(c, method)(path, headers={"Authorization": "Bearer junk"}, **kw).status_code == 401


@pytest.mark.parametrize("path", ["/projects/current", "/projects/current/files",
                                  "/projects/current/files/footage/cam_00.mp4"])
def test_no_folder_is_404(env, path):
    r = env.client().get(path, headers=auth())
    assert r.status_code == 404 and "no workstation folder" in r.json()["detail"]


def test_routes_answer_for_the_tickets_project_only(env):
    c = env.client()
    provision(c)
    assert c.get("/projects/current", headers=auth("proj-2")).status_code == 404
    assert c.get("/projects/current/files", headers=auth("proj-2")).status_code == 404


def test_cors_and_private_network_preflight_on_project_routes(env):
    c = env.client()
    for path, method in (("/projects", "POST"), ("/projects/current/files", "GET")):
        r = c.options(path, headers={"Origin": ORIGIN, "Access-Control-Request-Method": method,
                                     "Access-Control-Request-Headers": "authorization",
                                     "Access-Control-Request-Private-Network": "true"})
        assert r.status_code == 200
        assert r.headers["access-control-allow-origin"] == ORIGIN
        assert r.headers["access-control-allow-private-network"] == "true"


# ---------------------------------------------------------------- listing and download
@pytest.fixture
def files(env, tmp_path):
    c = env.client()
    provision(c, floorplan=True)
    folder = env.root / "living-room"
    (folder / "footage" / "cam_00.mp4").write_bytes(b"video-0")
    (folder / "footage" / "cam 01.mp4").write_bytes(b"video-1")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    (folder / "footage" / "escape.mp4").symlink_to(outside)
    (folder / "linked-dir").symlink_to(tmp_path)
    return c, folder


def test_listing(files):
    c, folder = files
    r = c.get("/projects/current/files", headers=auth())
    assert r.status_code == 200
    paths = [e["Path"] for e in r.json()]
    assert paths == ["floorplan/layout.png", "footage/cam 01.mp4", "footage/cam_00.mp4"]
    entry = r.json()[2]
    assert entry["Size"] == len(b"video-0") and isinstance(entry["ModifiedAt"], int)


def test_download_with_header(files):
    c, _ = files
    r = c.get("/projects/current/files/footage/cam_00.mp4", headers=auth())
    assert r.status_code == 200 and r.content == b"video-0"
    assert r.headers["content-type"] == "application/octet-stream"
    assert r.headers["content-disposition"].startswith("attachment")
    assert "cam_00.mp4" in r.headers["content-disposition"]


def test_download_with_percent_encoded_segment(files):
    c, _ = files
    r = c.get("/projects/current/files/footage/cam%2001.mp4", headers=auth())
    assert r.status_code == 200 and r.content == b"video-1"


def test_download_with_query_ticket(files):
    c, _ = files
    r = c.get(f"/projects/current/files/footage/cam_00.mp4?ticket={make_ticket()}")
    assert r.status_code == 200 and r.content == b"video-0"
    assert c.get("/projects/current/files/footage/cam_00.mp4?ticket=junk").status_code == 401


@pytest.mark.parametrize("path", [
    "%2E%2E/secret.txt", "footage/%2E%2E/%2E%2E/secret.txt", "%2Fetc%2Fpasswd",
    "footage/escape.mp4", "linked-dir/secret.txt", "project.json", "footage", "footage/missing.mp4",
])
def test_unsafe_or_missing_paths_are_404(files, path):
    c, _ = files
    r = c.get(f"/projects/current/files/{path}", headers=auth())
    assert r.status_code == 404, path


# ---------------------------------------------------------------- CLI
def test_cli_projects(env, monkeypatch, capsys):
    monkeypatch.setenv("AMC_PROJECTS_DIR", str(env.root))
    assert cli.main(["projects"]) == 0
    assert "no project folders" in capsys.readouterr().out

    provision(env.client())
    env.store.update("living-room", last_run_at=1794879242, calib_runs=2)
    env.store.create("proj-2", "kitchen", "Kitchen")
    env.store.update("kitchen", state="failed", error="AMC did not become ready")
    assert cli.main(["projects"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split()[:2] == ["FOLDER", "TITLE"] and lines[0].rstrip().endswith("PATH")
    kitchen, living = lines[1], lines[2]
    assert kitchen.startswith("kitchen") and "proj-2" in kitchen and "failed" in kitchen
    assert living.startswith("living-room") and "Living Room" in living and "proj-1" in living
    amc_id = env.store.get_by_folder("living-room").amc_project_id
    assert amc_id in living and "ready" in living and "2026-11-17 01:34 UTC" in living
    assert living.split()[-2] == "2" and living.endswith(str(env.root / "living-room"))
    assert "kitchen: failed: AMC did not become ready" in lines


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read any directory")
def test_cli_projects_reports_an_unreadable_root(env, monkeypatch, capsys):
    provision(env.client())
    monkeypatch.setenv("AMC_PROJECTS_DIR", str(env.root))
    env.root.chmod(0o000)
    try:
        assert cli.main(["projects"]) == 1
    finally:
        env.root.chmod(0o750)
    out = capsys.readouterr()
    assert "cannot read the projects root" in out.err and "no project folders" not in out.out
