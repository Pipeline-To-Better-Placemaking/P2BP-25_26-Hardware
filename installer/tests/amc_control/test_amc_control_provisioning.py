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
        # Like AmcController: start() clears the error and may set its own
        # (start_error); a start process that already exited with an error is
        # only recorded when status() looks (exited_error).
        self.error = ""
        self.start_error = ""
        self.exited_error = ""

    def start(self):
        self.started += 1
        self.error = self.start_error
        if self.comes_up:
            self.fake.ready = True

    def stop(self):
        self.stopped += 1

    def status(self):
        if self.exited_error:
            self.error, self.exited_error = self.exited_error, ""
        return {"State": "ready" if self.fake.ready else "down", "UiUrl": "", "Error": self.error, "UpdatedAt": 0}

    def ports(self):
        return 18000, 15000


class FakeRunner:
    def __init__(self):
        self.active = False
        self.started = []
        self.resumed = 0
        self.inside_start = None  # a test hook run inside start()
        self.job_pid = None  # set by a test: status() then reports a run of this project
        self.job_state = "running"

    def uses_amc(self):
        return self.active

    def status(self):
        if not self.job_pid:
            return None
        return {**self._job(self.job_pid), "State": self.job_state}

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
            runner_job=lambda: self.runner.status(),
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
    (env.root / "living-room" / "calibration" / "result.yml").write_bytes(b"r")
    h = auth(scope="amc-provision")
    assert c.get("/projects/current/files", headers=h).status_code == 403
    assert c.get("/projects/current/files/calibration/result.yml", headers=h).status_code == 403
    q = make_ticket(scope="amc-provision")
    assert c.get(f"/projects/current/files/calibration/result.yml?ticket={q}").status_code == 403


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
                                  "/projects/current/files/floorplan/layout.png"])
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
    (folder / "footage" / "notes.txt").write_bytes(b"footage notes")
    (folder / "calibration" / "a.MP4").write_bytes(b"video-a")
    (folder / "calibration" / "result.yml").write_bytes(b"result-0")
    (folder / "calibration" / "run 1.yml").write_bytes(b"result-1")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    (folder / "calibration" / "escape.yml").symlink_to(outside)
    (folder / "linked-dir").symlink_to(tmp_path)
    (folder / "calibration" / "clip-link.yml").symlink_to(folder / "footage" / "cam_00.mp4")
    (folder / "calibration" / "notes-link.yml").symlink_to(folder / "footage" / "notes.txt")
    (folder / "calibration" / "result-link.yml").symlink_to(folder / "calibration" / "result.yml")
    return c, folder


def test_listing(files):
    c, folder = files
    r = c.get("/projects/current/files", headers=auth())
    assert r.status_code == 200
    paths = [e["Path"] for e in r.json()]
    assert paths == ["calibration/a.MP4", "calibration/result.yml", "calibration/run 1.yml",
                     "floorplan/layout.png", "footage/cam_00.mp4", "footage/notes.txt"]
    entry = r.json()[4]
    assert entry["Size"] == len(b"video-0") and isinstance(entry["ModifiedAt"], int)


def test_listing_marks_video_files_not_downloadable(files):
    c, _ = files
    flags = {e["Path"]: e["Downloadable"] for e in c.get("/projects/current/files", headers=auth()).json()}
    assert flags == {"calibration/a.MP4": False, "calibration/result.yml": True, "calibration/run 1.yml": True,
                     "floorplan/layout.png": True, "footage/cam_00.mp4": False, "footage/notes.txt": False}


def test_download_with_header(files):
    c, _ = files
    r = c.get("/projects/current/files/calibration/result.yml", headers=auth())
    assert r.status_code == 200 and r.content == b"result-0"
    assert r.headers["content-type"] == "application/octet-stream"
    assert r.headers["content-disposition"].startswith("attachment")
    assert "result.yml" in r.headers["content-disposition"]


def test_download_with_percent_encoded_segment(files):
    c, _ = files
    r = c.get("/projects/current/files/calibration/run%201.yml", headers=auth())
    assert r.status_code == 200 and r.content == b"result-1"


def test_download_with_query_ticket(files):
    c, _ = files
    r = c.get(f"/projects/current/files/calibration/result.yml?ticket={make_ticket()}")
    assert r.status_code == 200 and r.content == b"result-0"
    assert c.get("/projects/current/files/calibration/result.yml?ticket=junk").status_code == 401


@pytest.mark.parametrize("path", [
    "footage/cam_00.mp4", "footage/notes.txt", "calibration/a.MP4", "footage", "footage/missing.txt",
    "x.mkv", "a/b.MOV", "c.avi", "d.ts", "e.h264", "f.H265", "footage%2Fcam_00.mp4",
    "calibration/%2E%2E/footage/cam_00.mp4", "./footage/notes.txt",
    "calibration/clip-link.yml", "calibration/notes-link.yml",
])
def test_video_files_are_403(files, path):
    c, _ = files
    r = c.get(f"/projects/current/files/{path}", headers=auth())
    assert r.status_code == 403, path
    assert r.json()["detail"] == "Video files stay on the workstation"
    assert c.get(f"/projects/current/files/{path}?ticket={make_ticket()}").status_code == 403, path


def test_link_to_a_non_video_file_still_downloads(files):
    c, _ = files
    r = c.get("/projects/current/files/calibration/result-link.yml", headers=auth())
    assert r.status_code == 200 and r.content == b"result-0"


def test_video_rule_is_decided_before_any_disk_access(env, monkeypatch):
    """No folder, and the store is never consulted: the path alone decides."""
    c = env.client()

    def boom(*a, **kw):
        raise AssertionError("touched the disk")

    monkeypatch.setattr(env.store, "get_by_pid", boom)
    monkeypatch.setattr(env.store, "resolve_file", boom)
    assert c.get("/projects/current/files/footage/cam_00.mp4", headers=auth()).status_code == 403


@pytest.mark.parametrize("path", [
    "%2E%2E/secret.txt", "calibration/%2E%2E/%2E%2E/secret.txt", "%2Fetc%2Fpasswd",
    "calibration/escape.yml", "linked-dir/secret.txt", "project.json", "calibration", "calibration/missing.yml",
])
def test_unsafe_or_missing_paths_are_404(files, path):
    c, _ = files
    r = c.get(f"/projects/current/files/{path}", headers=auth())
    assert r.status_code == 404, path


# ---------------------------------------------------------------- deleting a project
def delete(client, pid="proj-1", scope="amc-delete"):
    return client.delete("/projects/current", headers=auth(pid, scope))


def test_delete_removes_the_amc_project_and_the_folder(env):
    c = env.client()
    provision(c)
    amc_id = env.store.get_by_pid("proj-1").amc_project_id
    (env.root / "living-room" / "footage" / "cam_00.mp4").write_bytes(b"v")
    r = delete(c)
    assert r.status_code == 200
    assert r.json() == {"Folder": "living-room", "AmcProjectId": amc_id, "AmcProjectDeleted": True,
                        "Leftover": None}
    assert calls(env.fake, "delete_project") == [{"pid": amc_id}]
    assert amc_id not in env.fake.projects
    assert list(env.root.iterdir()) == []
    assert env.store.get_by_pid("proj-1") is None
    assert c.get("/projects/current", headers=auth()).status_code == 404
    assert env.controller.stopped == 0  # AMC is left running


def test_delete_without_an_amc_project_skips_amc(env):
    env.store.create("proj-1", "living-room", "Living Room")
    env.fake.ready = False
    r = delete(env.client())
    assert r.status_code == 200
    assert r.json() == {"Folder": "living-room", "AmcProjectId": None, "AmcProjectDeleted": False,
                        "Leftover": None}
    assert env.controller.started == 0 and calls(env.fake, "delete_project") == []
    assert not (env.root / "living-room").exists()


def test_delete_while_amc_is_down_starts_it_and_is_503_without_waiting(env):
    c = env.client()
    provision(c)
    env.fake.ready = False
    env.controller.comes_up = False
    started, now = env.controller.started, env.now[0]
    r = delete(c)
    assert r.status_code == 503 and r.headers["retry-after"] == "15"
    assert "AMC is starting" in r.json()["detail"] and "last start error" not in r.json()["detail"]
    assert env.controller.started == started + 1 and env.now[0] == now  # never waited
    assert calls(env.fake, "delete_project") == []
    assert env.store.get_by_pid("proj-1") is not None and (env.root / "living-room").is_dir()
    assert not env.provisioner.busy()
    env.fake.ready = True  # AMC came up; the client retries
    assert delete(c).status_code == 200
    assert not (env.root / "living-room").exists()


@pytest.mark.parametrize("exited,at_start,shown", [
    ("amc start exited 1: pull failed", "", "amc start exited 1: pull failed"),  # an earlier start's failure
    ("", "Tailscale address unavailable", "Tailscale address unavailable"),       # this start's failure
    ("amc start exited 1: pull failed", "Tailscale address unavailable", "Tailscale address unavailable"),
])
def test_delete_503_includes_the_controllers_last_start_error(env, exited, at_start, shown):
    c = env.client()
    provision(c)
    env.fake.ready = False
    env.controller.comes_up = False
    env.controller.exited_error, env.controller.start_error = exited, at_start
    r = delete(c)
    assert r.status_code == 503 and r.headers["retry-after"] == "15"
    detail = r.json()["detail"]
    assert detail.endswith(f"(last start error: {shown})")
    assert detail.count("last start error") == 1


def test_delete_503_keeps_a_failed_earlier_start_with_the_real_controller(env, monkeypatch, tmp_path):
    """Against AmcController itself: start() clears its error, so it must be read first."""
    from mv3dt_amc_control import controller as controller_mod
    from mv3dt_amc_control.amc_api import AmcApi

    monkeypatch.setenv("MV3DT_INSTALL_DIR", str(tmp_path / "mv3dt"))
    monkeypatch.setenv("AMC_STATE_DIR", str(tmp_path / "amc-state"))
    monkeypatch.setenv("AMC_PUBLIC_HOST", "100.64.0.7")

    class Proc:
        def __init__(self, rc):
            self.rc = rc

        def poll(self):
            return self.rc

    procs = [Proc(1), Proc(None)]  # the earlier start exits 1; the delete's start keeps running
    launched = []

    def popen(cmd, **kw):
        launched.append(cmd)
        return procs.pop(0)

    monkeypatch.setattr(controller_mod.subprocess, "Popen", popen)
    real = controller_mod.AmcController()
    monkeypatch.setattr(real, "ready", lambda: False)
    real.start()  # the earlier, failed start; nobody has looked at status() since
    (tmp_path / "amc-state" / "amc_service.log").write_text("docker pull failed\n")

    env.store.create("proj-1", "living-room", "Living Room")
    env.store.update("living-room", amc_project_id="20260001", state="ready")
    env.fake.ready = False
    prov = Provisioner(env.store, real, config_loader=env.provisioner._load_config,
                       api_factory=lambda cfg: AmcApi("http://amc/v1",
                                                      client=TestClient(env.fake.app(), base_url="http://amc")))
    with pytest.raises(ProvisionError) as exc:
        prov.delete("proj-1")
    assert exc.value.status == 503 and exc.value.retry_after == 15
    assert "last start error: amc start exited 1" in exc.value.message
    assert "docker pull failed" in exc.value.message
    assert len(launched) == 2  # the delete asked AMC to start again
    assert (env.root / "living-room").is_dir()


def test_delete_503_exposes_retry_after_to_the_browser(env):
    c = env.client()
    provision(c)
    env.fake.ready = False
    env.controller.comes_up = False
    r = c.delete("/projects/current", headers={**auth(scope="amc-delete"), "Origin": ORIGIN})
    assert r.status_code == 503 and r.headers["retry-after"] == "15"
    assert r.headers["access-control-allow-origin"] == ORIGIN
    exposed = [h.strip().lower() for h in r.headers["access-control-expose-headers"].split(",")]
    assert "retry-after" in exposed


def test_amc_404_on_delete_counts_as_deleted(env):
    c = env.client()
    provision(c)
    env.fake.projects.clear()
    r = delete(c)
    assert r.status_code == 200 and r.json()["AmcProjectDeleted"] is True
    assert not (env.root / "living-room").exists()


@pytest.mark.parametrize("status", [400, 409, 500])
def test_amc_refusing_the_delete_is_502_and_removes_nothing(env, status):
    c = env.client()
    provision(c)
    (env.root / "living-room" / "footage" / "cam_00.mp4").write_bytes(b"v")
    env.fake.delete_status = status
    r = delete(c)
    assert r.status_code == 502 and f"HTTP {status}" in r.json()["detail"]
    assert (env.root / "living-room" / "footage" / "cam_00.mp4").read_bytes() == b"v"
    assert env.store.get_by_pid("proj-1").state == "ready"
    assert [p.name for p in env.root.iterdir()] == ["living-room"]
    env.fake.delete_status = None  # the user retries
    assert delete(c).status_code == 200


def test_delete_without_a_folder_is_404(env):
    r = delete(env.client())
    assert r.status_code == 404 and "no workstation folder" in r.json()["detail"]


def test_delete_answers_for_the_tickets_project_only(env):
    c = env.client()
    provision(c)
    assert delete(c, pid="proj-2").status_code == 404
    assert env.store.get_by_pid("proj-1") is not None


@pytest.mark.parametrize("state", ["running", "waiting_for_user", "failed", "interrupted"])
def test_delete_is_409_while_this_project_has_an_unfinished_run(env, state):
    c = env.client()
    provision(c)
    env.runner.job_pid, env.runner.job_state = "proj-1", state
    r = delete(c)
    assert r.status_code == 409 and "Cancel this project's run first" in r.json()["detail"]
    assert calls(env.fake, "delete_project") == [] and env.store.get_by_pid("proj-1") is not None


@pytest.mark.parametrize("state", ["completed", "cancelled"])
def test_delete_is_allowed_after_this_projects_run_finished(env, state):
    c = env.client()
    provision(c)
    env.runner.job_pid, env.runner.job_state = "proj-1", state
    assert delete(c).status_code == 200


def test_delete_is_allowed_while_another_projects_run_is_active(env):
    c = env.client()
    provision(c)
    env.runner.active, env.runner.job_pid = True, "proj-2"
    assert delete(c).status_code == 200


def test_delete_is_409_while_provisioning(slow):
    c = slow.client()
    provision(c)
    r = delete(c)
    assert r.status_code == 409 and "being set up" in r.json()["detail"]
    assert (slow.root / "living-room").is_dir()


def test_create_app_wires_the_runners_project_into_delete(env):
    """create_app builds its own provisioner from the runner's status()."""
    env.store.create("proj-1", "living-room", "Living Room")
    env.runner.job_pid, env.runner.job_state = "proj-1", "failed"
    app = create_app(SETTINGS, controller=env.controller, runner=env.runner, store=env.store)
    r = TestClient(app).delete("/projects/current", headers=auth(scope="amc-delete"))
    assert r.status_code == 409 and "Cancel this project's run first" in r.json()["detail"]


@pytest.mark.parametrize("scope", ["amc", "amc-provision", "tracking"])
def test_only_a_delete_ticket_can_delete(env, scope):
    c = env.client()
    provision(c)
    r = delete(c, scope=scope)
    assert r.status_code == 403 and "project deletion" in r.json()["detail"]
    assert env.store.get_by_pid("proj-1") is not None


def test_delete_ticket_is_refused_elsewhere(env):
    c = env.client()
    provision(c)
    h = auth(scope="amc-delete")
    assert c.get("/projects/current", headers=h).status_code == 403
    assert c.get("/projects/current/files", headers=h).status_code == 403
    assert c.post("/amc/start", headers=h).status_code == 403
    assert c.post("/projects", json={"Spec": make_spec()}, headers=h).status_code == 403


def test_delete_needs_a_ticket(env):
    c = env.client()
    assert c.delete("/projects/current").status_code == 401
    assert c.delete("/projects/current", headers={"Authorization": "Bearer junk"}).status_code == 401


def test_folder_name_is_reusable_after_delete(env):
    c = env.client()
    provision(c)
    assert delete(c).status_code == 200
    assert provision(c, pid="proj-2").status_code == 202
    b = env.store.get_by_pid("proj-2")
    assert b.folder == "living-room" and b.state == "ready"


def test_removal_error_is_reported_as_leftover(env, monkeypatch):
    c = env.client()
    provision(c)

    def fail(path, *a, **kw):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr("mv3dt_amc_control.projects.shutil.rmtree", fail)
    r = delete(c)
    assert r.status_code == 200
    leftover = r.json()["Leftover"]
    assert leftover and os.path.basename(leftover).startswith(".deleting-living-room-")
    assert os.path.isdir(leftover)
    assert env.store.get_by_pid("proj-1") is None and env.store.list() == []
    assert provision(c, pid="proj-2").status_code == 202  # the name is free anyway


def test_delete_cors_preflight_allows_delete(env):
    r = env.client().options("/projects/current", headers={
        "Origin": ORIGIN, "Access-Control-Request-Method": "DELETE",
        "Access-Control-Request-Headers": "authorization", "Access-Control-Request-Private-Network": "true"})
    assert r.status_code == 200 and "DELETE" in r.headers["access-control-allow-methods"]


def test_amc_api_delete_project_and_stop_calibration():
    from mv3dt_amc_control.amc_api import AmcApi, AmcApiError

    fake = FakeAmc()
    fake.add_project("p1")
    fake.projects["p1"]["state"] = "RUNNING"
    api = AmcApi("http://amc/v1", client=TestClient(fake.app(), base_url="http://amc"))
    api.stop_calibration("p1")
    assert fake.projects["p1"]["state"] == "INIT"
    assert api.delete_project("p1") is True and "p1" not in fake.projects
    assert api.delete_project("p1") is False  # AMC's 404: already gone
    with pytest.raises(AmcApiError) as exc:
        api.stop_calibration("p1")
    assert exc.value.status == 404
    fake.add_project("p2")
    fake.delete_status = 500
    with pytest.raises(AmcApiError) as exc:
        api.delete_project("p2")
    assert exc.value.status == 500 and "delete_project failed" in str(exc.value)


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
