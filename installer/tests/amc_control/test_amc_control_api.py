"""mv3dt_amc_control: AMC on/off control, tickets, CORS and the CLI.

Tested against a fake `amc` wrapper script and fake AMC HTTP servers, so no
GPU, Docker, sudo or tailscale is involved.

Run from installer/: `python3 -m pytest tests/amc_control -v`
"""

from __future__ import annotations

import json
import stat
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

jwt = pytest.importorskip("jwt")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("yaml")

from fastapi.testclient import TestClient  # noqa: E402

from mv3dt_amc_control import cli  # noqa: E402
from mv3dt_amc_control.api import create_app  # noqa: E402
from mv3dt_amc_control.controller import AmcController  # noqa: E402
from mv3dt_amc_control.settings import Settings  # noqa: E402

SECRET = "ticket-secret-ticket-secret-ticket-secret"
ORIGIN = "https://app.example.org"
READY = {"on": False}


def make_ticket(secret=SECRET, exp_delta=900, aud="p2bp-workstation", scope="amc", pid="proj-1", sub="user-1",
                iss=None):
    claims = {"sub": sub, "pid": pid, "scope": scope, "aud": aud, "exp": int(time.time()) + exp_delta}
    if iss:
        claims["iss"] = iss
    return jwt.encode(claims, secret, algorithm="HS256")


class FakeAmcHandler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"code": 0}).encode() if self.path == "/v1/ready" else b"<html/>"
        self.send_response(200 if READY["on"] else 503)
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def amc_env(monkeypatch, tmp_path):
    """An install tree, a fake amc wrapper and fake AMC servers. Returns (calls log, UI port)."""
    READY["on"] = False
    servers = [HTTPServer(("127.0.0.1", 0), FakeAmcHandler) for _ in range(2)]
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    ms, ui = (s.server_address[1] for s in servers)
    (tmp_path / "installer.conf").write_text(
        f"AUTO_MAGIC_CALIB_MS_PORT={ms}\nAUTO_MAGIC_CALIB_UI_PORT={ui}\nAMC_PROJECT_ID=20261019_151133_2885\n"
    )
    calls = tmp_path / "calls.log"
    amc = tmp_path / "fake_amc"
    amc.write_text(
        "#!/bin/sh\n"
        f'echo "$@" >> {calls}\n'
        '[ "$1" = "--down" ] && exit 0\n'
        '[ -n "$FAKE_FAIL" ] && { echo boom >&2; exit 3; }\n'
        "sleep 0.5\nexit 0\n"
    )
    amc.chmod(amc.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("MV3DT_INSTALL_DIR", str(tmp_path))
    monkeypatch.setenv("AMC_BIN", str(amc))
    monkeypatch.setenv("AMC_USE_SUDO", "0")
    monkeypatch.setenv("AMC_PUBLIC_HOST", "100.64.0.7")
    monkeypatch.delenv("AMC_STATE_DIR", raising=False)
    yield calls, ui
    for s in servers:
        s.shutdown()


def make_client(settings=None, ticket=None):
    settings = settings or Settings(ticket_secret=SECRET, allowed_origins=[ORIGIN])
    client = TestClient(create_app(settings, controller=AmcController()))
    client.headers["Authorization"] = f"Bearer {ticket or make_ticket()}"
    return client


@pytest.fixture
def client(amc_env):
    return make_client()


def wait_state(client, want, secs=5):
    end = time.time() + secs
    last = None
    while time.time() < end:
        last = client.get("/amc/status").json()
        if last["State"] == want:
            return last
        time.sleep(0.1)
    raise AssertionError(f"never reached {want}; last={last}")


# ---------------------------------------------------------------- tickets
@pytest.mark.parametrize("method,path", [
    ("get", "/amc/status"), ("post", "/amc/start"), ("post", "/amc/stop"),
    ("get", "/amc/automation/status"), ("post", "/amc/automation/continue"),
    ("post", "/amc/automation/cancel"), ("get", "/amc/automation/config"),
])
def test_every_route_needs_a_ticket(amc_env, method, path):
    c = TestClient(create_app(Settings(ticket_secret=SECRET), controller=AmcController()))
    assert getattr(c, method)(path).status_code == 401
    assert getattr(c, method)(path, headers={"Authorization": "Bearer junk"}).status_code == 401


@pytest.mark.parametrize("kw", [{"secret": "wrong-secret-wrong-secret-wrong-secret"}, {"exp_delta": -10},
                                {"aud": "authenticated"}])
def test_bad_tickets_are_a_401(amc_env, kw):
    assert make_client(ticket=make_ticket(**kw)).get("/amc/status").status_code == 401


def test_a_tracking_ticket_cannot_control_amc(amc_env):
    """Both services share the secret; the scope keeps their tickets apart."""
    c = make_client(ticket=make_ticket(scope="tracking"))
    assert c.post("/amc/start").status_code == 403
    assert not amc_env[0].exists()


def test_project_comes_from_tracking_project_id_not_amcs_own_id(amc_env):
    """installer.conf's AMC_PROJECT_ID is AMC's id and must not be read as the web project."""
    settings = Settings.from_env({
        "WORKSTATION_TICKET_SECRET": SECRET, "TRACKING_PROJECT_ID": "proj-1", "ALLOWED_ORIGINS": ORIGIN,
        "AMC_PROJECT_ID": "20261019_151133_2885",
    })
    assert settings.project_id == "proj-1"
    assert make_client(settings, make_ticket(pid="proj-1")).get("/amc/status").status_code == 200
    assert make_client(settings, make_ticket(pid="other")).get("/amc/status").status_code == 403


def test_issuer_is_checked_when_configured(amc_env):
    settings = Settings(ticket_secret=SECRET, ticket_issuer="betterplacemaking")
    assert make_client(settings, make_ticket(iss="betterplacemaking")).get("/amc/status").status_code == 200
    assert make_client(settings, make_ticket(iss="someone-else")).get("/amc/status").status_code == 401
    assert make_client(settings, make_ticket()).get("/amc/status").status_code == 401


def test_missing_secret_is_a_503_not_open_access(amc_env):
    c = make_client(Settings(ticket_secret=""))
    assert c.get("/amc/status").status_code == 503


def test_health_is_open_and_holds_no_secret(amc_env):
    c = TestClient(create_app(Settings(ticket_secret=SECRET), controller=AmcController()))
    body = c.get("/health").json()
    assert body["status"] == "ok" and body["ticket_configured"] is True
    assert SECRET not in json.dumps(body)


# ---------------------------------------------------------------- browser access
def preflight(client, origin, private_network=True):
    headers = {"Origin": origin, "Access-Control-Request-Method": "POST",
               "Access-Control-Request-Headers": "authorization"}
    if private_network:
        headers["Access-Control-Request-Private-Network"] = "true"
    return client.options("/amc/start", headers=headers)


def test_cors_allows_only_the_configured_origins(client):
    ok = preflight(client, ORIGIN, private_network=False)
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == ORIGIN
    assert "access-control-allow-private-network" not in ok.headers
    bad = preflight(client, "https://evil.example", private_network=False)
    assert "access-control-allow-origin" not in bad.headers


def test_chrome_private_network_preflight_is_answered_for_allowed_origins_only(client):
    ok = preflight(client, ORIGIN)
    assert ok.status_code == 200          # Chrome needs a 2xx, not only the header
    assert ok.headers["access-control-allow-private-network"] == "true"
    assert ok.headers["access-control-allow-origin"] == ORIGIN
    bad = preflight(client, "https://evil.example")
    assert bad.status_code == 400 and "access-control-allow-origin" not in bad.headers


def test_wildcard_origin_is_refused():
    with pytest.raises(ValueError):
        create_app(Settings(ticket_secret=SECRET, allowed_origins=["*"]))


# ---------------------------------------------------------------- AMC on/off
def test_start_then_ready_then_stop(client, amc_env):
    calls, ui = amc_env
    assert client.get("/amc/status").json()["State"] == "down"

    r = client.post("/amc/start")
    assert r.status_code == 202 and r.json()["State"] == "starting"
    READY["on"] = True
    d = wait_state(client, "ready")
    assert d["UiUrl"] == f"http://100.64.0.7:{ui}"
    # Exactly the argv the Step 9 sudoers rule allows.
    assert calls.read_text().splitlines() == ["--keep-up --no-open --non-interactive --host-ip 100.64.0.7"]

    READY["on"] = False
    assert client.post("/amc/stop").status_code == 202
    assert wait_state(client, "down")["UiUrl"] == ""
    assert calls.read_text().splitlines()[-1] == "--down"


def test_start_is_idempotent(client, amc_env):
    for _ in range(3):
        client.post("/amc/start")
    assert len(amc_env[0].read_text().splitlines()) == 1


def test_start_when_already_ready_launches_nothing(client, amc_env):
    READY["on"] = True
    assert client.post("/amc/start").json()["State"] == "ready"
    assert not amc_env[0].exists()


def test_failed_start_reports_the_log(client, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL", "1")
    client.post("/amc/start")
    d = wait_state(client, "error")
    assert "exited 3" in d["Error"] and "boom" in d["Error"]


def test_no_tailscale_address_is_a_clear_error(client, amc_env, monkeypatch):
    monkeypatch.setenv("AMC_PUBLIC_HOST", "")
    monkeypatch.setattr(AmcController, "_tailscale_host", lambda self: None)
    d = client.post("/amc/start").json()
    assert d["State"] == "error" and "Tailscale" in d["Error"]
    assert not amc_env[0].exists()


def test_bad_host_is_refused_not_passed_to_argv(client, amc_env, monkeypatch):
    monkeypatch.setenv("AMC_PUBLIC_HOST", "x; rm -rf /")
    assert client.post("/amc/start").json()["State"] == "error"
    assert not amc_env[0].exists()


def test_stop_during_start_does_not_report_the_aborted_start(client):
    client.post("/amc/start")
    client.post("/amc/stop")
    assert wait_state(client, "down")["Error"] == ""


def test_missing_install_tree_is_down_not_a_crash(client, monkeypatch, tmp_path):
    monkeypatch.setenv("MV3DT_INSTALL_DIR", str(tmp_path / "nope"))
    assert client.get("/amc/status").json()["State"] == "down"


def test_logs_go_to_the_services_own_state_folder(client, amc_env, tmp_path):
    """<install_dir>/run belongs to root; the service writes under amc_control/state."""
    client.post("/amc/start")
    assert (tmp_path / "amc_control" / "state" / "amc_service.log").exists()
    assert not (tmp_path / "run").exists()


def test_fallback_address_is_the_tailscale_ipv4(monkeypatch):
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setenv("AMC_PUBLIC_HOST", "")
    seen = []

    def fake_run(argv, **kwargs):
        seen.append(argv)
        return SimpleNamespace(returncode=0, stdout="100.101.102.103\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert AmcController().public_host() == "100.101.102.103"
    assert seen == [["tailscale", "ip", "-4"]]


# ---------------------------------------------------------------- settings and CLI
def test_settings_from_env():
    s = Settings.from_env({
        "WORKSTATION_TICKET_SECRET": SECRET, "TRACKING_PROJECT_ID": " proj-1 ",
        "ALLOWED_ORIGINS": "https://a.example/, http://localhost:5173", "TICKET_ISSUER": "",
        "MV3DT_INSTALL_DIR": "/srv/mv3dt", "AMC_AUTOMATION_CONFIG": "/srv/mv3dt/amc_control/amc_automation.yaml",
    })
    assert s.project_id == "proj-1" and s.ticket_issuer is None
    assert s.allowed_origins == ["https://a.example", "http://localhost:5173"]
    assert str(s.install_dir) == "/srv/mv3dt" and s.bind_address() == ("127.0.0.1", 8091)


@pytest.mark.parametrize("env,message", [
    ({}, "WORKSTATION_TICKET_SECRET"),
    ({"WORKSTATION_TICKET_SECRET": SECRET, "TRACKING_PROJECT_ID": "p"}, "ALLOWED_ORIGINS"),
    ({"WORKSTATION_TICKET_SECRET": SECRET, "TRACKING_PROJECT_ID": "p", "ALLOWED_ORIGINS": "*"}, "not *"),
    ({"WORKSTATION_TICKET_SECRET": SECRET, "TRACKING_PROJECT_ID": "p", "ALLOWED_ORIGINS": ORIGIN,
      "BIND": "0.0.0.0:8091"}, "only loopback"),
    ({"WORKSTATION_TICKET_SECRET": SECRET, "TRACKING_PROJECT_ID": "p", "ALLOWED_ORIGINS": ORIGIN,
      "BIND": "nonsense"}, "host:port"),
])
def test_serve_refuses_an_unsafe_or_incomplete_setup(env, message, capsys):
    assert cli._serve(Settings.from_env(env)) == 2
    assert message in capsys.readouterr().err


def test_check_config_reports_a_bad_override(tmp_path, capsys):
    good = tmp_path / "amc_automation.yaml"
    good.write_text("project:\n  mode: create_new\n")
    assert cli._check_config(Settings(automation_config=good)) == 0
    assert "project.mode=create_new" in capsys.readouterr().out
    good.write_text("project:\n  mood: create_new\n")
    assert cli._check_config(Settings(automation_config=good)) == 1


def test_missing_override_file_means_base_settings(tmp_path, capsys):
    assert cli._check_config(Settings(automation_config=tmp_path / "absent.yaml")) == 0
    assert "project.mode=reuse_installer_project" in capsys.readouterr().out
