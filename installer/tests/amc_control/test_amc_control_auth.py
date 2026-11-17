"""mv3dt_amc_control: ticket scopes and the calibration spec's folder.

Run from installer/: `python3 -m pytest tests/amc_control -v`
"""

from __future__ import annotations

import time

import pytest

jwt = pytest.importorskip("jwt")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi import Depends, FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from mv3dt_amc_control.auth import PROVISION_SCOPE, SCOPE, TicketError, ticket_dependency, verify_ticket  # noqa: E402
from mv3dt_amc_control.calibration_spec import SpecError, verify_spec  # noqa: E402

SECRET = "ticket-secret-ticket-secret-ticket-secret"


def mint(scope=SCOPE, pid="proj-1", **extra):
    claims = {"sub": "user-1", "pid": pid, "scope": scope, "aud": "p2bp-workstation",
              "exp": int(time.time()) + 900, **extra}
    return jwt.encode(claims, SECRET, algorithm="HS256")


# ---------------------------------------------------------------- scopes
def test_scope_constants():
    assert SCOPE == "amc" and PROVISION_SCOPE == "amc-provision"


@pytest.mark.parametrize("pid", ["proj-1", "another-project", "42"])
def test_any_project_is_accepted(pid):
    assert verify_ticket(mint(pid=pid), SECRET)["pid"] == pid


def test_verify_ticket_defaults_to_the_amc_scope():
    assert verify_ticket(mint(), SECRET)["scope"] == SCOPE
    with pytest.raises(TicketError) as exc:
        verify_ticket(mint(scope=PROVISION_SCOPE), SECRET)
    assert exc.value.status == 403


@pytest.mark.parametrize("want,given", [
    (SCOPE, PROVISION_SCOPE), (PROVISION_SCOPE, SCOPE), (PROVISION_SCOPE, "tracking"), (SCOPE, "tracking"),
])
def test_each_check_accepts_exactly_one_scope(want, given):
    assert verify_ticket(mint(scope=want), SECRET, scope=want)["scope"] == want
    with pytest.raises(TicketError) as exc:
        verify_ticket(mint(scope=given), SECRET, scope=want)
    assert exc.value.status == 403


@pytest.mark.parametrize("pid", [None, ""])
def test_a_ticket_without_a_project_is_refused(pid):
    claims = {"sub": "user-1", "scope": SCOPE, "aud": "p2bp-workstation", "exp": int(time.time()) + 900}
    if pid is not None:
        claims["pid"] = pid
    with pytest.raises(TicketError) as exc:
        verify_ticket(jwt.encode(claims, SECRET, algorithm="HS256"), SECRET)
    assert exc.value.status == 401


def scoped_app():
    app = FastAPI()
    amc = ticket_dependency(SECRET, None)
    provision = ticket_dependency(SECRET, None, PROVISION_SCOPE)

    @app.get("/amc")
    def amc_route(claims: dict = Depends(amc)):
        return claims

    @app.post("/projects")
    def provision_route(claims: dict = Depends(provision)):
        return claims

    return TestClient(app)


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_provision_dependency_accepts_only_provision_tickets():
    c = scoped_app()
    assert c.post("/projects", headers=bearer(mint(scope=PROVISION_SCOPE, pid="any"))).status_code == 200
    assert c.post("/projects", headers=bearer(mint(scope=SCOPE))).status_code == 403
    assert c.post("/projects", headers=bearer(mint(scope="tracking"))).status_code == 403
    assert c.post("/projects").status_code == 401


def test_amc_dependency_refuses_provision_tickets():
    c = scoped_app()
    assert c.get("/amc", headers=bearer(mint(scope=SCOPE, pid="any"))).status_code == 200
    assert c.get("/amc", headers=bearer(mint(scope=PROVISION_SCOPE))).status_code == 403


# ---------------------------------------------------------------- spec folder
def spec_token(spec, pid="proj-1"):
    claims = {"sub": "user-1", "pid": pid, "scope": "amc-calibration-spec", "aud": "p2bp-workstation",
              "exp": int(time.time()) + 900, "spec": spec}
    return jwt.encode(claims, SECRET, algorithm="HS256")


@pytest.mark.parametrize("folder", ["a", "0", "plaza-north", "site-2026", "a" * 48, "a" + "-" * 46 + "b"])
def test_a_valid_folder_is_kept(folder):
    assert verify_spec(spec_token({"title": "T", "folder": folder}), SECRET, "proj-1").folder == folder


def test_absent_or_null_folder_is_none():
    assert verify_spec(spec_token({"title": "T"}), SECRET, "proj-1").folder is None
    assert verify_spec(spec_token({"title": "T", "folder": None}), SECRET, "proj-1").folder is None


@pytest.mark.parametrize("folder", [
    "", "-a", "a-", "Plaza", "plaza_north", "plaza north", "a/b", "..", "a" * 49, "é", "plaza\n",
])
def test_an_invalid_folder_is_a_spec_error(folder):
    with pytest.raises(SpecError):
        verify_spec(spec_token({"title": "T", "folder": folder}), SECRET, "proj-1")


@pytest.mark.parametrize("folder", [1, True, [], {"name": "a"}])
def test_a_non_string_folder_is_a_spec_error(folder):
    with pytest.raises(SpecError):
        verify_spec(spec_token({"title": "T", "folder": folder}), SECRET, "proj-1")
