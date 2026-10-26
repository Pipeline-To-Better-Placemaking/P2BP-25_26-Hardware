"""Tests for `mv3dt_tracking.auth` (HS256 ticket verification).

Run from installer/: `python3 -m pytest tests/tracking/test_auth.py -v`
"""

from __future__ import annotations

import time

import pytest

jwt = pytest.importorskip("jwt")
if not hasattr(jwt, "encode"):  # a different `jwt` distribution, not PyJWT
    pytest.skip("PyJWT is not installed", allow_module_level=True)
pytest.importorskip("fastapi")

from mv3dt_tracking.auth import TicketError, bearer_token, verify_ticket  # noqa: E402

SECRET = "test-secret-not-real-0123456789abcdef"
PID = "proj-1"


def mint(secret=SECRET, **overrides) -> str:
    claims = {
        "aud": "p2bp-workstation",
        "scope": "tracking",
        "pid": PID,
        "sub": "user-1",
        "name": "Ada",
        "exp": int(time.time()) + 300,
    }
    claims.update(overrides)
    claims = {k: v for k, v in claims.items() if v is not None}
    return jwt.encode(claims, secret, algorithm="HS256")


def status(token, **kw) -> int:
    with pytest.raises(TicketError) as exc:
        verify_ticket(token, SECRET, PID, **kw)
    return exc.value.status


def test_valid_ticket_returns_claims():
    claims = verify_ticket(mint(), SECRET, PID)
    assert (claims.sub, claims.pid, claims.name) == ("user-1", PID, "Ada")


def test_numeric_pid_and_missing_name():
    claims = verify_ticket(mint(pid=42, name=None), SECRET, "42")
    assert claims.pid == "42"
    assert claims.name == "user-1"


def test_scope_may_be_a_list_or_space_separated():
    assert verify_ticket(mint(scope=["read", "tracking"]), SECRET, PID)
    assert verify_ticket(mint(scope="read tracking"), SECRET, PID)


@pytest.mark.parametrize(
    "token",
    [
        "not-a-jwt",
        "",
        mint(secret="another-secret-also-not-real-0123"),
        mint(exp=int(time.time()) - 10),
        mint(exp=None),
        mint(aud="someone-else"),
        mint(aud=None),
        mint(sub=None),
        jwt.encode({"aud": "p2bp-workstation", "scope": "tracking", "pid": PID, "sub": "u",
                    "exp": int(time.time()) + 300}, SECRET + "x" * 32, algorithm="HS512"),
    ],
    ids=["garbage", "empty", "bad-signature", "expired", "no-exp", "wrong-aud", "no-aud",
         "no-sub", "wrong-alg"],
)
def test_401_cases(token):
    with pytest.raises(TicketError) as exc:
        verify_ticket(token, SECRET, PID)
    assert exc.value.status == 401
    assert exc.value.message == "Invalid or expired ticket"


@pytest.mark.parametrize(
    "overrides",
    [{"scope": "replay"}, {"scope": None}, {"pid": "proj-2"}, {"pid": None}],
    ids=["wrong-scope", "no-scope", "wrong-pid", "no-pid"],
)
def test_403_cases(overrides):
    assert status(mint(**overrides)) == 403


def test_issuer_checked_only_when_configured():
    token = mint(iss="https://web.example")
    assert verify_ticket(token, SECRET, PID)
    assert verify_ticket(token, SECRET, PID, issuer="https://web.example")
    assert status(token, issuer="https://other.example") == 401
    assert status(mint(), issuer="https://web.example") == 401


def test_empty_secret_never_verifies():
    token = jwt.encode({"aud": "p2bp-workstation"}, "", algorithm="HS256")
    with pytest.raises(TicketError) as exc:
        verify_ticket(token, "", PID)
    assert exc.value.status == 401


@pytest.mark.parametrize("header", [None, "", "Basic abc", "Bearer", "Bearer   "])
def test_bearer_header_rejects(header):
    with pytest.raises(TicketError) as exc:
        bearer_token(header)
    assert exc.value.status == 401


def test_bearer_header_accepts_any_case():
    assert bearer_token("bearer abc.def") == "abc.def"
