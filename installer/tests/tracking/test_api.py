"""Tests for `mv3dt_tracking.api` (the ticket-authenticated history API).

Run from installer/: `python3 -m pytest tests/tracking/test_api.py -v`
"""

from __future__ import annotations

import csv
import io
import json
import struct
import time
import zlib
from datetime import datetime, timezone

import pytest

jwt = pytest.importorskip("jwt")
if not hasattr(jwt, "encode"):  # a different `jwt` distribution, not PyJWT
    pytest.skip("PyJWT is not installed", allow_module_level=True)
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("yaml")

from fastapi.testclient import TestClient  # noqa: E402

from mv3dt_tracking.api import create_app  # noqa: E402
from mv3dt_tracking.lock import LeaseManager  # noqa: E402
from mv3dt_tracking.settings import Settings  # noqa: E402
from mv3dt_tracking.store import Detection, TrackStore  # noqa: E402

SECRET = "test-secret-not-real-0123456789abcdef"
PID = "proj-1"
ORIGIN = "https://app.example"
T0 = 1_760_000_000_000  # an exact multiple of the 200 ms bucket
HOUR = 3600 * 1000

T_GT2PX = [[10.0, 0.0, 5.0], [0.0, -10.0, 300.0], [0.0, 0.0, 1.0]]
# Identity rotation with translation t: the camera centre is -t.
CAMINFO = """projectionMatrix_3x4_w2p:
- 1.0
- 0.0
- 0.0
- -2.0
- 0.0
- 1.0
- 0.0
- -3.0
- 0.0
- 0.0
- 1.0
- -4.0
modelInfo:
  - classID: 1
    height: 1.7
    radius: 0.3
"""


def iso(seconds: float) -> str:
    stamp = datetime.fromtimestamp(seconds, tz=timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.") + f"{stamp.microsecond // 1000:03d}Z"


class Clock:
    def __init__(self) -> None:
        self.t = 1_000.0

    def __call__(self) -> float:
        return self.t


def png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = b"".join(b"\x00" + b"\x00" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def mint(**overrides) -> str:
    claims = {
        "aud": "p2bp-workstation",
        "scope": "tracking",
        "pid": PID,
        "sub": "user-1",
        "name": "Ada",
        "exp": int(time.time()) + 300,
    }
    claims.update(overrides)
    return jwt.encode({k: v for k, v in claims.items() if v is not None}, SECRET, algorithm="HS256")


def auth(**overrides) -> dict:
    return {"Authorization": f"Bearer {mint(**overrides)}"}


@pytest.fixture
def calib_dirs(tmp_path):
    calib = tmp_path / "calib"
    models = tmp_path / "models"
    calib.mkdir()
    models.mkdir()
    (calib / "map.png").write_bytes(png(4, 3))
    (calib / "transforms.yml").write_text(
        "%YAML:1.0\n---\nT_gt2px: !!opencv-matrix\n  rows: 3\n  cols: 3\n  dt: d\n"
        f"  data: {json.dumps([v for row in T_GT2PX for v in row])}\n"
    )
    (models / "camInfo_00.yml").write_text(CAMINFO)
    (models / "camInfo_01.yml").write_text("projectionMatrix_3x4_w2p: [1, 2]\n")
    return calib, models


@pytest.fixture
def store(tmp_path):
    s = TrackStore(tmp_path / "data")
    rows = []
    # Track 1 walks from (0,0) for 2 s; track 2 continues it 1 s later and
    # 0.5 m away; track 9 is far away the whole time.
    for i in range(20):
        rows.append(Detection(T0 + i * 100, "cam1", 1, 0.1 * i, 0.0, conf=0.9))
        rows.append(Detection(T0 + i * 100, "cam2", 9, 10.0, 10.0))
    for i in range(10):
        rows.append(Detection(T0 + 3000 + i * 100, "cam2", 2, 2.4 + 0.1 * i, 0.0))
    rows.append(Detection(T0 + 3500, "cam1", 3, None, None))  # no position
    s.insert_many(rows)
    return s


@pytest.fixture
def clock():
    return Clock()


def make_client(store, clock, calib=None, models=None, recorder=None, **settings_kw):
    settings = Settings(
        ticket_secret=SECRET,
        project_id=PID,
        allowed_origins=[ORIGIN],
        location_id="loc-a",
        calib_dir=calib,
        camera_models_dir=models,
        **settings_kw,
    )
    app = create_app(settings, store, recorder=recorder, leases=LeaseManager(clock=clock))
    return TestClient(app)


@pytest.fixture
def client(store, clock, calib_dirs):
    return make_client(store, clock, *calib_dirs)


def session(client, sub="user-1", name="Ada", kind="replay") -> str:
    r = client.post("/tracking/session", json={"kind": kind}, headers=auth(sub=sub, name=name))
    assert r.status_code == 200, r.text
    return r.json()["SessionId"]


def data_headers(sid, **overrides):
    return {**auth(**overrides), "X-Tracking-Session": sid}


# -- open routes and auth ------------------------------------------------------


def test_health_needs_no_ticket(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["version"] and body["recorder"] is None


def test_health_reports_recorder_stats(store, clock):
    class FakeRecorder:
        def stats(self):
            return {"connected": True, "messages": 3}

    r = make_client(store, clock, recorder=FakeRecorder()).get("/health")
    assert r.json()["recorder"] == {"connected": True, "messages": 3}


TRACKING_GETS = ["/tracking/meta", "/tracking/floorplan.png",
                 f"/tracking/range?from={T0}&to={T0 + 1000}",
                 f"/tracking/export?from={T0}&to={T0 + 1000}"]


@pytest.mark.parametrize("path", TRACKING_GETS)
@pytest.mark.parametrize(
    "headers,code",
    [
        ({}, 401),
        ({"Authorization": "Bearer garbage"}, 401),
        ({"Authorization": "Basic dXNlcjpwYXNz"}, 401),
        ({"Authorization": f"Bearer {mint(exp=int(time.time()) - 5)}"}, 401),
        ({"Authorization": "Bearer " + jwt.encode(
            {"aud": "p2bp-workstation", "scope": "tracking", "pid": PID, "sub": "u",
             "exp": int(time.time()) + 300}, "wrong-secret-also-not-real-0123456", algorithm="HS256")}, 401),
        (auth(scope="other"), 403),
        (auth(pid="proj-2"), 403),
    ],
    ids=["missing", "malformed", "not-bearer", "expired", "bad-signature", "wrong-scope", "wrong-pid"],
)
def test_auth_matrix(client, path, headers, code):
    r = client.get(path, headers=headers)
    assert r.status_code == code
    assert "Message" in r.json()
    if code == 401:
        assert r.headers["www-authenticate"] == "Bearer"


def test_session_routes_need_a_ticket(client):
    assert client.post("/tracking/session", json={"kind": "replay"}).status_code == 401
    assert client.post("/tracking/session/x/heartbeat").status_code == 401
    assert client.delete("/tracking/session/x").status_code == 401


def test_issuer_enforced_when_configured(store, clock):
    c = make_client(store, clock, ticket_issuer="https://web.example")
    assert c.get("/tracking/meta", headers=auth()).status_code == 401
    assert c.get("/tracking/meta", headers=auth(iss="https://web.example")).status_code == 200


# -- meta and floor plan ---------------------------------------------------------


def test_meta_with_calibration(client):
    r = client.get("/tracking/meta", headers=auth())
    assert r.status_code == 200
    body = r.json()
    assert body["locationId"] == "loc-a"
    assert body["span"] == {"from": T0, "to": T0 + 3900}
    assert body["floorplan"] == {"width": 4, "height": 3}
    assert body["T_gt2px"] == T_GT2PX
    # camInfo_01 has an unusable matrix and is skipped.
    assert body["cameras"] == [{"name": "camInfo_00", "x": 2.0, "y": 3.0, "z": 4.0}]


def test_meta_without_calibration_is_nulls(tmp_path, clock):
    c = make_client(TrackStore(tmp_path / "empty"), clock, tmp_path / "nope", tmp_path / "nope2")
    r = c.get("/tracking/meta", headers=auth())
    assert r.status_code == 200
    assert r.json() == {"locationId": "loc-a", "span": None, "floorplan": None,
                        "T_gt2px": None, "cameras": None}
    c = make_client(TrackStore(tmp_path / "empty"), clock)
    assert c.get("/tracking/meta", headers=auth()).json()["cameras"] is None


def test_meta_with_broken_transforms(client, calib_dirs):
    (calib_dirs[0] / "transforms.yml").write_text("T_gt2px: [1, 2, :\n")
    r = client.get("/tracking/meta", headers=auth())
    assert r.status_code == 200 and r.json()["T_gt2px"] is None


def test_floorplan(client, calib_dirs):
    r = client.get("/tracking/floorplan.png", headers=auth())
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content == (calib_dirs[0] / "map.png").read_bytes()
    (calib_dirs[0] / "map.png").unlink()
    r = client.get("/tracking/floorplan.png", headers=auth())
    assert r.status_code == 404 and "Message" in r.json()


# -- lease routes ----------------------------------------------------------------


def test_session_lifecycle(client, clock):
    r = client.post("/tracking/session", json={"kind": "replay"}, headers=auth())
    assert r.status_code == 200
    sid, expires = r.json()["SessionId"], r.json()["ExpiresAt"]
    assert expires == iso(clock.t + 45)

    # Second holder is refused with who holds it and until when.
    r = client.post("/tracking/session", json={"kind": "export"}, headers=auth(sub="user-2", name="Bob"))
    assert r.status_code == 423
    assert r.json() == {"HeldBy": "Ada", "Until": expires}

    # Same holder gets the same lease back.
    clock.t += 10
    r = client.post("/tracking/session", json={"kind": "export"}, headers=auth())
    assert r.json()["SessionId"] == sid

    clock.t += 20
    r = client.post(f"/tracking/session/{sid}/heartbeat", headers=auth())
    assert r.status_code == 200 and r.json() == {"ExpiresAt": iso(clock.t + 45)}
    assert client.post(f"/tracking/session/{sid}/heartbeat", headers=auth(sub="user-2")).status_code == 403
    assert client.post("/tracking/session/unknown/heartbeat", headers=auth()).status_code == 404

    r = client.delete(f"/tracking/session/{sid}", headers=auth(sub="user-2", name="Bob"))
    assert r.status_code == 403
    r = client.delete(f"/tracking/session/{sid}", headers=auth())
    assert r.status_code == 204
    assert client.delete(f"/tracking/session/{sid}", headers=auth()).status_code == 404
    assert client.post(f"/tracking/session/{sid}/heartbeat", headers=auth()).status_code == 404

    r = client.post("/tracking/session", json={"kind": "replay"}, headers=auth(sub="user-2", name="Bob"))
    assert r.status_code == 200


def test_session_expires_without_heartbeat(client, clock):
    sid = session(client)
    clock.t += 45
    assert client.post(f"/tracking/session/{sid}/heartbeat", headers=auth()).status_code == 404
    session(client, sub="user-2", name="Bob")


def test_session_kind_validated(client):
    for body in ({"kind": "upload"}, {}):
        r = client.post("/tracking/session", json=body, headers=auth())
        assert r.status_code == 400 and "Message" in r.json()


# -- range -----------------------------------------------------------------------


def rng(client, sid, **params):
    q = {"from": T0, "to": T0 + 5000, **params}
    return client.get("/tracking/range", params=q, headers=data_headers(sid))


def test_range_requires_session(client):
    r = client.get("/tracking/range", params={"from": T0, "to": T0 + 1000}, headers=auth())
    assert r.status_code == 423
    sid = session(client)
    r = client.get("/tracking/range", params={"from": T0, "to": T0 + 1000},
                   headers={**auth(), "X-Tracking-Session": "wrong"})
    assert r.status_code == 423
    # A valid id presented by a different ticket holder is refused too.
    r = client.get("/tracking/range", params={"from": T0, "to": T0 + 1000},
                   headers=data_headers(sid, sub="user-2"))
    assert r.status_code == 423


def test_range_stitched_shape(client):
    sid = session(client)
    r = rng(client, sid)
    assert r.status_code == 200
    body = r.json()
    assert (body["from"], body["to"], body["hz"], body["mode"]) == (T0, T0 + 5000, 5.0, "stitched")
    people = body["people"]
    assert [p["trackIds"] for p in people] == [[1, 2], [9]]
    first = people[0]
    assert set(first) == {"id", "trackIds", "t", "x", "y"}
    assert len(first["t"]) == len(first["x"]) == len(first["y"])
    assert first["t"][0] == T0 and all(b - a >= 200 for a, b in zip(first["t"], first["t"][1:]))


def test_range_raw_shape(client):
    sid = session(client)
    body = rng(client, sid, mode="raw", hz=10).json()
    assert body["mode"] == "raw" and body["hz"] == 10
    assert [(p["id"], p["trackIds"]) for p in body["people"]] == [(1, [1]), (9, [9]), (2, [2])]
    assert len(body["people"][0]["t"]) == 20


@pytest.mark.parametrize(
    "params",
    [
        {"to": T0 + 6 * HOUR + 1},
        {"to": T0 - 1},
        {"hz": 0},
        {"hz": 25.5},
        {"hz": -1},
        {"mode": "fused"},
        {"from": "yesterday"},
    ],
    ids=["over-6h", "backwards", "hz-0", "hz-over-25", "hz-negative", "bad-mode", "bad-from"],
)
def test_range_validation(client, params):
    sid = session(client)
    r = rng(client, sid, **params)
    assert r.status_code == 400
    assert "Message" in r.json()


def test_range_exactly_6h_and_25hz_allowed(client):
    sid = session(client)
    assert rng(client, sid, to=T0 + 6 * HOUR, hz=25).status_code == 200


def test_range_concurrent_request_gets_429(client):
    sid = session(client)
    leases = client.app.state.leases
    guard = leases.begin(sid, "user-1", "range")
    try:
        r = rng(client, sid)
        assert r.status_code == 429 and "Retry-After" in r.headers
    finally:
        leases.end(guard)
    assert rng(client, sid).status_code == 200


def test_range_31st_request_in_a_minute_gets_429(client, clock):
    sid = session(client)
    for _ in range(30):
        assert rng(client, sid, to=T0 + 100).status_code == 200
        clock.t += 1
        client.post(f"/tracking/session/{sid}/heartbeat", headers=auth())
    r = rng(client, sid, to=T0 + 100)
    assert r.status_code == 429
    assert r.headers["Retry-After"] == "30"
    clock.t += 30
    client.post(f"/tracking/session/{sid}/heartbeat", headers=auth())
    assert rng(client, sid, to=T0 + 100).status_code == 200


# -- export ----------------------------------------------------------------------


def export(client, sid, **params):
    q = {"from": T0, "to": T0 + 5000, **params}
    return client.get("/tracking/export", params=q, headers=data_headers(sid))


def test_export_csv_raw(client):
    sid = session(client, kind="export")
    r = export(client, sid, mode="raw", format="csv")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert r.headers["content-disposition"] == f'attachment; filename="tracking_{T0}_{T0 + 5000}_raw.csv"'
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert list(rows[0]) == ["ts_ms", "sensor", "track_id", "x", "y", "z", "w", "l", "h", "conf"]
    assert len(rows) == 51
    assert rows[0]["ts_ms"] == str(T0) and rows[0]["sensor"] == "cam1" and rows[0]["conf"] == "0.9"
    assert rows[0]["z"] == ""
    positionless = [r for r in rows if r["track_id"] == "3"]
    assert positionless and positionless[0]["x"] == ""


def test_export_jsonl_stitched(client):
    sid = session(client, kind="export")
    r = export(client, sid, mode="stitched", format="jsonl")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    lines = [json.loads(line) for line in r.text.splitlines()]
    assert len(lines) == 51
    assert set(lines[0]) == {"ts_ms", "sensor", "track_id", "x", "y", "z", "w", "l", "h", "conf", "person_id"}
    person = {d["track_id"]: d["person_id"] for d in lines}
    assert person[1] == person[2] != person[9]
    assert person[3] is None
    ts = [d["ts_ms"] for d in lines]
    assert ts == sorted(ts)


def test_export_spans_chunks_in_order(store, client):
    store.insert_many([Detection(T0 + 2 * HOUR + 5, "cam1", 50, 1.0, 1.0)])
    sid = session(client, kind="export")
    r = export(client, sid, mode="raw", format="jsonl", to=T0 + 3 * HOUR)
    lines = [json.loads(line) for line in r.text.splitlines()]
    assert len(lines) == 52 and lines[-1]["track_id"] == 50


def test_export_csv_stitched_has_person_id(client):
    sid = session(client, kind="export")
    rows = list(csv.DictReader(io.StringIO(export(client, sid, mode="stitched").text)))
    assert "person_id" in rows[0]


def test_export_limits_and_session(client):
    r = client.get("/tracking/export", params={"from": T0, "to": T0 + 1000}, headers=auth())
    assert r.status_code == 423
    sid = session(client, kind="export")
    assert export(client, sid, to=T0 + 24 * HOUR + 1).status_code == 400
    assert export(client, sid, format="xlsx").status_code == 400
    assert export(client, sid, to=T0 + 24 * HOUR).status_code == 200


def test_export_releases_guard_and_blocks_concurrent(client):
    sid = session(client, kind="export")
    leases = client.app.state.leases
    guard = leases.begin(sid, "user-1", "export")
    r = export(client, sid)
    assert r.status_code == 429 and r.headers["Retry-After"] == "1"
    leases.end(guard)
    assert export(client, sid).status_code == 200
    # The finished export freed the lease for the next request.
    assert export(client, sid).status_code == 200
    assert rng(client, sid).status_code == 200


# -- CORS ------------------------------------------------------------------------


def preflight(client, origin):
    return client.options(
        "/tracking/range",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "authorization,x-tracking-session",
        },
    )


def test_cors_preflight_allowed_origin(client):
    r = preflight(client, ORIGIN)
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == ORIGIN
    allowed = r.headers["access-control-allow-headers"].lower()
    assert "authorization" in allowed and "x-tracking-session" in allowed
    assert "DELETE" in r.headers["access-control-allow-methods"]


def test_cors_preflight_other_origin_refused(client):
    r = preflight(client, "https://evil.example")
    assert r.status_code == 400
    assert "access-control-allow-origin" not in r.headers


def test_cors_simple_request_headers(client):
    r = client.get("/health", headers={"Origin": ORIGIN})
    assert r.headers["access-control-allow-origin"] == ORIGIN
    exposed = r.headers["access-control-expose-headers"].lower()
    assert "content-disposition" in exposed and "retry-after" in exposed
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers
