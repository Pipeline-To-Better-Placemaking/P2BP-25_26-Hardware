"""Tests for `mv3dt_tracking.api` (the ticket-authenticated history API).

Run from installer/: `python3 -m pytest tests/tracking/test_api.py -v`
"""

from __future__ import annotations

import csv
import io
import json
import os
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

from mv3dt_tracking import api  # noqa: E402
from mv3dt_tracking.api import create_app  # noqa: E402
from mv3dt_tracking.fuse import fuse  # noqa: E402
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
    if code == 401:
        assert r.json()["Message"] in ("Invalid or expired ticket", "Missing bearer ticket")
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
    assert set(body) == {"location_id", "span", "floorplan", "cameras"}
    assert body["location_id"] == "loc-a"
    assert body["span"] == [T0, T0 + 3900]
    assert body["floorplan"] == {"width": 4, "height": 3, "T_gt2px": T_GT2PX}
    # camInfo_01 has an unusable matrix and is skipped.
    assert body["cameras"] == [{"name": "camInfo_00", "x": 2.0, "y": 3.0, "z": 4.0}]


def test_meta_without_calibration_is_nulls(tmp_path, clock):
    c = make_client(TrackStore(tmp_path / "empty"), clock, tmp_path / "nope", tmp_path / "nope2")
    r = c.get("/tracking/meta", headers=auth())
    assert r.status_code == 200
    assert r.json() == {"location_id": "loc-a", "span": None, "floorplan": None, "cameras": []}
    c = make_client(TrackStore(tmp_path / "empty"), clock)
    assert c.get("/tracking/meta", headers=auth()).json()["cameras"] == []


def test_meta_with_broken_transforms(client, calib_dirs):
    (calib_dirs[0] / "transforms.yml").write_text("T_gt2px: [1, 2, :\n")
    r = client.get("/tracking/meta", headers=auth())
    assert r.status_code == 200 and r.json()["floorplan"] is None


def test_meta_floorplan_needs_the_image_too(client, calib_dirs):
    (calib_dirs[0] / "map.png").unlink()
    body = client.get("/tracking/meta", headers=auth()).json()
    assert body["floorplan"] is None and len(body["cameras"]) == 1


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
    r = client.post(f"/tracking/session/{sid}/heartbeat", headers=auth(sub="user-2"))
    assert r.status_code == 423 and r.json() == {"HeldBy": "Ada", "Until": iso(clock.t + 45)}
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


def test_range_default_is_fused(client):
    sid = session(client)
    body = rng(client, sid).json()
    assert body["mode"] == "fused" and "summary" in body and "deprecated" not in body


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
        {"mode": "bogus"},
        {"from": "yesterday"},
    ],
    ids=["over-6h", "backwards", "hz-0", "hz-over-25", "hz-negative", "bad-mode", "bad-from"],
)
def test_range_validation(client, params):
    sid = session(client)
    r = rng(client, sid, **params)
    assert r.status_code == 400
    assert "Message" in r.json()


def test_range_sample_cap(client):
    sid = session(client)
    assert rng(client, sid, to=T0 + 6 * HOUR, hz=5).status_code == 200
    assert rng(client, sid, to=T0 + 72 * 60 * 1000, hz=25).status_code == 200
    for to, hz in ((T0 + 6 * HOUR, 25), (T0 + 72 * 60 * 1000 + 1, 25), (T0 + 6 * HOUR, 5.01)):
        r = rng(client, sid, to=to, hz=hz)
        assert r.status_code == 400
        assert "108000" in r.json()["Message"]


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


def test_export_spans_chunks_in_order(store, client):
    store.insert_many([Detection(T0 + 2 * HOUR + 5, "cam1", 50, 1.0, 1.0)])
    sid = session(client, kind="export")
    r = export(client, sid, mode="raw", format="jsonl", to=T0 + 3 * HOUR)
    lines = [json.loads(line) for line in r.text.splitlines()]
    assert len(lines) == 52 and lines[-1]["track_id"] == 50


def test_export_reads_in_bounded_windows(store, monkeypatch):
    spans = []
    real = store.query_range

    def spy(lo, hi, hz=None):
        spans.append((hi - lo, hz))
        return real(lo, hi, hz)

    monkeypatch.setattr(store, "query_range", spy)
    list(api.export_records(store, T0 - 7, T0 + 3 * HOUR, fused=False))
    assert max(w for w, _ in spans) < 60 * 1000
    spans.clear()
    list(api.export_records(store, T0 - 7, T0 + 3 * HOUR, fused=True))
    assert max(w for w, _ in spans) < 10 * 60 * 1000
    assert all(hz is None for _, hz in spans)


def test_wildcard_origin_rejected(store):
    with pytest.raises(ValueError):
        create_app(Settings(ticket_secret=SECRET, project_id=PID, allowed_origins=["*"]), store)


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


@pytest.mark.parametrize("op", ["range", "export"])
def test_running_data_request_never_blocks_other_routes(client, op):
    sid = session(client)
    leases = client.app.state.leases
    guard = leases.begin(sid, "user-1", op)
    try:
        assert client.post(f"/tracking/session/{sid}/heartbeat", headers=auth()).status_code == 200
        r = client.post("/tracking/session", json={"kind": "export"}, headers=auth())
        assert r.status_code == 200 and r.json()["SessionId"] == sid
        assert client.get("/tracking/meta", headers=auth()).status_code == 200
        assert client.get("/tracking/floorplan.png", headers=auth()).status_code == 200
        assert client.get("/health").status_code == 200
        assert client.delete(f"/tracking/session/{sid}", headers=auth()).status_code == 204
    finally:
        leases.end(guard)


def test_rate_limit_ignores_non_data_routes(client, clock):
    sid = session(client)
    for _ in range(40):
        assert client.post(f"/tracking/session/{sid}/heartbeat", headers=auth()).status_code == 200
        assert client.get("/tracking/meta", headers=auth()).status_code == 200
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


# -- client disconnect during export ---------------------------------------------


class SlowDenseStore:
    """A store whose raw reads are slow and full, so a 24 h export would take
    far longer than the test if generation did not stop on disconnect."""

    def __init__(self) -> None:
        self.raw_reads = 0

    def span(self):
        return None

    def query_range(self, lo, hi, hz=None):
        if hz is not None:
            return []
        self.raw_reads += 1
        time.sleep(0.02)
        step = max(1, (hi - lo) // 2000)
        return [Detection(t, "cam1", 7, 1.0, 2.0, 0.0, 0.5, 0.5, 1.7, 0.9) for t in range(lo, hi, step)]


def test_abandoned_export_stops_and_releases_guard(clock):
    uvicorn = pytest.importorskip("uvicorn")
    import socket
    import threading

    store = SlowDenseStore()
    settings = Settings(ticket_secret=SECRET, project_id=PID, allowed_origins=[ORIGIN])
    leases = LeaseManager(clock=clock)
    app = create_app(settings, store, leases=leases)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert time.monotonic() < deadline, "server did not start"
            time.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        lease = leases.acquire("user-1", "Ada", "export")

        request = (
            f"GET /tracking/export?from={T0}&to={T0 + 24 * HOUR}&format=jsonl HTTP/1.1\r\n"
            f"Host: 127.0.0.1\r\nAuthorization: Bearer {mint()}\r\n"
            f"X-Tracking-Session: {lease.id}\r\n\r\n"
        ).encode()
        sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        sock.sendall(request)
        received = b""
        while len(received) < 256 * 1024:
            data = sock.recv(65536)
            assert data, "server closed early"
            received += data
        assert received.startswith(b"HTTP/1.1 200")
        assert lease.in_flight
        sock.close()

        deadline = time.monotonic() + 1.5
        while lease.in_flight or leases._exporting:
            assert time.monotonic() < deadline, "guard still held after disconnect"
            time.sleep(0.02)
        reads = store.raw_reads
        time.sleep(0.3)
        # Generation stopped: at most the chunk in flight finished, far
        # short of the 1440 one-minute reads of the whole window.
        assert store.raw_reads <= reads + 1
        assert store.raw_reads < 100
    finally:
        server.should_exit = True
        thread.join(5)


# -- fused range and export --------------------------------------------------------

FUSED_TO = T0 + 60_000


def _walker(rows, track, cam, t0, seconds, x0, y, vx=0.5, dx=0.0, hz=10):
    for i in range(int(seconds * hz)):
        t = i / hz
        rows.append(Detection(T0 + t0 + int(t * 1000), cam, track, x0 + vx * t + dx, y))


@pytest.fixture
def fused_store(tmp_path):
    """Person A walks on both cameras under one MV3DT id (camera2 0.2 m off);
    person B sits still while the tracker churns ids 5 -> 6 across a 4 s gap;
    a static object is seen by camera2 only under id 20; track 30 lives 1 s."""
    s = TrackStore(tmp_path / "fused")
    rows = []
    _walker(rows, 1, "camera1", 2000, 10, 0.5, -1.0)
    _walker(rows, 1, "camera2", 2000, 10, 0.5, -1.0, dx=0.2)
    _walker(rows, 5, "camera1", 0, 10, 5.0, -4.0, vx=0.0)
    _walker(rows, 6, "camera1", 14_000, 10, 5.2, -4.0, vx=0.0)
    _walker(rows, 20, "camera2", 0, 40, 3.2, -3.7, vx=0.0)
    _walker(rows, 30, "camera1", 30_000, 1, 1.0, -2.0, vx=0.0)
    s.insert_many(rows)
    return s


@pytest.fixture
def files(tmp_path):
    return tmp_path / "cleaning.json", tmp_path / "offsets.json"


def write_json(path, data):
    """Write and push the mtime forward so the reload is seen at once."""
    path.write_text(json.dumps(data) if not isinstance(data, str) else data)
    st = path.stat()
    bump = getattr(write_json, "n", 0) + 1
    write_json.n = bump
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + bump * 1_000_000_000))


ROOM = {"x": [0, 7.41], "y": [-5.02, 0]}
TV = {"name": "tv", "sensors": ["camera2"], "rect": {"x": [2.75, 3.75], "y": [-4.25, -3.25]}}


@pytest.fixture
def fclient(fused_store, clock, files):
    cleaning, offsets = files
    write_json(cleaning, {"version": 1, "room": ROOM})
    return make_client(fused_store, clock, cleaning_file=cleaning, offsets_file=offsets)


def frng(client, sid=None, **params):
    sid = sid or session(client)
    q = {"from": T0, "to": FUSED_TO, **params}
    r = client.get("/tracking/range", params=q, headers=data_headers(sid))
    assert r.status_code == 200, r.text
    return r.json()


def test_range_fused_shape_and_summary(fclient):
    body = frng(fclient)
    assert (body["from"], body["to"], body["hz"], body["mode"]) == (T0, FUSED_TO, 5.0, "fused")
    people = body["people"]
    # B (ids 5 and 6 stitched) first, then the static object, then A.
    assert [p["id"] for p in people] == [1, 2, 3]
    assert [p["trackIds"] for p in people] == [[5, 6], [20], [1]]
    for p in people:
        assert set(p) == {"id", "trackIds", "cameras", "t", "x", "y", "n_cams"}
        assert len(p["t"]) == len(p["x"]) == len(p["y"]) == len(p["n_cams"])
        assert all(b - a >= 200 for a, b in zip(p["t"], p["t"][1:]))
    assert [p["t"][0] for p in people] == sorted(p["t"][0] for p in people)
    a = people[2]
    assert a["cameras"] == ["camera1", "camera2"] and max(a["n_cams"]) == 2
    s = body["summary"]
    assert {"people", "max_at_once", "occupied_s", "persons", "dropped", "offsets", "preset",
            "config_error", "config_warnings"} <= set(s)
    assert s["people"] == 3 and s["max_at_once"] == 3 and s["preset"] == "strict"
    assert [p["id"] for p in s["persons"]] == [1, 2, 3]
    assert s["dropped"]["short_tracklet"] == 1 and s["dropped"]["shared_id_joins"] == 1
    assert s["config_error"] is None and s["config_warnings"] == []
    assert s["offsets"] == {"applied": False, "needed": None, "gain_m": None, "learned_at_ms": None,
                            "enabled": True, "learned": False, "error": None}


def test_range_stitched_is_a_deprecated_alias_of_fused(fclient):
    sid = session(fclient)
    fused = frng(fclient, sid)
    stitched = frng(fclient, sid, mode="stitched")
    assert stitched["mode"] == "fused" and "stitched" in stitched["deprecated"]
    assert stitched["people"] == fused["people"] and stitched["summary"] == fused["summary"]


def test_range_raw_mode_is_unchanged(fclient):
    body = frng(fclient, mode="raw")
    assert body["mode"] == "raw" and "summary" not in body
    assert sorted(p["id"] for p in body["people"]) == [1, 5, 6, 20, 30]


def test_range_fused_downsamples_only_when_hz_is_coarser(fclient):
    sid = session(fclient)
    slow = frng(fclient, sid, hz=1)
    assert slow["hz"] == 1.0
    assert all(b - a >= 1000 for p in slow["people"] for a, b in zip(p["t"], p["t"][1:]))
    assert [p["trackIds"] for p in slow["people"]] == [[5, 6], [20], [1]]
    fast = frng(fclient, sid, hz=25)
    assert fast["hz"] == 5.0 and fast["people"] == frng(fclient, sid)["people"]


def test_cleaning_reload_changes_results_without_restart(fclient, files):
    cleaning, _ = files
    sid = session(fclient)
    assert frng(fclient, sid)["summary"]["people"] == 3
    write_json(cleaning, {"version": 1, "room": ROOM, "zones": [TV]})
    body = frng(fclient, sid)
    assert body["summary"]["people"] == 2
    assert body["summary"]["dropped"]["zones"] == {"tv": 400}
    write_json(cleaning, {"version": 1, "room": ROOM, "zones": [TV], "preset": "relaxed"})
    assert frng(fclient, sid)["summary"]["preset"] == "relaxed"


def test_config_error_and_warnings_are_reported(fclient, files):
    cleaning, _ = files
    sid = session(fclient)
    write_json(cleaning, {"version": 1, "room": ROOM, "zones": [TV], "foo": 1})
    body = frng(fclient, sid)
    assert body["summary"]["config_warnings"] == ["ignored unknown key foo"]
    assert body["summary"]["people"] == 2
    write_json(cleaning, "{not json")
    body = frng(fclient, sid)
    assert "invalid JSON" in body["summary"]["config_error"]
    assert body["summary"]["people"] == 2  # the last valid config still applies
    write_json(cleaning, {"version": 1, "room": ROOM})
    assert frng(fclient, sid)["summary"]["config_error"] is None


OFFSETS = {
    "version": 1, "learned_at_ms": 123, "model": "translation", "pairs": 500,
    "median_dist_before_m": 0.2, "median_dist_after_m": 0.0, "gain_m": 0.2, "needed": True,
    "sensors": {"camera1": {"dx": 0.1, "dy": 0.0, "theta_deg": 0.0, "scale": 1.0, "sigma_m": 0.05},
                "camera2": {"dx": -0.1, "dy": 0.0, "theta_deg": 0.0, "scale": 1.0, "sigma_m": 0.05}},
}


def test_offsets_applied_and_toggled(fclient, files):
    cleaning, offsets = files
    sid = session(fclient)
    before = frng(fclient, sid)
    write_json(offsets, OFFSETS)
    after = frng(fclient, sid)
    assert after["summary"]["offsets"] == {"applied": True, "needed": True, "gain_m": 0.2,
                                           "learned_at_ms": 123, "enabled": True, "learned": True,
                                           "error": None}
    seated = lambda body: body["people"][0]["x"][0]  # noqa: E731, person B is camera1 only
    assert seated(after) == pytest.approx(seated(before) + 0.1)
    write_json(cleaning, {"version": 1, "room": ROOM, "offsets_apply": False})
    off = frng(fclient, sid)
    assert off["summary"]["offsets"]["applied"] is False
    assert off["summary"]["offsets"]["enabled"] is False and off["summary"]["offsets"]["learned"] is True
    assert off["people"] == before["people"]
    write_json(cleaning, {"version": 1, "room": ROOM})
    write_json(offsets, "[]")
    bad = frng(fclient, sid)["summary"]["offsets"]
    assert bad["applied"] is False and bad["learned"] is False and bad["error"]


def test_export_fused_rows(fclient):
    sid = session(fclient, kind="export")
    q = {"from": T0, "to": FUSED_TO, "mode": "fused"}
    r = fclient.get("/tracking/export", params=q, headers=data_headers(sid))
    assert r.status_code == 200
    assert r.headers["content-disposition"] == f'attachment; filename="tracking_{T0}_{FUSED_TO}_fused.csv"'
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert list(rows[0]) == ["ts_ms", "person_id", "x", "y", "cameras"]
    body = frng(fclient, sid)
    assert len(rows) == sum(len(p["t"]) for p in body["people"])
    a = [row for row in rows if row["person_id"] == "3"]
    assert {row["cameras"] for row in a} >= {"camera1;camera2"}
    r = fclient.get("/tracking/export", params={**q, "mode": "stitched", "format": "jsonl"},
                    headers=data_headers(sid))
    lines = [json.loads(line) for line in r.text.splitlines()]
    assert set(lines[0]) == {"ts_ms", "person_id", "x", "y", "cameras"}
    assert len(lines) == len(rows)
    assert "_fused.jsonl" in r.headers["content-disposition"]


def test_export_fused_chunks_equal_one_shot_and_range_ids(tmp_path, clock):
    store = TrackStore(tmp_path / "chunked")
    edge = T0 - T0 % (10 * 60 * 1000) + 10 * 60 * 1000  # a fusion window edge
    rows = []
    base = edge - T0
    # One person whose ids churn across the edge, one far away later, and
    # one that straddles the edge alone.
    _walker(rows, 1, "camera1", base - 6000, 5, 1.0, -1.0, vx=0.1)
    _walker(rows, 2, "camera1", base + 1000, 6, 1.6, -1.0, vx=0.1)
    _walker(rows, 3, "camera2", base - 8000, 20, 6.0, -4.0, vx=0.0)
    _walker(rows, 4, "camera1", base + 20 * 60 * 1000 + 7, 5, 2.0, -2.0, vx=0.2)
    store.insert_many(rows)
    lo, hi = edge - 60_000 + 13, edge + 25 * 60 * 1000

    one_shot = fuse([r for r in store.query_range(lo, hi)])
    recs = list(api.export_records(store, lo, hi, fused=True))
    by_person = {}
    for rec in recs:
        by_person.setdefault(rec["person_id"], []).append((rec["ts_ms"], rec["x"], rec["y"]))
    expected = {p["id"]: list(zip(p["t"], p["x"], p["y"])) for p in one_shot.people}
    assert {k: sorted(v) for k, v in by_person.items()} == expected
    assert len(expected) == 3

    client = make_client(store, clock)
    body = client.get("/tracking/range", params={"from": lo, "to": hi},
                      headers=data_headers(session(client))).json()
    assert {p["id"]: p["t"] for p in body["people"]} == {k: [t for t, _, _ in v] for k, v in expected.items()}
