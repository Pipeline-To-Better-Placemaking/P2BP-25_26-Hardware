"""Tests for `mv3dt_tracking.cli` and `mv3dt_tracking.settings`.

Run from installer/: `python3 -m pytest tests/tracking/test_cli.py -v`
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from mv3dt_tracking.cli import main, rows_to_detections
from mv3dt_tracking.settings import Settings, is_loopback
from mv3dt_tracking.store import Detection, TrackStore

DAY1 = 1_760_918_400_000  # 2025-10-20T00:00Z
DAY2 = DAY1 + 86_400_000


def _env(tmp_path, **extra):
    return {"DATA_DIR": str(tmp_path / "data"), **extra}


def test_settings_from_env_defaults_and_parsing():
    s = Settings.from_env({
        "LOCATION_ID": "loc-a",
        "ALLOWED_ORIGINS": " https://a.example/ ,https://b.example,, ",
        "MQTT_PORT": "1884",
        "TICKET_ISSUER": "",
    })
    assert s.mqtt_topic == "mv3dt/loc-a/sv3d"
    assert s.allowed_origins == ["https://a.example", "https://b.example"]
    assert s.mqtt_port == 1884
    assert s.ticket_issuer is None
    assert s.data_dir == Path("/var/lib/mv3dt-tracking")
    assert s.bind_address() == ("127.0.0.1", 8090)
    assert Settings.from_env({"MQTT_TOPIC": "x/y", "LOCATION_ID": "z"}).mqtt_topic == "x/y"
    assert Settings(bind="[::1]:9000").bind_address() == ("::1", 9000)
    with pytest.raises(ValueError):
        Settings(bind="8090").bind_address()


@pytest.mark.parametrize(
    "host,expected",
    [("127.0.0.1", True), ("127.1.2.3", True), ("::1", True), ("localhost", True),
     ("0.0.0.0", False), ("::", False), ("192.0.2.1", False), ("127.example.com", False),
     ("localhost.example.com", False), ("", False)],
)
def test_is_loopback(host, expected):
    assert is_loopback(host) is expected


def test_rows_to_detections():
    rows = [
        {"ts_ms": 1, "sensor": "cam1", "track_id": 7, "x": 1.0, "y": 2.0, "z": None,
         "w": None, "l": None, "h": None, "conf": 0.5},
        {"ts_ms": None, "sensor": "cam1", "track_id": 8, "x": 1.0, "y": 2.0},
    ]
    assert rows_to_detections(rows) == [Detection(1, "cam1", 7, 1.0, 2.0, conf=0.5)]


def test_stats_prints_json(tmp_path, capsys):
    TrackStore(tmp_path / "data").insert_many([Detection(DAY1 + 5, "cam1", 1, 0.0, 0.0)])
    assert main(["stats"], env=_env(tmp_path)) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["rows"] == 1 and out["files"] == 1
    assert out["span"] == {"from": DAY1 + 5, "to": DAY1 + 5}


def test_wipe_refuses_without_yes(tmp_path, capsys):
    store = TrackStore(tmp_path / "data")
    store.insert_many([Detection(DAY1, "cam1", 1, 0.0, 0.0)])
    assert main(["wipe"], env=_env(tmp_path)) == 2
    assert store.stats()["rows"] == 1


def test_wipe_before_and_all(tmp_path, capsys):
    store = TrackStore(tmp_path / "data")
    store.insert_many([Detection(DAY1, "cam1", 1, 0.0, 0.0), Detection(DAY2, "cam1", 2, 0.0, 0.0)])
    assert main(["wipe", "--yes", "--before", "2025-10-21"], env=_env(tmp_path)) == 0
    assert json.loads(capsys.readouterr().out)["removed"] == 1
    assert store.stats()["rows"] == 1
    assert main(["wipe", "--yes"], env=_env(tmp_path)) == 0
    assert store.stats()["rows"] == 0


def test_wipe_rejects_bad_date(tmp_path):
    with pytest.raises(SystemExit):
        main(["wipe", "--yes", "--before", "21/10/2025"], env=_env(tmp_path))


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"WORKSTATION_TICKET_SECRET": "s" * 32, "LOCATION_ID": "loc-a"},
        {"WORKSTATION_TICKET_SECRET": "s" * 32, "TRACKING_PROJECT_ID": "p", "LOCATION_ID": "loc-a",
         "BIND": "0.0.0.0:8090"},
        {"WORKSTATION_TICKET_SECRET": "s" * 32, "TRACKING_PROJECT_ID": "p", "LOCATION_ID": "loc-a",
         "BIND": "192.0.2.1:8090", "ALLOW_NON_LOCAL_BIND": "0"},
        {"WORKSTATION_TICKET_SECRET": "s" * 32, "TRACKING_PROJECT_ID": "p", "LOCATION_ID": "loc-a",
         "BIND": "127.example.com:8090"},
        {"WORKSTATION_TICKET_SECRET": "s" * 32, "TRACKING_PROJECT_ID": "p", "LOCATION_ID": "loc-a",
         "ALLOWED_ORIGINS": "https://a.example,*"},
        {"WORKSTATION_TICKET_SECRET": "s" * 32, "TRACKING_PROJECT_ID": "p", "LOCATION_ID": "loc-a",
         "BIND": "nonsense"},
    ],
    ids=["no-secret", "no-project", "all-interfaces", "non-loopback-address", "127-hostname",
         "wildcard-origin", "bad-bind"],
)
def test_serve_refuses_unsafe_config(tmp_path, extra, capsys):
    assert main(["serve"], env=_env(tmp_path, **extra)) == 2
    assert "refusing" in capsys.readouterr().err


# -- settings for the cleaning files ---------------------------------------------------


def test_cleaning_and_offsets_paths_default_beside_the_package(tmp_path):
    from mv3dt_tracking import settings as settings_mod

    s = Settings.from_env({})
    assert s.cleaning_file == settings_mod.TRACKING_DIR / "cleaning.json"
    assert s.offsets_file == settings_mod.TRACKING_DIR / "offsets.json"
    assert (settings_mod.TRACKING_DIR / "mv3dt_tracking").is_dir()
    s = Settings.from_env({"CLEANING_FILE": str(tmp_path / "c.json"), "OFFSETS_FILE": str(tmp_path / "o.json")})
    assert (s.cleaning_file, s.offsets_file) == (tmp_path / "c.json", tmp_path / "o.json")
    assert Settings().cleaning_file is None and Settings().offsets_file is None


# -- cleaning commands -------------------------------------------------------------------

T0 = DAY1 + 3_600_000
ROOM = {"x": [0, 7.41], "y": [-5.02, 0]}


def _cleaning_env(tmp_path, cleaning=None):
    path = tmp_path / "cleaning.json"
    path.write_text(json.dumps(cleaning or {"version": 1, "room": ROOM}))
    return _env(tmp_path, CLEANING_FILE=str(path), OFFSETS_FILE=str(tmp_path / "offsets.json"))


def _static_object(rows, cam="camera2", x=3.2, y=-3.7, minutes=12, life_s=20):
    """A fixed object one camera sees for `minutes`, its id churning every `life_s`."""
    for k in range(minutes * 60 // life_s):
        for i in range(life_s * 5):
            rows.append(Detection(T0 + (k * life_s) * 1000 + i * 200, cam, 100 + k, x, y))


def _shared_walk(rows, seconds=60, offset=0.4):
    """One person on both cameras under one id; camera2 sits `offset` m east."""
    for i in range(seconds * 5):
        t = i / 5
        x = 1.0 + 0.05 * t
        rows.append(Detection(T0 + int(t * 1000), "camera1", 1, x, -2.0))
        rows.append(Detection(T0 + int(t * 1000), "camera2", 1, x + offset, -2.0))


def _iso(ms):
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def test_zones_suggest_prints_warning_stats_and_zone(tmp_path, capsys):
    rows = []
    _static_object(rows)
    _shared_walk(rows)
    TrackStore(tmp_path / "data").insert_many(rows)
    env = _cleaning_env(tmp_path)
    window = ["--from", str(T0 - 1000), "--to", str(T0 + 13 * 60_000)]
    assert main(["zones-suggest", *window], env=env) == 0
    out = capsys.readouterr().out
    assert out.startswith("WARNING") and "seated person" in out and "Replay" in out
    assert "suggested-1: camera camera2" in out and "36 tracker ids" in out and "median speed" in out
    zone = json.loads(out.strip().splitlines()[-1].strip().rstrip(","))
    assert zone["sensors"] == ["camera2"] and zone["name"] == "suggested-1"
    assert zone["rect"]["x"][0] <= 3.2 <= zone["rect"]["x"][1]
    assert not (tmp_path / "cleaning.json").read_text().count("suggested")

    assert main(["zones-suggest", "--json", *window], env=env) == 0
    data = json.loads(capsys.readouterr().out)
    assert "seated person" in data["warning"]
    assert [s["zone"]["name"] for s in data["suggestions"]] == ["suggested-1"]
    assert {"occupied_s", "ids", "camera", "median_speed_mps"} <= set(data["suggestions"][0]["stats"])


def test_zones_suggest_hours_window_and_existing_zone(tmp_path, capsys):
    rows = []
    _static_object(rows)
    TrackStore(tmp_path / "data").insert_many(rows)
    zone = {"name": "tv", "sensors": ["camera2"], "rect": {"x": [2.75, 3.75], "y": [-4.25, -3.25]}}
    env = _cleaning_env(tmp_path, {"version": 1, "room": ROOM, "zones": [zone]})
    to = str(T0 + 13 * 60_000)
    assert main(["zones-suggest", "--hours", "1", "--to", to], env=env) == 0
    assert "no static single-camera hotspots found" in capsys.readouterr().out
    # The default window (24 h before now) does not reach this old data.
    assert main(["zones-suggest"], env=_cleaning_env(tmp_path)) == 0
    assert "no static" in capsys.readouterr().out


def test_learn_offsets_prints_and_saves_only_with_save(tmp_path, capsys):
    rows = []
    _shared_walk(rows)
    TrackStore(tmp_path / "data").insert_many(rows)
    env = _cleaning_env(tmp_path)
    window = ["--from", _iso(T0 - 1000), "--to", _iso(T0 + 120_000)]
    assert main(["learn-offsets", *window], env=env) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["pairs"] >= 200 and report["needed"] is True and report["saved"] is None
    assert report["median_dist_before_m"] == pytest.approx(0.4, abs=0.01)
    assert report["median_dist_after_m"] < 0.05 and report["gain_m"] > 0.3
    assert report["sensors"]["camera1"]["dx"] == pytest.approx(0.2, abs=0.01)
    assert not (tmp_path / "offsets.json").exists()

    assert main(["learn-offsets", "--model", "similarity", "--save", *window], env=env) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["saved"] == str(tmp_path / "offsets.json")
    saved = json.loads((tmp_path / "offsets.json").read_text())
    assert saved["needed"] is True and set(saved["sensors"]) == {"camera1", "camera2"}


def test_learn_offsets_refuses_to_save_too_few_pairs(tmp_path, capsys):
    rows = []
    _shared_walk(rows, seconds=10)
    TrackStore(tmp_path / "data").insert_many(rows)
    env = _cleaning_env(tmp_path)
    assert main(["learn-offsets", "--save", "--from", str(T0), "--to", str(T0 + 60_000)], env=env) == 1
    captured = capsys.readouterr()
    assert "not saved" in captured.err and json.loads(captured.out)["saved"] is None
    assert not (tmp_path / "offsets.json").exists()


def test_clean_report_prints_summary_and_counters(tmp_path, capsys):
    rows = []
    _static_object(rows, minutes=2)
    _shared_walk(rows)
    rows.append(Detection(T0 + 5, "camera1", 77, 50.0, 50.0))  # out of the room
    TrackStore(tmp_path / "data").insert_many(rows)
    env = _cleaning_env(tmp_path, {"version": 1, "room": ROOM, "bogus": True})
    assert main(["clean-report", "--from", _iso(T0 - 1000), "--to", _iso(T0 + 180_000)], env=env) == 0
    captured = capsys.readouterr()
    out = json.loads(captured.out)["summary"]
    assert out["people"] == 2 and out["dropped"]["out_of_bounds"] == 1
    assert out["config_warnings"] == ["ignored unknown key bogus"] and "bogus" in captured.err
    assert out["offsets"]["applied"] is False and out["preset"] == "strict"


def test_clean_report_needs_a_window(tmp_path):
    with pytest.raises(SystemExit):
        main(["clean-report", "--from", "2025-10-20T00:00"], env=_env(tmp_path))
    with pytest.raises(SystemExit):
        main(["clean-report", "--from", "yesterday", "--to", "today"], env=_env(tmp_path))


def test_hours_and_from_together_is_a_usage_error(tmp_path, capsys):
    for cmd in ("zones-suggest", "learn-offsets"):
        with pytest.raises(SystemExit) as exc:
            main([cmd, "--hours", "2", "--from", str(T0)], env=_cleaning_env(tmp_path))
        assert exc.value.code == 2
        assert "not allowed with argument" in capsys.readouterr().err


# -- room source ---------------------------------------------------------------

WALLS = {"x": [-1.0, 8.0], "y": [-6.0, 1.0]}


def _calib(tmp_path):
    """A calibration whose floor plan covers x 20..24, y 20..23: none of the
    test data, so only a config room keeps anything."""
    d = tmp_path / "calib"
    d.mkdir()
    (d / "transforms.yml").write_text(
        "%YAML:1.0\nT_gt2px: !!opencv-matrix\n  rows: 3\n  cols: 3\n  dt: d\n"
        "  data: [100.0, 0.0, -2000.0, 0.0, 100.0, -2000.0, 0.0, 0.0, 1.0]\n"
    )
    (d / "map.png").write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
                                + struct.pack(">II", 400, 300) + b"\x08\x02\x00\x00\x00")
    return str(d)


@pytest.mark.parametrize("cleaning, source", [
    ({"version": 1, "room": WALLS}, "config"),
    ({"version": 1, "zones": []}, "calibration"),
    ({"version": 1, "room": None}, "calibration"),
])
def test_clean_report_states_the_room_source(tmp_path, capsys, cleaning, source):
    rows = []
    _shared_walk(rows)
    TrackStore(tmp_path / "data").insert_many(rows)
    env = {**_cleaning_env(tmp_path, cleaning), "CALIB_DIR": _calib(tmp_path)}
    assert main(["clean-report", "--from", str(T0 - 1000), "--to", str(T0 + 120_000)], env=env) == 0
    out = json.loads(capsys.readouterr().out)["summary"]
    assert out["room_source"] == source
    if source == "config":
        assert out["room"] == WALLS and out["people"] == 1
    else:
        assert out["room"]["x"] == [pytest.approx(20.0), pytest.approx(24.0)]
        assert out["room"]["y"] == [pytest.approx(20.0), pytest.approx(23.0)]
        assert out["people"] == 0 and out["dropped"]["out_of_bounds"] > 0


def test_clean_report_without_room_or_calibration_reports_none(tmp_path, capsys):
    rows = []
    _shared_walk(rows)
    TrackStore(tmp_path / "data").insert_many(rows)
    env = _cleaning_env(tmp_path, {"version": 1, "zones": []})
    assert main(["clean-report", "--from", str(T0 - 1000), "--to", str(T0 + 120_000)], env=env) == 0
    out = json.loads(capsys.readouterr().out)["summary"]
    assert out["room_source"] == "none" and out["room"] is None and out["people"] == 1


@pytest.mark.parametrize("room, found", [(WALLS, True), (None, False)])
def test_zones_suggest_and_learn_offsets_use_the_config_room(tmp_path, capsys, room, found):
    rows = []
    _static_object(rows)
    _shared_walk(rows)
    TrackStore(tmp_path / "data").insert_many(rows)
    env = {**_cleaning_env(tmp_path, {"version": 1, "room": room}), "CALIB_DIR": _calib(tmp_path)}
    window = ["--from", str(T0 - 1000), "--to", str(T0 + 13 * 60_000)]
    assert main(["zones-suggest", "--json", *window], env=env) == 0
    assert bool(json.loads(capsys.readouterr().out)["suggestions"]) is found
    assert main(["learn-offsets", *window], env=env) == 0
    pairs = json.loads(capsys.readouterr().out)["pairs"]
    assert (pairs >= 200) if found else pairs == 0
