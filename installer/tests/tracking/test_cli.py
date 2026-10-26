"""Tests for `mv3dt_tracking.cli` and `mv3dt_tracking.settings`.

Run from installer/: `python3 -m pytest tests/tracking/test_cli.py -v`
"""

from __future__ import annotations

import json
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
