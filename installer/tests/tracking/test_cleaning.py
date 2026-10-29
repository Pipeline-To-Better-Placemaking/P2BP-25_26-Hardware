"""Tests for `mv3dt_tracking.cleaning` (config, zones, room, suggestions).

Run from installer/: `python3 -m pytest tests/tracking/test_cleaning.py -v`
"""

from __future__ import annotations

import json
import os
import random
import struct

import pytest

from mv3dt_tracking.cleaning import (
    RELAXED,
    STRICT,
    CleaningConfig,
    CleaningConfigError,
    CleaningFile,
    load_config,
    parse_config,
    parse_zone,
    resolve_room,
    resolve_room_source,
    room_from_calib,
    room_summary,
    suggest_zones,
)
from mv3dt_tracking.store import Detection

T0 = 1_760_000_000_000


# -- config ------------------------------------------------------------------


def test_defaults_are_the_strict_prototype_values():
    cfg = parse_config({})
    assert cfg == CleaningConfig()
    assert cfg.preset == "strict"
    assert (cfg.room_margin_m, cfg.vmax_mps, cfg.bucket_ms) == (0.3, 3.0, 200)
    assert (cfg.min_tracklet_s, cfg.min_tracklet_samples) == (1.0, 5)
    assert (cfg.stitch_gap_s, cfg.stitch_d0_m, cfg.stitch_dv_mps, cfg.stitch_dmax_m) == (120.0, 0.8, 0.1, 3.5)
    assert (cfg.dup_dist_m, cfg.dup_frac, cfg.dup_min_overlap_s) == (1.2, 0.7, 1.0)
    assert (cfg.smooth_window, cfg.min_person_s) == (5, 3.0)
    assert cfg.shared_id_hint and cfg.offsets_apply and cfg.stitch_exclusive
    assert cfg.shared_id_gate_m == 1.5 and cfg.zones == ()
    # The dataclass defaults and the STRICT table agree.
    for key, value in STRICT.items():
        if key not in ("zones", "room"):
            assert getattr(cfg, key) == value, key
    assert cfg.stitch_dist(0) == 0.8
    assert cfg.stitch_dist(10) == pytest.approx(1.8)
    assert cfg.stitch_dist(100) == 3.5


def test_relaxed_preset_and_overrides():
    cfg = parse_config({"preset": "relaxed"})
    for key, value in RELAXED.items():
        if key not in ("zones", "room"):
            assert getattr(cfg, key) == value, key
    assert cfg.vmax_mps == 4.0 and cfg.min_tracklet_s == 0.5 and cfg.min_tracklet_samples == 3
    assert cfg.stitch_gap_s == 30.0 and cfg.min_person_s == 1.0
    # A key present in the file always wins over the preset.
    pinned = parse_config({"preset": "relaxed", "vmax_mps": 2.5, "min_person_s": 3})
    assert pinned.vmax_mps == 2.5 and pinned.min_person_s == 3.0
    assert pinned.stitch_gap_s == 30.0


def test_unknown_keys_are_ignored_and_reported():
    cfg = parse_config({"version": 1, "vmaxx": 9, "preset": "strict"})
    assert cfg.ignored_keys == ("vmaxx",)
    assert cfg.vmax_mps == 3.0


@pytest.mark.parametrize("data, needle", [
    ({"preset": "loose"}, "preset"),
    ({"vmax_mps": "fast"}, "vmax_mps"),
    ({"vmax_mps": 0}, "vmax_mps"),
    ({"stitch_gap_s": -1}, "stitch_gap_s"),
    ({"bucket_ms": 200.5}, "bucket_ms"),
    ({"smooth_window": True}, "smooth_window"),
    ({"shared_id_hint": "yes"}, "shared_id_hint"),
    ({"dup_frac": 1.5}, "dup_frac"),
    ({"zones": {}}, "zones"),
    ({"zones": [{"rect": {"x": [0, 1], "y": [0, 1]}}]}, "zones[0].name"),
    ({"zones": [{"name": "a", "rect": {"x": [1, 0], "y": [0, 1]}}]}, "zones[0].rect.x"),
    ({"zones": [{"name": "a"}]}, "exactly one of rect or polygon"),
    ({"zones": [{"name": "a", "rect": {"x": [0, 1], "y": [0, 1]}, "polygon": [[0, 0], [1, 0], [1, 1]]}]},
     "exactly one of rect or polygon"),
    ({"zones": [{"name": "a", "polygon": [[0, 0], [1, 0]]}]}, "zones[0].polygon"),
    ({"zones": [{"name": "a", "sensors": "cam2", "rect": {"x": [0, 1], "y": [0, 1]}}]}, "zones[0].sensors"),
    ({"zones": [{"name": "a", "rect": {"x": [0, 1], "y": [0, 1]}, "colour": "red"}]}, "zones[0]"),
    ({"room": {"x": [0, 7]}}, "room"),
    ({"smooth_window": 4}, "smooth_window"),
    ({"smooth_window": 0}, "smooth_window"),
    ({"version": 2}, "version"),
    ([1, 2], "JSON object"),
])
def test_validation_errors_name_the_key(data, needle):
    with pytest.raises(CleaningConfigError) as err:
        parse_config(data)
    assert needle in str(err.value)


def test_load_config_from_file(tmp_path):
    assert load_config(None) == CleaningConfig()
    assert load_config(tmp_path / "missing.json") == CleaningConfig()
    path = tmp_path / "cleaning.json"
    path.write_text(json.dumps({"version": 1, "preset": "relaxed", "zones": []}))
    assert load_config(path).preset == "relaxed"
    path.write_text("{not json")
    with pytest.raises(CleaningConfigError, match="invalid JSON"):
        load_config(path)


def _write(path, data, bump):
    path.write_text(json.dumps(data))
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + bump * 1_000_000_000))


def test_cleaning_file_reloads_on_mtime_change_and_keeps_last_valid(tmp_path):
    path = tmp_path / "cleaning.json"
    cf = CleaningFile(path)
    assert cf.get() == CleaningConfig() and cf.error is None      # missing: strict defaults
    _write(path, {"preset": "strict"}, 1)
    first = cf.get()
    assert first.preset == "strict"
    assert cf.get() is first                                      # cached while unchanged
    _write(path, {"preset": "relaxed"}, 2)
    assert cf.get().preset == "relaxed"
    _write(path, {"preset": "relaxed", "vmax_mps": -1}, 3)
    kept = cf.get()
    assert kept.preset == "relaxed" and kept.vmax_mps == 4.0      # last valid config kept
    assert "vmax_mps" in cf.error
    _write(path, {"preset": "strict"}, 4)
    assert cf.get().preset == "strict" and cf.error is None
    path.unlink()
    assert cf.get() == CleaningConfig()


# -- zones -------------------------------------------------------------------


def test_zone_per_sensor_rect_and_polygon():
    tv = parse_zone({"name": "tv", "sensors": ["cam2"], "rect": {"x": [2.75, 3.75], "y": [-4.25, -3.25]}})
    assert tv.contains("cam2", 3.25, -3.75)
    assert not tv.contains("cam1", 3.25, -3.75)                  # other cameras unaffected
    assert not tv.contains("cam2", 4.0, -3.75)
    assert tv.contains("cam2", 2.75, -3.25)                      # inclusive edges
    tri = parse_zone({"name": "corner", "polygon": [[0, 0], [2, 0], [0, -2]]})
    assert tri.sensors is None
    assert tri.contains("cam1", 0.5, -0.5) and tri.contains("cam2", 0.5, -0.5)
    assert not tri.contains("cam1", 1.5, -1.5)
    every = parse_zone({"name": "all", "sensors": [], "rect": {"x": [0, 1], "y": [0, 1]}})
    assert every.contains("cam9", 0.5, 0.5)                      # empty sensors means all


# -- room --------------------------------------------------------------------


def _calib(tmp_path, w=791, h=522):
    d = tmp_path / "calib"
    d.mkdir()
    # u = 100 x + 50, v = -100 y + 20
    (d / "transforms.yml").write_text(
        "%YAML:1.0\nT_gt2px: !!opencv-matrix\n  rows: 3\n  cols: 3\n  dt: d\n"
        "  data: [100.0, 0.0, 50.0, 0.0, -100.0, 20.0, 0.0, 0.0, 1.0]\n"
    )
    (d / "map.png").write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
                                + struct.pack(">II", w, h) + b"\x08\x02\x00\x00\x00")
    return d


def test_room_from_calibration_inverse_homography(tmp_path):
    d = _calib(tmp_path)
    (x0, x1), (y0, y1) = room_from_calib(d)
    assert (x0, x1) == (pytest.approx(-0.5), pytest.approx(7.41))
    assert (y0, y1) == (pytest.approx(-5.02), pytest.approx(0.2))
    assert resolve_room(parse_config({}), d) == room_from_calib(d)


def test_config_room_overrides_calibration(tmp_path):
    d = _calib(tmp_path)
    cfg = parse_config({"room": {"x": [0.2, 6.5], "y": [-4.6, -0.1]}})
    assert resolve_room_source(cfg, d) == (((0.2, 6.5), (-4.6, -0.1)), "config")
    assert resolve_room(cfg, d) == ((0.2, 6.5), (-4.6, -0.1))


@pytest.mark.parametrize("data", [{}, {"room": None}, {"preset": "relaxed", "room": None}])
def test_absent_or_null_room_keeps_calibration(tmp_path, data):
    d = _calib(tmp_path)
    room, source = resolve_room_source(parse_config(data), d)
    assert source == "calibration" and room == room_from_calib(d)
    assert resolve_room_source(parse_config(data), None) == (None, "none")


def test_room_summary_names_the_source_and_bounds():
    assert room_summary(((0.2, 6.5), (-4.6, -0.1)), "config") == {
        "room_source": "config", "room": {"x": [0.2, 6.5], "y": [-4.6, -0.1]}}
    assert room_summary(None, "none") == {"room_source": "none", "room": None}


def test_room_falls_back_to_config_then_none(tmp_path):
    cfg = parse_config({"room": {"x": [0, 7.41], "y": [-5.02, 0]}})
    assert resolve_room(cfg, tmp_path / "nothing") == ((0.0, 7.41), (-5.02, 0.0))
    assert resolve_room(cfg, None) == ((0.0, 7.41), (-5.02, 0.0))
    assert resolve_room(parse_config({}), None) is None
    assert room_from_calib(None) is None


# -- suggestions ---------------------------------------------------------------


def _static(rng, cams, tid0, x, y, t0, t1, life, noise=0.02):
    out, tid, t = [], tid0, t0
    while t < t1:
        end = min(t + life, t1)
        k = t
        while k < end:
            for cam in cams:
                out.append(Detection(T0 + int(k * 1000), cam, tid, x + rng.gauss(0, noise), y + rng.gauss(0, noise)))
            k += 0.2
        tid += 1
        t = end + 0.5
    return out


def test_suggests_the_static_single_camera_object_only():
    rng = random.Random(1)
    rows = _static(rng, ["cam2"], 900, 3.25, -3.75, 0, 600, life=20)        # a TV: 30 ids, cam2 only
    rows += _static(rng, ["cam1", "cam2"], 100, 5.5, -1.0, 0, 600, life=2)  # seated, seen by both
    rows += _static(rng, ["cam1"], 400, 1.0, -4.0, 0, 600, life=300)        # static but only 2 ids
    rows += _static(rng, ["cam1"], 600, 6.0, -4.0, 0, 60, life=2)           # brief: 10% occupancy
    # A walker crossing the room, one camera.
    rows += [Detection(T0 + i * 200, "cam1", 50 + i // 25, 0.5 + 0.01 * i, -2.5) for i in range(600)]
    out = suggest_zones(rows)
    assert len(out) == 1
    s = out[0]
    assert s["zone"]["name"] == "suggested-1"
    assert s["zone"]["sensors"] == ["cam2"]
    x0, x1 = s["zone"]["rect"]["x"]
    y0, y1 = s["zone"]["rect"]["y"]
    assert x0 < 3.25 < x1 and y0 < -3.75 < y1 and x1 - x0 <= 1.5 and y1 - y0 <= 1.5
    assert s["stats"]["camera_share"] == 1.0 and s["stats"]["ids"] >= 10
    assert s["stats"]["occupancy_frac"] > 0.9
    # Ready to paste: it parses as a zone and covers the object.
    assert parse_zone(s["zone"]).contains("cam2", 3.25, -3.75)
    assert suggest_zones([]) == []


def test_long_window_still_finds_an_object_that_is_on_part_of_the_time():
    rng = random.Random(2)
    # A TV on for 12 minutes of a 2 h window that is otherwise busy elsewhere.
    rows = _static(rng, ["cam2"], 900, 3.25, -3.75, 0, 720, life=30)
    rows += _static(rng, ["cam1", "cam2"], 100, 5.5, -1.0, 0, 7200, life=2)
    out = suggest_zones(rows)
    assert len(out) == 1 and out[0]["zone"]["sensors"] == ["cam2"]
    assert out[0]["stats"]["occupancy_frac"] < 0.3 and out[0]["stats"]["occupied_s"] >= 600
    assert suggest_zones(rows, min_occupancy_s=900) == []
