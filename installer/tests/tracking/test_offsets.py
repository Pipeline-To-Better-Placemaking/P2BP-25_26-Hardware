"""Tests for `mv3dt_tracking.offsets` (learned per-camera correction).

Run from installer/: `python3 -m pytest tests/tracking/test_offsets.py -v`
"""

from __future__ import annotations

import json
import math
import random

import pytest

from mv3dt_tracking.offsets import (
    MIN_PAIRS,
    apply_offsets,
    learn_offsets,
    load_offsets,
    read_offsets,
    save_offsets,
    validate_offsets,
)
from mv3dt_tracking.store import Detection

T0 = 1_760_000_000_000


def _pairs(n_ids=3, seconds=40.0, offset=(0.0, 0.0), rotate_deg=0.0, noise=0.03, seed=1):
    """Several people walking across the room, each seen by both cameras under
    one shared id; cam2 reads `offset` (and `rotate_deg` about the origin) off."""
    rng = random.Random(seed)
    th = math.radians(rotate_deg)
    rows = []
    for k in range(n_ids):
        for i in range(int(seconds * 10)):
            t = i / 10
            x = 0.5 + (6.0 * ((t / seconds + k / n_ids) % 1.0))
            y = -1.0 - 1.5 * k + 0.5 * math.sin(t / 3)
            ts = T0 + int(k * seconds * 1000) + i * 100
            rows.append(Detection(ts, "cam1", 10 + k, x + rng.gauss(0, noise), y + rng.gauss(0, noise)))
            x2 = math.cos(th) * x - math.sin(th) * y + offset[0]
            y2 = math.sin(th) * x + math.cos(th) * y + offset[1]
            rows.append(Detection(ts, "cam2", 10 + k, x2 + rng.gauss(0, noise), y2 + rng.gauss(0, noise)))
    return rows


def _relative(o, x=3.0, y=-2.0):
    a = apply_offsets((x, y), "cam1", o)
    b = apply_offsets((x, y), "cam2", o)
    return a[0] - b[0], a[1] - b[1]


def test_learns_an_injected_one_metre_offset_split_between_cameras():
    o = learn_offsets(_pairs(offset=(1.0, 0.0)))
    assert o["pairs"] >= MIN_PAIRS
    s = o["sensors"]
    # Each camera moves half way toward the other.
    assert s["cam1"]["dx"] == pytest.approx(0.5, abs=0.025)
    assert s["cam2"]["dx"] == pytest.approx(-0.5, abs=0.025)
    assert s["cam1"]["dy"] == pytest.approx(0.0, abs=0.025)
    rel = _relative(o)
    assert rel[0] == pytest.approx(1.0, abs=0.05) and rel[1] == pytest.approx(0.0, abs=0.05)
    assert o["median_dist_before_m"] == pytest.approx(1.0, abs=0.05)
    assert o["median_dist_after_m"] < 0.1
    assert o["gain_m"] > 0.9
    assert o["needed"] is True
    assert o["model"] == "translation"
    assert 0 < s["cam1"]["sigma_m"] < 0.05
    assert o["window"]["from_ms"] == T0


def test_zero_offset_is_flagged_not_needed():
    o = learn_offsets(_pairs())
    assert o["needed"] is False
    assert abs(o["gain_m"]) < 0.1
    assert abs(o["sensors"]["cam1"]["dx"]) < 0.02


def test_similarity_model_fits_a_small_rotation_and_falls_back_on_a_large_one():
    small = learn_offsets(_pairs(offset=(0.3, -0.2), rotate_deg=3.0), model="similarity")
    trans = learn_offsets(_pairs(offset=(0.3, -0.2), rotate_deg=3.0), model="translation")
    assert {v["model"] for v in small["sensors"].values()} == {"similarity"}
    assert small["median_dist_after_m"] < trans["median_dist_after_m"]
    assert small["median_dist_after_m"] < 0.08
    # Half of the rotation each way.
    assert small["sensors"]["cam1"]["theta_deg"] == pytest.approx(1.5, abs=0.2)
    assert small["sensors"]["cam2"]["theta_deg"] == pytest.approx(-1.5, abs=0.2)
    big = learn_offsets(_pairs(rotate_deg=20.0, n_ids=3), model="similarity", gate_m=10.0)
    assert {v["model"] for v in big["sensors"].values()} == {"translation"}
    with pytest.raises(ValueError):
        learn_offsets([], model="affine")


def test_gate_excludes_far_pairs_and_mismatched_ids():
    rows = _pairs(offset=(0.5, 0.0))
    # A different id on cam2 never pairs; a same-id point 5 m away is gated out.
    rows.append(Detection(T0, "cam2", 999, 1.0, -1.0))
    rows.append(Detection(T0, "cam1", 77, 1.0, -1.0))
    rows.append(Detection(T0, "cam2", 77, 6.0, -1.0))
    o = learn_offsets(rows)
    assert o["pairs"] == 3 * 40 * 5          # one pair per id per 200 ms bucket
    assert _relative(o)[0] == pytest.approx(0.5, abs=0.05)


def test_too_few_pairs_learns_nothing_and_is_not_saved(tmp_path):
    o = learn_offsets(_pairs(seconds=5.0, n_ids=1))
    assert o["pairs"] < MIN_PAIRS
    assert o["sensors"] == {} and o["needed"] is False
    with pytest.raises(ValueError):
        save_offsets(tmp_path / "offsets.json", o)
    assert not (tmp_path / "offsets.json").exists()
    assert learn_offsets([])["median_dist_before_m"] is None


def test_save_load_round_trip_and_apply(tmp_path):
    o = learn_offsets(_pairs(offset=(0.0, 0.8)))
    path = tmp_path / "sub" / "offsets.json"
    save_offsets(path, o)
    assert json.loads(path.read_text()) == o
    back = load_offsets(path)
    assert back == o
    assert [p.name for p in path.parent.iterdir()] == ["offsets.json"]
    assert load_offsets(tmp_path / "missing.json") is None
    assert load_offsets(None) is None
    # Unknown cameras and empty offsets leave points alone.
    assert apply_offsets((1.0, 2.0), "cam9", back) == (1.0, 2.0)
    assert apply_offsets((1.0, 2.0), "cam1", None) == (1.0, 2.0)
    assert apply_offsets((1.0, 2.0), "cam1", back)[1] == pytest.approx(2.4, abs=0.03)


def test_default_gate_is_the_shared_id_gate():
    near = learn_offsets(_pairs(offset=(1.4, 0.0)))
    far = learn_offsets(_pairs(offset=(1.6, 0.0)))
    assert near["pairs"] == 600 and near["needed"] is True
    assert far["pairs"] == 0 and far["sensors"] == {}
    assert learn_offsets(_pairs(offset=(1.6, 0.0)), gate_m=2.0)["pairs"] == 600


def test_three_cameras_get_the_zero_sum_least_squares_translations():
    rng = random.Random(8)
    true = {"cam1": 0.3, "cam2": 0.0, "cam3": -0.3}        # what each camera reads off in x
    rows = []
    for i in range(600):
        t = i / 10
        x, y = 0.5 + 0.1 * t, -2.0 + 0.5 * math.sin(t / 4)
        for cam, off in true.items():
            rows.append(Detection(T0 + i * 100, cam, 5, x + off + rng.gauss(0, 0.02), y + rng.gauss(0, 0.02)))
    o = learn_offsets(rows)
    for cam, off in true.items():
        assert o["sensors"][cam]["dx"] == pytest.approx(-off, abs=0.01), cam
        assert o["sensors"][cam]["dy"] == pytest.approx(0.0, abs=0.01), cam
    assert sum(v["dx"] for v in o["sensors"].values()) == pytest.approx(0.0, abs=1e-6)
    assert o["median_dist_after_m"] < 0.05


@pytest.mark.parametrize("content, needle", [
    ('[1]', "not a JSON object"),
    ('{"sensors": []}', "sensors"),
    ('{"sensors": {"cam1": {"dx": 0, "dy": 0, "scale": 1}}}', "sensors.cam1.theta_deg"),
    ('{"sensors": {"cam1": {"dx": NaN, "dy": 0, "theta_deg": 0, "scale": 1}}}', "sensors.cam1.dx"),
    ('{"sensors": {"cam1": {"dx": 0, "dy": Infinity, "theta_deg": 0, "scale": 1}}}', "sensors.cam1.dy"),
    ('{"sensors": {"cam1": {"dx": 0, "dy": 0, "theta_deg": 0, "scale": 0}}}', "sensors.cam1.scale"),
    ('{"sensors": {"cam1": {"dx": 0, "dy": 0, "theta_deg": 0, "scale": 1, "sigma_m": "x"}}}', "sigma_m"),
    ('{not json', "invalid JSON"),
])
def test_invalid_offsets_files_are_rejected_with_a_reason(tmp_path, content, needle):
    path = tmp_path / "offsets.json"
    path.write_text(content)
    with pytest.raises(ValueError, match=needle):
        load_offsets(path)
    data, error = read_offsets(path)
    assert data is None and needle in error


def test_valid_offsets_validate_and_read():
    o = learn_offsets(_pairs(offset=(0.4, 0.0)))
    assert validate_offsets(o) is None
    assert read_offsets(None) == (None, None)


def test_saved_offsets_file_is_mode_644(tmp_path):
    import stat

    path = tmp_path / "offsets.json"
    save_offsets(path, learn_offsets(_pairs(offset=(0.4, 0.0))))
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
