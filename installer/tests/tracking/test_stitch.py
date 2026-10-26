"""Tests for `mv3dt_tracking.stitch` (query-time tracker-ID stitching).

Run from installer/: `python3 -m pytest tests/tracking/test_stitch.py -v`
"""

from __future__ import annotations

import pytest

from mv3dt_tracking.stitch import Stitcher, people_series, raw_series, stitch
from mv3dt_tracking.store import Detection

T0 = 1_760_000_000_000


def _det(t_s, tid, x, y, sensor="cam1") -> Detection:
    return Detection(T0 + int(round(t_s * 1000)), sensor, tid, x, y)


def test_one_person_churning_through_seven_ids_is_one_person():
    # Each ID lives for 0.5 s, then the next appears 1.5 s after the last
    # sighting, at roughly the same spot, alternating cameras.
    rows = []
    for k in range(7):
        start = k * 2.0
        cam = "cam1" if k % 2 == 0 else "cam2"
        rows += [_det(start, 100 + k, 2.0 + 0.05 * k, 3.0), _det(start + 0.5, 100 + k, 2.0 + 0.05 * k, 3.1, cam)]
    mapping = stitch(rows)
    assert set(mapping) == {100 + k for k in range(7)}
    assert set(mapping.values()) == {1}


def test_stationary_object_2m_away_stays_separate():
    rows = []
    for k in range(10):
        rows.append(_det(k * 0.5, 1, 0.0, 0.0))        # walker
        rows.append(_det(k * 0.5, 2, 2.1, 0.0))        # stationary object, 2.1 m away
    rows.append(_det(5.5, 3, 0.1, 0.0))                # walker re-acquired
    mapping = stitch(rows)
    assert mapping[1] != mapping[2]
    assert mapping[3] == mapping[1]


def test_new_track_is_given_to_the_nearest_candidate():
    rows = [
        _det(0, 1, 10.0, 0.0), _det(0, 2, 3.0, 0.0),   # 7 m apart: two people
        _det(2, 1, 0.0, 0.0), _det(2, 2, 3.0, 0.0),    # person 1 walks over
        _det(2.5, 3, 2.0, 0.0),                        # 2 m from 1, 1 m from 2
    ]
    mapping = stitch(rows, dist_m=4.0)
    assert mapping[1] != mapping[2]
    assert mapping[3] == mapping[2]


def test_person_appearing_3m_away_later_stays_separate():
    rows = [_det(0, 1, 0.0, 0.0), _det(1, 1, 0.0, 0.0), _det(2, 2, 3.0, 0.0)]
    mapping = stitch(rows)
    assert mapping == {1: 1, 2: 2}


@pytest.mark.parametrize("gap, same", [(3.0, True), (3.001, False)])
def test_gap_boundary(gap, same):
    rows = [_det(0, 1, 0.0, 0.0), _det(gap, 2, 0.0, 0.0)]
    mapping = stitch(rows, gap_s=3.0)
    assert (mapping[1] == mapping[2]) is same


def test_same_track_on_two_cameras_is_one_person():
    rows = [_det(0, 5, 1.0, 1.0, "cam1"), _det(0, 5, 1.1, 1.0, "cam2")]
    assert stitch(rows) == {5: 1}


def test_people_series_averages_two_cameras_in_one_bucket():
    rows = [
        _det(0.00, 1, 1.0, 2.0, "cam1"),
        _det(0.05, 1, 3.0, 4.0, "cam2"),
        _det(0.25, 2, 3.5, 4.0, "cam1"),   # stitched into person 1, next bucket
    ]
    mapping = stitch(rows)
    assert mapping == {1: 1, 2: 1}
    (person,) = people_series(rows, mapping, hz=5)
    assert person == {
        "id": 1, "trackIds": [1, 2], "t": [T0, T0 + 200], "x": [2.0, 3.5], "y": [3.0, 4.0],
    }


def test_series_sorted_by_first_appearance():
    rows = [_det(1, 10, 0.0, 0.0), _det(0, 20, 9.0, 9.0), _det(2, 10, 0.0, 0.0)]
    mapping = stitch(rows)
    out = people_series(rows, mapping, hz=1)
    assert [p["trackIds"] for p in out] == [[20], [10]]
    assert [p["id"] for p in out] == [mapping[20], mapping[10]]


def test_raw_series_is_keyed_by_track_id():
    rows = [_det(0, 7, 1.0, 1.0, "cam1"), _det(0.1, 7, 3.0, 3.0, "cam2"), _det(0.1, 8, 5.0, 5.0)]
    out = raw_series(rows, hz=2)
    assert out == [
        {"id": 7, "trackIds": [7], "t": [T0], "x": [2.0], "y": [2.0]},
        {"id": 8, "trackIds": [8], "t": [T0], "x": [5.0], "y": [5.0]},
    ]


def test_series_rejects_non_positive_hz():
    with pytest.raises(ValueError):
        raw_series([], hz=0)


def test_negative_timestamps_bucket_by_floor_like_the_store():
    rows = [Detection(-150, "cam1", 1, 1.0, 0.0), Detection(50, "cam1", 1, 3.0, 0.0)]
    (series,) = raw_series(rows, hz=5)
    assert series["t"] == [-200, 0]
    assert series["x"] == [1.0, 3.0]


def _walk_rows():
    # Three people, each churning through new IDs every 1.5 s, plus one ID
    # that reappears 2.9 s later (inside the gap window) at the same spot.
    rows = []
    tid = 1
    for person, (x0, y0) in enumerate([(0.0, 0.0), (5.0, 0.0), (0.0, 5.0)]):
        t = person * 0.3
        for k in range(8):
            rows += [_det(t, tid, x0 + 0.1 * k, y0), _det(t + 0.5, tid, x0 + 0.1 * k, y0, "cam2")]
            tid += 1
            t += 2.0
    rows += [_det(30.0, 900, 9.0, 9.0), _det(32.9, 901, 9.1, 9.0), _det(36.0, 902, 9.1, 9.0)]
    return sorted(rows, key=lambda r: r.ts_ms)


@pytest.mark.parametrize("chunk", [1, 2, 5, 7, 1000])
def test_stitcher_in_chunks_matches_one_call(chunk):
    rows = _walk_rows()
    expected = stitch(rows)
    s = Stitcher()
    returned: dict[int, int] = {}
    for i in range(0, len(rows), chunk):
        new = s.feed(rows[i:i + chunk])
        assert not set(new) & set(returned)  # each track reported once
        returned.update(new)
    assert s.mapping == expected == returned
    assert len(set(expected.values())) == 5  # 3 walkers, 900+901, 902


def test_stitcher_boundary_inside_gap_window():
    # Split exactly between the two sightings 2.9 s apart.
    a, b = [_det(30.0, 900, 9.0, 9.0)], [_det(32.9, 901, 9.1, 9.0)]
    s = Stitcher()
    assert s.feed(a) == {900: 1}
    assert s.feed(b) == {901: 1}
    assert s.feed([]) == {}
    assert s.mapping == stitch(a + b) == {900: 1, 901: 1}
