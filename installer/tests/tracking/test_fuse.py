"""Tests for `mv3dt_tracking.fuse` on synthetic recordings that mirror what
the real cameras produce: churning tracker ids, a static object seen by one
camera, a seated person with flickering ids, concurrent walkers, points
outside the room and teleport jumps.

Run from installer/: `python3 -m pytest tests/tracking/test_fuse.py -v`
"""

from __future__ import annotations

import math
import random

import pytest

from mv3dt_tracking.cleaning import parse_config
from mv3dt_tracking.fuse import Fuser, fuse
from mv3dt_tracking.store import Detection

T0 = 1_760_000_000_000
ROOM = {"x": [0.0, 7.0], "y": [-5.0, 0.0]}
TV = {"name": "tv", "sensors": ["cam2"], "rect": {"x": [2.75, 3.75], "y": [-4.25, -3.25]}}


def _track(tid, cam, t0, t1, pos, rng, hz=10.0, noise=0.03):
    out = []
    t = t0
    while t < t1:
        x, y = pos(t)
        out.append(Detection(T0 + int(round(t * 1000)), cam, tid,
                             x + rng.gauss(0, noise), y + rng.gauss(0, noise)))
        t += 1.0 / hz
    return out


def _churn(first_tid, cams, t0, t1, pos, rng, life=(1.0, 2.0), gap=0.3, noise=0.03):
    """One target whose tracker id changes every 1-2 s, with short gaps."""
    out, tid, t = [], first_tid, t0
    while t < t1:
        end = min(t + rng.uniform(*life), t1)
        for cam in cams:
            out += _track(tid, cam, t, end, pos, rng, noise=noise)
        tid += 1
        t = end + gap
    return out


def _walk_loop(cx, cy, r, period):
    return lambda t: (cx + r * math.cos(2 * math.pi * t / period), cy + r * math.sin(2 * math.pi * t / period))


def _static(x, y):
    return lambda t: (x, y)


def _living_room(rng):
    rows = []
    # A walker looping the room for 60 s, seen by both cameras with churning ids.
    rows += _churn(100, ["cam1", "cam2"], 0.0, 60.0, _walk_loop(2.0, -1.5, 1.0, 15.0), rng)
    # A seated person on the far side for 120 s, cam1 only, flickering ids.
    rows += _churn(500, ["cam1"], 10.0, 130.0, _static(5.5, -4.0), rng, life=(0.8, 2.0), gap=0.4, noise=0.05)
    # A TV that camera 2 sees as a person, 150 s, a few long ids.
    rows += _churn(900, ["cam2"], 0.0, 150.0, _static(3.25, -3.75), rng, life=(20.0, 40.0), gap=1.0, noise=0.02)
    # Points outside the room.
    rows += [Detection(T0 + 500 * k, "cam1", 2000 + k, -3.0, 2.0) for k in range(50)]
    rows += [Detection(T0 + 700 * k, "cam2", 3000 + k, 12.0, -2.0) for k in range(50)]
    # Teleport jumps inside the walker's tracks.
    walker = [r for r in rows if 100 <= r.track_id < 500 and r.sensor == "cam1"]
    for r in walker[5::40]:
        rows.append(Detection(r.ts_ms + 50, r.sensor, r.track_id, r.x + 3.0, r.y - 1.5))
    return sorted(rows, key=lambda r: (r.ts_ms, r.sensor, r.track_id))


def _max_step_speed(people, gap_ms=2000):
    v = 0.0
    for p in people:
        for i in range(1, len(p["t"])):
            dt = p["t"][i] - p["t"][i - 1]
            if 0 < dt <= gap_ms:
                v = max(v, math.hypot(p["x"][i] - p["x"][i - 1], p["y"][i] - p["y"][i - 1]) / (dt / 1000))
    return v


def _out_of_bounds(people, room=ROOM, margin=0.3):
    return sum(
        1 for p in people for x, y in zip(p["x"], p["y"])
        if not (room["x"][0] - margin <= x <= room["x"][1] + margin
                and room["y"][0] - margin <= y <= room["y"][1] + margin)
    )


@pytest.fixture
def living_room():
    return _living_room(random.Random(7))


def test_living_room_becomes_two_people_with_the_tv_zone(living_room):
    cfg = parse_config({"zones": [TV]})
    res = fuse(living_room, cfg, room=ROOM)
    s = res.summary
    assert s["people"] == 2
    assert s["max_at_once"] == 2
    assert _out_of_bounds(res.people) == 0
    assert _max_step_speed(res.people) <= cfg.vmax_mps * 1.2
    assert s["dropped"]["out_of_bounds"] == 100
    assert s["dropped"]["zone"] > 0 and s["dropped"]["zones"] == {"tv": s["dropped"]["zone"]}
    assert s["dropped"]["speed"] > 0                       # the teleports
    assert s["dropped"]["stitch_joins"] > 0                # churning ids were joined
    assert s["offsets"]["applied"] is False
    assert s["preset"] == "strict"
    # Numbered 1..N by first appearance: the walker starts at t=0, the seated person at 10 s.
    assert [p["id"] for p in res.people] == [1, 2]
    walker, seated = res.people
    assert walker["cameras"] == ["cam1", "cam2"] and max(walker["n_cams"]) == 2
    assert seated["cameras"] == ["cam1"]
    assert all(abs(x - 5.5) < 0.3 for x in seated["x"])
    assert len(walker["trackIds"]) > 20 and len(seated["trackIds"]) > 40
    shares = {p["id"]: p for p in s["persons"]}
    assert shares[1]["camera_share"] == {"cam1": 1.0, "cam2": 1.0}
    assert shares[2]["duration_s"] == pytest.approx(120, abs=3)


def test_without_the_zone_the_tv_is_a_third_person(living_room):
    res = fuse(living_room, parse_config({}), room=ROOM)
    assert res.summary["people"] == 3
    assert res.summary["max_at_once"] == 3
    tv = [p for p in res.people if p["cameras"] == ["cam2"]]
    assert len(tv) == 1 and all(abs(x - 3.25) < 0.2 for x in tv[0]["x"])


def test_no_room_means_no_bounds_filter(living_room):
    res = fuse(living_room, parse_config({"zones": [TV]}))
    assert res.summary["dropped"]["out_of_bounds"] == 0


def test_two_concurrent_walkers_stay_separate():
    rng = random.Random(3)
    rows = _churn(10, ["cam1", "cam2"], 0.0, 40.0, lambda t: (0.5 + 0.15 * t, -1.0), rng)
    rows += _churn(50, ["cam1", "cam2"], 0.0, 40.0, lambda t: (6.5 - 0.15 * t, -3.2), rng)
    res = fuse(rows, parse_config({}), room=ROOM)
    assert res.summary["people"] == 2
    assert res.summary["max_at_once"] == 2
    for p in res.people:
        ys = {round(y) for y in p["y"]}
        assert len(ys) == 1          # each person stays on its own line


def test_never_joins_a_person_tracked_elsewhere_at_the_same_time():
    rng = random.Random(5)
    # A tracked continuously; B starts near where A was 1 s earlier while A
    # is still being tracked elsewhere: B must not join A.
    rows = _track(1, "cam1", 0.0, 20.0, lambda t: (1.0 + 0.2 * t, -1.0), rng)
    rows += _track(2, "cam1", 0.0, 2.0, lambda t: (3.0, -3.0), rng)
    rows += _track(3, "cam1", 3.0, 20.0, lambda t: (3.2, -3.0), rng)
    res = fuse(rows, parse_config({}), room=ROOM)
    assert res.summary["people"] == 2
    ids = sorted(sorted(p["trackIds"]) for p in res.people)
    assert ids == [[1], [2, 3]]


def test_shared_id_join_only_within_the_gate():
    rng = random.Random(11)
    # 1.4 m apart: too far for a duplicate (1.2 m), close enough for the shared id (1.5 m).
    near = _track(7, "cam1", 0.0, 10.0, lambda t: (2.0, -1.8), rng)
    near += _track(7, "cam2", 0.0, 10.0, lambda t: (2.0, -3.2), rng)
    far = _track(8, "cam1", 0.0, 10.0, lambda t: (5.0, -1.0), rng)
    far += _track(8, "cam2", 0.0, 10.0, lambda t: (5.0, -3.0), rng)       # 2.0 m apart
    res = fuse(near + far, parse_config({}), room=ROOM)
    assert res.summary["people"] == 3
    assert res.summary["dropped"]["shared_id_joins"] == 1
    assert res.summary["dropped"]["duplicate_joins"] == 0
    joined = [p for p in res.people if p["trackIds"] == [7]]
    assert len(joined) == 1 and set(joined[0]["n_cams"]) == {2}
    assert all(abs(y + 2.5) < 0.1 for y in joined[0]["y"])          # midway between the cameras
    off = fuse(near, parse_config({"shared_id_hint": False}), room=ROOM)
    assert off.summary["people"] == 2


def test_relaxed_preset_keeps_what_strict_drops():
    rng = random.Random(2)
    rows = _track(1, "cam1", 0.0, 2.0, lambda t: (2.0, -2.0), rng)            # a 2 s person
    rows += _track(2, "cam1", 30.0, 30.7, lambda t: (5.0, -4.0), rng)         # a 0.7 s tracklet
    rows += _track(3, "cam1", 31.0, 33.0, lambda t: (5.0, -4.0), rng)
    strict = fuse(rows, parse_config({}), room=ROOM)
    relaxed = fuse(rows, parse_config({"preset": "relaxed"}), room=ROOM)
    assert strict.summary["people"] == 0
    assert strict.summary["dropped"]["short_tracklet"] == 1
    assert strict.summary["dropped"]["short_person"] == 2
    assert relaxed.summary["people"] == 2
    assert relaxed.summary["preset"] == "relaxed"
    assert sorted(p["trackIds"] for p in relaxed.people) == [[1], [2, 3]]
    # A pinned key overrides the preset.
    pinned = fuse(rows, parse_config({"preset": "relaxed", "min_person_s": 3.0}), room=ROOM)
    assert pinned.summary["people"] == 0


def test_merged_path_respects_the_post_merge_speed_bound():
    rng = random.Random(9)
    # One walker; for 2 s a second camera reports it 1.1 m off (inside the
    # shared-id gate), so the fused mean jumps 0.55 m in and out of that window.
    path = lambda t: (1.0 + 0.3 * t, -2.0)  # noqa: E731
    rows = _track(1, "cam1", 0.0, 20.0, path, rng, noise=0.02)
    rows += _track(1, "cam2", 8.0, 10.0, lambda t: (path(t)[0], -3.1), rng, noise=0.02)
    res = fuse(rows, parse_config({}), room=ROOM)
    assert res.summary["people"] == 1
    cfg = parse_config({})
    p = res.people[0]
    for i in range(1, len(p["t"])):
        dt = max((p["t"][i] - p["t"][i - 1]) / 1000, cfg.bucket_ms / 1000)
        step = math.hypot(p["x"][i] - p["x"][i - 1], p["y"][i] - p["y"][i - 1])
        assert step <= cfg.vmax_mps * dt + cfg.speed_slack_m
    # The smoothed path never reports the jump as a teleport.
    assert _max_step_speed(res.people) <= cfg.vmax_mps * 1.2
    # With a tight bound the jump itself is dropped and counted.
    tight = parse_config({"vmax_mps": 1.0, "speed_slack_m": 0.0})
    res = fuse(rows, tight, room=ROOM)
    assert res.summary["people"] == 1
    assert res.summary["dropped"]["post_merge_speed"] > 0
    assert _max_step_speed(res.people) <= 1.0 * 1.2


def _chunks(rows, size):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


@pytest.mark.parametrize("size", [1, 7, 250, 1000, 4999])
def test_chunked_feeding_matches_one_shot(living_room, size):
    rng = random.Random(1)
    # A second session after a long quiet gap, so a segment flushes mid-feed.
    later = _churn(7000, ["cam1"], 400.0, 430.0, _walk_loop(4.0, -2.5, 1.0, 10.0), rng)
    rows = living_room + later
    cfg = parse_config({"zones": [TV]})

    one = Fuser(cfg, room=ROOM)
    expected = one.feed(rows) + one.finish()

    f = Fuser(cfg, room=ROOM)
    got, emitted_before_finish = [], 0
    for chunk in _chunks(rows, size):
        got += f.feed(chunk)
    emitted_before_finish = len(got)
    got += f.finish()
    assert got == expected
    assert f.dropped == one.dropped
    assert emitted_before_finish > 0                    # the first session was final early
    assert sorted({s.person_id for s in got}) == [1, 2, 3]


def test_chunk_boundary_inside_a_person_and_on_equal_timestamps():
    rng = random.Random(4)
    rows = _churn(1, ["cam1", "cam2"], 0.0, 30.0, lambda t: (1.0 + 0.15 * t, -2.0), rng)
    rows.sort(key=lambda r: (r.ts_ms, r.sensor, r.track_id))
    cfg = parse_config({})
    one = Fuser(cfg, room=ROOM)
    expected = one.feed(rows) + one.finish()
    # Split between cam1 and cam2 rows of the same timestamp, mid-person.
    mid = next(i for i in range(len(rows) // 2, len(rows)) if rows[i].sensor == "cam2")
    assert rows[mid].ts_ms == rows[mid - 1].ts_ms
    f = Fuser(cfg, room=ROOM)
    got = f.feed(rows[:mid])
    assert got == []                                     # the person is not final yet
    got += f.feed(rows[mid:]) + f.finish()
    assert got == expected and len({s.person_id for s in got}) == 1


def test_feed_rejects_out_of_order_chunks():
    f = Fuser(parse_config({}))
    f.feed([Detection(T0 + 1000, "cam1", 1, 1.0, -1.0)])
    with pytest.raises(ValueError):
        f.feed([Detection(T0, "cam1", 1, 1.0, -1.0)])
    f.finish()
    with pytest.raises(RuntimeError):
        f.feed([])


def test_offsets_are_applied_and_weighted():
    rng = random.Random(6)
    # The same id on both cameras, cam2 reading 2.0 m to the +x side: too far
    # for the shared-id gate unless the learned correction is applied.
    rows = _track(1, "cam1", 0.0, 10.0, lambda t: (2.0, -2.0), rng)
    rows += _track(1, "cam2", 0.0, 10.0, lambda t: (4.0, -2.0), rng)
    offsets = {"needed": True, "gain_m": 1.9, "learned_at_ms": 5, "sensors": {
        "cam1": {"dx": 0.9, "dy": 0.0, "theta_deg": 0.0, "scale": 1.0, "sigma_m": 0.1},
        "cam2": {"dx": -0.9, "dy": 0.0, "theta_deg": 0.0, "scale": 1.0, "sigma_m": 0.2},
    }}
    plain = fuse(rows, parse_config({}), room=ROOM)
    assert plain.summary["people"] == 2 and plain.summary["max_at_once"] == 2
    corrected = fuse(rows, parse_config({}), offsets, ROOM)
    assert corrected.summary["people"] == 1
    assert corrected.summary["offsets"] == {"applied": True, "needed": True, "gain_m": 1.9, "learned_at_ms": 5}
    p = corrected.people[0]
    assert set(p["n_cams"]) == {2}
    # cam1 lands at 2.9 with weight 1/0.1^2, cam2 at 3.1 with 1/0.2^2: 2.94, not 3.0.
    assert sum(p["x"]) / len(p["x"]) == pytest.approx(2.94, abs=0.015)
    toggled = fuse(rows, parse_config({"offsets_apply": False}), offsets, ROOM)
    assert toggled.summary["offsets"]["applied"] is False and toggled.summary["people"] == 2
