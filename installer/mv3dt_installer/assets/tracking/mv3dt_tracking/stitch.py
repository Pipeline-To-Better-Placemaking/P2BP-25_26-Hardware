"""Query-time ID stitching: join tracker IDs into stable person IDs.

The tracker re-issues IDs when a person is briefly lost. A new tracker ID
that first appears within `gap_s` seconds and `dist_m` metres of where an
existing person was last seen (on any camera) continues that person. Tracker
IDs are already shared across cameras, so one ID seen by two sensors is one
person. Stored rows are never modified; this runs on query results only.
"""

from __future__ import annotations

from collections import defaultdict
from math import hypot
from typing import Callable, Iterable

from .store import Detection, bucket_ms


def _ordered(rows: Iterable[Detection]) -> list[Detection]:
    return sorted(rows, key=lambda r: (r.ts_ms, r.sensor, r.track_id, r.x, r.y))


def stitch(rows: list[Detection], gap_s: float = 3.0, dist_m: float = 1.2) -> dict[int, int]:
    """Map each track_id to a person id (1, 2, 3, ... in order of creation)."""
    gap_ms = gap_s * 1000.0
    person_of: dict[int, int] = {}
    last: dict[int, tuple[int, float, float]] = {}
    next_person = 1
    for r in _ordered(rows):
        if r.track_id not in person_of:
            best, best_d = None, None
            for p, (lt, lx, ly) in last.items():
                d = hypot(r.x - lx, r.y - ly)
                if r.ts_ms - lt <= gap_ms and d <= dist_m and (best_d is None or d < best_d):
                    best, best_d = p, d
            if best is None:
                best, next_person = next_person, next_person + 1
            person_of[r.track_id] = best
        last[person_of[r.track_id]] = (r.ts_ms, r.x, r.y)
    return person_of


def _series(rows: Iterable[Detection], key: Callable[[Detection], int], hz: float) -> list[dict]:
    width = bucket_ms(hz)
    # id -> bucket start -> [sum_x, sum_y, n]
    acc: dict[int, dict[int, list[float]]] = defaultdict(dict)
    tracks: dict[int, set[int]] = defaultdict(set)
    first: dict[int, int] = {}
    for r in _ordered(rows):
        k = key(r)
        b = r.ts_ms - r.ts_ms % width  # floor, same edges as the store
        cell = acc[k].setdefault(b, [0.0, 0.0, 0])
        cell[0] += r.x
        cell[1] += r.y
        cell[2] += 1
        tracks[k].add(r.track_id)
        first.setdefault(k, r.ts_ms)

    out = []
    for k in sorted(acc, key=lambda k: (first[k], k)):
        buckets = sorted(acc[k].items())
        out.append({
            "id": k,
            "trackIds": sorted(tracks[k]),
            "t": [b for b, _ in buckets],
            "x": [sx / n for _, (sx, _, n) in buckets],
            "y": [sy / n for _, (_, sy, n) in buckets],
        })
    return out


def people_series(rows: list[Detection], mapping: dict[int, int], hz: float) -> list[dict]:
    """Per stitched person, samples averaged per bucket of `bucket_ms(hz)` ms
    across cameras and track ids, sorted by first appearance. Rows whose
    track_id is not in `mapping` are skipped."""
    return _series((r for r in rows if r.track_id in mapping), lambda r: mapping[r.track_id], hz)


def raw_series(rows: list[Detection], hz: float) -> list[dict]:
    """Same shape as `people_series`, keyed by track_id (no stitching)."""
    return _series(rows, lambda r: r.track_id, hz)
