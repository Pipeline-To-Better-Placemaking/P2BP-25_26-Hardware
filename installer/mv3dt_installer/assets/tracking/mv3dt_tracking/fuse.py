"""Combine both cameras' tracker output into cleaned, stable people.

DeepStream publishes one position per camera per tracker ID, IDs churn
when someone is briefly lost, and part of every recording is noise
(points outside the room, a TV detected as a person, teleporting feet).
`Fuser` turns raw detections into one path per person, in these stages:

1. bounds and zones: drop points outside the room (plus a margin) and
   inside an ignore zone for their camera;
2. speed gate per (track, camera): drop a point the track could not have
   reached from its last kept point at `vmax_mps`;
3. tracklets keyed by (track, camera), split after `tracklet_gap_s` of
   silence, bucketed (median per bucket); short ones are dropped;
4. per-camera offset correction (see `offsets`), when enabled;
5. association into people: the same MV3DT ID on two cameras when their
   median distance is within `shared_id_gate_m`, concurrent duplicates that
   stay together, and
   sequential continuation (gap and distance limits) that never joins a
   person who is being tracked elsewhere at the same time;
6. per person per bucket, cameras fused by an inverse-variance weighted
   mean (each camera's `sigma_m` from the applied offsets, else equal
   weights), with the cameras that contributed kept;
7. rolling-median smoothing, then a second speed gate on the merged path;
8. people shorter than `min_person_s` are dropped.

Counters (`Fuser.dropped`): out_of_bounds, zone (and per zone name in
`zones`) and speed count raw detections; short_tracklet counts tracklets
(their detections in short_tracklet_samples); shared_id_joins,
duplicate_joins and stitch_joins count association joins;
post_merge_speed counts fused buckets; short_person counts people (their
buckets in short_person_buckets).

Feeding works in chunks (in timestamp order). Association can only link
tracklets less than `stitch_gap_s` apart, so once nothing has been seen for
longer than that after the last kept tracklet, everything before is final
and is emitted; any chunking gives the same output as one call.
"""

from __future__ import annotations

import bisect
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .cleaning import CleaningConfig
from .offsets import apply_offsets
from .store import Detection

DROP_KEYS = (
    "out_of_bounds", "zone", "speed", "short_tracklet", "short_tracklet_samples",
    "shared_id_joins", "duplicate_joins", "stitch_joins", "post_merge_speed",
    "short_person", "short_person_buckets",
)
_MIN_SIGMA_M = 0.05

Room = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class FusedSample:
    ts_ms: int                 # bucket start
    person_id: int
    x: float
    y: float
    n_cams: int
    cameras: tuple[str, ...]
    track_ids: tuple[int, ...]


@dataclass
class FuseResult:
    people: list[dict]
    summary: dict


@dataclass
class _Tracklet:
    track_id: int
    camera: str
    start: int
    last_raw: tuple[int, float, float]          # last kept raw sample, for the speed gate
    buckets: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    n_raw: int = 0
    pts: list[tuple[int, float, float, int]] = field(default_factory=list)  # (bucket, x, y, n) once closed

    @property
    def key(self) -> tuple[int, str, int]:
        return (self.start, self.camera, self.track_id)

    @property
    def end(self) -> int:
        return self.pts[-1][0]


def _room(room) -> Room | None:
    if room is None:
        return None
    if isinstance(room, dict):
        return (tuple(room["x"]), tuple(room["y"]))
    (x0, x1), (y0, y1) = room
    return ((x0, x1), (y0, y1))


class Fuser:
    """Resumable fusion; see the module docstring. Feed detections in
    timestamp order, then call `finish()` once."""

    def __init__(self, config: CleaningConfig | None = None, offsets: dict | None = None, room=None) -> None:
        self.config = config or CleaningConfig()
        self.offsets = offsets if (offsets and offsets.get("sensors") and self.config.offsets_apply) else None
        self.room = _room(room)
        self.dropped: dict = {k: 0 for k in DROP_KEYS}
        self.dropped["zones"] = {z.name: 0 for z in self.config.zones}
        # Inverse-variance weight per camera from the learned residual spread
        # when offsets are applied, else equal weights.
        self._weight: dict[str, float] = {}
        if self.offsets:
            for cam, t in self.offsets["sensors"].items():
                sigma = t.get("sigma_m")
                if isinstance(sigma, (int, float)) and sigma >= 0:
                    self._weight[cam] = 1.0 / max(sigma, _MIN_SIGMA_M) ** 2
        self._open: dict[tuple[int, str], _Tracklet] = {}
        self._pending: list[_Tracklet] = []
        self._watermark: int | None = None
        self._next_person = 1
        self._finished = False

    @property
    def offsets_applied(self) -> bool:
        return self.offsets is not None

    # -- stages 1-4: per sample ---------------------------------------------

    def _in_room(self, x: float, y: float) -> bool:
        if self.room is None:
            return True
        (x0, x1), (y0, y1) = self.room
        m = self.config.room_margin_m
        return x0 - m <= x <= x1 + m and y0 - m <= y <= y1 + m

    def _accept(self, r: Detection) -> tuple[float, float] | None:
        """Stage 1, plus the stage 4 correction: the point to keep, or None.
        Bounds are checked before and after the correction, so a corrected
        point never leaves the room; zones apply to the observed position."""
        if not self._in_room(r.x, r.y):
            self.dropped["out_of_bounds"] += 1
            return None
        for z in self.config.zones:
            if z.contains(r.sensor, r.x, r.y):
                self.dropped["zone"] += 1
                self.dropped["zones"][z.name] += 1
                return None
        if self.offsets is None:
            return (r.x, r.y)
        p = apply_offsets((r.x, r.y), r.sensor, self.offsets)
        if not self._in_room(*p):
            self.dropped["out_of_bounds"] += 1
            return None
        return p

    def _add(self, r: Detection, point: tuple[float, float]) -> None:
        cfg = self.config
        key = (r.track_id, r.sensor)
        t = self._open.get(key)
        if t is not None and r.ts_ms - t.last_raw[0] > cfg.tracklet_gap_s * 1000:
            self._close(key)
            t = None
        if t is not None:
            lt, lx, ly = t.last_raw
            dt = max((r.ts_ms - lt) / 1000.0, cfg.min_dt_s)
            if math.hypot(r.x - lx, r.y - ly) > cfg.vmax_mps * dt + cfg.speed_slack_m:
                self.dropped["speed"] += 1
                return
            t.last_raw = (r.ts_ms, r.x, r.y)
        else:
            t = self._open[key] = _Tracklet(r.track_id, r.sensor, r.ts_ms, (r.ts_ms, r.x, r.y))
        x, y = point
        b = r.ts_ms - r.ts_ms % cfg.bucket_ms
        t.buckets.setdefault(b, []).append((x, y))
        t.n_raw += 1

    def _close(self, key: tuple[int, str]) -> None:
        cfg = self.config
        t = self._open.pop(key)
        t.pts = [
            (b, statistics.median(p[0] for p in ps), statistics.median(p[1] for p in ps), len(ps))
            for b, ps in sorted(t.buckets.items())
        ]
        t.buckets = {}
        if len(t.pts) >= cfg.min_tracklet_samples and (t.end - t.pts[0][0]) / 1000.0 >= cfg.min_tracklet_s:
            self._pending.append(t)
        else:
            self.dropped["short_tracklet"] += 1
            self.dropped["short_tracklet_samples"] += t.n_raw

    # -- public --------------------------------------------------------------

    def feed(self, detections: Iterable[Detection]) -> list[FusedSample]:
        """Process one chunk; return the samples of people now final."""
        if self._finished:
            raise RuntimeError("feed() after finish()")
        rows = sorted(
            (r for r in detections if r.x is not None and r.y is not None),
            key=lambda r: (r.ts_ms, r.sensor, r.track_id, r.x, r.y),
        )
        if not rows:
            return []
        if self._watermark is not None and rows[0].ts_ms < self._watermark:
            raise ValueError(
                f"detections out of order: {rows[0].ts_ms} is before {self._watermark} already fed"
            )
        for r in rows:
            point = self._accept(r)
            if point is not None:
                self._add(r, point)
        self._watermark = rows[-1].ts_ms
        return self._advance(final=False)

    def finish(self) -> list[FusedSample]:
        """Close everything still open and return the remaining samples."""
        if self._finished:
            return []
        self._finished = True
        return self._advance(final=True)

    # -- segments ------------------------------------------------------------

    def _advance(self, final: bool) -> list[FusedSample]:
        cfg = self.config
        w = self._watermark
        gap_ms = cfg.tracklet_gap_s * 1000
        for key in [k for k, t in self._open.items() if final or w - t.last_raw[0] > gap_ms]:
            self._close(key)
        if not self._pending:
            return []
        if final:
            batch, self._pending = self._pending, []
            return self._people(batch)
        # Split the pending tracklets into groups no association can bridge
        # (more than stitch_gap_s between one group's last bucket and the
        # next one's first), and emit each group, oldest first, once nothing
        # open or still to come can reach it.
        hold = cfg.stitch_gap_s * 1000
        bms = cfg.bucket_ms
        pending = sorted(self._pending, key=lambda t: t.key)
        # Earliest bucket an open or future tracklet can start in (a later
        # sample may still land in the watermark's bucket).
        first_new = w - w % bms
        if self._open:
            first_new = min(first_new, min(t.start - t.start % bms for t in self._open.values()))
        out: list[FusedSample] = []
        i = 0
        while i < len(pending):
            j, end = i, pending[i].end
            while j + 1 < len(pending) and pending[j + 1].pts[0][0] - end <= hold:
                j += 1
                end = max(end, pending[j].end)
            if first_new <= end + hold:
                break
            out += self._people(pending[i:j + 1])
            i = j + 1
        self._pending = pending[i:]
        return out

    def _people(self, tracklets: list[_Tracklet]) -> list[FusedSample]:
        cfg = self.config
        tracklets.sort(key=lambda t: t.key)
        n = len(tracklets)
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, c: int, counter: str) -> None:
            ra, rc = find(a), find(c)
            if ra != rc:
                parent[max(ra, rc)] = min(ra, rc)
                self.dropped[counter] += 1

        pos = [{b: (x, y) for b, x, y, _ in t.pts} for t in tracklets]

        def common(a: int, c: int) -> list[int]:
            return [b for b in pos[c] if b in pos[a]]

        # (0) MV3DT's shared ID on two cameras, when the positions agree.
        if cfg.shared_id_hint:
            by_id: dict[int, list[int]] = defaultdict(list)
            for i, t in enumerate(tracklets):
                by_id[t.track_id].append(i)
            for idx in by_id.values():
                for k, a in enumerate(idx):
                    for c in idx[k + 1:]:
                        if tracklets[a].camera == tracklets[c].camera:
                            continue
                        bs = common(a, c)
                        if not bs:
                            continue
                        gap = statistics.median(math.dist(pos[a][b], pos[c][b]) for b in bs)
                        if gap <= cfg.shared_id_gate_m:
                            union(a, c, "shared_id_joins")

        # (a) concurrent duplicates: overlapping tracklets that stay close.
        for a in range(n):
            for c in range(a + 1, n):
                if tracklets[c].start > tracklets[a].end:
                    break
                bs = common(a, c)
                if not bs or len(bs) * cfg.bucket_ms / 1000.0 < cfg.dup_min_overlap_s:
                    continue
                close = sum(math.dist(pos[a][b], pos[c][b]) <= cfg.dup_dist_m for b in bs)
                if close / len(bs) >= cfg.dup_frac:
                    union(a, c, "duplicate_joins")

        # (b) sequential continuation. With stitch_exclusive, never join two
        # people tracked at the same time: the prototype checked only the
        # newcomer's own lifetime, which is the same while the newcomer is a
        # lone tracklet, but here it may already carry its other camera's
        # history from pass (0) or (a).
        spans: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for i, t in enumerate(tracklets):
            spans[find(i)].append((t.pts[0][0], t.end))
        spans = {r: _merge_spans(v) for r, v in spans.items()}
        ends = sorted((t.end, i) for i, t in enumerate(tracklets))
        end_ts = [e for e, _ in ends]
        gap_ms = cfg.stitch_gap_s * 1000
        for c, tc in enumerate(tracklets):
            t0, x0, y0, _ = tc.pts[0]
            rc = find(c)
            best, best_d = None, None
            lo = bisect.bisect_left(end_ts, t0 - gap_ms)
            hi = bisect.bisect_left(end_ts, t0)
            for te, a in ends[lo:hi]:
                ra = find(a)
                if ra == rc:
                    continue
                _, xa, ya, _ = tracklets[a].pts[-1]
                d = math.hypot(x0 - xa, y0 - ya)
                if d > cfg.stitch_dist((t0 - te) / 1000.0) or (best_d is not None and d >= best_d):
                    continue
                if cfg.stitch_exclusive and _overlaps(spans[ra], spans[rc]):
                    continue
                best, best_d = a, d
            if best is not None:
                rb = find(best)
                merged = _merge_spans(spans.pop(rb) + spans.pop(rc))
                union(rb, rc, "stitch_joins")
                spans[find(rc)] = merged

        # (6) per person per bucket, fuse cameras: within a camera the
        # tracklets' points weighted by their sample counts, across cameras
        # by the camera weight.
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)
        people = []
        for members in groups.values():
            acc: dict[int, dict[str, list]] = {}
            tids: dict[int, set[int]] = defaultdict(set)
            for i in members:
                t = tracklets[i]
                for b, x, y, k in t.pts:
                    cell = acc.setdefault(b, {}).setdefault(t.camera, [0.0, 0.0, 0])
                    cell[0] += x * k
                    cell[1] += y * k
                    cell[2] += k
                    tids[b].add(t.track_id)
            ts = sorted(acc)
            raw = []
            for b in ts:
                sx = sy = sw = 0.0
                for cam, (cx, cy, k) in acc[b].items():
                    w = self._weight.get(cam, 1.0)
                    sx += w * cx / k
                    sy += w * cy / k
                    sw += w
                raw.append((sx / sw, sy / sw))
            # (7) smoothing, then the speed gate on the merged path.
            half = cfg.smooth_window // 2
            sm = []
            for k in range(len(raw)):
                win = raw[max(0, k - half): k + half + 1]
                sm.append((statistics.median(q[0] for q in win), statistics.median(q[1] for q in win)))
            kept = [0]
            for k in range(1, len(ts)):
                lk = kept[-1]
                dt = max((ts[k] - ts[lk]) / 1000.0, cfg.bucket_ms / 1000.0)
                if math.dist(sm[k], sm[lk]) <= cfg.vmax_mps * dt + cfg.speed_slack_m:
                    kept.append(k)
                else:
                    self.dropped["post_merge_speed"] += 1
            # (8) minimum person duration.
            if (ts[kept[-1]] - ts[kept[0]]) / 1000.0 < cfg.min_person_s:
                self.dropped["short_person"] += 1
                self.dropped["short_person_buckets"] += len(kept)
                continue
            people.append((ts[kept[0]], members[0], [
                (ts[k], sm[k], tuple(sorted(acc[ts[k]])), tuple(sorted(tids[ts[k]]))) for k in kept
            ]))

        out = []
        for _, _, samples in sorted(people, key=lambda p: (p[0], p[1])):
            pid, self._next_person = self._next_person, self._next_person + 1
            for b, (x, y), cams, tids in samples:
                out.append(FusedSample(b, pid, x, y, len(cams), cams, tids))
        out.sort(key=lambda s: (s.ts_ms, s.person_id))
        return out


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorted, merged closed intervals."""
    out: list[list[int]] = []
    for s0, s1 in sorted(spans):
        if out and s0 <= out[-1][1]:
            out[-1][1] = max(out[-1][1], s1)
        else:
            out.append([s0, s1])
    return [(a, b) for a, b in out]


def _overlaps(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    """Whether two merged interval lists share any instant."""
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][0] <= b[j][1] and b[j][0] <= a[i][1]:
            return True
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return False


def summarize(samples: list[FusedSample]) -> tuple[list[dict], dict]:
    """People series (sorted by id) and the occupancy part of a summary."""
    by_person: dict[int, list[FusedSample]] = defaultdict(list)
    for s in samples:
        by_person[s.person_id].append(s)
    people, per_person = [], []
    seconds: dict[int, set[int]] = defaultdict(set)
    for pid in sorted(by_person):
        ss = sorted(by_person[pid], key=lambda s: s.ts_ms)
        cams = sorted({c for s in ss for c in s.cameras})
        people.append({
            "id": pid,
            "trackIds": sorted({t for s in ss for t in s.track_ids}),
            "cameras": cams,
            "t": [s.ts_ms for s in ss],
            "x": [s.x for s in ss],
            "y": [s.y for s in ss],
            "n_cams": [s.n_cams for s in ss],
        })
        per_person.append({
            "id": pid,
            "duration_s": round((ss[-1].ts_ms - ss[0].ts_ms) / 1000.0, 1),
            "camera_share": {c: round(sum(c in s.cameras for s in ss) / len(ss), 3) for c in cams},
        })
        for s in ss:
            seconds[s.ts_ms // 1000].add(pid)
    occ = {
        "people": len(people),
        "max_at_once": max((len(v) for v in seconds.values()), default=0),
        "occupied_s": len(seconds),
        "persons": per_person,
    }
    return people, occ


def fuse(
    detections: Iterable[Detection],
    config: CleaningConfig | None = None,
    offsets: dict | None = None,
    room=None,
) -> FuseResult:
    """One-shot fusion of a whole range."""
    f = Fuser(config, offsets, room)
    samples = f.feed(detections)
    samples += f.finish()
    people, summary = summarize(samples)
    summary["dropped"] = {**f.dropped, "zones": dict(f.dropped["zones"])}
    summary["offsets"] = offsets_summary(offsets, f.offsets_applied)
    summary["preset"] = f.config.preset
    return FuseResult(people, summary)


def offsets_summary(offsets: dict | None, applied: bool) -> dict:
    """The `summary.offsets` block of a fused response."""
    offsets = offsets or {}
    return {
        "applied": applied,
        "needed": offsets.get("needed"),
        "gain_m": offsets.get("gain_m"),
        "learned_at_ms": offsets.get("learned_at_ms"),
    }
