"""Cleaning configuration, room bounds and ignore-zone suggestions.

The operator-managed `cleaning.json` holds the knobs `fuse` uses. Every key
is optional: a `preset` ("strict", the default, or "relaxed") supplies the
base values and any other key overrides one of them, so relaxing a single
rule is a one-line edit. Bad values raise `CleaningConfigError` naming the
key; unknown keys are logged and ignored. `CleaningFile` re-reads the file
whenever its mtime changes, so edits apply without a restart, and keeps the
last valid config while the file is invalid.

Zones are areas whose detections are dropped: `{"name", "sensors"?,
"rect": {"x": [a, b], "y": [a, b]}}` or `{"name", "sensors"?, "polygon":
[[x, y], ...]}` in metres. `sensors` limits a zone to some cameras (a TV
seen as a person by one camera only). `suggest_zones` proposes such zones
from static single-camera hotspots in recorded detections.
"""

from __future__ import annotations

import json
import logging
import math
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

from . import calib
from .store import Detection

log = logging.getLogger(__name__)

STRICT: dict = {
    "version": 1,
    "room_margin_m": 0.3,
    "room": None,              # fallback {"x": [a, b], "y": [a, b]} when there is no calibration
    "zones": [],
    "vmax_mps": 3.0,           # fastest plausible walk between samples of one track
    "speed_slack_m": 0.3,      # jitter allowance added to every speed check
    "min_dt_s": 0.04,          # floor on dt in the per-track speed gate
    "bucket_ms": 200,
    "tracklet_gap_s": 5.0,     # a (track, camera) silent this long starts a new tracklet
    "min_tracklet_s": 1.0,
    "min_tracklet_samples": 5,  # buckets, not raw detections
    "offsets_apply": True,
    "shared_id_hint": True,
    "shared_id_gate_m": 1.5,
    "dup_dist_m": 1.2,
    "dup_frac": 0.7,
    "dup_min_overlap_s": 1.0,
    "stitch_gap_s": 120.0,
    "stitch_d0_m": 0.8,        # allowed jump at a zero gap ...
    "stitch_dv_mps": 0.1,      # ... growing by this much per second of gap ...
    "stitch_dmax_m": 3.5,      # ... up to this cap
    "stitch_exclusive": True,  # never stitch onto a person tracked elsewhere at the same time
    "smooth_window": 5,        # rolling median, in buckets
    "min_person_s": 3.0,
}

# Starting points, re-tuned once the cameras agree after re-recording.
RELAXED: dict = {
    **STRICT,
    "room_margin_m": 0.5,
    "vmax_mps": 4.0,
    "speed_slack_m": 0.5,
    "min_tracklet_s": 0.5,
    "min_tracklet_samples": 3,
    "shared_id_gate_m": 1.0,
    "dup_dist_m": 0.8,
    "dup_frac": 0.6,
    "stitch_gap_s": 30.0,
    "stitch_dmax_m": 2.0,
    "smooth_window": 3,
    "min_person_s": 1.0,
}

PRESETS = {"strict": STRICT, "relaxed": RELAXED}

_POSITIVE = {
    "vmax_mps", "bucket_ms", "tracklet_gap_s", "stitch_dmax_m", "dup_dist_m",
    "smooth_window", "shared_id_gate_m", "version",
}
_NON_NEGATIVE = {
    "room_margin_m", "speed_slack_m", "min_dt_s", "min_tracklet_s", "min_tracklet_samples",
    "stitch_gap_s", "stitch_d0_m", "stitch_dv_mps", "dup_min_overlap_s", "min_person_s",
}
_INTS = {"bucket_ms", "min_tracklet_samples", "smooth_window", "version"}
_BOOLS = {"offsets_apply", "shared_id_hint", "stitch_exclusive"}


class CleaningConfigError(ValueError):
    """cleaning.json is unreadable or holds an invalid value."""


@dataclass(frozen=True)
class Zone:
    name: str
    sensors: tuple[str, ...] | None
    polygon: tuple[tuple[float, float], ...]  # a rect is stored as its 4 corners
    rect: tuple[tuple[float, float], tuple[float, float]] | None = None

    def contains(self, sensor: str, x: float, y: float) -> bool:
        if self.sensors and sensor not in self.sensors:
            return False
        if self.rect is not None:
            (x0, x1), (y0, y1) = self.rect
            return x0 <= x <= x1 and y0 <= y <= y1
        return point_in_polygon(x, y, self.polygon)


@dataclass(frozen=True)
class CleaningConfig:
    version: int = 1
    preset: str = "strict"
    room_margin_m: float = 0.3
    room: tuple[tuple[float, float], tuple[float, float]] | None = None
    zones: tuple[Zone, ...] = field(default_factory=tuple)
    vmax_mps: float = 3.0
    speed_slack_m: float = 0.3
    min_dt_s: float = 0.04
    bucket_ms: int = 200
    tracklet_gap_s: float = 5.0
    min_tracklet_s: float = 1.0
    min_tracklet_samples: int = 5
    offsets_apply: bool = True
    shared_id_hint: bool = True
    shared_id_gate_m: float = 1.5
    dup_dist_m: float = 1.2
    dup_frac: float = 0.7
    dup_min_overlap_s: float = 1.0
    stitch_gap_s: float = 120.0
    stitch_d0_m: float = 0.8
    stitch_dv_mps: float = 0.1
    stitch_dmax_m: float = 3.5
    stitch_exclusive: bool = True
    smooth_window: int = 5
    min_person_s: float = 3.0
    ignored_keys: tuple[str, ...] = ()   # unknown keys found in the file

    def stitch_dist(self, gap_s: float) -> float:
        """How far a person may have moved across a gap of `gap_s` seconds."""
        return min(self.stitch_d0_m + self.stitch_dv_mps * gap_s, self.stitch_dmax_m)

    def with_overrides(self, **kw) -> "CleaningConfig":
        return replace(self, **kw)


def point_in_polygon(x: float, y: float, poly: Iterable[tuple[float, float]]) -> bool:
    """Ray casting; points on an edge may fall either way."""
    pts = list(poly)
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _number(key: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CleaningConfigError(f"{key}: expected a number, got {value!r}")
    return float(value)


def _range(key: str, value) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise CleaningConfigError(f"{key}: expected [min, max], got {value!r}")
    a, b = (_number(f"{key}[{i}]", v) for i, v in enumerate(value))
    if a >= b:
        raise CleaningConfigError(f"{key}: min {a} must be below max {b}")
    return a, b


def _rect(key: str, value) -> tuple[tuple[float, float], tuple[float, float]]:
    if not isinstance(value, dict) or set(value) != {"x", "y"}:
        raise CleaningConfigError(f"{key}: expected {{\"x\": [a, b], \"y\": [a, b]}}, got {value!r}")
    return _range(f"{key}.x", value["x"]), _range(f"{key}.y", value["y"])


def parse_zone(value, key: str = "zone") -> Zone:
    if not isinstance(value, dict):
        raise CleaningConfigError(f"{key}: expected an object, got {value!r}")
    unknown = set(value) - {"name", "sensors", "rect", "polygon"}
    if unknown:
        raise CleaningConfigError(f"{key}: unknown key(s) {sorted(unknown)}")
    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise CleaningConfigError(f"{key}.name: expected a non-empty string")
    sensors = value.get("sensors")
    if sensors is not None:
        if not isinstance(sensors, list) or not all(isinstance(s, str) and s for s in sensors):
            raise CleaningConfigError(f"{key}.sensors: expected a list of camera names")
        sensors = tuple(sensors)
    if ("rect" in value) == ("polygon" in value):
        raise CleaningConfigError(f"{key}: give exactly one of rect or polygon")
    if "rect" in value:
        rect = _rect(f"{key}.rect", value["rect"])
        (x0, x1), (y0, y1) = rect
        return Zone(name, sensors, ((x0, y0), (x1, y0), (x1, y1), (x0, y1)), rect)
    poly = value["polygon"]
    if not isinstance(poly, list) or len(poly) < 3:
        raise CleaningConfigError(f"{key}.polygon: expected at least 3 [x, y] points")
    pts = []
    for i, p in enumerate(poly):
        if not isinstance(p, list) or len(p) != 2:
            raise CleaningConfigError(f"{key}.polygon[{i}]: expected [x, y], got {p!r}")
        pts.append((_number(f"{key}.polygon[{i}][0]", p[0]), _number(f"{key}.polygon[{i}][1]", p[1])))
    return Zone(name, sensors, tuple(pts))


def parse_config(data) -> CleaningConfig:
    """Build a config from parsed cleaning.json content (a dict).

    Each key takes the file's value when present, else the selected preset's.
    Unknown keys are logged and listed in `ignored_keys`; invalid values
    raise `CleaningConfigError` naming the key.
    """
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise CleaningConfigError(f"expected a JSON object, got {type(data).__name__}")
    preset = data.get("preset", "strict")
    if preset not in PRESETS:
        raise CleaningConfigError(f"preset: expected one of {sorted(PRESETS)}, got {preset!r}")
    base = PRESETS[preset]
    unknown = sorted(set(data) - set(base) - {"preset"})
    if unknown:
        log.warning("cleaning config: ignoring unknown key(s) %s", unknown)
    merged = {**base, **{k: v for k, v in data.items() if k in base}}

    out: dict = {"preset": preset, "ignored_keys": tuple(unknown)}
    for key, value in merged.items():
        if key == "zones":
            if not isinstance(value, list):
                raise CleaningConfigError("zones: expected a list")
            out[key] = tuple(parse_zone(z, f"zones[{i}]") for i, z in enumerate(value))
        elif key == "room":
            out[key] = None if value is None else _rect("room", value)
        elif key in _BOOLS:
            if not isinstance(value, bool):
                raise CleaningConfigError(f"{key}: expected true or false, got {value!r}")
            out[key] = value
        else:
            num = _number(key, value)
            if key in _INTS:
                if num != int(num):
                    raise CleaningConfigError(f"{key}: expected a whole number, got {value!r}")
                num = int(num)
            if key in _POSITIVE and num <= 0:
                raise CleaningConfigError(f"{key}: must be > 0, got {value!r}")
            if key in _NON_NEGATIVE and num < 0:
                raise CleaningConfigError(f"{key}: must be >= 0, got {value!r}")
            out[key] = num
    if out["version"] != 1:
        raise CleaningConfigError(f"version: only 1 is supported, got {out['version']!r}")
    if out["smooth_window"] % 2 == 0:
        raise CleaningConfigError(f"smooth_window: must be a positive odd number, got {out['smooth_window']!r}")
    if not 0 < out["dup_frac"] <= 1:
        raise CleaningConfigError(f"dup_frac: must be in (0, 1], got {out['dup_frac']!r}")
    return CleaningConfig(**out)


def load_config(path: Path | None) -> CleaningConfig:
    """Read cleaning.json; a missing file (or None) gives the strict defaults.
    Raises `CleaningConfigError` for unreadable JSON or invalid values."""
    if path is None:
        return CleaningConfig()
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return CleaningConfig()
    except OSError as exc:
        raise CleaningConfigError(f"{path}: {exc}") from exc
    try:
        data = json.loads(text) if text.strip() else {}
    except json.JSONDecodeError as exc:
        raise CleaningConfigError(f"{path}: invalid JSON: {exc}") from exc
    try:
        return parse_config(data)
    except CleaningConfigError as exc:
        raise CleaningConfigError(f"{path}: {exc}") from exc


class CleaningFile:
    """Cached loader that re-reads cleaning.json when it changes on disk.

    `get()` returns the current config. When the file becomes invalid the
    last valid config is kept (strict defaults if there never was one), the
    error is logged once and kept in `error` until a valid version appears.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = None if path is None else Path(path)
        self._stamp: object = object()
        self._config = CleaningConfig()
        self.error: str | None = None

    def _current_stamp(self):
        if self.path is None:
            return None
        try:
            st = os.stat(self.path)
        except FileNotFoundError:
            return None
        return (st.st_mtime_ns, st.st_size, st.st_ino)

    def get(self) -> CleaningConfig:
        stamp = self._current_stamp()
        if stamp != self._stamp:
            self._stamp = stamp
            try:
                self._config = load_config(self.path)
                self.error = None
            except CleaningConfigError as exc:
                self.error = str(exc)
                log.error("cleaning config: %s; keeping the previous config", exc)
        return self._config


# -- room bounds -------------------------------------------------------------


def room_from_calib(calib_dir: Path | None) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """World-space ((x0, x1), (y0, y1)) covered by the floor-plan image:
    the image corners mapped through the inverse of `T_gt2px`."""
    if calib_dir is None:
        return None
    t = calib.read_t_gt2px(calib_dir)
    size = calib.png_size(Path(calib_dir) / "map.png")
    if t is None or size is None:
        return None
    w, h = size
    xs, ys = [], []
    for u, v in ((0, 0), (w, 0), (w, h), (0, h)):
        p = calib._solve3(t, [float(u), float(v), 1.0])
        if p is None or abs(p[2]) < 1e-12:
            return None
        xs.append(p[0] / p[2])
        ys.append(p[1] / p[2])
    if max(xs) - min(xs) <= 0 or max(ys) - min(ys) <= 0:
        return None
    return (min(xs), max(xs)), (min(ys), max(ys))


def resolve_room(config: CleaningConfig, calib_dir: Path | None = None):
    """Room bounds to filter with: calibration first, then the configured
    fallback room, else None (no bounds filter)."""
    return room_from_calib(calib_dir) or config.room


# -- zone suggestions --------------------------------------------------------


def suggest_zones(
    detections: Iterable[Detection],
    cell_m: float = 0.5,
    min_camera_share: float = 0.95,
    max_median_speed_mps: float = 0.1,
    min_occupancy_frac: float = 0.3,
    min_occupancy_s: float = 600.0,
    min_ids: int = 10,
    pad_m: float = 0.25,
) -> list[dict]:
    """Static hotspots worth ignoring, as `{"zone": <zone dict>, "stats": {...}}`.

    Pass the detections that survive bounds, zones and the speed gate. The
    window is gridded into `cell_m` cells; a cell is a candidate when it is
    occupied in at least `min_occupancy_frac` of the window's occupied
    seconds or for at least `min_occupancy_s` seconds (so a long window still
    finds an object that is only on part of the time), and at least
    `min_camera_share` of its detections come from one camera. Touching candidates of the same camera form a group, kept when
    the median per-tracklet speed (net displacement over duration) is at
    most `max_median_speed_mps` and it saw at least `min_ids` tracker ids.
    Each group becomes a `rect` padded by `pad_m` and limited to its camera,
    named `suggested-<n>`, largest occupancy first.
    """
    rows = sorted(detections, key=lambda r: (r.ts_ms, r.sensor, r.track_id))
    if not rows:
        return []
    window_s: set[int] = {r.ts_ms // 1000 for r in rows}
    seconds: dict[tuple[int, int], set[int]] = defaultdict(set)
    cams: dict[tuple[int, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tracks: dict[tuple[int, int], dict[tuple[int, str], list]] = defaultdict(dict)
    for r in rows:
        cell = (math.floor(r.x / cell_m), math.floor(r.y / cell_m))
        seconds[cell].add(r.ts_ms // 1000)
        cams[cell][r.sensor] += 1
        tracks[cell].setdefault((r.track_id, r.sensor), []).append((r.ts_ms, r.x, r.y))

    hot: dict[tuple[int, int], str] = {}
    for cell, secs in seconds.items():
        total = sum(cams[cell].values())
        cam, n = max(cams[cell].items(), key=lambda kv: (kv[1], kv[0]))
        occupied = len(secs) >= min_occupancy_frac * len(window_s) or len(secs) >= min_occupancy_s
        if occupied and n / total >= min_camera_share:
            hot[cell] = cam

    out = []
    seen: set[tuple[int, int]] = set()
    for start in sorted(hot):
        if start in seen:
            continue
        cam = hot[start]
        group, stack = [], [start]
        seen.add(start)
        while stack:
            c = stack.pop()
            group.append(c)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (c[0] + dx, c[1] + dy)
                    if hot.get(nb) == cam and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        per_track: dict[tuple[int, str], list] = defaultdict(list)
        for c in group:
            for k, pts in tracks[c].items():
                per_track[k].extend(pts)
        speeds = []
        for pts in per_track.values():
            pts.sort()
            dur = (pts[-1][0] - pts[0][0]) / 1000.0
            if dur >= 1.0:
                speeds.append(math.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2]) / dur)
        med = statistics.median(speeds) if speeds else 0.0
        ids = {tid for tid, _ in per_track}
        samples = sum(sum(cams[c].values()) for c in group)
        share = sum(cams[c][cam] for c in group) / samples
        occ = set().union(*(seconds[c] for c in group))
        if med > max_median_speed_mps or len(ids) < min_ids or share < min_camera_share:
            continue
        x0 = min(c[0] for c in group) * cell_m - pad_m
        x1 = (max(c[0] for c in group) + 1) * cell_m + pad_m
        y0 = min(c[1] for c in group) * cell_m - pad_m
        y1 = (max(c[1] for c in group) + 1) * cell_m + pad_m
        out.append({
            "zone": {
                "sensors": [cam],
                "rect": {"x": [round(x0, 3), round(x1, 3)], "y": [round(y0, 3), round(y1, 3)]},
            },
            "stats": {
                "camera": cam,
                "camera_share": round(share, 3),
                "occupied_s": len(occ),
                "occupancy_frac": round(len(occ) / len(window_s), 3),
                "ids": len(ids),
                "samples": samples,
                "median_speed_mps": round(med, 3),
                "cells": len(group),
            },
        })
    out.sort(key=lambda s: -s["stats"]["occupied_s"])
    for i, s in enumerate(out, 1):
        s["zone"] = {"name": f"suggested-{i}", **s["zone"]}
    return out
