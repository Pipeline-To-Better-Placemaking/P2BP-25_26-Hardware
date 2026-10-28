"""Learn and apply a per-camera 2D correction from co-observed detections.

When MV3DT gives one person the same track ID on two cameras at once, both
positions should be the same floor point; in practice each camera's foot
estimate is off by a steady few tens of centimetres. From such pairs (same
track_id, two cameras, same time bucket, within `gate_m`), taken after
bounds, zones and the speed gate, we fit per camera a correction that moves
its points onto the pair midpoints. Each camera therefore moves half way and
neither is treated as the reference; with two cameras the translations sum
to zero.

A correction maps (x, y) to `scale * R(theta) @ (x, y) + (dx, dy)`, with the
rotation about the world origin. The default model is a translation; with
the similarity model, pairs too concentrated to pin down rotation and scale,
or a fit that is not small, fall back to a translation. `needed` is false
when the correction would shrink the median cross-camera distance by less
than `MIN_GAIN_M`. Each sensor also gets `sigma_m`, its residual spread,
which `fuse` uses as an inverse-variance weight.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .store import Detection

MIN_GAIN_M = 0.1
MIN_PAIRS = 200
MIN_SPREAD_M = 0.5           # rms pair spread below which only a translation is fitted
MAX_THETA_DEG = 5.0
SCALE_RANGE = (0.95, 1.05)
SCHEMA_VERSION = 1


def _pairs(rows: Iterable[Detection], bucket_ms: int, gate_m: float):
    """[(cam_a, (xa, ya), cam_b, (xb, yb))] per track per bucket, cam_a < cam_b."""
    acc: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(dict)
    for r in rows:
        cell = acc[(r.ts_ms // bucket_ms, r.track_id)].setdefault(r.sensor, [0.0, 0.0, 0])
        cell[0] += r.x
        cell[1] += r.y
        cell[2] += 1
    out = []
    for key in sorted(acc):
        cams = sorted(acc[key])
        pts = {c: (acc[key][c][0] / acc[key][c][2], acc[key][c][1] / acc[key][c][2]) for c in cams}
        for i, a in enumerate(cams):
            for b in cams[i + 1:]:
                if math.dist(pts[a], pts[b]) <= gate_m:
                    out.append((a, pts[a], b, pts[b]))
    return out


def _fit(src: list[tuple[float, float]], dst: list[tuple[float, float]], similarity: bool) -> dict:
    """Least-squares 2D similarity (or translation) taking src onto dst,
    closed form (Umeyama in 2D)."""
    n = len(src)
    sx = sum(p[0] for p in src) / n
    sy = sum(p[1] for p in src) / n
    tx = sum(p[0] for p in dst) / n
    ty = sum(p[1] for p in dst) / n
    theta, scale = 0.0, 1.0
    if similarity:
        a = b = var = 0.0
        for (px, py), (qx, qy) in zip(src, dst):
            px, py, qx, qy = px - sx, py - sy, qx - tx, qy - ty
            a += px * qx + py * qy
            b += px * qy - py * qx
            var += px * px + py * py
        if var > 0:
            theta = math.atan2(b, a)
            scale = math.hypot(a, b) / var
    c, s = math.cos(theta) * scale, math.sin(theta) * scale
    return {
        "dx": tx - (c * sx - s * sy),
        "dy": ty - (s * sx + c * sy),
        "theta_deg": math.degrees(theta),
        "scale": scale,
    }


def _transform(t: dict, x: float, y: float) -> tuple[float, float]:
    th = math.radians(t["theta_deg"])
    c, s = math.cos(th) * t["scale"], math.sin(th) * t["scale"]
    return c * x - s * y + t["dx"], s * x + c * y + t["dy"]


def apply_offsets(point: tuple[float, float], camera: str, offsets: dict | None) -> tuple[float, float]:
    """Correct one (x, y) seen by `camera`; unchanged when there is no entry."""
    if not offsets:
        return point
    t = (offsets.get("sensors") or {}).get(camera)
    if not t:
        return point
    return _transform(t, point[0], point[1])


def _fit_camera(pts, targets, model: str) -> tuple[dict, str]:
    if model == "similarity":
        n = len(pts)
        cx = sum(p[0] for p in pts) / n
        cy = sum(p[1] for p in pts) / n
        spread = math.sqrt(sum((p[0] - cx) ** 2 + (p[1] - cy) ** 2 for p in pts) / n)
        if n >= 3 and spread >= MIN_SPREAD_M:
            t = _fit(pts, targets, True)
            if abs(t["theta_deg"]) <= MAX_THETA_DEG and SCALE_RANGE[0] <= t["scale"] <= SCALE_RANGE[1]:
                return t, "similarity"
    return _fit(pts, targets, False), "translation"


def learn_offsets(
    detections: Iterable[Detection],
    bucket_ms: int = 200,
    gate_m: float = 1.5,
    model: str = "translation",
    min_pairs: int = MIN_PAIRS,
    window: dict | None = None,
) -> dict:
    """Fit per-sensor corrections from same-ID cross-camera pairs.

    Pass detections that survived bounds, zones and the speed gate. Only
    pairs within `gate_m` (default: the strict `shared_id_gate_m`, 1.5 m, so
    only plausibly-same-person pairs count) are used. Each
    sensor's points are fitted onto the pair midpoints, so with two sensors
    each moves half the mean difference toward the other (the corrections
    sum to zero). `model="similarity"` also fits a small rotation and
    uniform scale per sensor, falling back to a translation outside
    +-5 degrees or a 0.95..1.05 scale. Returns the offsets.json content;
    with fewer than `min_pairs` pairs, `sensors` is empty and `needed` false
    (`save_offsets` refuses to write it).
    """
    if model not in ("translation", "similarity"):
        raise ValueError(f"model must be translation or similarity, got {model!r}")
    rows = list(detections)
    if window is None:
        window = {"from_ms": min((r.ts_ms for r in rows), default=None),
                  "to_ms": max((r.ts_ms for r in rows), default=None)}
    pairs = _pairs(rows, bucket_ms, gate_m)
    result = {
        "version": SCHEMA_VERSION,
        "learned_at_ms": int(time.time() * 1000),
        "window": window,
        "model": model,
        "pairs": len(pairs),
        "min_pairs": min_pairs,
        "median_dist_before_m": None,
        "median_dist_after_m": None,
        "gain_m": 0.0,
        "needed": False,
        "sensors": {},
    }
    if not pairs:
        return result
    before = statistics.median(math.dist(pa, pb) for _, pa, _, pb in pairs)
    result["median_dist_before_m"] = round(before, 4)
    result["median_dist_after_m"] = round(before, 4)
    if len(pairs) < min_pairs:
        return result

    def fit_all(use):
        # Zero-sum least-squares translations first; each camera's points are
        # then fitted onto the midpoints of the translation-corrected pairs,
        # which reproduces those translations for the translation model and
        # adds the small rotation and scale for the similarity model.
        shift = _zero_sum_translations(use)
        src: dict[str, list] = defaultdict(list)
        dst: dict[str, list] = defaultdict(list)
        for a, pa, b, pb in use:
            ta, tb = shift[a], shift[b]
            mid = ((pa[0] + ta[0] + pb[0] + tb[0]) / 2, (pa[1] + ta[1] + pb[1] + tb[1]) / 2)
            src[a].append(pa)
            dst[a].append(mid)
            src[b].append(pb)
            dst[b].append(mid)
        fitted = {cam: _fit_camera(src[cam], dst[cam], model) for cam in sorted(src)}
        return {c: t for c, (t, _) in fitted.items()}, {c: m for c, (_, m) in fitted.items()}

    def residuals(cams, use):
        return [math.dist(_transform(cams[a], *pa), _transform(cams[b], *pb)) for a, pa, b, pb in use]

    cams, models = fit_all(pairs)
    # One refit without gross outliers (ID swaps, mismatched feet).
    res = residuals(cams, pairs)
    cut = max(0.3, 3 * statistics.median(res))
    inliers = [p for p, r in zip(pairs, res) if r <= cut]
    if len(inliers) >= min_pairs and len(inliers) < len(pairs):
        cams, models = fit_all(inliers)

    res = residuals(cams, pairs)
    after = statistics.median(res)
    # sigma: median distance of a sensor's corrected point from the corrected
    # pair midpoint (half the corrected cross-camera distance). With exactly
    # two cameras every pair involves both, so their sigmas are equal by
    # construction and the stage 6 weights come out equal; they differ only
    # with three or more cameras.
    half: dict[str, list[float]] = defaultdict(list)
    for (a, _, b, _), r in zip(pairs, res):
        half[a].append(r / 2)
        half[b].append(r / 2)
    result["sensors"] = {
        c: {"dx": round(t["dx"], 4), "dy": round(t["dy"], 4),
            "theta_deg": round(t["theta_deg"], 4), "scale": round(t["scale"], 5),
            "sigma_m": round(statistics.median(half[c]), 4), "model": models[c]}
        for c, t in cams.items()
    }
    result["median_dist_after_m"] = round(after, 4)
    result["gain_m"] = round(before - after, 4)
    result["needed"] = before - after >= MIN_GAIN_M
    return result


def _solve(m: list[list[float]], v: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting for a small dense system."""
    n = len(v)
    a = [row[:] + [v[i]] for i, row in enumerate(m)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[piv] = a[piv], a[col]
        p = a[col][col]
        if abs(p) < 1e-12:
            raise ValueError("singular system")
        for r in range(n):
            if r != col and a[r][col]:
                f = a[r][col] / p
                for k in range(col, n + 1):
                    a[r][k] -= f * a[col][k]
    return [a[i][n] / a[i][i] for i in range(n)]


def _zero_sum_translations(pairs) -> dict[str, tuple[float, float]]:
    """Per-camera translations t minimising sum |(p_a + t_a) - (p_b + t_b)|^2
    over all pairs, with the translations of each connected group of
    cameras summing to zero (the normal equations are a graph Laplacian;
    the constraint removes its constant null space)."""
    cams = sorted({c for a, _, b, _ in pairs for c in (a, b)})
    idx = {c: i for i, c in enumerate(cams)}
    n = len(cams)
    lap = [[0.0] * n for _ in range(n)]
    rx, ry = [0.0] * n, [0.0] * n
    group = list(range(n))

    def root(i):
        while group[i] != i:
            i = group[i]
        return i

    for a, pa, b, pb in pairs:
        i, j = idx[a], idx[b]
        lap[i][i] += 1
        lap[j][j] += 1
        lap[i][j] -= 1
        lap[j][i] -= 1
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        rx[i] += dx
        ry[i] += dy
        rx[j] -= dx
        ry[j] -= dy
        group[root(i)] = root(j)
    for i in range(n):
        for j in range(n):
            if root(i) == root(j):
                lap[i][j] += 1.0
    tx, ty = _solve(lap, rx), _solve(lap, ry)
    return {c: (tx[i], ty[i]) for c, i in idx.items()}


_SENSOR_FIELDS = ("dx", "dy", "theta_deg", "scale")


def validate_offsets(data) -> str | None:
    """None when `data` is usable offsets.json content, else why not."""
    if not isinstance(data, dict):
        return "not a JSON object"
    sensors = data.get("sensors")
    if not isinstance(sensors, dict):
        return "sensors: expected an object"
    for name, entry in sensors.items():
        if not isinstance(entry, dict):
            return f"sensors.{name}: expected an object"
        for key in _SENSOR_FIELDS + (("sigma_m",) if "sigma_m" in entry else ()):
            v = entry.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                return f"sensors.{name}.{key}: expected a finite number, got {v!r}"
        if entry["scale"] <= 0:
            return f"sensors.{name}.scale: must be > 0, got {entry['scale']!r}"
        if "sigma_m" in entry and entry["sigma_m"] < 0:
            return f"sensors.{name}.sigma_m: must be >= 0, got {entry['sigma_m']!r}"
    return None


def load_offsets(path: Path | None) -> dict | None:
    """Read offsets.json; None when the path is unset or the file is missing.
    Raises ValueError naming the problem when the file is unreadable or
    invalid (see `read_offsets` for a variant that never raises)."""
    if path is None:
        return None
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    problem = validate_offsets(data)
    if problem:
        raise ValueError(f"{path}: {problem}")
    return data


def read_offsets(path: Path | None) -> tuple[dict | None, str | None]:
    """(offsets, None), (None, None) when missing, or (None, error) when
    invalid: an invalid file means no correction, never a crash."""
    try:
        return load_offsets(path), None
    except ValueError as exc:
        return None, str(exc)


def save_offsets(path: Path, offsets: dict) -> None:
    """Write offsets.json atomically (temp file in the same directory, then
    rename). Refuses a result learned from too few pairs."""
    if not offsets.get("sensors") or offsets.get("pairs", 0) < offsets.get("min_pairs", MIN_PAIRS):
        raise ValueError(
            f"not writing offsets learned from {offsets.get('pairs', 0)} pairs"
            f" (need {offsets.get('min_pairs', MIN_PAIRS)})"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        os.fchmod(fd, 0o644)  # mkstemp creates 0600; the file is meant to be world-readable
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(offsets, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
