"""Read the calibration files `GET /tracking/meta` reports.

Every reader returns None (or skips a file) when its input is missing or
unreadable, so a workstation without a finished calibration still answers
`meta` with null fields rather than an error.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MATRIX_KEYS = ("projectionMatrix_3x4_w2p", "projectionMatrix_3x4")


class _Loader(yaml.SafeLoader):
    """SafeLoader that reads OpenCV-style tagged nodes as plain data."""


def _construct_any(loader: yaml.SafeLoader, _suffix: str, node: yaml.Node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_Loader.add_multi_constructor("!", _construct_any)
_Loader.add_multi_constructor("tag:yaml.org,2002:opencv", _construct_any)


def load_yaml(path: Path):
    """Parse a YAML file, tolerating OpenCV's `%YAML:1.0` header and tags."""
    text = Path(path).read_text(encoding="utf-8")
    if text.startswith("%YAML:"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
    return yaml.load(text, Loader=_Loader)  # noqa: S506 (SafeLoader subclass)


def _numbers(value) -> list[float] | None:
    """Flatten a matrix given as a flat list, nested lists or an OpenCV
    `{rows, cols, data}` mapping."""
    if isinstance(value, dict) and "data" in value:
        value = value["data"]
    if not isinstance(value, list):
        return None
    flat: list = []
    for item in value:
        flat.extend(item if isinstance(item, list) else [item])
    try:
        return [float(v) for v in flat]
    except (TypeError, ValueError):
        return None


def _find_key(data, key: str):
    if isinstance(data, dict):
        if key in data:
            return data[key]
        for v in data.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    elif isinstance(data, list):
        for v in data:
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


def read_t_gt2px(calib_dir: Path | None) -> list[list[float]] | None:
    """The 3x3 world-to-pixel matrix `T_gt2px` from transforms.yml."""
    if calib_dir is None:
        return None
    path = Path(calib_dir) / "transforms.yml"
    try:
        nums = _numbers(_find_key(load_yaml(path), "T_gt2px"))
    except (OSError, yaml.YAMLError) as exc:
        log.info("no T_gt2px from %s: %s", path, exc)
        return None
    if nums is None or len(nums) != 9:
        return None
    return [nums[0:3], nums[3:6], nums[6:9]]


def png_size(path: Path) -> tuple[int, int] | None:
    """(width, height) from a PNG's IHDR chunk, or None."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
    except OSError:
        return None
    if len(head) < 24 or not head.startswith(_PNG_SIGNATURE) or head[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", head[16:24])


def _solve3(m: list[list[float]], b: list[float]) -> list[float] | None:
    """Solve m @ x = b for a 3x3 m by Cramer's rule; None if singular."""

    def det(a):
        return (
            a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
        )

    d = det(m)
    if abs(d) < 1e-12:
        return None
    out = []
    for col in range(3):
        a = [row[:] for row in m]
        for r in range(3):
            a[r][col] = b[r]
        out.append(det(a) / d)
    return out


def camera_centre(p: list[float]) -> list[float] | None:
    """World position of a camera from its row-major 3x4 projection matrix:
    C = -inv(P[:, :3]) @ P[:, 3]."""
    if len(p) != 12:
        return None
    m = [p[0:3], p[4:7], p[8:11]]
    t = [-p[3], -p[7], -p[11]]
    return _solve3(m, t)


def read_cameras(models_dir: Path | None) -> list[dict] | None:
    """[{name, x, y, z}] for each camInfo_*.yml, or None if the directory is missing."""
    if models_dir is None or not Path(models_dir).is_dir():
        return None
    cameras = []
    for path in sorted(Path(models_dir).glob("camInfo_*.yml")):
        try:
            data = load_yaml(path)
        except (OSError, yaml.YAMLError) as exc:
            log.info("skipping camera model %s: %s", path, exc)
            continue
        if not isinstance(data, dict):
            continue
        nums = next((_numbers(data[k]) for k in _MATRIX_KEYS if k in data), None)
        centre = camera_centre(nums) if nums else None
        if centre is None:
            continue
        x, y, z = centre
        cameras.append({"name": path.stem, "x": x, "y": y, "z": z})
    return cameras
