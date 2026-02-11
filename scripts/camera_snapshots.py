#!/usr/bin/env python3
"""scripts.camera_snapshots

Grab and save a single frame from each camera stream.

- Camera list and RTSP URLs are sourced from scripts.camera_handler.
- Does not read or write config.json.

Examples:
    python3 -m scripts.camera_snapshots
    python3 -m scripts.camera_snapshots --out snapshots --timeout 6 --warmup 10

Optional debug:
    python3 -m scripts.camera_snapshots --enabled-only --ext .jpg

Dependencies:
    pip install opencv-python (or opencv-contrib-python)
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception as e:
    raise SystemExit(
        "OpenCV is required. Install opencv-python (or opencv-contrib-python).\n\n"
        f"Import error: {e}"
    )

try:
    import scripts.camera_handler as camera_handler  # type: ignore
except Exception as e:
    raise SystemExit(
        "camera_handler module is required (scripts.camera_handler).\n\n"
        f"Import error: {e}"
    )


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _resolve_base_dir() -> Path:
    """Resolve the app root.

    Prefer the current working directory but fall back to repo root derived from this file location.
    """
    cwd = Path.cwd().resolve()
    if (cwd / "config").exists() or (cwd / "homographies").exists():
        return cwd
    script_root = Path(__file__).resolve().parent.parent
    return script_root


def safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(s))


def open_capture(rtsp_url: str) -> cv2.VideoCapture:
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|stimeout;5000000|rw_timeout;5000000",
    )
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    try:
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            cap.set(getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"), 5000)
    except Exception:
        pass
    try:
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            cap.set(getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"), 5000)
    except Exception:
        pass
    return cap


def _grab_one_frame(cap: cv2.VideoCapture, timeout_s: float, warmup: int) -> Optional[np.ndarray]:
    """Read frames until we have one usable frame.

    - warmup: number of initial frames to discard after first successful reads.
    """
    start = time.time()
    discarded = 0
    last_frame: Optional[np.ndarray] = None

    while time.time() - start < timeout_s:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue
        last_frame = frame
        if discarded < warmup:
            discarded += 1
            continue
        return frame

    return last_frame


def _iter_camera_keys(enabled_only: bool) -> List[str]:
    states = camera_handler.get_camera_states()
    keys = list(states.keys())
    if not enabled_only:
        return keys

    enabled: List[str] = []
    for k in keys:
        cam = camera_handler.get_camera(k)
        if cam is None:
            continue
        if bool(getattr(cam, "enabled", False)):
            enabled.append(k)
    return enabled


def main() -> int:
    parser = argparse.ArgumentParser(description="Save a single frame from all cameras")
    parser.add_argument(
        "--out",
        default="snapshots",
        help="Output directory (relative to base dir unless absolute). Default: snapshots",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Seconds to wait per camera before giving up. Default: 5.0",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=8,
        help="Number of initial frames to discard (helps avoid stale/black frames). Default: 8",
    )
    parser.add_argument(
        "--enabled-only",
        action="store_true",
        help="Only snapshot cameras with camera_handler Camera.enabled == true",
    )
    parser.add_argument(
        "--ext",
        default=".png",
        help="Output extension: .png or .jpg/.jpeg. Default: .png",
    )
    args = parser.parse_args()

    base_dir = _resolve_base_dir()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = base_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = str(args.ext).strip().lower()
    if not ext.startswith("."):
        ext = "." + ext
    if ext not in {".png", ".jpg", ".jpeg"}:
        raise SystemExit(f"Unsupported --ext {args.ext!r}; use .png or .jpg/.jpeg")

    keys = _iter_camera_keys(enabled_only=bool(args.enabled_only))
    log(f"Saving snapshots for {len(keys)} camera(s) -> {out_dir}")

    ok_count = 0
    fail_count = 0

    for cam_key in keys:
        cam = camera_handler.get_camera(cam_key)
        if cam is None:
            log(f"[{cam_key}] missing camera from camera_handler")
            fail_count += 1
            continue

        rtsp = getattr(cam, "rtsp", None)
        if not isinstance(rtsp, str) or not rtsp.strip():
            log(f"[{cam_key}] missing rtsp")
            fail_count += 1
            continue

        name = safe_filename(cam_key) if os.name == "nt" else str(cam_key)
        out_path = out_dir / f"{name}{ext}"

        log(f"[{cam_key}] opening stream...")
        cap = open_capture(rtsp)
        if not cap.isOpened():
            log(f"[{cam_key}] failed to open rtsp")
            fail_count += 1
            continue

        try:
            frame = _grab_one_frame(cap, timeout_s=float(args.timeout), warmup=int(args.warmup))
        finally:
            cap.release()

        if frame is None:
            log(f"[{cam_key}] no frame received")
            fail_count += 1
            continue

        try:
            wrote = bool(cv2.imwrite(str(out_path), frame))
        except Exception as e:
            wrote = False
            log(f"[{cam_key}] failed to save {out_path}: {e}")

        if wrote:
            h, w = frame.shape[:2]
            log(f"[{cam_key}] saved {out_path} ({w}x{h})")
            ok_count += 1
        else:
            log(f"[{cam_key}] failed to save {out_path}")
            fail_count += 1

    log(f"Done. OK={ok_count}, failed={fail_count}")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
