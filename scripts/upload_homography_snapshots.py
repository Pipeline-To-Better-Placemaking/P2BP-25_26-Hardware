#!/usr/bin/env python3
"""scripts.upload_homography_snapshots

One-shot script: grab a snapshot from each enabled camera and upload it to
GCS at /homography-snapshots/<mac>.jpg.

This is a manual testing tool to backfill snapshot images for homography
records that were created without one (e.g. via upload_local_homographies).

Usage:
    python3 -m scripts.upload_homography_snapshots
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

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

from scripts import cloud_storage_media


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _resolve_base_dir() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "config" / "config.json").exists():
        return cwd
    script_root = Path(__file__).resolve().parent.parent
    if (script_root / "config" / "config.json").exists():
        return script_root
    return cwd


def safe_filename(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(s))


def enabled_camera_macs(config: Dict, available_macs: List[str]) -> List[str]:
    """Return MACs to process: those explicitly enabled in TrackingCameras, or all if not set."""
    tc = config.get("TrackingCameras")
    if isinstance(tc, dict) and tc:
        return [str(mac) for mac, enabled in tc.items() if bool(enabled) and str(mac) in set(available_macs)]
    return list(available_macs)


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


def grab_frame(cap: cv2.VideoCapture, timeout_s: float = 10.0, warmup: int = 8) -> Optional[np.ndarray]:
    """Read frames until a usable one is available, discarding the first `warmup` frames."""
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


def main() -> int:
    base_dir = _resolve_base_dir()
    config_path = base_dir / "config" / "config.json"
    if not config_path.exists():
        raise SystemExit(f"config.json not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    api_key, endpoint = cloud_storage_media.load_env()

    project_id = config.get("ProjectId") or None
    if not project_id:
        raise SystemExit("ProjectId missing from config.json — run a heartbeat first so the backend populates it")

    states = camera_handler.get_camera_states()
    available_macs = list(states.keys())
    cam_keys = enabled_camera_macs(config, available_macs)
    log(f"Processing {len(cam_keys)} enabled camera(s).")

    ok_count = 0
    fail_count = 0

    for cam_key in cam_keys:
        cam = camera_handler.get_camera(cam_key)
        if cam is None:
            log(f"[{cam_key}] missing from camera_handler")
            fail_count += 1
            continue

        rtsp = getattr(cam, "rtsp", None)
        if not isinstance(rtsp, str) or not rtsp.strip():
            log(f"[{cam_key}] missing rtsp")
            fail_count += 1
            continue

        log(f"[{cam_key}] opening stream...")
        cap = open_capture(rtsp)
        if not cap.isOpened():
            log(f"[{cam_key}] failed to open rtsp")
            fail_count += 1
            continue

        try:
            frame = grab_frame(cap)
        finally:
            cap.release()

        if frame is None:
            log(f"[{cam_key}] no frame received")
            fail_count += 1
            continue

        remote_path = f"/vision/{project_id}/homography-snapshots/{safe_filename(cam_key)}.jpg"
        tmp_fd, tmp_snap = tempfile.mkstemp(suffix=".jpg")
        try:
            os.close(tmp_fd)
            cv2.imwrite(tmp_snap, frame)
            result = cloud_storage_media.upload(tmp_snap, remote_path, api_key=api_key, endpoint=endpoint)
            log(f"[{cam_key}] uploaded: {result.remote_path}")
            ok_count += 1
        except Exception as e:
            log(f"[{cam_key}] upload failed: {e}")
            fail_count += 1
        finally:
            try:
                os.unlink(tmp_snap)
            except OSError:
                pass

    log(f"Done. OK={ok_count}, failed={fail_count}")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
