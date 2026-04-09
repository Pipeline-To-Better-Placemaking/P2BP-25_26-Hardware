#!/usr/bin/env python3
"""scripts.intrinsics_calibrator

Continuously collects ChArUco sightings for camera intrinsics calibration.
Runs when config["Intrinsics"]["BeginCalibration"] is true.

Workflow:
  1. Poll config.json mtime (same pattern as aruco_scanner.py).
  2. When BeginCalibration is set, open each camera stream and detect ChArUco
     boards continuously.
  3. Accept sightings that pass quality gates (min corners, RMSE threshold,
     pose diversity). Track coverage across a spatial grid.
  4. Every upload cycle: POST accumulated sightings to /api/Intrinsics/submit-sightings
     and include a CalibrationState summary for the health report.
  5. When commanded (via BeginCalibration remaining true and enough sightings),
     run cv2.calibrateCamera locally and POST the result to
     /api/Intrinsics/submit-result, then stop collection.

To run:
    python3 -m scripts.intrinsics_calibrator --once
    python3 -m scripts.intrinsics_calibrator
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

try:
    import cv2
    from cv2 import aruco as cv2_aruco  # type: ignore
except Exception as e:
    raise SystemExit(
        "OpenCV with aruco is required. Install opencv-contrib-python.\n"
        f"Import error: {e}"
    )

try:
    import scripts.camera_handler as camera_handler  # type: ignore
except Exception as e:
    raise SystemExit(
        "camera_handler module is required.\n"
        f"Import error: {e}"
    )

from scripts import cloud_storage_media
from scripts.camera_onboard import _detect_camera_type
from scripts.homography import FrameUndistorter, grab_frames, open_capture
from scripts.json_models.intrinsics_calibration import (
    IntrinsicsSightingDto,
    IntrinsicsResultResponseDto,
    SubmitIntrinsicsResultDto,
    SubmitIntrinsicsSightingsDto,
)

# --- constants -----------------------------------------------------------

POLL_SECONDS = 1.0
MIN_CORNERS_FRACTION = 0.5      # accept if >= 50% of board corners detected
MAX_RMSE_ACCEPT = 1.5           # reject sighting if per-view RMSE is above this
MIN_ROTATION_DIFF_DEG = 10.0    # reject sighting if pose too similar to existing ones
MAX_SIGHTINGS_PER_CAM = 100     # sliding window; evict worst-RMSE when full
UPLOAD_EVERY_N_SIGHTINGS = 5    # flush to server after accumulating this many new ones

REGION_NAMES = [
    "top-left",    "top-center",    "top-right",
    "middle-left", "center",        "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
]


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


# --- board utilities -----------------------------------------------------

def _make_charuco_board(board_cfg: dict) -> Tuple[cv2_aruco.CharucoBoard, int]:
    """Build a CharucoBoard from config and return (board, total_inner_corners)."""
    details = board_cfg.get("Board") or {}
    squares_x: int = int(details.get("SquaresX", 7))
    squares_y: int = int(details.get("SquaresY", 5))
    square_len: float = float(details.get("SquareSize", 0.04))
    marker_len: float = float(details.get("ArucoSize", 0.03))
    dict_name: str = str(details.get("Dictionary", "DICT_4X4_50"))

    attr = getattr(cv2_aruco, dict_name, None)
    if attr is None and not dict_name.startswith("DICT_"):
        attr = getattr(cv2_aruco, f"DICT_{dict_name}", None)
    if attr is None:
        raise ValueError(f"Unknown ArUco dictionary: {dict_name!r}")

    dictionary = cv2_aruco.getPredefinedDictionary(int(attr))
    board = cv2_aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, dictionary)
    total_corners = (squares_x - 1) * (squares_y - 1)
    return board, total_corners


# --- quality gates -------------------------------------------------------

def _board_center(charuco_corners: np.ndarray) -> Tuple[float, float]:
    """Return (cx, cy) of the centroid of detected ChArUco corners."""
    pts = charuco_corners.reshape(-1, 2)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _grid_cell(cx: float, cy: float, frame_w: int, frame_h: int, n_cells: int) -> int:
    """Map board center to a grid cell index (row-major, sqrt(n_cells) x sqrt(n_cells))."""
    cols = int(round(math.sqrt(n_cells)))
    rows = cols
    col = min(int(cx / frame_w * cols), cols - 1)
    row = min(int(cy / frame_h * rows), rows - 1)
    return row * cols + col


def _rotation_angle_deg(rvec: np.ndarray) -> float:
    """Return the rotation magnitude in degrees from a Rodrigues rotation vector."""
    return float(np.linalg.norm(rvec) * 180.0 / math.pi)


def _is_pose_diverse(rvec: np.ndarray, accepted_rvecs: List[np.ndarray]) -> bool:
    """Return True if this pose is sufficiently different from all accepted ones."""
    angle = _rotation_angle_deg(rvec)
    for prev in accepted_rvecs:
        if abs(angle - _rotation_angle_deg(prev)) < MIN_ROTATION_DIFF_DEG:
            return False
    return True


# --- pose suggestion -----------------------------------------------------

def _suggest(coverage: List[int], accepted_rvecs: List[np.ndarray]) -> Tuple[Optional[str], Optional[str]]:
    """Return (suggested_region, suggested_tilt) or (None, None) if collection is complete."""
    for i, covered in enumerate(coverage):
        if not covered:
            return REGION_NAMES[i % len(REGION_NAMES)], None

    # All cells covered — check tilt diversity.
    if accepted_rvecs:
        avg_angle = float(np.mean([_rotation_angle_deg(r) for r in accepted_rvecs]))
        if avg_angle < 20.0:
            return None, "Tilt the board 30-45° toward the camera for better coverage"

    return None, None


# --- per-camera calibration state ----------------------------------------

class CameraCalibState:
    def __init__(self, mac: str, n_cells: int) -> None:
        self.mac = mac
        self.n_cells = n_cells
        self.sightings: List[IntrinsicsSightingDto] = []
        self.accepted_rvecs: List[np.ndarray] = []
        self.coverage: List[int] = [0] * n_cells
        self.new_since_upload: int = 0
        self.done: bool = False

    @property
    def current_rmse(self) -> float:
        if not self.sightings:
            return 0.0
        return float(np.mean([s.Rmse for s in self.sightings]))

    def to_health_dict(self) -> dict:
        if self.done:
            return {
                "Status": "done",
                "SightingsCollected": len(self.sightings),
                "CoverageGrid": list(self.coverage),
                "SuggestedRegion": None,
                "SuggestedTilt": None,
                "CurrentRmse": round(self.current_rmse, 3),
            }
        region, tilt = _suggest(self.coverage, self.accepted_rvecs)
        return {
            "Status": "collecting",
            "SightingsCollected": len(self.sightings),
            "CoverageGrid": list(self.coverage),
            "SuggestedRegion": region,
            "SuggestedTilt": tilt,
            "CurrentRmse": round(self.current_rmse, 3),
        }

    def add_sighting(
        self,
        charuco_corners: np.ndarray,
        charuco_ids: np.ndarray,
        rvec: np.ndarray,
        frame_w: int,
        frame_h: int,
        rmse: float,
        captured_at: str,
    ) -> bool:
        """Attempt to add sighting. Returns True if accepted."""
        if rmse > MAX_RMSE_ACCEPT:
            return False
        if not _is_pose_diverse(rvec, self.accepted_rvecs):
            return False

        pts = charuco_corners.reshape(-1, 2).tolist()
        ids = charuco_ids.flatten().tolist()

        dto = IntrinsicsSightingDto(
            CornerCount=len(pts),
            ImagePoints=pts,
            CornerIds=ids,
            FrameSize=[frame_w, frame_h],
            Rmse=rmse,
            CapturedAt=captured_at,
        )

        if len(self.sightings) >= MAX_SIGHTINGS_PER_CAM:
            # Evict the worst-RMSE sighting.
            worst_idx = max(range(len(self.sightings)), key=lambda i: self.sightings[i].Rmse)
            self.sightings.pop(worst_idx)

        self.sightings.append(dto)
        self.accepted_rvecs.append(rvec.copy())
        self.new_since_upload += 1

        # Mark grid cell covered.
        cx, cy = _board_center(charuco_corners)
        cell = _grid_cell(cx, cy, frame_w, frame_h, self.n_cells)
        self.coverage[cell] = 1

        return True


# --- sighting detection on a single frame --------------------------------

def _detect_charuco_sighting(
    frame_bgr: np.ndarray,
    board: cv2_aruco.CharucoBoard,
    min_corners: int,
    K_np: Optional[np.ndarray],
    dist_np: Optional[np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Detect ChArUco corners in a frame and estimate per-view RMSE.

    Returns (charuco_corners, charuco_ids, rvec, rmse) or None if detection fails
    or quality gates aren't met.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    dictionary = board.getDictionary()
    params = cv2_aruco.DetectorParameters()

    try:
        detector = cv2_aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(gray)
    except AttributeError:
        corners, ids, _ = cv2_aruco.detectMarkers(gray, dictionary, parameters=params)

    if ids is None or len(ids) == 0:
        return None

    ret, charuco_corners, charuco_ids = cv2_aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )
    if ret < min_corners or charuco_corners is None or charuco_ids is None:
        return None

    if K_np is None or dist_np is None:
        # No intrinsics available yet — estimate pose without undistortion.
        # RMSE approximation via corner reprojection is skipped; use 0 as placeholder.
        return charuco_corners, charuco_ids, np.zeros((3, 1), dtype=np.float64), 0.0

    ok, rvec, tvec = cv2_aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board, K_np, dist_np, None, None
    )
    if not ok or rvec is None:
        return None

    # Compute per-view reprojection RMSE.
    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
    if obj_pts is None or len(obj_pts) < 4:
        return None
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K_np, dist_np)
    rmse = float(np.sqrt(np.mean((projected.reshape(-1, 2) - img_pts.reshape(-1, 2)) ** 2)))

    return charuco_corners, charuco_ids, rvec, rmse


# --- calibration compute -------------------------------------------------

def _compute_and_upload(
    state: CameraCalibState,
    board: cv2_aruco.CharucoBoard,
    frame_w: int,
    frame_h: int,
    api_key: str,
    endpoint: str,
    is_per_unit: bool,
    model_id: Optional[str],
) -> bool:
    """Run cv2.calibrateCamera on accumulated sightings and upload the result."""
    if len(state.sightings) < 6:
        log(f"[{state.mac}] not enough sightings to calibrate ({len(state.sightings)} < 6)")
        return False

    obj_points_list = []
    img_points_list = []

    for s in state.sightings:
        charuco_corners = np.array(s.ImagePoints, dtype=np.float32).reshape(-1, 1, 2)
        charuco_ids = np.array(s.CornerIds, dtype=np.int32).reshape(-1, 1)
        obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
        if obj_pts is None or len(obj_pts) < 4:
            continue
        obj_points_list.append(obj_pts.reshape(-1, 1, 3).astype(np.float32))
        img_points_list.append(img_pts.reshape(-1, 1, 2).astype(np.float32))

    if len(obj_points_list) < 6:
        log(f"[{state.mac}] too few valid sightings after filtering ({len(obj_points_list)})")
        return False

    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points_list,
        img_points_list,
        (frame_w, frame_h),
        None,
        None,
    )
    log(f"[{state.mac}] calibration complete: rms={rms:.4f} sightings={len(obj_points_list)}")

    dto = SubmitIntrinsicsResultDto(
        CameraMac=state.mac,
        IsPerUnit=is_per_unit,
        ModelId=model_id,
        CameraMatrix=K.tolist(),
        DistortionCoefficients=dist.flatten().tolist(),
        ReprojectionError=float(rms),
        SightingsUsed=len(obj_points_list),
    )
    url = cloud_storage_media._join_url(endpoint, "/api/Intrinsics/submit-result")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=dto.to_dict(), timeout=10)
    r.raise_for_status()
    log(f"[{state.mac}] intrinsics result uploaded")
    return True


def _upload_sightings(
    state: CameraCalibState,
    api_key: str,
    endpoint: str,
    is_per_unit: bool,
    model_id: Optional[str],
) -> None:
    if not state.sightings or state.new_since_upload == 0:
        return
    dto = SubmitIntrinsicsSightingsDto(
        CameraMac=state.mac,
        IsPerUnit=is_per_unit,
        ModelId=model_id,
        Sightings=list(state.sightings),
    )
    url = cloud_storage_media._join_url(endpoint, "/api/Intrinsics/submit-sightings")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=dto.to_dict(), timeout=15)
    r.raise_for_status()
    state.new_since_upload = 0
    log(f"[{state.mac}] uploaded {len(state.sightings)} sightings")


# --- main collection loop ------------------------------------------------

def run_collection(base_dir: Path, cfg_path: Path, config: dict) -> None:
    intrinsics_cfg = config.get("Intrinsics") or {}
    n_cells: int = int(intrinsics_cfg.get("GridCells", 9))
    min_sightings: int = int(intrinsics_cfg.get("MinSightings", 40))
    # model_id from config is the override; if absent, each camera's model is
    # auto-detected from its MAC prefix (see camera_onboard.MAC_PREFIX_TO_MODEL).
    config_model_id: Optional[str] = intrinsics_cfg.get("ModelId") or None
    per_unit_macs: List[str] = [m.lower() for m in (intrinsics_cfg.get("PerUnitOverrideMacs") or [])]

    charuco_cfg = config.get("CharucoBoard") or {}
    try:
        board, total_corners = _make_charuco_board(charuco_cfg)
    except Exception as e:
        log(f"Failed to build ChArUco board from config: {e}")
        return

    min_corners = max(4, int(total_corners * MIN_CORNERS_FRACTION))

    try:
        api_key, endpoint = cloud_storage_media.load_env()
    except Exception as e:
        log(f"Upload credentials unavailable: {e}")
        return

    states = camera_handler.get_camera_states()
    tc = config.get("TrackingCameras", {})
    cam_macs: List[str] = (
        [mac for mac, enabled in tc.items() if bool(enabled) and mac in states]
        if isinstance(tc, dict) and tc
        else list(states.keys())
    )

    if not cam_macs:
        log("No cameras available for intrinsics calibration")
        return

    log(f"Starting intrinsics collection: cameras={cam_macs} min_sightings={min_sightings}")

    cam_states: Dict[str, CameraCalibState] = {}
    for mac in cam_macs:
        cam_states[mac] = CameraCalibState(mac, n_cells)

    # Write initial health state file so heartbeat can pick it up.
    _write_calibration_state(base_dir, cam_states)

    while True:
        for mac in cam_macs:
            cam = camera_handler.get_camera(mac)
            if cam is None:
                continue
            rtsp = getattr(cam, "rtsp", None)
            if not isinstance(rtsp, str) or not rtsp.strip():
                continue

            K_raw = getattr(cam, "camera_matrix", None)
            dist_raw = getattr(cam, "distortion_coefficients", None)
            K_np: Optional[np.ndarray] = None
            dist_np: Optional[np.ndarray] = None
            if K_raw is not None:
                try:
                    K_np = np.array(K_raw, dtype=np.float64)
                    if K_np.shape != (3, 3):
                        K_np = None
                except Exception:
                    K_np = None
            if dist_raw is not None:
                try:
                    dist_np = np.array(dist_raw, dtype=np.float64).reshape(-1)
                except Exception:
                    dist_np = None

            declared_res = getattr(cam, "resolution", None)
            declared_size = None
            if isinstance(declared_res, (list, tuple)) and len(declared_res) == 2:
                try:
                    declared_size = (int(declared_res[0]), int(declared_res[1]))
                except Exception:
                    pass

            undistorter = FrameUndistorter(K_np, dist_np, expected_size=declared_size)

            cap = open_capture(rtsp)
            if not cap.isOpened():
                continue

            try:
                frames = grab_frames(cap, max_frames=3, max_seconds=3.0)
            finally:
                cap.release()

            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or (declared_size[0] if declared_size else 1920)
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or (declared_size[1] if declared_size else 1080)

            state = cam_states[mac]
            captured_at = datetime.now(timezone.utc).isoformat()
            for frame in frames:
                undist = undistorter.undistort(frame)
                result = _detect_charuco_sighting(undist, board, min_corners, K_np, dist_np)
                if result is None:
                    continue
                charuco_corners, charuco_ids, rvec, rmse = result
                accepted = state.add_sighting(
                    charuco_corners, charuco_ids, rvec,
                    frame_w, frame_h, rmse, captured_at,
                )
                if accepted:
                    log(
                        f"[{mac}] sighting accepted: total={len(state.sightings)} "
                        f"rmse={rmse:.3f} coverage={sum(state.coverage)}/{n_cells}"
                    )

            # Resolve model ID: config override > MAC prefix auto-detection > None (per-unit).
            is_per_unit = mac in per_unit_macs
            if is_per_unit:
                effective_model_id = None
            else:
                effective_model_id = config_model_id or _detect_camera_type(mac)
                if effective_model_id is None:
                    # Unknown model — fall back to per-unit so calibration still works.
                    log(f"[{mac}] unknown camera model (MAC prefix unrecognised); storing as per-unit intrinsics")
                    is_per_unit = True

            # Flush sightings to server periodically.
            if state.new_since_upload >= UPLOAD_EVERY_N_SIGHTINGS:
                try:
                    _upload_sightings(state, api_key, endpoint, is_per_unit, effective_model_id)
                except Exception as e:
                    log(f"[{mac}] sightings upload failed: {e}")

            # Run calibration if enough sightings collected.
            if len(state.sightings) >= min_sightings and sum(state.coverage) >= n_cells:
                log(f"[{mac}] coverage complete — computing intrinsics")
                try:
                    ok = _compute_and_upload(
                        state, board, frame_w, frame_h,
                        api_key, endpoint, is_per_unit, effective_model_id,
                    )
                    if ok:
                        state.done = True
                except Exception as e:
                    log(f"[{mac}] calibration compute failed: {e}")

        _write_calibration_state(base_dir, cam_states)

        # Remove completed cameras after writing so the dashboard sees the "done" status.
        for mac in [m for m, s in cam_states.items() if s.done]:
            cam_states.pop(mac)

        if not cam_states:
            log("All cameras calibrated — collection done")
            return

        time.sleep(POLL_SECONDS)


def _write_calibration_state(base_dir: Path, cam_states: Dict[str, CameraCalibState]) -> None:
    """Write per-camera calibration state to a file for the heartbeat to include."""
    try:
        state_dir = base_dir / "run"
        state_dir.mkdir(parents=True, exist_ok=True)
        payload = {mac: s.to_health_dict() for mac, s in cam_states.items()}
        tmp = state_dir / "intrinsics_calibration_state.json.tmp"
        dst = state_dir / "intrinsics_calibration_state.json"
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
        tmp.replace(dst)
    except Exception:
        pass


def run_service(base_dir: Path) -> None:
    cfg_path = base_dir / "config" / "config.json"
    log(f"Watching: {cfg_path}")
    last_known_mtime: Optional[float] = None
    last_triggered_mtime: Optional[float] = None

    while True:
        try:
            if cfg_path.exists():
                try:
                    current_mtime = cfg_path.stat().st_mtime
                except OSError:
                    current_mtime = None

                if current_mtime is not None and current_mtime != last_known_mtime:
                    begin = False
                    config = {}
                    try:
                        with cfg_path.open("r", encoding="utf-8") as f:
                            config = json.load(f)
                        begin = bool((config.get("Intrinsics") or {}).get("BeginCalibration"))
                    except Exception:
                        pass

                    last_known_mtime = current_mtime

                    if begin and current_mtime != last_triggered_mtime:
                        last_triggered_mtime = current_mtime
                        try:
                            run_collection(base_dir=base_dir, cfg_path=cfg_path, config=config)
                        except Exception as e:
                            log(f"Collection run failed: {e}")
                            traceback.print_exc(file=sys.stdout)

        except KeyboardInterrupt:
            log("Exiting.")
            return
        except Exception as e:
            log(f"Error in intrinsics_calibrator service loop: {e}")
            traceback.print_exc(file=sys.stdout)

        time.sleep(POLL_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Guided intrinsics calibration service.")
    parser.add_argument("--once", action="store_true", help="Run one collection pass then exit.")
    args = parser.parse_args()

    base_dir = _resolve_base_dir()
    try:
        log(f"script: {Path(__file__).resolve()}")
        log(f"cwd: {Path.cwd().resolve()}")
        log(f"base_dir: {base_dir}")
    except Exception:
        pass

    cfg_path = base_dir / "config" / "config.json"
    if args.once:
        if not cfg_path.exists():
            raise SystemExit(f"Missing config file: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        run_collection(base_dir=base_dir, cfg_path=cfg_path, config=config)
    else:
        run_service(base_dir=base_dir)


if __name__ == "__main__":
    main()
