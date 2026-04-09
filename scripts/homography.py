#!/usr/bin/env python3
"""scripts.homography

Computes a per-camera homography from image pixels to *board-local* coordinates using a ChArUco board.

Key behaviors:
- Reads ./config/config.json for CharucoBoard.BeginScanning and CharucoBoard.Board settings.
- Camera streams + intrinsics are sourced from scripts.camera_handler (not cameras_runtime.json directly).
- Frames are undistorted (when intrinsics are available) before ChArUco detection.
- Writes homography YAML to ./homographies/<mac>_homography.yml.
- The only config.json write performed by this script is setting CharucoBoard.BeginScanning to false.

Dependencies:
    pip install opencv-contrib-python numpy

To run:
    python3 -m scripts.homography --once
    python3 -m scripts.homography
"""


from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    from cv2 import aruco  # type: ignore
except Exception as e:
    raise SystemExit(
        "OpenCV with aruco is required. Install opencv-contrib-python.\n"
        "Example: pip install opencv-contrib-python\n\n"
        f"Import error: {e}"
    )


try:
    import scripts.camera_handler as camera_handler  # type: ignore
except Exception as e:
    raise SystemExit(
        "camera_handler module is required (scripts.camera_handler).\n\n"
        f"Import error: {e}"
    )

import requests
from scripts import cloud_storage_media
from scripts.json_models.homography_upload import LocalHomographyResponseDto, SubmitLocalHomographyDto


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


#file functions
def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_base_dir() -> Path:
    """Resolve the app root that contains config/ and homographies/.

    Prefer the current working directory (systemd sets WorkingDirectory=/opt/p2bp/camera),
    but fall back to the repo root derived from this file location.
    """
    cwd = Path.cwd().resolve()
    if (cwd / "config" / "config.json").exists():
        return cwd
    script_root = Path(__file__).resolve().parent.parent
    if (script_root / "config" / "config.json").exists():
        return script_root
    return cwd


def _get_float(d: Dict, key: str, default: float) -> float:
    try:
        return float(d.get(key, default))
    except Exception:
        return default


def _get_int(d: Dict, key: str, default: int) -> int:
    try:
        return int(d.get(key, default))
    except Exception:
        return default

def atomic_write_json(path: Path, data: Dict) -> None:  # atomic writing to avoid partial writes
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preferred: atomic replace via temp file in the same directory.
    # Important: preserve mode/owner/group of the existing file.
    # systemd services often run as root; without this, os.replace() would
    # replace config.json with a new 0600 root-owned file, making it appear
    # inaccessible to non-root users ("red X" symptom in file explorers).
    tmp_path: Optional[str] = None
    existing_stat = None
    try:
        if path.exists():
            existing_stat = path.stat()
    except OSError:
        existing_stat = None

    def _apply_perms(fd: int) -> None:
        try:
            if existing_stat is not None:
                try:
                    os.fchmod(fd, int(existing_stat.st_mode) & 0o777)
                except Exception:
                    pass
                try:
                    if hasattr(os, "fchown"):
                        os.fchown(fd, int(existing_stat.st_uid), int(existing_stat.st_gid))  # type: ignore[attr-defined]
                except Exception:
                    pass
            else:
                # Reasonable default for config files.
                try:
                    os.fchmod(fd, 0o644)
                except Exception:
                    pass
        except Exception:
            pass

    def _fsync_dir(parent: Path) -> None:
        # Best-effort: ensure directory entry is committed (Linux).
        try:
            if hasattr(os, "O_DIRECTORY"):
                dfd = os.open(str(parent), os.O_RDONLY | os.O_DIRECTORY)  # type: ignore[attr-defined]
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
        except Exception:
            pass

    try:
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
        _apply_perms(tmp_fd)
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _fsync_dir(path.parent)
        tmp_path = None
    except PermissionError:
        # Fallback: write directly to the file (non-atomic).
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_filename(s: str) -> str: #windows sucks bro
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in s)


def topdown_filename_for_mac(mac: str) -> str:
    # Requirement: "<mac>_Top-Down.png".
    # Windows cannot create ':' filenames, so sanitize only on Windows.
    if os.name == "nt":
        return f"{safe_filename(mac)}_Top-Down.png"
    return f"{mac}_Top-Down.png"


#config parsing functions
def parse_resolution_str(s: str) -> Tuple[int, int]:
    #expecting FHD rez=
    if not isinstance(s, str) or "x" not in s:
        raise ValueError(f"Invalid resolution string: {s!r}")
    w, h = s.lower().split("x", 1)
    return int(w), int(h)

def dict_constant_from_name(name: str):
    #ex: "DICT_4X4_50"
    if not isinstance(name, str):
        raise ValueError("Dictionary name must be a string")
    if not name.startswith("DICT_"):
        name = "DICT_" + name
    if not hasattr(aruco, name):
        raise ValueError(f"Unknown ArUco dictionary constant: {name}")
    return getattr(aruco, name)


def _set_attr_if_present(obj, name: str, value) -> None:
    try:
        if hasattr(obj, name):
            setattr(obj, name, value)
    except Exception:
        pass


def _create_detector_params():
    """Create and tune detector parameters for more robust marker detection."""
    try:
        params = aruco.DetectorParameters_create()  # type: ignore[attr-defined]
    except Exception:
        params = aruco.DetectorParameters()  # type: ignore[call-arg]

    # These defaults are intentionally conservative; they mainly help with
    # blur/low-contrast and reduce missed detections.
    _set_attr_if_present(params, "adaptiveThreshWinSizeMin", 3)
    _set_attr_if_present(params, "adaptiveThreshWinSizeMax", 53)
    _set_attr_if_present(params, "adaptiveThreshWinSizeStep", 4)
    _set_attr_if_present(params, "minDistanceToBorder", 3)
    _set_attr_if_present(params, "minMarkerPerimeterRate", 0.03)
    _set_attr_if_present(params, "maxMarkerPerimeterRate", 4.0)
    _set_attr_if_present(params, "polygonalApproxAccuracyRate", 0.03)
    _set_attr_if_present(params, "minCornerDistanceRate", 0.05)
    _set_attr_if_present(params, "minMarkerDistanceRate", 0.05)
    _set_attr_if_present(params, "perspectiveRemoveIgnoredMarginPerCell", 0.13)
    _set_attr_if_present(params, "maxErroneousBitsInBorderRate", 0.35)
    _set_attr_if_present(params, "errorCorrectionRate", 0.6)

    # Corner refinement improves ChArUco interpolation stability.
    try:
        refine = getattr(aruco, "CORNER_REFINE_SUBPIX", None)
        if refine is not None:
            _set_attr_if_present(params, "cornerRefinementMethod", int(refine))
    except Exception:
        pass
    _set_attr_if_present(params, "cornerRefinementWinSize", 5)
    _set_attr_if_present(params, "cornerRefinementMaxIterations", 50)
    _set_attr_if_present(params, "cornerRefinementMinAccuracy", 0.1)

    return params


def _detect_markers(gray: np.ndarray, aruco_dict, detector_params):
    """Detect ArUco markers using either the new ArucoDetector API or legacy detectMarkers."""
    try:
        detector = aruco.ArucoDetector(aruco_dict, detector_params)  # type: ignore[attr-defined]
        corners, ids, rejected = detector.detectMarkers(gray)
        return corners, ids, rejected
    except Exception:
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
        return corners, ids, rejected


def _maybe_clahe(gray: np.ndarray) -> np.ndarray:
    """Lightweight contrast boost to help marker detection in dim scenes."""
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    except Exception:
        return gray

@dataclass
class BoardSpec:
    dictionary_name: str
    squares_x: int
    squares_y: int
    square_length: float #world units (in mm?)
    marker_length: float

def board_from_new_config(charuco_cfg: Dict) -> BoardSpec:
    b = charuco_cfg["Board"]
    return BoardSpec(
        dictionary_name=b["Dictionary"],
        squares_x=int(b["SquaresX"]),
        squares_y=int(b["SquaresY"]),
        square_length=float(b["SquareSize"]),
        marker_length=float(b["ArucoSize"]),
    )


def _get_chessboard_corners(board) -> np.ndarray:  #return board corners in local coords
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float64)
    return np.asarray(board.chessboardCorners, dtype=np.float64)


#detection and calculations
@dataclass
class HomographyResult:
    H: np.ndarray
    inliers: int
    rmse_board: float
    corners_used: int
    frame_size: Tuple[int, int] #(w,h)
    markers_detected: int
    charuco_detected: int
    used_undistorted_image: bool

def build_charuco_board(board_spec: BoardSpec):
    dict_const = dict_constant_from_name(board_spec.dictionary_name)
    aruco_dict = aruco.getPredefinedDictionary(dict_const)

    if hasattr(aruco, "CharucoBoard_create"):
        board = aruco.CharucoBoard_create(
            board_spec.squares_x,
            board_spec.squares_y,
            board_spec.square_length,
            board_spec.marker_length,
            aruco_dict,
        )
    else:
        board = aruco.CharucoBoard(
            (board_spec.squares_x, board_spec.squares_y),
            board_spec.square_length,
            board_spec.marker_length,
            aruco_dict,
        )
    return board, aruco_dict

def detect_charuco(
    gray: np.ndarray,
    aruco_dict,
    board,
    detector_params,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """Return (charuco_corners, charuco_ids, markers_detected)."""
    corners, ids, _ = _detect_markers(gray, aruco_dict, detector_params)
    markers_detected = 0 if ids is None else int(len(ids))
    if ids is None or len(ids) == 0:
        return None, None, markers_detected

    # Some OpenCV builds accept cameraMatrix/distCoeffs here; others don't.
    try:
        n, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board,
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
        )
    except TypeError:
        n, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board,
        )

    if charuco_ids is None or charuco_corners is None or int(n) <= 0:
        return None, None, markers_detected
    return charuco_corners, charuco_ids, markers_detected

def compute_homography_from_frame(
    frame_bgr: np.ndarray,
    board,
    aruco_dict,
    detector_params,
    ransac_thresh_px: float,
    min_corners: int,
    camera_matrix: Optional[np.ndarray] = None,
    dist_coeffs: Optional[np.ndarray] = None,
    use_undistorted_image: bool = True,
) -> Optional[HomographyResult]:
    h_img, w_img = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # First try marker/charuco detection on the provided frame.
    gray2 = _maybe_clahe(gray)
    charuco_corners, charuco_ids, markers_detected = detect_charuco(
        gray2, aruco_dict, board, detector_params, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs
    )
    if charuco_ids is None or charuco_corners is None:
        return None

    ids = charuco_ids.reshape(-1).astype(int)
    if len(ids) < min_corners:
        return None

    chess = _get_chessboard_corners(board)[:, :2]  # (N,2) board-local coordinates
    if np.max(ids) >= chess.shape[0] or np.min(ids) < 0:
        return None

    dst_local = chess[ids]  # (N,2)

    src_pts = np.asarray(charuco_corners, dtype=np.float64)  # (N,1,2)
    # If detection was performed on a distorted frame (i.e., not an already-undistorted image),
    # undistort the points before solving. This accounts for lens distortion without requiring
    # image warping that can harm detection.
    if (not use_undistorted_image) and camera_matrix is not None and dist_coeffs is not None:
        try:
            src_pts = cv2.undistortPoints(src_pts, camera_matrix, dist_coeffs, P=camera_matrix)
        except Exception:
            pass
    dst_pts = dst_local.reshape(-1, 1, 2).astype(np.float64)  # (N,1,2)

    H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh_px)
    if H is None or mask is None:
        return None

    mask = mask.reshape(-1).astype(bool)
    inliers = int(mask.sum())
    if inliers < max(4, min_corners // 2):
        return None

    pred = cv2.perspectiveTransform(src_pts.astype(np.float64), H)  #(N,1,2)
    err = np.linalg.norm((pred - dst_pts), axis=2).reshape(-1)      #(N,2)
    rmse = float(np.sqrt(np.mean((err[mask]) ** 2))) if inliers > 0 else float("inf")

    return HomographyResult(
        H=H.astype(np.float64),
        inliers=inliers,
        rmse_board=rmse,
        corners_used=int(len(ids)),
        frame_size=(w_img, h_img),
        markers_detected=int(markers_detected),
        charuco_detected=int(len(ids)),
        used_undistorted_image=bool(use_undistorted_image),
    )


class FrameUndistorter:
    def __init__(
        self,
        K: Optional[np.ndarray],
        dist: Optional[np.ndarray],
        expected_size: Optional[Tuple[int, int]] = None,
    ):
        self.K = K
        self.dist = dist
        self.expected_size = expected_size
        self.map1: Optional[np.ndarray] = None
        self.map2: Optional[np.ndarray] = None
        self._size: Optional[Tuple[int, int]] = None

    def ready(self) -> bool:
        return self.K is not None and self.dist is not None

    def _scaled_K_for_size(self, size: Tuple[int, int]) -> np.ndarray:
        """Return intrinsics scaled from expected_size to the given frame size."""
        if self.K is None:
            raise ValueError("Intrinsics not loaded")
        if not self.expected_size:
            return np.array(self.K, dtype=np.float64)

        exp_w, exp_h = self.expected_size
        w, h = size
        if exp_w <= 0 or exp_h <= 0 or (w == exp_w and h == exp_h):
            return np.array(self.K, dtype=np.float64)

        rx = float(w) / float(exp_w)
        ry = float(h) / float(exp_h)
        K2 = np.array(self.K, dtype=np.float64).copy()
        K2[0, 0] *= rx
        K2[0, 2] *= rx
        K2[1, 1] *= ry
        K2[1, 2] *= ry
        return K2

    def scaled_intrinsics_for_size(self, size: Tuple[int, int]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self.ready():
            return None, None
        try:
            K_use = self._scaled_K_for_size(size)
            dist_use = np.array(self.dist, dtype=np.float64).reshape(-1) if self.dist is not None else None
            return K_use, dist_use
        except Exception:
            return None, None

    def _ensure_maps(self, frame: np.ndarray) -> None:
        if not self.ready():
            return
        h, w = frame.shape[:2]
        size = (w, h)
        if self._size == size and self.map1 is not None and self.map2 is not None:
            return

        K_use = self._scaled_K_for_size(size)
        dist_use = np.array(self.dist, dtype=np.float64).reshape(-1)
        newK, _ = cv2.getOptimalNewCameraMatrix(K_use, dist_use, size, 1.0, size)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            K_use, dist_use, None, newK, size, cv2.CV_16SC2
        )
        self._size = size

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        if not self.ready():
            return frame
        self._ensure_maps(frame)
        if self.map1 is None or self.map2 is None:
            return frame
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)


def _render_top_down(
    frame_bgr: np.ndarray,
    H_img_to_local: np.ndarray,
    board,
    board_spec: BoardSpec,
    pixels_per_square: int,
    margin_px: int,
) -> np.ndarray:
    """Warp an image into a board-local top-down view.

    H_img_to_local maps image pixels (undistorted) -> board-local coordinates.

    IMPORTANT: This function intentionally warps the *entire frame*, not just the board.
    The output canvas is chosen from the warped locations of the image corners.
    """
    h_img, w_img = frame_bgr.shape[:2]
    if h_img <= 0 or w_img <= 0:
        raise ValueError("Invalid frame size")

    H = np.asarray(H_img_to_local, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError("Invalid homography shape")

    # Warp the 4 image corners into board-local coordinates.
    img_corners = np.array(
        [[[0.0, 0.0]], [[float(w_img - 1), 0.0]], [[float(w_img - 1), float(h_img - 1)]], [[0.0, float(h_img - 1)]]],
        dtype=np.float64,
    )
    warped_corners = cv2.perspectiveTransform(img_corners, H).reshape(-1, 2)
    finite = np.isfinite(warped_corners).all(axis=1)
    if not bool(np.any(finite)):
        raise ValueError("Homography produced non-finite warped corners")
    warped_corners = warped_corners[finite]

    # Include board corners too (helps keep the board in view if the warped image corners
    # do something odd due to numerical instability).
    try:
        chess = _get_chessboard_corners(board)[:, :2]
        if chess.size > 0:
            warped_corners = np.vstack([warped_corners, np.asarray(chess, dtype=np.float64)])
    except Exception:
        pass

    min_xy = np.min(warped_corners, axis=0)
    max_xy = np.max(warped_corners, axis=0)
    extent = (max_xy - min_xy).astype(np.float64)

    # Choose pixels-per-unit derived from pixels_per_square and square_length.
    sq_len = float(board_spec.square_length) if float(board_spec.square_length) > 1e-9 else 1.0
    px_per_unit = float(max(1, int(pixels_per_square))) / sq_len

    def _compute_size(px_per_unit_use: float) -> Tuple[int, int]:
        out_w2 = int(math.ceil(extent[0] * px_per_unit_use)) + int(2 * max(0, int(margin_px)))
        out_h2 = int(math.ceil(extent[1] * px_per_unit_use)) + int(2 * max(0, int(margin_px)))
        return max(32, out_w2), max(32, out_h2)

    out_w, out_h = _compute_size(px_per_unit)

    # Safety clamp: avoid gigantic images if the homography maps corners far away.
    max_dim = 4096
    big = float(max(out_w, out_h))
    if big > float(max_dim):
        scale = float(max_dim) / big
        px_per_unit = max(1e-6, px_per_unit * scale)
        out_w, out_h = _compute_size(px_per_unit)

    # Local -> output pixel mapping: translate so min_xy maps to margin, then scale.
    tx = float(margin_px) - float(min_xy[0]) * px_per_unit
    ty = float(margin_px) - float(min_xy[1]) * px_per_unit
    T_local_to_px = np.array(
        [[px_per_unit, 0.0, tx], [0.0, px_per_unit, ty], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # Full mapping: image -> local -> output pixels.
    M = T_local_to_px @ H
    warped = cv2.warpPerspective(frame_bgr, M, (out_w, out_h), flags=cv2.INTER_LINEAR)
    return warped

def open_capture(rtsp_url: str) -> cv2.VideoCapture:
    # Best-effort: configure FFmpeg RTSP timeouts to avoid hangs.
    # OpenCV's FFmpeg backend honors OPENCV_FFMPEG_CAPTURE_OPTIONS in many builds.
    # Only set it if user/system hasn't already.
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "rtsp_transport;tcp|stimeout;5000000|rw_timeout;5000000",
    )
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # Some OpenCV builds expose explicit timeout properties.
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

def grab_frames(cap: cv2.VideoCapture, max_frames: int, max_seconds: float) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    start = time.time()
    while len(frames) < max_frames and (time.time() - start) < max_seconds:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue
        frames.append(frame)
    return frames

#homography output
def save_homography_yaml(
    out_path: Path,
    result: HomographyResult,
    cam_key: str,
    rtsp_url: str,
    config_resolution: Tuple[int, int],
    board_spec: BoardSpec,
    ransac_thresh_px: float,
    undistorted: bool,
) -> None:
    ensure_dir(out_path.parent)
    fs = cv2.FileStorage(str(out_path), cv2.FILE_STORAGE_WRITE)
    try:
        fs.write("homography", result.H)
        fs.write("camera_key", cam_key)
        fs.write("rtsp", rtsp_url)
        fs.write("config_resolution", np.array(config_resolution, dtype=np.int32))
        fs.write("frame_size", np.array(result.frame_size, dtype=np.int32))
        fs.write("inliers", int(result.inliers))
        fs.write("corners_used", int(result.corners_used))
        fs.write("rmse_board", float(result.rmse_board))
        fs.write("ransac_thresh_px", float(ransac_thresh_px))
        fs.write("undistorted", int(1 if undistorted else 0))
        fs.write("markers_detected", int(getattr(result, "markers_detected", 0)))
        fs.write("charuco_detected", int(getattr(result, "charuco_detected", 0)))
        fs.write("used_undistorted_image", int(1 if getattr(result, "used_undistorted_image", False) else 0))

        fs.write("board_dictionary", board_spec.dictionary_name)
        fs.write("board_squares", np.array([board_spec.squares_x, board_spec.squares_y], dtype=np.int32))
        fs.write("board_square_length", float(board_spec.square_length))
        fs.write("board_marker_length", float(board_spec.marker_length))
        fs.write("timestamp_unix", float(time.time()))
    finally:
        fs.release()


def enabled_camera_macs(config: Dict, available_macs: List[str]) -> List[str]:
    """Determine which cameras to scan.

    If TrackingCameras exists, only scan MACs explicitly set to true.
    Otherwise, scan all available cameras.
    """
    tc = config.get("TrackingCameras")
    if isinstance(tc, dict) and tc:
        return [str(mac) for mac, enabled in tc.items() if bool(enabled) and str(mac) in set(available_macs)]
    return list(available_macs)


def _submit_local_homography(api_key: str, endpoint: str, dto: SubmitLocalHomographyDto) -> None:
    url = cloud_storage_media._join_url(endpoint, "/api/Homography/submit-local")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for attempt in range(1, 4):
        try:
            r = requests.post(url, headers=headers, json=dto.to_dict(), timeout=10)
            r.raise_for_status()
            return
        except Exception as e:
            if attempt < 3:
                time.sleep(float(2 ** attempt))
            else:
                raise


def run_once(base_dir: Path) -> None:
    cfg_path = base_dir / "config" / "config.json"

    if not cfg_path.exists():
        raise SystemExit(f"Missing config file: {cfg_path}")

    log(f"run_once starting (base_dir={base_dir})")
    config = load_json(cfg_path)

    if "CharucoBoard" not in config or not isinstance(config["CharucoBoard"], dict):
        raise SystemExit("Missing required config section: CharucoBoard")
    cbcfg = config["CharucoBoard"]
    if not bool(cbcfg["BeginScanning"]):
        log("CharucoBoard.BeginScanning is false; nothing to do.")
        return

    completed = False
    try:
        if "Board" not in cbcfg or not isinstance(cbcfg["Board"], dict):
            raise SystemExit("Missing required config section: CharucoBoard.Board")
        board_spec = board_from_new_config(cbcfg)

        log(
            "Charuco board loaded: "
            f"SquaresX={board_spec.squares_x}, SquaresY={board_spec.squares_y}, "
            f"SquareSize={board_spec.square_length}, ArucoSize={board_spec.marker_length}, "
            f"Dictionary={board_spec.dictionary_name}"
        )

        # Tuning parameters. Provide defaults so older/minimal config.json still works.
        # If RansacReprojThresholdPx isn't present, choose a default relative to square size.
        if "RansacReprojThresholdPx" in cbcfg:
            ransac_thresh_px = _get_float(cbcfg, "RansacReprojThresholdPx", 3.0)
        else:
            ransac_thresh_px = max(3.0, 0.05 * float(board_spec.square_length))
        min_corners = _get_int(cbcfg, "MinCorners", 8)
        frames_to_try = _get_int(cbcfg, "FramesToAverage", 30)
        max_seconds_per_cam = _get_float(cbcfg, "MaxSecondsPerCam", 5.0)
        # Top-down rendering uses internal defaults (no config changes assumed).
        topdown_px_per_square = 150
        topdown_margin_px = 20

        # Optional: record expected config resolution in the homography YAML (debugging).
        res_w, res_h = 0, 0
        try:
            res_str = config.get("Camera", {}).get("Resolution")
            if isinstance(res_str, str):
                res_w, res_h = parse_resolution_str(res_str)
        except Exception:
            res_w, res_h = 0, 0

        board, aruco_dict = build_charuco_board(board_spec)
        detector_params = _create_detector_params()

        log("Querying camera_handler for cameras...")
        states = camera_handler.get_camera_states()
        available_macs = list(states.keys())
        cam_keys = enabled_camera_macs(config, available_macs)
        log(f"Running scan for {len(cam_keys)} camera(s).")
        if not cam_keys:
            log("No cameras selected for scanning.")
            completed = True
            return

        # Load upload credentials once for all cameras (best-effort; scan continues if unavailable).
        _api_key: Optional[str] = None
        _endpoint: Optional[str] = None
        _project_id: Optional[str] = config.get("ProjectId") or None
        if _project_id is None:
            log("ProjectId missing from config — snapshot upload disabled until next heartbeat")
        try:
            _api_key, _endpoint = cloud_storage_media.load_env()
        except Exception as _cred_err:
            log(f"Upload credentials unavailable, server sync disabled: {_cred_err}")

        out_dir = base_dir / "homographies"
        ensure_dir(out_dir)

        debug_enabled = os.environ.get("HOMOGRAPHY_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
        debug_dir = out_dir / "debug"
        if debug_enabled:
            ensure_dir(debug_dir)

        for cam_key in cam_keys:
            cam = camera_handler.get_camera(cam_key)
            if cam is None:
                log(f"[{cam_key}] missing camera from camera_handler")
                continue

            rtsp = getattr(cam, "rtsp", None)
            if not isinstance(rtsp, str) or not rtsp.strip():
                log(f"[{cam_key}] missing rtsp")
                continue

            K_raw = getattr(cam, "camera_matrix", None)
            dist_raw = getattr(cam, "distortion_coefficients", None)
            declared_size: Optional[Tuple[int, int]] = None
            try:
                res = getattr(cam, "resolution", None)
                if isinstance(res, (list, tuple)) and len(res) == 2:
                    declared_size = (int(res[0]), int(res[1]))
            except Exception:
                declared_size = None
            K_np: Optional[np.ndarray] = None
            dist_np: Optional[np.ndarray] = None
            if K_raw is not None and dist_raw is not None:
                try:
                    K_np = np.array(K_raw, dtype=np.float64)
                    if K_np.shape != (3, 3):
                        K_np = None
                except Exception:
                    K_np = None
                try:
                    dist_np = np.array(dist_raw, dtype=np.float64).reshape(-1)
                except Exception:
                    dist_np = None

            undistorter = FrameUndistorter(K_np, dist_np, expected_size=declared_size)
            if undistorter.ready():
                log(f"[{cam_key}] using intrinsics from camera_handler (undistort enabled)")
            else:
                log(f"[{cam_key}] intrinsics missing/invalid (undistort disabled)")

            log(f"[{cam_key}] opening stream...")
            cap = open_capture(rtsp)
            if not cap.isOpened():
                log(f"[{cam_key}] failed to open rtsp")
                continue

            try:
                t0 = time.time()
                frames = grab_frames(cap, max_frames=frames_to_try, max_seconds=max_seconds_per_cam)
            finally:
                cap.release()

            log(f"[{cam_key}] grabbed {len(frames)} frame(s) in {time.time() - t0:.2f}s")

            if not frames:
                log(f"[{cam_key}] no frames received")
                continue

            best: Optional[HomographyResult] = None
            best_frame: Optional[np.ndarray] = None
            max_markers = 0
            max_charuco = 0
            debug_best_markers = -1
            debug_frame_raw: Optional[np.ndarray] = None
            for f in frames:
                res: Optional[HomographyResult] = None
                try:
                    # Preferred path: undistort image before ChArUco detection.
                    f_undist = undistorter.undistort(f)
                    K_use, dist_use = undistorter.scaled_intrinsics_for_size((f.shape[1], f.shape[0]))

                    # Update diagnostics even if we can't solve a homography.
                    try:
                        gray_u = cv2.cvtColor(f_undist, cv2.COLOR_BGR2GRAY)
                        _, ids_u, markers_u = detect_charuco(
                            _maybe_clahe(gray_u),
                            aruco_dict,
                            board,
                            detector_params,
                            camera_matrix=None,
                            dist_coeffs=None,
                        )
                        charuco_u = 0 if ids_u is None else int(len(ids_u.reshape(-1)))
                        max_markers = max(max_markers, int(markers_u))
                        max_charuco = max(max_charuco, int(charuco_u))
                    except Exception:
                        pass

                    try:
                        gray_r = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                        _, ids_r, markers_r = detect_charuco(
                            _maybe_clahe(gray_r),
                            aruco_dict,
                            board,
                            detector_params,
                            camera_matrix=K_use,
                            dist_coeffs=dist_use,
                        )
                        charuco_r = 0 if ids_r is None else int(len(ids_r.reshape(-1)))
                        max_markers = max(max_markers, int(markers_r))
                        max_charuco = max(max_charuco, int(charuco_r))
                        if debug_enabled and int(markers_r) > debug_best_markers:
                            debug_best_markers = int(markers_r)
                            debug_frame_raw = f.copy()
                    except Exception:
                        pass

                    res = compute_homography_from_frame(
                        frame_bgr=f_undist,
                        board=board,
                        aruco_dict=aruco_dict,
                        detector_params=detector_params,
                        ransac_thresh_px=ransac_thresh_px,
                        min_corners=min_corners,
                        camera_matrix=None,
                        dist_coeffs=None,
                        use_undistorted_image=True,
                    )

                    # If undistorted-image detection fails, try raw detection and undistort points.
                    if res is None:
                        res = compute_homography_from_frame(
                            frame_bgr=f,
                            board=board,
                            aruco_dict=aruco_dict,
                            detector_params=detector_params,
                            ransac_thresh_px=ransac_thresh_px,
                            min_corners=min_corners,
                            camera_matrix=K_use,
                            dist_coeffs=dist_use,
                            use_undistorted_image=False,
                        )
                        if res is not None:
                            # Keep best_frame consistent with homography domain (undistorted pixels).
                            best_frame = f_undist
                except Exception:
                    res = None

                # Track best observed detection stats even if homography fails.
                if res is not None:
                    max_markers = max(max_markers, int(res.markers_detected))
                    max_charuco = max(max_charuco, int(res.charuco_detected))

                if res is None:
                    continue
                if best is None or res.rmse_board < best.rmse_board:
                    best = res
                    if best_frame is None:
                        # When the successful solve came from the undistorted-image path,
                        # use that undistorted frame for top-down rendering.
                        try:
                            best_frame = undistorter.undistort(f)
                        except Exception:
                            best_frame = f

            if best is None:
                # Diagnostic hint: distinguish dictionary mismatch vs board config mismatch.
                log(
                    f"[{cam_key}] board not detected / homography failed "
                    f"(max_markers={max_markers}, max_charuco={max_charuco}). "
                    "If max_markers is 0 across all frames, the ArUco Dictionary likely doesn't match the printed board, "
                    "or the board isn't visible / is too small/blurred."
                )

                if debug_enabled:
                    try:
                        key = safe_filename(cam_key)
                        raw = debug_frame_raw if debug_frame_raw is not None else frames[0]
                        und = undistorter.undistort(raw)
                        raw_path = debug_dir / f"{key}_raw.png"
                        und_path = debug_dir / f"{key}_undist.png"
                        cv2.imwrite(str(raw_path), raw)
                        cv2.imwrite(str(und_path), und)

                        # Overlay detections (raw frame) for quick inspection.
                        try:
                            K_use, dist_use = undistorter.scaled_intrinsics_for_size((raw.shape[1], raw.shape[0]))
                            gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                            corners, ids, _rej = _detect_markers(_maybe_clahe(gray), aruco_dict, detector_params)
                            overlay = raw.copy()
                            if ids is not None and len(ids) > 0:
                                aruco.drawDetectedMarkers(overlay, corners, ids)
                                try:
                                    n, cc, ci = aruco.interpolateCornersCharuco(
                                        markerCorners=corners,
                                        markerIds=ids,
                                        image=gray,
                                        board=board,
                                        cameraMatrix=K_use,
                                        distCoeffs=dist_use,
                                    )
                                except TypeError:
                                    n, cc, ci = aruco.interpolateCornersCharuco(
                                        markerCorners=corners,
                                        markerIds=ids,
                                        image=gray,
                                        board=board,
                                    )
                                if ci is not None and cc is not None and int(n) > 0:
                                    aruco.drawDetectedCornersCharuco(overlay, cc, ci)
                            overlay_path = debug_dir / f"{key}_overlay.png"
                            cv2.imwrite(str(overlay_path), overlay)
                        except Exception:
                            pass

                        log(f"[{cam_key}] debug frames saved under: {debug_dir}")
                    except Exception as e:
                        log(f"[{cam_key}] debug frame dump failed: {e}")
                continue

            out_name = f"{safe_filename(cam_key)}_homography.yml"
            out_path = out_dir / out_name

            save_homography_yaml(
                out_path=out_path,
                result=best,
                cam_key=cam_key,
                rtsp_url=rtsp,
                config_resolution=(res_w, res_h),
                board_spec=board_spec,
                ransac_thresh_px=ransac_thresh_px,
                undistorted=undistorter.ready(),
            )

            # Upload raw undistorted snapshot to cloud storage for the puzzle-piece UI.
            snapshot_path: Optional[str] = None
            if best_frame is not None and _api_key is not None and _project_id is not None:
                try:
                    remote_snapshot = f"/vision/{_project_id}/homography-snapshots/{safe_filename(cam_key)}.jpg"
                    tmp_fd, tmp_snap = tempfile.mkstemp(suffix=".jpg")
                    try:
                        os.close(tmp_fd)
                        cv2.imwrite(tmp_snap, best_frame)
                        upload_result = cloud_storage_media.upload(
                            tmp_snap, remote_snapshot, api_key=_api_key, endpoint=_endpoint
                        )
                        snapshot_path = upload_result.remote_path
                        log(f"[{cam_key}] snapshot uploaded: {snapshot_path}")
                    finally:
                        try:
                            os.unlink(tmp_snap)
                        except OSError:
                            pass
                except Exception as e:
                    log(f"[{cam_key}] snapshot upload failed (continuing): {e}")

            # Submit local homography matrix + metadata to server.
            if _api_key is not None and _endpoint is not None:
                try:
                    dto = SubmitLocalHomographyDto(
                        CameraMac=cam_key,
                        Matrix=best.H.tolist(),
                        FrameSize=list(best.frame_size),
                        Inliers=best.inliers,
                        RmseBoard=best.rmse_board,
                        CornersUsed=best.corners_used,
                        MarkersDetected=best.markers_detected,
                        ArucoDict=board_spec.dictionary_name,
                        SquaresX=board_spec.squares_x,
                        SquaresY=board_spec.squares_y,
                        SquareLength=board_spec.square_length,
                        MarkerLength=board_spec.marker_length,
                        TimestampUnix=time.time(),
                        SnapshotPath=snapshot_path,
                        CameraMatrix=K_np.tolist() if K_np is not None else None,
                        DistortionCoefficients=dist_np.tolist() if dist_np is not None else None,
                    )
                    _submit_local_homography(_api_key, _endpoint, dto)
                    log(f"[{cam_key}] homography submitted to server")
                except Exception as e:
                    log(f"[{cam_key}] homography submit failed (continuing): {e}")

            # Render a quick top-down confirmation image using the same (undistorted) frame.
            try:
                if best_frame is not None:
                    topdown = _render_top_down(
                        frame_bgr=best_frame,
                        H_img_to_local=best.H,
                        board=board,
                        board_spec=board_spec,
                        pixels_per_square=topdown_px_per_square,
                        margin_px=topdown_margin_px,
                    )
                    topdown_path = out_dir / topdown_filename_for_mac(cam_key)
                    cv2.imwrite(str(topdown_path), topdown)
                    log(f"[{cam_key}] top-down saved: {topdown_path}")
            except Exception as e:
                log(f"[{cam_key}] top-down render failed: {e}")

            log(f"[{cam_key}] saved: {out_path} (inliers={best.inliers}, rmse={best.rmse_board:.3f})")

        completed = True
    except SystemExit:
        raise
    except Exception as e:
        log(f"run_once failed: {e}")
        traceback.print_exc(file=sys.stdout)

    if completed:
        log("Scan complete.")
    else:
        log("Scan aborted.")


def run_service(base_dir: Path, poll_seconds: float) -> None:
    cfg_path = base_dir / "config" / "config.json"
    log(f"Watching: {cfg_path}")
    last_begin_scanning: Optional[bool] = None
    last_cfg_mtime: Optional[float] = None
    while True:
        try:
            if cfg_path.exists():
                try:
                    mtime = cfg_path.stat().st_mtime
                except OSError:
                    mtime = None
                config = load_json(cfg_path)
                cbcfg = config.get("CharucoBoard") if isinstance(config, dict) else None
                if isinstance(cbcfg, dict):
                    begin = bool(cbcfg.get("BeginScanning"))

                    # Avoid infinite rescans if we cannot write BeginScanning back to false.
                    # Only trigger when BeginScanning transitions false->true, OR when the
                    # config file changes on disk while BeginScanning is true.
                    should_trigger = False
                    if begin:
                        if last_begin_scanning is False or last_begin_scanning is None:
                            should_trigger = True
                        elif mtime is not None and last_cfg_mtime is not None and mtime != last_cfg_mtime:
                            should_trigger = True

                    last_begin_scanning = begin
                    if mtime is not None:
                        last_cfg_mtime = mtime

                    if should_trigger:
                        run_once(base_dir=base_dir)
        except KeyboardInterrupt:
            log("Exiting.")
            return
        except Exception as e:
            log(f"Error in homography service loop: {e}")
            traceback.print_exc(file=sys.stdout)

        time.sleep(poll_seconds)

def main():
    parser = argparse.ArgumentParser(description="Multi-camera ChArUco homography calibration service (new config layout)")
    parser.add_argument("--once", action="store_true", help="Run one scan if BeginScanning is true, then exit.")
    parser.add_argument("--poll", type=float, default=1.0, help="Polling interval in seconds.")
    args = parser.parse_args()

    base_dir = _resolve_base_dir()
    try:
        log(f"script: {Path(__file__).resolve()}")
        log(f"cwd: {Path.cwd().resolve()}")
        log(f"base_dir: {base_dir}")
        ff = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        if ff:
            log(f"OPENCV_FFMPEG_CAPTURE_OPTIONS={ff}")
    except Exception:
        pass
    try:
        import sys as _sys
        log(f"Python binary: {_sys.executable}")
        log(f"Python version: {_sys.version.splitlines()[0]}")
        import cv2 as _cv2_diag
        log(f"cv2 version: {_cv2_diag.__version__}")
        _ = _cv2_diag.aruco.DICT_4X4_50
        log("cv2.aruco: available")
    except AttributeError:
        log("cv2.aruco: MISSING — ArUco module not built into this cv2")
    except Exception as _e:
        log(f"cv2.aruco: ERROR — {_e}")
    if args.once:
        run_once(base_dir=base_dir)
    else:
        run_service(base_dir=base_dir, poll_seconds=args.poll)


if __name__ == "__main__":
    main()
