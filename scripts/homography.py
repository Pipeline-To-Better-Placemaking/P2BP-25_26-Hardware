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
import tempfile
import time
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

def detect_charuco(gray: np.ndarray, aruco_dict, board) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]: #return corners and ID
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict)
    if ids is None or len(ids) == 0:
        return None, None

    n, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=board,
    )
    if charuco_ids is None or charuco_corners is None or int(n) <= 0:
        return None, None
    return charuco_corners, charuco_ids

def compute_homography_from_frame(
    frame_bgr: np.ndarray,
    board,
    aruco_dict,
    ransac_thresh_px: float,
    min_corners: int,
) -> Optional[HomographyResult]:
    h_img, w_img = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    charuco_corners, charuco_ids = detect_charuco(gray, aruco_dict, board)
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
    For visualization, we map board-local coordinates into an output pixel canvas.
    """
    chess = _get_chessboard_corners(board)[:, :2]
    if chess.size == 0:
        raise ValueError("Board chessboard corners not available")

    # Board-local extents (in board units, e.g., square_length units).
    min_xy = np.min(chess, axis=0)
    max_xy = np.max(chess, axis=0)
    extent = (max_xy - min_xy).astype(np.float64)

    # Choose pixels-per-unit derived from pixels_per_square and square_length.
    # This keeps the image size stable regardless of unit choice.
    sq_len = float(board_spec.square_length) if float(board_spec.square_length) > 1e-9 else 1.0
    px_per_unit = float(max(1, int(pixels_per_square))) / sq_len

    out_w = int(math.ceil(extent[0] * px_per_unit)) + int(2 * max(0, int(margin_px)))
    out_h = int(math.ceil(extent[1] * px_per_unit)) + int(2 * max(0, int(margin_px)))
    out_w = max(32, out_w)
    out_h = max(32, out_h)

    # Local -> output pixel mapping: translate so min_xy maps to margin, then scale.
    tx = float(margin_px) - float(min_xy[0]) * px_per_unit
    ty = float(margin_px) - float(min_xy[1]) * px_per_unit
    T_local_to_px = np.array(
        [[px_per_unit, 0.0, tx], [0.0, px_per_unit, ty], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # Full mapping: image -> local -> output pixels.
    M = T_local_to_px @ np.asarray(H_img_to_local, dtype=np.float64)
    warped = cv2.warpPerspective(frame_bgr, M, (out_w, out_h), flags=cv2.INTER_LINEAR)
    return warped

def open_capture(rtsp_url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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


def _set_begin_scanning_false(cfg_path: Path) -> None:
    try:
        config = load_json(cfg_path)
    except Exception:
        return
    if not isinstance(config, dict):
        return
    cb = config.get("CharucoBoard")
    if not isinstance(cb, dict):
        return
    if cb.get("BeginScanning") is False:
        return
    cb["BeginScanning"] = False
    config["CharucoBoard"] = cb
    try:
        atomic_write_json(cfg_path, config)
    except Exception:
        pass

def run_once(base_dir: Path) -> None:
    cfg_path = base_dir / "config" / "config.json"

    if not cfg_path.exists():
        raise SystemExit(f"Missing config file: {cfg_path}")

    config = load_json(cfg_path)

    if "CharucoBoard" not in config or not isinstance(config["CharucoBoard"], dict):
        raise SystemExit("Missing required config section: CharucoBoard")
    cbcfg = config["CharucoBoard"]
    if not bool(cbcfg["BeginScanning"]):
        print("CharucoBoard.BeginScanning is false; nothing to do.")
        return

    try:
        if "Board" not in cbcfg or not isinstance(cbcfg["Board"], dict):
            raise SystemExit("Missing required config section: CharucoBoard.Board")
        board_spec = board_from_new_config(cbcfg)

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

        states = camera_handler.get_camera_states()
        available_macs = list(states.keys())
        cam_keys = enabled_camera_macs(config, available_macs)
        print(f"Running scan for {len(cam_keys)} camera(s).")

        out_dir = base_dir / "homographies"
        ensure_dir(out_dir)

        for cam_key in cam_keys:
            cam = camera_handler.get_camera(cam_key)
            if cam is None:
                print(f"[{cam_key}] missing camera from camera_handler")
                continue

            rtsp = getattr(cam, "rtsp", None)
            if not isinstance(rtsp, str) or not rtsp.strip():
                print(f"[{cam_key}] missing rtsp")
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
                print(f"[{cam_key}] using intrinsics from camera_handler (undistort enabled)")
            else:
                print(f"[{cam_key}] intrinsics missing/invalid (undistort disabled)")

            print(f"[{cam_key}] opening stream...")
            cap = open_capture(rtsp)
            if not cap.isOpened():
                print(f"[{cam_key}] failed to open rtsp")
                continue

            try:
                frames = grab_frames(cap, max_frames=frames_to_try, max_seconds=max_seconds_per_cam)
            finally:
                cap.release()

            if not frames:
                print(f"[{cam_key}] no frames received")
                continue

            best: Optional[HomographyResult] = None
            best_frame: Optional[np.ndarray] = None
            for f in frames:
                try:
                    f_use = undistorter.undistort(f)
                    res = compute_homography_from_frame(
                        frame_bgr=f_use,
                        board=board,
                        aruco_dict=aruco_dict,
                        ransac_thresh_px=ransac_thresh_px,
                        min_corners=min_corners,
                    )
                except Exception:
                    res = None

                if res is None:
                    continue
                if best is None or res.rmse_board < best.rmse_board:
                    best = res
                    best_frame = f_use

            if best is None:
                print(f"[{cam_key}] board not detected / homography failed")
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
                    print(f"[{cam_key}] top-down saved: {topdown_path}")
            except Exception as e:
                print(f"[{cam_key}] top-down render failed: {e}")

            print(f"[{cam_key}] saved: {out_path} (inliers={best.inliers}, rmse={best.rmse_board:.3f})")
    finally:
        # Only config.json write this script performs.
        _set_begin_scanning_false(cfg_path)

    print("Scan complete. CharucoBoard.BeginScanning set to false.")


def run_service(base_dir: Path, poll_seconds: float) -> None:
    cfg_path = base_dir / "config" / "config.json"
    print(f"Watching: {cfg_path}")
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
            print("Exiting.")
            return
        except Exception as e:
            print(f"Error in homography service loop: {e}")
            # Best-effort crash control: only flip BeginScanning back to false.
            try:
                _set_begin_scanning_false(cfg_path)
            except Exception:
                pass

        time.sleep(poll_seconds)

def main():
    parser = argparse.ArgumentParser(description="Multi-camera ChArUco homography calibration service (new config layout)")
    parser.add_argument("--once", action="store_true", help="Run one scan if BeginScanning is true, then exit.")
    parser.add_argument("--poll", type=float, default=1.0, help="Polling interval in seconds.")
    args = parser.parse_args()

    base_dir = _resolve_base_dir()
    print(f"Base dir: {base_dir}")
    if args.once:
        run_once(base_dir=base_dir)
    else:
        run_service(base_dir=base_dir, poll_seconds=args.poll)


if __name__ == "__main__":
    main()
