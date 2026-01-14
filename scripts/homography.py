#!/usr/bin/env python3
"""
relies on ./config/config.json and ./config/cameras_runtime.json

new dependency for this one: 
pip install opencv-contrib-python numpy

config.json:
    Camera.Resolution (e.g., "3072x1728")
    TrackingCameras: { (optional) <camera_key>: true/false }  
    CharucoBoard:
        BeginScanning: bool
        ReferencePoints: { P1: {x,y}, P2: {x,y} }  (world coords)
        Board: { SquaresX, SquaresY, SquareSize, ArucoSize, Dictionary, (optional) P1CornerId, (optional) P2CornerId }

cameras_runtime.json: { <camera_key>: { rtsp, ip, mac, resolution:[w,h] } }

looks for when ./config/config.json Homography.BeginScanning == true
load camera RTSP via ./config/cameras_runtime.json (now keyed by cam MAC)
each RTSP with detected ChArUco board corners computes a homography
saves at ./homographies/<camera_key>_homography_<WxH>.yml
returns status/results to config.json and switches state Homography.BeginScanning to false

to run: (first will check configs once and act accordingly, second enables continuous watching)
    python3 multicam_charuco_homography.py --once
    python3 multicam_charuco_homography.py
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


#file functions
def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def atomic_write_json(path: Path, data: Dict) -> None: #atomic wriitng to avoid partial writes
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_filename(s: str) -> str: #windows sucks bro
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in s)


#config parsing functions
def parse_xy(v) -> np.ndarray: #takes in coordinates ([x, y] || {"x": x, "y": y} || {"X": X, "Y": Y} || "(x, y)" || "x,y" string) and returns np.array([x, y], dtype=float64)
    if v is None:
        raise ValueError("Missing required point value")
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return np.array([float(v[0]), float(v[1])], dtype=np.float64)
    if isinstance(v, dict):
        # Accept either lowercase or capitalized keys (matches current config.json).
        key_map = {str(k).lower(): k for k in v.keys()}
        if "x" in key_map and "y" in key_map:
            return np.array(
                [float(v[key_map["x"]]), float(v[key_map["y"]])], dtype=np.float64
            )
    if isinstance(v, str):
        s = v.strip().replace("(", "").replace(")", "")
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) == 2:
            return np.array([float(parts[0]), float(parts[1])], dtype=np.float64)
    raise ValueError(f"Unsupported point format: {v!r}")

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
    
    #backup being right side top and bottom inner corners
    p1_corner_id: Optional[int] = None
    p2_corner_id: Optional[int] = None

def board_from_new_config(charuco_cfg: Dict) -> BoardSpec:
    b = charuco_cfg["Board"]
    return BoardSpec(
        dictionary_name=b["Dictionary"],
        squares_x=int(b["SquaresX"]),
        squares_y=int(b["SquaresY"]),
        square_length=float(b["SquareSize"]),
        marker_length=float(b["ArucoSize"]),
        p1_corner_id=(int(b["P1CornerId"]) if b.get("P1CornerId") is not None else None),
        p2_corner_id=(int(b["P2CornerId"]) if b.get("P2CornerId") is not None else None),
    )


#geometry functions
def _get_chessboard_corners(board) -> np.ndarray:  #return board corners in local coords
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float64)
    return np.asarray(board.chessboardCorners, dtype=np.float64)

def infer_p1_p2_ids_from_board(board) -> Tuple[int, int]: #backup corners
    corners = _get_chessboard_corners(board)[:, :2] #(N,2)
    x = corners[:, 0]
    max_x = np.max(x)
    right_idxs = np.where(np.isclose(x, max_x))[0]
    ys = corners[right_idxs, 1]
    p1 = int(right_idxs[np.argmin(ys)])
    p2 = int(right_idxs[np.argmax(ys)])
    return p1, p2

def similarity_from_two_points(l1: np.ndarray, l2: np.ndarray, w1: np.ndarray, w2: np.ndarray):  #return s, R, t, so world = s * R @ local + t
    dl = (l2 - l1).astype(np.float64)
    dw = (w2 - w1).astype(np.float64)
    nl = float(np.linalg.norm(dl))
    nw = float(np.linalg.norm(dw))
    if nl < 1e-9 or nw < 1e-9:
        raise ValueError("P1 and P2 are identical or too close in local/world space")

    s = nw / nl
    a_l = math.atan2(dl[1], dl[0])
    a_w = math.atan2(dw[1], dw[0])
    ang = a_w - a_l
    c, si = math.cos(ang), math.sin(ang)
    R = np.array([[c, -si], [si, c]], dtype=np.float64)
    t = w1 - (s * (R @ l1))
    return s, R, t

def local_to_world(local_pts: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (s * (local_pts @ R.T)) + t #(N,2)


#detection and calculations
@dataclass
class HomographyResult:
    H: np.ndarray
    inliers: int
    rmse_world: float
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
    p1_id: int,
    p2_id: int,
    p1_world: np.ndarray,
    p2_world: np.ndarray,
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

    chess = _get_chessboard_corners(board)[:, :2] #(N,2)
    if p1_id < 0 or p1_id >= chess.shape[0] or p2_id < 0 or p2_id >= chess.shape[0]:
        raise ValueError(f"P1/P2 corner IDs out of range. ids: {p1_id}, {p2_id}, max: {chess.shape[0]-1}")

    l1 = chess[p1_id]
    l2 = chess[p2_id]
    s, R, t = similarity_from_two_points(l1, l2, p1_world, p2_world)

    local_pts = chess[ids]                          #(N,2)
    world_pts = local_to_world(local_pts, s, R, t)  #(N,2)

    src_pts = np.asarray(charuco_corners, dtype=np.float64)     #(N,1,2)
    dst_pts = world_pts.reshape(-1, 1, 2).astype(np.float64)    #(N,1,2)

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
        rmse_world=rmse,
        corners_used=int(len(ids)),
        frame_size=(w_img, h_img),
    )

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
    p1_id: int,
    p2_id: int,
    p1_world: np.ndarray,
    p2_world: np.ndarray,
    ransac_thresh_px: float,
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
        fs.write("rmse_world", float(result.rmse_world))
        fs.write("ransac_thresh_px", float(ransac_thresh_px))

        fs.write("board_dictionary", board_spec.dictionary_name)
        fs.write("board_squares", np.array([board_spec.squares_x, board_spec.squares_y], dtype=np.int32))
        fs.write("board_square_length", float(board_spec.square_length))
        fs.write("board_marker_length", float(board_spec.marker_length))

        fs.write("p1_corner_id", int(p1_id))
        fs.write("p2_corner_id", int(p2_id))
        fs.write("p1_world", p1_world.astype(np.float64))
        fs.write("p2_world", p2_world.astype(np.float64))
        fs.write("timestamp_unix", float(time.time()))
    finally:
        fs.release()

def enabled_cameras(config: Dict, cams_runtime: Dict) -> List[str]: #when TrackingCameras, only run cams set to true
    tc = config.get("TrackingCameras")
    if isinstance(tc, dict) and tc:
        return [k for k, v in tc.items() if bool(v) and k in cams_runtime]
    return list(cams_runtime.keys())

def run_once(base_dir: Path) -> None:
    cfg_path = base_dir / "config" / "config.json"
    cams_path = base_dir / "config" / "cameras_runtime.json"

    if not cfg_path.exists():
        raise SystemExit(f"Missing config file: {cfg_path}")
    if not cams_path.exists():
        raise SystemExit(f"Missing cameras runtime file: {cams_path}")

    config = load_json(cfg_path)
    cams = load_json(cams_path)

    cbcfg = config["CharucoBoard"]
    if not bool(cbcfg["BeginScanning"]):
        print("CharucoBoard.BeginScanning is false; nothing to do.")
        return

    ref = cbcfg["ReferencePoints"]
    p1_world = parse_xy(ref["P1"])
    p2_world = parse_xy(ref["P2"])

    board_spec = board_from_new_config(cbcfg)

    #tuning (beyond config)
    ransac_thresh_px = float(cbcfg["RansacReprojThresholdPx"])
    min_corners = int(cbcfg["MinCorners"])
    frames_to_try = int(cbcfg["FramesToAverage"])
    max_seconds_per_cam = float(cbcfg["MaxSecondsPerCam"])

    #output naming from config.json
    res_str = config["Camera"]["Resolution"]
    res_w, res_h = parse_resolution_str(res_str)

    board, aruco_dict = build_charuco_board(board_spec)

    if board_spec.p1_corner_id is not None and board_spec.p2_corner_id is not None:
        p1_id, p2_id = int(board_spec.p1_corner_id), int(board_spec.p2_corner_id)
    else:
        p1_id, p2_id = infer_p1_p2_ids_from_board(board)

    cam_keys = enabled_cameras(config, cams)
    print(f"Running scan for {len(cam_keys)} camera(s). P1_id={p1_id}, P2_id={p2_id}")

    #CharucoBoard status
    config["CharucoBoard"]["Status"] = "running"
    config["CharucoBoard"]["LastRunUnix"] = time.time()
    config["CharucoBoard"]["Results"] = {}
    config["CharucoBoard"].pop("Error", None)
    atomic_write_json(cfg_path, config)

    out_dir = base_dir / "homographies"
    ensure_dir(out_dir)

    any_success = False

    for cam_key in cam_keys:
        cam_info = cams[cam_key]
        rtsp = cam_info.get("rtsp")
        if not rtsp:
            config["CharucoBoard"]["Results"][cam_key] = {"ok": False, "error": "missing rtsp"}
            atomic_write_json(cfg_path, config)
            continue

        print(f"[{cam_key}] opening stream...")
        cap = open_capture(rtsp)
        if not cap.isOpened():
            config["CharucoBoard"]["Results"][cam_key] = {"ok": False, "error": "failed to open rtsp"}
            atomic_write_json(cfg_path, config)
            continue

        try:
            frames = grab_frames(cap, max_frames=frames_to_try, max_seconds=max_seconds_per_cam)
        finally:
            cap.release()

        if not frames:
            config["CharucoBoard"]["Results"][cam_key] = {"ok": False, "error": "no frames received"}
            atomic_write_json(cfg_path, config)
            continue

        best: Optional[HomographyResult] = None
        for f in frames:
            try:
                res = compute_homography_from_frame(
                    frame_bgr=f,
                    board=board,
                    aruco_dict=aruco_dict,
                    p1_id=p1_id,
                    p2_id=p2_id,
                    p1_world=p1_world,
                    p2_world=p2_world,
                    ransac_thresh_px=ransac_thresh_px,
                    min_corners=min_corners,
                )
            except Exception:
                res = None

            if res is None:
                continue
            if best is None or res.rmse_world < best.rmse_world:
                best = res

        if best is None:
            print(f"[{cam_key}] board not detected / homography failed")
            config["CharucoBoard"]["Results"][cam_key] = {"ok": False, "error": "board not detected / homography failed"}
            atomic_write_json(cfg_path, config)
            continue

        out_name = f"{safe_filename(cam_key)}_homography_{res_w}x{res_h}.yml"
        out_path = out_dir / out_name

        save_homography_yaml(
            out_path=out_path,
            result=best,
            cam_key=cam_key,
            rtsp_url=rtsp,
            config_resolution=(res_w, res_h),
            board_spec=board_spec,
            p1_id=p1_id,
            p2_id=p2_id,
            p1_world=p1_world,
            p2_world=p2_world,
            ransac_thresh_px=ransac_thresh_px,
        )

        print(f"[{cam_key}] saved: {out_path} (inliers={best.inliers}, rmse={best.rmse_world:.3f})")
        any_success = True

        config["CharucoBoard"]["Results"][cam_key] = {
            "ok": True,
            "output": str(out_path.as_posix()),
            "inliers": best.inliers,
            "corners_used": best.corners_used,
            "rmse_world": best.rmse_world,
            "frame_size": list(best.frame_size),
        }
        atomic_write_json(cfg_path, config)

    #finalize
    config = load_json(cfg_path)
    config["CharucoBoard"]["BeginScanning"] = False
    config["CharucoBoard"]["Status"] = "done" if any_success else "failed"
    config["CharucoBoard"]["LastRunUnix"] = time.time()
    atomic_write_json(cfg_path, config)

    print("Scan complete. CharucoBoard.BeginScanning set to false.")


def run_service(base_dir: Path, poll_seconds: float) -> None:
    cfg_path = base_dir / "config" / "config.json"
    print(f"Watching: {cfg_path}")
    while True:
        try:
            if cfg_path.exists():
                config = load_json(cfg_path)
                cbcfg = config["CharucoBoard"]
                if bool(cbcfg["BeginScanning"]):
                    run_once(base_dir=base_dir)
        except KeyboardInterrupt:
            print("Exiting.")
            return
        except Exception as e:
            #crash control
            try:
                config = load_json(cfg_path) if cfg_path.exists() else {}
                config["CharucoBoard"]["Status"] = "failed"
                config["CharucoBoard"]["Error"] = str(e)
                config["CharucoBoard"]["LastRunUnix"] = time.time()
                if config["CharucoBoard"].get("BeginScanning") is True:
                    config["CharucoBoard"]["BeginScanning"] = False
                atomic_write_json(cfg_path, config)
            except Exception:
                pass

        time.sleep(poll_seconds)

def main():
    parser = argparse.ArgumentParser(description="Multi-camera ChArUco homography calibration service (new config layout)")
    parser.add_argument("--once", action="store_true", help="Run one scan if BeginScanning is true, then exit.")
    parser.add_argument("--poll", type=float, default=1.0, help="Polling interval in seconds.")
    args = parser.parse_args()

    # Resolve to repo/app root (one level above scripts/).
    base_dir = Path(__file__).resolve().parent.parent
    if args.once:
        run_once(base_dir=base_dir)
    else:
        run_service(base_dir=base_dir, poll_seconds=args.poll)


if __name__ == "__main__":
    main()
