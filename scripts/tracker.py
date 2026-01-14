import json
import os
import threading
import time
from collections import deque
from queue import Queue
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from osnet_ibn1_lite.encoder import OsNetEncoder

try:
    # When running as `python scripts/tracker.py`.
    import camera_handler
except Exception:
    # When `scripts` is imported as a package.
    from scripts import camera_handler  # type: ignore


# ---------------- PATHS ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "config", "config.json")
CONFIG_DIR = os.path.join(BASE_DIR, "config")


# ---------------- DEFAULTS (only used if missing in config.json) ----------------
DEFAULT_MODEL = "Yolov10n"
DEFAULT_CONF_THRESH = 0.35
DEFAULT_MAX_FPS = 30
DEFAULT_IMG_SIZE = 640
DEFAULT_MIN_BOX = 56
DEFAULT_TRACK_TTL = 2.0
EMBED_INTERVAL_SEC = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- MODELS ----------------
YOLO_MODEL: Optional[YOLO] = None
YOLO_LOCK = threading.Lock()

osnet = OsNetEncoder(
    input_width=704,
    input_height=480,
    weight_filepath=os.path.join(BASE_DIR, "osnet_ibn1_lite", "model_weights.pth.tar-40"),
    batch_size=32,
    num_classes=2022,
    patch_height=256,
    patch_width=128,
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    GPU=(DEVICE == "cuda"),
)


def _load_app_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("config.json must be a JSON object")
    return data


def _resolve_yolo_weights(model_setting: str) -> str:
    # Allows either friendly names (from config.json) or direct paths.
    mapping = {
        "Yolov10n": "yolov10n.pt",
        "Yolov10m": "yolov10m.pt",
        "Yolov8n": "yolov8n.pt",
    }
    candidate = mapping.get(model_setting, model_setting)
    # Prefer workspace-root-relative paths.
    if os.path.isabs(candidate):
        return candidate
    return os.path.join(BASE_DIR, candidate)


def _safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _homography_transform(H: np.ndarray, point: Tuple[float, float]) -> Tuple[float, float]:
    x, y = point
    p = np.array([x, y, 1.0], dtype=np.float64)
    p = H @ p
    if abs(p[2]) < 1e-9:
        return float(x), float(y)
    p /= p[2]
    return float(p[0]), float(p[1])


def _try_load_intrinsics_json(path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    K = data.get("camera_matrix")
    dist = data.get("distortion_coefficients")
    if K is None or dist is None:
        return None

    K = np.array(K, dtype=np.float64)
    dist = np.array(dist, dtype=np.float64)
    # Common formats: (1,5), (5,), (1,8), etc.
    dist = dist.reshape(-1)
    return K, dist


class Undistorter:
    def __init__(self, intrinsics_path: Optional[str]):
        self.intrinsics_path = intrinsics_path
        self.K: Optional[np.ndarray] = None
        self.dist: Optional[np.ndarray] = None
        self.map1: Optional[np.ndarray] = None
        self.map2: Optional[np.ndarray] = None
        self._size: Optional[Tuple[int, int]] = None

        if intrinsics_path:
            loaded = _try_load_intrinsics_json(intrinsics_path)
            if loaded:
                self.K, self.dist = loaded

    def ready(self) -> bool:
        return self.K is not None and self.dist is not None

    def _ensure_maps(self, frame: np.ndarray) -> None:
        if not self.ready():
            return
        h, w = frame.shape[:2]
        size = (w, h)
        if self._size == size and self.map1 is not None and self.map2 is not None:
            return
        newK, _ = cv2.getOptimalNewCameraMatrix(self.K, self.dist, size, 1.0, size)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, newK, size, cv2.CV_16SC2
        )
        self._size = size

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        if not self.ready():
            return frame
        self._ensure_maps(frame)
        if self.map1 is None or self.map2 is None:
            return frame
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)

# ---------------- CAMERA THREAD ----------------
class CameraThread(threading.Thread):
    def __init__(
        self,
        mac: str,
        cam: Any,
        conf_thresh: float,
        imgsz: int,
        max_fps: int,
        min_box: int,
        undistorter: Undistorter,
    ):
        super().__init__(daemon=True)
        self.mac = mac
        self.cam = cam
        self.display_name = getattr(cam, "name", None) or getattr(cam, "ip", None) or mac
        self.cap = cv2.VideoCapture(getattr(cam, "rtsp"))
        self.frame_queue = Queue(maxsize=5)
        self.frame_id = 0
        self.frame = None
        self.bt_to_sid: Dict[int, int] = {}
        self.free_sids: deque[int] = deque()
        self.next_sid = 0
        self.tracks_local: Dict[int, Dict[str, Any]] = {}
        self.output_tracks: Dict[int, Dict[str, Any]] = {}

        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.max_fps = max(1, max_fps)
        self.min_box = min_box
        self.undistorter = undistorter
        self.homography = getattr(cam, "homography", None)
        if self.homography is not None and not isinstance(self.homography, np.ndarray):
            try:
                self.homography = np.array(self.homography, dtype=np.float64)
            except Exception:
                self.homography = None

        threading.Thread(target=self.read_frames, daemon=True).start()

    def read_frames(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame)

    def _get_sid(self, bt_id):
        if bt_id in self.bt_to_sid:
            return self.bt_to_sid[bt_id]
        sid = self.free_sids.popleft() if self.free_sids else self.next_sid
        if sid == self.next_sid:
            self.next_sid += 1
        self.bt_to_sid[bt_id] = sid
        return sid

    def run(self):
        last_process = 0.0
        while True:
            if self.frame_queue.empty():
                time.sleep(0.001)
                continue

            frame = self.frame_queue.get()
            self.frame_id += 1
            now = time.time()

            # Limit processing rate.
            min_dt = 1.0 / float(self.max_fps)
            if now - last_process < min_dt:
                continue
            last_process = now

            # Undistort (if intrinsics available for this camera).
            frame = self.undistorter.undistort(frame)

            if YOLO_MODEL is None:
                # Should be initialized in main() before threads start.
                time.sleep(0.01)
                continue

            with YOLO_LOCK:
                results = YOLO_MODEL.track(frame, conf=self.conf_thresh, imgsz=self.imgsz,
                                           classes=[0], persist=True,
                                           tracker="bytetrack.yaml", verbose=False)

            if not results or results[0].boxes is None or results[0].boxes.id is None:
                self.frame = frame
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, bt_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                if y2 - y1 < self.min_box:
                    continue

                sid = self._get_sid(bt_id)
                t = self.tracks_local.get(bt_id)
                if t is None:
                    t = self.tracks_local[bt_id] = {
                        "sid": sid,
                        "last": now,
                        "last_embed": 0.0,
                        "feats": deque(maxlen=20),
                    }
                t["last"] = now

                feet = ((x1+x2)/2, y2)
                if isinstance(self.homography, np.ndarray) and self.homography.shape == (3, 3):
                    gx, gy = _homography_transform(self.homography, feet)
                else:
                    gx, gy = float(feet[0]), float(feet[1])

                out = self.output_tracks.get(sid)
                if out is None:
                    out = self.output_tracks[sid] = {
                        "track": [],
                        "vectors": [],
                        "mac": self.mac,
                    }
                out["track"].append({"time": int(now * 1000), "x": gx, "y": gy})

                # Compute OSNet embedding ~1Hz (for offline fusion script).
                if now - float(t["last_embed"]) >= EMBED_INTERVAL_SEC:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    feat = osnet.get_features([crop])[0].tolist()

                    t["feats"].append(feat)
                    out["vectors"].append({
                        "time": int(now * 1000),
                        "vector": [float(x) for x in feat],
                    })
                    t["last_embed"] = now

                # Visualize local SID only (no cross-camera fusion here).
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"SID {sid}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            self.frame = frame

# ---------------- MAIN ----------------
def main():
    cfg = _load_app_config()
    tracking_cfg = cfg.get("Tracking", {}) if isinstance(cfg.get("Tracking"), dict) else {}
    enabled = bool(tracking_cfg.get("Enabled", True))
    if not enabled:
        print("[INFO] Tracking.Enabled is false; exiting.")
        return

    model_setting = str(tracking_cfg.get("Model", DEFAULT_MODEL))
    conf_thresh = _safe_float(tracking_cfg.get("ConfidenceThreshold"), DEFAULT_CONF_THRESH)
    max_fps = _safe_int(tracking_cfg.get("MaxFps"), DEFAULT_MAX_FPS)

    weights_path = _resolve_yolo_weights(model_setting)
    if not os.path.exists(weights_path):
        print(f"[WARN] YOLO weights not found at {weights_path}; falling back to model_setting={model_setting}")
        weights_path = model_setting

    global YOLO_MODEL
    YOLO_MODEL = YOLO(weights_path)

    tracking_cams = cfg.get("TrackingCameras", {}) if isinstance(cfg.get("TrackingCameras"), dict) else {}
    enabled_macs = [mac for mac, on in tracking_cams.items() if bool(on)]
    if not enabled_macs:
        print("[WARN] No cameras enabled in TrackingCameras; exiting.")
        return

    cams: list[CameraThread] = []
    for i, mac in enumerate(enabled_macs):
        cam = camera_handler.get_camera(mac)
        if cam is None:
            print(f"[WARN] Camera not found in runtime file for MAC: {mac}")
            continue

        # Intrinsics file per camera: /config/<cameraname>.json
        # We try a few reasonable names since config.json only has MACs.
        cam_name = getattr(cam, "name", None) or f"cam{i}"
        candidates = [
            os.path.join(CONFIG_DIR, f"{mac}.json"),
            os.path.join(CONFIG_DIR, f"{cam_name}.json"),
            os.path.join(CONFIG_DIR, f"{cam_name}.intrinsics.json"),
            os.path.join(CONFIG_DIR, f"cam{i}.json"),
            os.path.join(CONFIG_DIR, f"{mac.replace(':', '_')}.json"),
            os.path.join(CONFIG_DIR, f"{getattr(cam, 'ip', '')}.json") if getattr(cam, "ip", "") else "",
        ]
        candidates = [p for p in candidates if p]
        intrinsics_path = next((p for p in candidates if os.path.exists(p)), None)
        if intrinsics_path is None:
            print(f"[WARN] No intrinsics found for {mac} (tried: {', '.join(os.path.basename(p) for p in candidates)})")
        else:
            print(f"[INFO] Using intrinsics for {mac}: {os.path.relpath(intrinsics_path, BASE_DIR)}")

        undistorter = Undistorter(intrinsics_path)
        if intrinsics_path and not undistorter.ready():
            print(f"[WARN] Intrinsics file exists but could not be parsed: {intrinsics_path}")

        c = CameraThread(
            mac=mac,
            cam=cam,
            conf_thresh=conf_thresh,
            imgsz=DEFAULT_IMG_SIZE,
            max_fps=max_fps,
            min_box=DEFAULT_MIN_BOX,
            undistorter=undistorter,
        )
        cams.append(c)
        c.start()

    if not cams:
        print("[WARN] No camera threads started; exiting.")
        return

    while True:
        for c in cams:
            if c.frame is not None:
                cv2.imshow(f"{c.display_name}", cv2.resize(c.frame, (640,360)))
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

    export: Dict[str, Dict[str, Any]] = {}
    for c in cams:
        export[c.mac] = {}
        # Best-effort TTL cleanup before export.
        cutoff = time.time() - DEFAULT_TRACK_TTL
        for sid, data in c.output_tracks.items():
            track = data.get("track", [])
            if track and (track[-1].get("time", 0) / 1000.0) < cutoff:
                continue
            export[c.mac][str(sid)] = {
                "track": data.get("track", []),
                "vectors": data.get("vectors", []),
            }

    with open(os.path.join(BASE_DIR, "tracks_by_camera.json"), "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)

    print("[INFO] Saved tracks_by_camera.json")

if __name__ == "__main__":
    main()
