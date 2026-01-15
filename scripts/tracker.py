import json
import os
import sys
import threading
import time
from collections import deque
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# PyTorch 2.6 changed torch.load() defaults/behavior around "weights_only" and safe
# unpickling, which can break older Ultralytics weight loading with errors like:
#   Unsupported global: GLOBAL ultralytics.nn.tasks.DetectionModel
# We proactively allowlist the Ultralytics model class when the API is available.
try:
    if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
        from ultralytics.nn.tasks import DetectionModel  # type: ignore

        torch.serialization.add_safe_globals([DetectionModel])  # type: ignore
except Exception:
    # Best-effort: if this fails, Ultralytics may still work (or fail with a clear error).
    pass

try:
    # Preferred layout: OSNet code lives under ./models/osnet/<variant>/
    _BASE_DIR_FOR_IMPORTS = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _MODELS_OSNET_DIR = os.path.join(_BASE_DIR_FOR_IMPORTS, "models", "osnet")
    if os.path.isdir(_MODELS_OSNET_DIR) and _MODELS_OSNET_DIR not in sys.path:
        sys.path.insert(0, _MODELS_OSNET_DIR)

    # Import directly from the variant folder (models/osnet/osnet_ibn1_lite)
    from osnet_ibn1_lite.encoder import OsNetEncoder  # type: ignore
except Exception as e1:
    try:
        # Legacy layout (older repo): ./osnet/osnet_ibn1_lite/ packaged as osnet.osnet_ibn1_lite
        from osnet.osnet_ibn1_lite.encoder import OsNetEncoder  # type: ignore
    except Exception as e2:
        OsNetEncoder = None  # type: ignore
        _OSNET_IMPORT_ERROR = e2
        _OSNET_IMPORT_ERROR_PRIMARY = e1
    else:
        _OSNET_IMPORT_ERROR = None
        _OSNET_IMPORT_ERROR_PRIMARY = None
else:
    _OSNET_IMPORT_ERROR = None
    _OSNET_IMPORT_ERROR_PRIMARY = None

try:
    # When running as `python scripts/tracker.py`.
    import camera_handler
except Exception:
    # When `scripts` is imported as a package.
    from scripts import camera_handler  # type: ignore


# ---------------- PATHS ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODELS_OSNET_DIR = os.path.join(MODELS_DIR, "osnet")
MODELS_YOLO_DIR = os.path.join(MODELS_DIR, "yolo")
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

osnet = None


def _require_osnet() -> bool:
    """Ensure OSNet is available; log and return False if not.

    We intentionally avoid raising a full traceback under systemd because Ubuntu's
    crash reporter may try to write to /var/crash, which is blocked by
    ProtectSystem=strict.
    """

    global osnet

    if _OSNET_IMPORT_ERROR is not None or OsNetEncoder is None:
        print(
            "[FATAL] OSNet is required but could not be imported. "
            "Ensure the OSNet code is present under /opt/p2bp/camera/models/osnet/osnet_ibn1_lite "
            "(or installed on PYTHONPATH). "
            f"Import error: {_OSNET_IMPORT_ERROR}"
        )
        if "_OSNET_IMPORT_ERROR_PRIMARY" in globals() and _OSNET_IMPORT_ERROR_PRIMARY is not None:
            print(f"[FATAL] Primary import attempt (osnet.osnet_ibn1_lite) failed: {_OSNET_IMPORT_ERROR_PRIMARY}")
        return False

    weights_path = os.environ.get("OSNET_WEIGHTS_PATH")
    if weights_path:
        if not os.path.isabs(weights_path):
            weights_path = os.path.join(BASE_DIR, weights_path)
        candidates = [weights_path]
    else:
        candidates = [
            os.path.join(MODELS_OSNET_DIR, "osnet_ibn1_lite", "model_weights.pth.tar-40"),
            # Back-compat locations
            os.path.join(BASE_DIR, "osnet", "osnet_ibn1_lite", "model_weights.pth.tar-40"),
            os.path.join(BASE_DIR, "osnet_ibn1_lite", "model_weights.pth.tar-40"),
        ]

    weights_path = next((p for p in candidates if os.path.exists(p)), None)
    if weights_path is None:
        print("[FATAL] OSNet weights missing. Tried:")
        for p in candidates:
            print(f"  - {p}")
        print("[FATAL] Set OSNET_WEIGHTS_PATH to override.")
        return False

    try:
        osnet = OsNetEncoder(
            input_width=704,
            input_height=480,
            weight_filepath=weights_path,
            batch_size=32,
            num_classes=2022,
            patch_height=256,
            patch_width=128,
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
            GPU=(DEVICE == "cuda"),
        )
    except Exception as e:
        print(f"[FATAL] OSNet initialization failed: {e}")
        return False

    return True


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
    if os.path.isabs(candidate):
        return candidate

    # If the config provides a subpath (e.g., "models/yolo/yolov10n.pt"), treat it as
    # workspace-root-relative.
    if "/" in candidate or "\\" in candidate:
        return os.path.join(BASE_DIR, candidate)

    # Prefer the new location: ./models/yolo/<file>.pt
    models_path = os.path.join(MODELS_YOLO_DIR, candidate)
    if os.path.exists(models_path):
        return models_path

    # Back-compat: workspace-root-relative paths.
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
    def __init__(
        self,
        K: Optional[np.ndarray] = None,
        dist: Optional[np.ndarray] = None,
        expected_size: Optional[Tuple[int, int]] = None,
        intrinsics_path: Optional[str] = None,
    ):
        self.intrinsics_path = intrinsics_path
        self.expected_size = expected_size
        self.K: Optional[np.ndarray] = K
        self.dist: Optional[np.ndarray] = dist
        self.map1: Optional[np.ndarray] = None
        self.map2: Optional[np.ndarray] = None
        self._size: Optional[Tuple[int, int]] = None

        if (self.K is None or self.dist is None) and intrinsics_path:
            loaded = _try_load_intrinsics_json(intrinsics_path)
            if loaded:
                self.K, self.dist = loaded

    def ready(self) -> bool:
        return self.K is not None and self.dist is not None

    def _scaled_K_for_size(self, size: Tuple[int, int]) -> np.ndarray:
        """Return intrinsics scaled from expected_size to the given frame size."""
        if self.K is None:
            raise ValueError("Intrinsics not loaded")
        if not self.expected_size:
            return self.K

        exp_w, exp_h = self.expected_size
        w, h = size
        if exp_w <= 0 or exp_h <= 0 or (w == exp_w and h == exp_h):
            return self.K

        rx = float(w) / float(exp_w)
        ry = float(h) / float(exp_h)
        K2 = self.K.astype(np.float64).copy()
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
        dist_use = self.dist
        if dist_use is None:
            return

        # OpenCV accepts a variety of distortion shapes; normalize to (N,).
        dist_use = np.array(dist_use, dtype=np.float64).reshape(-1)

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
            if osnet is None:
                print("[FATAL] OSNet is required but was not initialized; stopping camera thread")
                return

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
    if not _require_osnet():
        return

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

    cams: List[CameraThread] = []
    for i, mac in enumerate(enabled_macs):
        cam = camera_handler.get_camera(mac)
        if cam is None:
            print(f"[WARN] Camera not found in runtime file for MAC: {mac}")
            continue

        rtsp = getattr(cam, "rtsp", None)
        if not isinstance(rtsp, str) or not rtsp.strip():
            print(f"[WARN] Camera {mac} has no RTSP URL; skipping")
            continue

        # Preferred: use scaled intrinsics from camera_handler (already adjusted to config Camera.Resolution).
        K = getattr(cam, "camera_matrix", None)
        dist = getattr(cam, "distortion_coefficients", None)
        declared_size: Optional[Tuple[int, int]] = None
        try:
            res = getattr(cam, "resolution", None)
            if isinstance(res, (list, tuple)) and len(res) == 2:
                declared_size = (int(res[0]), int(res[1]))
        except Exception:
            declared_size = None

        K_np: Optional[np.ndarray] = None
        dist_np: Optional[np.ndarray] = None
        if K is not None and dist is not None:
            try:
                K_np = np.array(K, dtype=np.float64)
                if K_np.shape != (3, 3):
                    K_np = None
            except Exception:
                K_np = None
            try:
                dist_np = np.array(dist, dtype=np.float64).reshape(-1)
            except Exception:
                dist_np = None

        intrinsics_path: Optional[str] = None
        intrinsics_source = "camera_handler" if (K_np is not None and dist_np is not None) else "legacy"
        if K_np is None or dist_np is None:
            # Fallback (legacy): Intrinsics file per camera: /config/<cameraname>.json
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
                print(
                    f"[WARN] No intrinsics provided by camera_handler for {mac} and no legacy intrinsics JSON found "
                    f"(tried: {', '.join(os.path.basename(p) for p in candidates)})"
                )
            else:
                print(f"[INFO] Using legacy intrinsics for {mac}: {os.path.relpath(intrinsics_path, BASE_DIR)}")
        else:
            print(f"[INFO] Using camera_handler intrinsics for {mac}")

        # Prevent accidental double-scaling:
        # - camera_handler already scales intrinsics to the configured resolution.
        # - only apply dynamic scaling when using legacy per-camera JSON intrinsics.
        expected_size_for_scaling = declared_size if intrinsics_source == "legacy" else None

        undistorter = Undistorter(
            K=K_np,
            dist=dist_np,
            expected_size=expected_size_for_scaling,
            intrinsics_path=intrinsics_path,
        )
        if (K_np is not None or dist_np is not None) and not undistorter.ready():
            print(f"[WARN] camera_handler intrinsics present but invalid for {mac}")
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
