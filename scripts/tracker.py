import json
import os
import signal
import sys
import tempfile
import threading
import time
from collections import deque
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def _patch_torch_load_weights_only_default() -> None:
    """PyTorch 2.6 changed torch.load(weights_only=...) default behavior.

    Some older Ultralytics weights (and older ultralytics versions) expect
    torch.load(..., weights_only=False). When the default is weights_only=True,
    loading can fail with errors like:
      Weights only load failed ... Unsupported global: GLOBAL torch.nn.modules.container.Sequential

    This is a targeted compatibility shim for loading local model weights.
    """

    try:
        import inspect

        sig = inspect.signature(torch.load)
        if "weights_only" not in sig.parameters:
            return
    except Exception:
        # If we can't introspect, don't patch.
        return

    original_load = torch.load

    def _patched_load(*args, **kwargs):  # type: ignore
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = _patched_load  # type: ignore

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
DEFAULT_TRACK_TTL = 60.0
EMBED_INTERVAL_SEC = 1.0
DEFAULT_EXPORT_INTERVAL_SEC = 2.0
DEFAULT_MAX_TRACK_POINTS = 5000
DEFAULT_MAX_VECTORS = 300
DEFAULT_EVENTS_LOG_FLUSH_SEC = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- MODELS ----------------
_BOOTSTRAP_YOLO_MODEL: Optional[YOLO] = None

STOP_EVENT = threading.Event()

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


def _has_gui_display() -> bool:
    """Best-effort check whether a GUI display is available.

    Under systemd services there is usually no DISPLAY/WAYLAND_DISPLAY, and
    calling cv2.imshow() can hang or error.
    """

    # Windows/macOS generally have a GUI session.
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True

    # Linux/Unix: require an actual display environment.
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _should_show_preview(tracking_cfg: Dict[str, Any]) -> bool:
    # Allow explicit override via environment.
    env = os.environ.get("P2BP_TRACKER_PREVIEW")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")

    # Config flag (default False for systemd safety).
    return bool(tracking_cfg.get("Preview", False))


def _configure_ultralytics_dirs() -> None:
    """Configure Ultralytics settings/cache to be service-friendly.

    Ultralytics downloads model weights to a cache directory (not the working
    directory). Under systemd, HOME/XDG paths can be non-writable or unexpected.
    We prefer to keep caches inside the repo root so they are permitted by
    ReadWritePaths.
    """

    try:
        os.makedirs(MODELS_YOLO_DIR, exist_ok=True)
    except Exception:
        # If we can't create the directory, Ultralytics will fall back to its
        # default cache locations.
        return

    # Steer caches/config into the service working directory if not already set.
    # (Do not override if the unit file sets these explicitly.)
    os.environ.setdefault("HOME", BASE_DIR)
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(BASE_DIR, ".cache"))
    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(BASE_DIR, ".config", "Ultralytics"))

    # Best-effort: update Ultralytics internal settings if available.
    try:
        from ultralytics.utils import SETTINGS  # type: ignore

        try:
            SETTINGS["weights_dir"] = MODELS_YOLO_DIR
        except Exception:
            pass
    except Exception:
        pass


def _resolve_yolo_weights(model_setting: str) -> str:
    # Allows either friendly names (from config.json) or direct paths.
    mapping = {
        "yolov10n": "yolov10n.pt",
        "yolov10m": "yolov10m.pt",
        "yolov8n": "yolov8n.pt",
    }

    raw = str(model_setting).strip()
    key = raw.strip().lower()
    candidate = mapping.get(key, raw)

    # Normalize common cases like "yolov8n" -> "yolov8n.pt".
    # If the user already provided an extension (e.g., .pt), keep it.
    if isinstance(candidate, str):
        cand = candidate.strip()
        cand_lower = cand.lower()
        if cand_lower.startswith("yolov") and "." not in os.path.basename(cand) and not cand_lower.endswith(".pt"):
            candidate = f"{cand}.pt"

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


def _build_yolo_model(model_setting: str) -> YOLO:
    """Create a YOLO model, downloading weights if needed."""

    weights_path = _resolve_yolo_weights(model_setting)
    if os.path.exists(weights_path):
        return YOLO(weights_path)

    # If missing, attempt to let Ultralytics auto-download into our weights_dir.
    print(f"[WARN] YOLO weights not found at {weights_path}; attempting auto-download via Ultralytics")
    _configure_ultralytics_dirs()

    # Use a stem/filename for Ultralytics auto-download (it does not download to
    # arbitrary non-existent paths).
    mapping = {
        "yolov10n": "yolov10n.pt",
        "yolov10m": "yolov10m.pt",
        "yolov8n": "yolov8n.pt",
    }
    raw = str(model_setting).strip()
    candidate = mapping.get(raw.lower(), raw)
    candidate = os.path.basename(str(candidate))
    if candidate.lower().startswith("yolov") and "." not in candidate and not candidate.lower().endswith(".pt"):
        candidate = f"{candidate}.pt"
    return YOLO(candidate)


def _yolo_weights_spec_for_threads(model_setting: str) -> str:
    """Return a weights spec that each camera thread can pass to YOLO(...).

    Prefer an absolute path under models/yolo when present; otherwise return a
    model filename (e.g. "yolov8n.pt") which Ultralytics can resolve from its
    cache.
    """

    weights_path = _resolve_yolo_weights(model_setting)
    if os.path.exists(weights_path):
        return weights_path

    mapping = {
        "yolov10n": "yolov10n.pt",
        "yolov10m": "yolov10m.pt",
        "yolov8n": "yolov8n.pt",
    }
    raw = str(model_setting).strip()
    candidate = mapping.get(raw.lower(), raw)
    candidate = os.path.basename(str(candidate))
    if candidate.lower().startswith("yolov") and "." not in candidate and not candidate.lower().endswith(".pt"):
        candidate = f"{candidate}.pt"
    return candidate


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


def _write_json_atomic(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(path) + ".", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        tmp_path = ""
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


class _EventLogger:
    def __init__(self, path: str, flush_interval_sec: float = DEFAULT_EVENTS_LOG_FLUSH_SEC) -> None:
        self.path = path
        self.flush_interval_sec = max(0.0, float(flush_interval_sec))
        self._q: "Queue[Optional[Dict[str, Any]]]" = Queue(maxsize=20000)
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        try:
            self._q.put(None, timeout=1.0)
        except Exception:
            pass
        try:
            self._thread.join(timeout=timeout)
        except Exception:
            pass

    def log(self, event: Dict[str, Any]) -> None:
        # Requirement: save ALL data collected. If disk can't keep up, we block.
        self._q.put(event)

    def _run(self) -> None:
        last_flush = time.time()
        # Line-buffered for timely writes; still explicitly flush periodically.
        with open(self.path, "a", encoding="utf-8", buffering=1) as f:
            while True:
                item = self._q.get()
                if item is None:
                    break

                try:
                    f.write(json.dumps(item, separators=(",", ":")) + "\n")
                except Exception:
                    # Best-effort: if a single write fails, continue.
                    pass

                if self.flush_interval_sec > 0:
                    now = time.time()
                    if now - last_flush >= self.flush_interval_sec:
                        try:
                            f.flush()
                        except Exception:
                            pass
                        last_flush = now


def _export_tracks_by_camera(cams: List["CameraThread"], track_ttl_seconds: float) -> Dict[str, Dict[str, Any]]:
    export: Dict[str, Dict[str, Any]] = {}
    cutoff = time.time() - float(track_ttl_seconds)
    for c in cams:
        export[c.mac] = {}
        for sid, data in c.output_tracks.items():
            track = data.get("track", [])
            if not track:
                continue

            # Best-effort TTL cleanup before export. (track times are ms)
            last_t = float(track[-1].get("time", 0)) / 1000.0
            if last_t < cutoff:
                continue

            export[c.mac][str(sid)] = {
                "track": list(track) if isinstance(track, deque) else track,
                "vectors": list(data.get("vectors", [])) if isinstance(data.get("vectors", []), deque) else data.get("vectors", []),
            }
    return export


def _start_exporter_thread(
    cams: List["CameraThread"],
    output_path: str,
    track_ttl_seconds: float,
    export_interval_seconds: float,
    stop_event: threading.Event,
) -> threading.Thread:
    def _run() -> None:
        # Give camera threads a moment to warm up before the first write.
        next_write = time.time() + max(0.1, float(export_interval_seconds))
        while not stop_event.is_set():
            now = time.time()
            if now >= next_write:
                try:
                    export = _export_tracks_by_camera(cams, track_ttl_seconds)
                    _write_json_atomic(output_path, export)
                except Exception:
                    # Best-effort: avoid crashing the tracker due to exporter issues.
                    pass
                next_write = now + max(0.1, float(export_interval_seconds))
            time.sleep(0.1)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


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
        yolo_weights: str,
        conf_thresh: float,
        imgsz: int,
        max_fps: int,
        min_box: int,
        undistorter: Undistorter,
        output_retention_sec: float,
        max_track_points: int,
        max_vectors: int,
        event_logger: Optional[_EventLogger],
        stop_event: threading.Event,
    ):
        super().__init__(daemon=False)
        self.mac = mac
        self.cam = cam
        self.stop_event = stop_event
        self.display_name = getattr(cam, "name", None) or getattr(cam, "ip", None) or mac
        self.rtsp = str(getattr(cam, "rtsp"))
        self.cap = cv2.VideoCapture(self.rtsp, cv2.CAP_FFMPEG)
        try:
            # Reduce latency/backlog when possible.
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except Exception:
            pass
        self.frame_queue = Queue(maxsize=5)
        self.frame_id = 0
        self.frame = None
        self.bt_to_sid: Dict[int, int] = {}
        self.next_sid = 0
        self.tracks_local: Dict[int, Dict[str, Any]] = {}
        self.output_tracks: Dict[int, Dict[str, Any]] = {}

        self.output_retention_sec = float(output_retention_sec)
        self.max_track_points = max(1, int(max_track_points))
        self.max_vectors = max(1, int(max_vectors))
        self.event_logger = event_logger

        # Per-camera YOLO instance so ByteTrack state does not bleed across cameras.
        self.yolo = YOLO(yolo_weights)

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

        self._reader_thread = threading.Thread(target=self.read_frames, daemon=True)
        self._reader_thread.start()

    def read_frames(self):
        consecutive_failures = 0
        while not self.stop_event.is_set():
            try:
                ret, frame = self.cap.read()
            except Exception:
                ret, frame = False, None

            if ret and frame is not None:
                consecutive_failures = 0
                try:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)
                except Exception:
                    pass
                continue

            consecutive_failures += 1
            # Backoff a bit on read failures.
            time.sleep(0.05)

            # If the stream stalls, try reopening.
            if consecutive_failures >= 100 and not self.stop_event.is_set():
                consecutive_failures = 0
                try:
                    self.cap.release()
                except Exception:
                    pass
                try:
                    self.cap = cv2.VideoCapture(self.rtsp, cv2.CAP_FFMPEG)
                    try:
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                    except Exception:
                        pass
                except Exception:
                    pass

    def _get_sid(self, bt_id):
        if bt_id in self.bt_to_sid:
            return self.bt_to_sid[bt_id]
        # Do not reuse SIDs: keeps history unambiguous for append-only logs.
        sid = self.next_sid
        self.next_sid += 1
        self.bt_to_sid[bt_id] = sid
        return sid

    def run(self):
        last_process = 0.0
        last_prune = 0.0
        while not self.stop_event.is_set():
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

            results = self.yolo.track(
                frame,
                conf=self.conf_thresh,
                imgsz=self.imgsz,
                classes=[0],
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )

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
                        "track": deque(maxlen=self.max_track_points),
                        "vectors": deque(maxlen=self.max_vectors),
                        "mac": self.mac,
                    }
                out["track"].append({"time": int(now * 1000), "x": gx, "y": gy})
                out["last_ms"] = int(now * 1000)

                if self.event_logger is not None:
                    try:
                        self.event_logger.log({
                            "type": "track",
                            "mac": self.mac,
                            "sid": sid,
                            "time": int(now * 1000),
                            "x": gx,
                            "y": gy,
                        })
                    except Exception:
                        pass

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

                    if self.event_logger is not None:
                        try:
                            self.event_logger.log({
                                "type": "vector",
                                "mac": self.mac,
                                "sid": sid,
                                "time": int(now * 1000),
                                "vector": [float(x) for x in feat],
                            })
                        except Exception:
                            pass

                # Visualize local SID only (no cross-camera fusion here).
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"SID {sid}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            self.frame = frame

            # Periodically prune stale tracks to bound memory.
            if now - last_prune >= 1.0:
                last_prune = now
                if self.output_retention_sec > 0:
                    cutoff = now - self.output_retention_sec

                    # Remove stale bytetrack ids and free their SIDs.
                    stale_bt_ids: List[int] = []
                    for bt_id2, t2 in self.tracks_local.items():
                        try:
                            if (now - float(t2.get("last", 0.0))) > self.output_retention_sec:
                                stale_bt_ids.append(bt_id2)
                        except Exception:
                            stale_bt_ids.append(bt_id2)

                    for bt_id2 in stale_bt_ids:
                        t2 = self.tracks_local.pop(bt_id2, None)
                        sid2 = None
                        if isinstance(t2, dict):
                            sid2 = t2.get("sid")
                        if bt_id2 in self.bt_to_sid:
                            sid2 = self.bt_to_sid.pop(bt_id2, sid2)
                        if isinstance(sid2, int):
                            self.output_tracks.pop(sid2, None)

        try:
            self.cap.release()
        except Exception:
            pass

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

    _patch_torch_load_weights_only_default()

    # Bootstrap once to (a) validate weights load and (b) trigger an auto-download
    # if needed. Each camera thread will still use its own YOLO instance.
    global _BOOTSTRAP_YOLO_MODEL
    _BOOTSTRAP_YOLO_MODEL = _build_yolo_model(model_setting)
    yolo_weights_spec = _yolo_weights_spec_for_threads(model_setting)

    show_preview = _should_show_preview(tracking_cfg)
    if show_preview and not _has_gui_display():
        print("[WARN] Tracking preview requested but no GUI display detected; disabling preview.")
        show_preview = False

    tracking_cams = cfg.get("TrackingCameras", {}) if isinstance(cfg.get("TrackingCameras"), dict) else {}
    enabled_macs = [mac for mac, on in tracking_cams.items() if bool(on)]
    if not enabled_macs:
        print("[WARN] No cameras enabled in TrackingCameras; exiting.")
        return

    cams: List[CameraThread] = []

    export_path = os.path.join(BASE_DIR, "tracks_by_camera.json")
    export_interval = _safe_float(tracking_cfg.get("ExportIntervalSeconds"), DEFAULT_EXPORT_INTERVAL_SEC)
    export_ttl = _safe_float(tracking_cfg.get("ExportTrackTtlSeconds"), DEFAULT_TRACK_TTL)

    # Append-only full history log (JSONL). This is the durable source of truth.
    tracks_dir = os.path.join(BASE_DIR, "tracks")
    os.makedirs(tracks_dir, exist_ok=True)
    session_stamp = time.strftime("%Y%m%d-%H%M%S")
    events_path = os.path.join(tracks_dir, f"tracks_events-{session_stamp}.jsonl")
    events_flush = _safe_float(tracking_cfg.get("EventsLogFlushSeconds"), DEFAULT_EVENTS_LOG_FLUSH_SEC)
    event_logger = _EventLogger(events_path, flush_interval_sec=events_flush)
    event_logger.start()
    try:
        event_logger.log({"type": "session_start", "time": int(time.time() * 1000)})
    except Exception:
        pass

    # Bound in-memory growth.
    max_track_points_default = int(max(1.0, export_ttl) * max(1, max_fps)) + 100
    max_vectors_default = int(max(1.0, export_ttl) / max(EMBED_INTERVAL_SEC, 0.1)) + 10
    max_track_points = _safe_int(tracking_cfg.get("MaxTrackPoints"), min(DEFAULT_MAX_TRACK_POINTS, max_track_points_default))
    max_vectors = _safe_int(tracking_cfg.get("MaxVectors"), min(DEFAULT_MAX_VECTORS, max_vectors_default))
    exporter_thread: Optional[threading.Thread] = None

    def _handle_stop(signum, frame):  # type: ignore
        STOP_EVENT.set()

    try:
        signal.signal(signal.SIGINT, _handle_stop)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _handle_stop)
    except Exception:
        pass

    try:
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
                yolo_weights=yolo_weights_spec,
                conf_thresh=conf_thresh,
                imgsz=DEFAULT_IMG_SIZE,
                max_fps=max_fps,
                min_box=DEFAULT_MIN_BOX,
                undistorter=undistorter,
                output_retention_sec=export_ttl,
                max_track_points=max_track_points,
                max_vectors=max_vectors,
                event_logger=event_logger,
                stop_event=STOP_EVENT,
            )
            cams.append(c)
            c.start()

        if not cams:
            print("[WARN] No camera threads started; exiting.")
            return

        exporter_thread = _start_exporter_thread(
            cams=cams,
            output_path=export_path,
            track_ttl_seconds=export_ttl,
            export_interval_seconds=export_interval,
            stop_event=STOP_EVENT,
        )

        while not STOP_EVENT.is_set():
            if show_preview:
                for c in cams:
                    if c.frame is not None:
                        cv2.imshow(f"{c.display_name}", cv2.resize(c.frame, (640, 360)))
                if cv2.waitKey(1) == 27:
                    break
            else:
                time.sleep(0.25)

    except Exception as e:
        STOP_EVENT.set()
        print(f"[FATAL] tracker main loop failed: {e}")
        return

    finally:
        STOP_EVENT.set()
        for c in cams:
            try:
                c.join(timeout=2.0)
            except Exception:
                pass
            try:
                c.cap.release()
            except Exception:
                pass
        if show_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        # Stop event logger after camera threads stop producing events.
        try:
            event_logger.log({"type": "session_stop", "time": int(time.time() * 1000)})
        except Exception:
            pass
        event_logger.stop(timeout=5.0)

    # Final export on shutdown (best-effort). Exporter thread may already have written.
    try:
        export = _export_tracks_by_camera(cams, export_ttl)
        _write_json_atomic(export_path, export)
    except Exception:
        pass

    print("[INFO] Saved tracks_by_camera.json")

if __name__ == "__main__":
    main()
