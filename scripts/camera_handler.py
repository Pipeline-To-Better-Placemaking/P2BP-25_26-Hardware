import json
import os
import re
import tempfile
import threading
import time
from typing import Dict, List, Optional

import yaml

# cameras_runtime is at config/cameras_runtime.json
RUNTIME_PATH = "config/cameras_runtime.json"
CONFIG_PATH = "config/config.json"
REFRESH_INTERVAL = 30  # seconds between automatic file re-reads


# camera class
class Camera:
    def __init__(self, mac: str, camera_info: dict):
        self.mac = mac
        self.ip: str = camera_info.get('ip', "")
        self.resolution: List[int] = camera_info.get('resolution', [0, 0])
        self.rtsp: str = camera_info.get('rtsp', "")
        self.enabled: bool = camera_info.get('enabled', False)
        self.homography: Optional[List[List[float]]] = camera_info.get('homography', None)
        self.camera_matrix: Optional[List[List[float]]] = camera_info.get('camera_matrix', None)
        self.distortion_coefficients: Optional[List[float]] = camera_info.get('distortion_coefficients', None)
        self.reprojection_error: Optional[float] = camera_info.get('reprojection_error', None)

    def to_state_dict(self) -> dict:
        # In PascalCase to match the .NET backend
        return {
            "Mac": self.mac,
            "Ip": self.ip,
            "Resolution": self.resolution,
            "Enabled": self.enabled,
            "Homography": self.homography,
            "CameraMatrix": self.camera_matrix,
            "DistortionCoefficients": self.distortion_coefficients,
            "ReprojectionError": self.reprojection_error,
        }

# camera controller class (singleton)
class CameraController:
    _instance: Optional["CameraController"] = None

    def __init__(self, runtime_path: str = RUNTIME_PATH):
        self.runtime_path = runtime_path
        self.cameras: Dict[str, Camera] = {}
        self._last_refresh: float = 0
        self._refresh_interval_s: float = float(REFRESH_INTERVAL)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop,
            name="CameraControllerRefresh",
            daemon=True,
        )
        self._load_runtime()
        self._refresh_thread.start()

    def _refresh_loop(self) -> None:
        """Background loop to periodically refresh runtime using config.HeartbeatInterval."""

        while not self._stop_event.is_set():
            with self._lock:
                interval_s = self._refresh_interval_s if self._refresh_interval_s > 0 else float(REFRESH_INTERVAL)

            # Wait for the interval (or stop).
            if self._stop_event.wait(timeout=interval_s):
                break

            try:
                self.refresh_runtime()
            except Exception:
                # Never let the background loop crash the process.
                pass

    def close(self) -> None:
        """Stop the background refresh thread (best-effort)."""

        self._stop_event.set()
        try:
            if self._refresh_thread.is_alive():
                self._refresh_thread.join(timeout=2.0)
        except Exception:
            pass

    @classmethod
    def get_instance(cls, runtime_path: str = RUNTIME_PATH) -> "CameraController":
        """Returns singleton instance, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls(runtime_path)
        return cls._instance

    def _load_runtime(self) -> None:
        """Load cameras_runtime.json into Camera objects."""
        runtime_path = os.path.abspath(self.runtime_path)
        if not os.path.exists(runtime_path):
            return

        try:
            with open(runtime_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        if not isinstance(data, dict):
            return

        config_dir = os.path.dirname(runtime_path)
        base_dir = os.path.abspath(os.path.join(config_dir, os.pardir))
        homographies_dir = os.path.join(base_dir, "homographies")

        config_path = CONFIG_PATH
        if not os.path.isabs(config_path) and not os.path.exists(config_path):
            # Prefer config.json next to cameras_runtime.json if CWD differs.
            config_path = os.path.join(config_dir, os.path.basename(CONFIG_PATH))

        def _safe_filename(s: str) -> str:
            return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(s))

        def _parse_resolution(value) -> Optional[List[int]]:
            if value is None:
                return None
            if isinstance(value, (list, tuple)) and len(value) == 2:
                try:
                    return [int(value[0]), int(value[1])]
                except Exception:
                    return None
            if isinstance(value, str):
                s = value.strip().lower().replace(" ", "")
                m = re.search(r"(\d{3,5})\D+(\d{3,5})", s)
                if not m:
                    return None
                try:
                    return [int(m.group(1)), int(m.group(2))]
                except Exception:
                    return None
            return None

        desired_res: Optional[List[int]] = None
        tracking_cameras: Dict[str, bool] = {}
        config_data: Optional[dict] = None

        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)

                # Drive refresh cadence from config.HeartbeatInterval.
                # (Falls back to REFRESH_INTERVAL if missing/invalid.)
                try:
                    hb = config_data.get("HeartbeatInterval", REFRESH_INTERVAL)
                    hb_s = float(hb)
                    if hb_s > 0:
                        with self._lock:
                            self._refresh_interval_s = hb_s
                except Exception:
                    with self._lock:
                        self._refresh_interval_s = float(REFRESH_INTERVAL)

                if isinstance(config_data, dict):
                    cam_cfg = config_data.get("Camera", {})
                    if isinstance(cam_cfg, dict):
                        desired_res = _parse_resolution(cam_cfg.get("Resolution"))

                cameras_config = config_data.get("TrackingCameras", {})
                if isinstance(cameras_config, dict) and cameras_config is not None:
                    # Don't persist this into cameras_runtime.json; treat as derived state.
                    tracking_cameras = {
                        str(mac): bool(enabled)
                        for mac, enabled in cameras_config.items()
                        if isinstance(enabled, bool)
                    }

                if isinstance(data, dict):
                    self.update_camera_setting(data, config_data)
            except (OSError, json.JSONDecodeError):
                pass

        def _scale_homography(H, src_w: int, src_h: int, dst_w: int, dst_h: int):
            """Scale a pixel->world homography from src resolution to dst resolution."""
            import numpy as np

            if src_w == dst_w and src_h == dst_h:
                return H
            sx = float(src_w) / float(dst_w)
            sy = float(src_h) / float(dst_h)
            S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
            H2 = H @ S
            try:
                if abs(float(H2[2, 2])) > 1e-12:
                    H2 = H2 / float(H2[2, 2])
            except Exception:
                pass
            return H2

        def _scale_intrinsics(K, src_w: int, src_h: int, dst_w: int, dst_h: int):
            """Scale camera intrinsics K from src resolution to dst resolution."""
            import numpy as np

            if src_w == dst_w and src_h == dst_h:
                return K
            rx = float(dst_w) / float(src_w)
            ry = float(dst_h) / float(src_h)
            K2 = np.array(K, dtype=float).copy()
            K2[0, 0] *= rx
            K2[0, 2] *= rx
            K2[1, 1] *= ry
            K2[1, 2] *= ry
            return K2

        # Cache intrinsics per camera type.
        intrinsics_cache: Dict[str, dict] = {}

        with self._lock:
            self.cameras.clear()
            for mac, info in data.items():
                if isinstance(info, dict):
                    # Inject enabled into the in-memory Camera objects only.
                    camera_info = dict(info)
                    if mac in tracking_cameras:
                        camera_info["enabled"] = tracking_cameras[mac]

                    # Load/scale homography for this camera.
                    try:
                        homography_path = os.path.join(homographies_dir, f"{_safe_filename(mac)}_homography.yml")
                        if os.path.exists(homography_path):
                            import cv2

                            fs = cv2.FileStorage(homography_path, cv2.FILE_STORAGE_READ)
                            try:
                                H = fs.getNode("homography").mat()
                                frame_size = fs.getNode("frame_size").mat()
                            finally:
                                fs.release()

                            if H is not None and frame_size is not None and len(frame_size.reshape(-1)) >= 2:
                                src_w = int(frame_size.reshape(-1)[0])
                                src_h = int(frame_size.reshape(-1)[1])
                                if desired_res and len(desired_res) == 2:
                                    dst_w, dst_h = int(desired_res[0]), int(desired_res[1])
                                    Hs = _scale_homography(H, src_w, src_h, dst_w, dst_h)
                                else:
                                    Hs = H

                                camera_info["homography"] = Hs.tolist()
                    except Exception:
                        # Missing/malformed homography should not break camera loading.
                        pass

                    # Load/scale intrinsics for this camera type.
                    try:
                        from scripts.camera_onboard import _detect_camera_type

                        cam_type = _detect_camera_type(mac)
                        if cam_type:
                            intr = intrinsics_cache.get(cam_type)
                            if intr is None:
                                intr_path = os.path.join(config_dir, f"{cam_type}_camera_intrinsics.yml")
                                if os.path.exists(intr_path):
                                    with open(intr_path, "r", encoding="utf-8") as f:
                                        intr = yaml.safe_load(f) or {}
                                else:
                                    intr = {}
                                intrinsics_cache[cam_type] = intr

                            if isinstance(intr, dict) and intr:
                                K = intr.get("camera_matrix")
                                d = intr.get("distortion_coefficients")
                                w0 = intr.get("image_width")
                                h0 = intr.get("image_height")
                                reproj = intr.get("reprojection_error")

                                if K is not None:
                                    import numpy as np

                                    K_np = np.array(K, dtype=float)
                                    if K_np.shape == (3, 3) and w0 and h0 and desired_res and len(desired_res) == 2:
                                        K_np = _scale_intrinsics(
                                            K_np,
                                            int(w0),
                                            int(h0),
                                            int(desired_res[0]),
                                            int(desired_res[1]),
                                        )
                                    camera_info["camera_matrix"] = K_np.tolist()

                                if d is not None:
                                    # Stored as [[k1,k2,p1,p2,k3]] in your YAML; normalize to a flat list.
                                    if isinstance(d, list) and len(d) == 1 and isinstance(d[0], list):
                                        camera_info["distortion_coefficients"] = [float(x) for x in d[0]]
                                    elif isinstance(d, list):
                                        camera_info["distortion_coefficients"] = [float(x) for x in d]

                                if reproj is not None:
                                    try:
                                        camera_info["reprojection_error"] = float(reproj)
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                    self.cameras[mac] = Camera(mac, camera_info)

        with self._lock:
            self._last_refresh = time.time()

    def update_camera_setting(self, cameras_runtime: dict, config: dict) -> bool:
        def _parse_resolution(value) -> Optional[List[int]]:
            if value is None:
                return None
            if isinstance(value, (list, tuple)) and len(value) == 2:
                try:
                    return [int(value[0]), int(value[1])]
                except Exception:
                    return None
            if isinstance(value, str):
                s = value.strip().lower().replace(" ", "")
                # Accept forms like "1920x1080", "1920*1080", "1920x1080p".
                m = re.search(r"(\d{3,5})\D+(\d{3,5})", s)
                if not m:
                    return None
                try:
                    return [int(m.group(1)), int(m.group(2))]
                except Exception:
                    return None
            return None

        def _normalize_live_resolution(value) -> List[int]:
            parsed = _parse_resolution(value)
            return parsed if parsed is not None else [0, 0]

        setting_changed = False

        camera_settings = config.get("Camera") if isinstance(config, dict) else {}
        if not isinstance(camera_settings, dict):
            camera_settings = {}

        desired_res = _parse_resolution(camera_settings.get("Resolution")) or [0, 0]

        tracking_cams = config.get("TrackingCameras", {}) if isinstance(config, dict) else {}
        if not isinstance(tracking_cams, dict):
            tracking_cams = {}

        for mac, cam_info in cameras_runtime.items():
            if not isinstance(cam_info, dict):
                continue

            # Enabled is controlled by config.json (TrackingCameras), not by cameras_runtime.json.
            enabled = tracking_cams.get(mac, cam_info.get("enabled", False))
            if not isinstance(enabled, bool) or not enabled:
                continue

            live_res = _normalize_live_resolution(cam_info.get("resolution"))

            # Maybe add fps here later
            if desired_res != [0, 0] and live_res != desired_res:
                from scripts.camera_onboard import _detect_camera_type

                cam_type = _detect_camera_type(mac)
                if cam_type == "ANNKE":
                    from scripts.camera_controllers import annke_controller

                    ip = cam_info.get("ip", "")
                    if not ip:
                        print(f"[CAMERA_CONTROLLER] Missing IP for {mac}; cannot update resolution")
                        continue

                    try:
                        ok = annke_controller.set_video_resolution(
                            ip,
                            f"{desired_res[0]}x{desired_res[1]}",
                        )
                        if ok:
                            print(f"[CAMERA_CONTROLLER] Updated resolution for {mac} to {desired_res}")
                            cam_info["resolution"] = desired_res
                            setting_changed = True
                        else:
                            print(f"[CAMERA_CONTROLLER] Failed to update resolution for {mac}")
                    except Exception as e:
                        print(f"[CAMERA_CONTROLLER] Error updating resolution for {mac}: {e}")

        if setting_changed:
            # Persist updated runtime state so we don't re-apply every refresh.
            try:
                # Keep runtime schema stable: strip derived keys like "enabled".
                sanitized = {}
                for mac, info in cameras_runtime.items():
                    if isinstance(info, dict):
                        sanitized[mac] = {k: v for k, v in info.items() if k != "enabled"}
                    else:
                        sanitized[mac] = info

                os.makedirs(os.path.dirname(self.runtime_path) or ".", exist_ok=True)

                # Atomic write: temp file in same directory + replace.
                tmp_path: Optional[str] = None
                try:
                    tmp_fd, tmp_path = tempfile.mkstemp(
                        prefix=os.path.basename(self.runtime_path) + ".",
                        dir=os.path.dirname(self.runtime_path) or ".",
                    )
                    with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                        json.dump(sanitized, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_path, self.runtime_path)
                    tmp_path = None
                except PermissionError:
                    # Fallback: write directly (non-atomic).
                    with open(self.runtime_path, "w", encoding="utf-8") as f:
                        json.dump(sanitized, f, indent=2)
                        f.flush()
                        os.fsync(f.fileno())
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
            except OSError as e:
                print(f"[CAMERA_CONTROLLER] Warning: failed to write {self.runtime_path}: {e}")

            tracking_cfg = config.get("Tracking", {}) if isinstance(config, dict) else {}
            if not isinstance(tracking_cfg, dict):
                tracking_cfg = {}
            tracking_enabled = bool(tracking_cfg.get("Enabled", tracking_cfg.get("enabled", False)))

            if tracking_enabled:
                import scripts.signals as signals

                # Restart tracking script to force it to use new settings.
                signals.set_signal("", "tracking", False)
                time.sleep(1)
                signals.set_signal("", "tracking", True)

        return setting_changed

    def _maybe_refresh(self) -> None:
        """Refresh from disk if config.HeartbeatInterval seconds have passed."""
        with self._lock:
            interval_s = self._refresh_interval_s if self._refresh_interval_s > 0 else float(REFRESH_INTERVAL)
            last = self._last_refresh

        if time.time() - last > interval_s:
            self._load_runtime()

    def get_camera(self, mac: str) -> Optional[Camera]:
        self._maybe_refresh()
        with self._lock:
            return self.cameras.get(mac)

    def get_camera_states(self) -> Dict[str, dict]:
        """Returns camera states for heartbeat payload."""
        self._maybe_refresh()
        with self._lock:
            return {mac: cam.to_state_dict() for mac, cam in self.cameras.items()}

    def refresh_runtime(self) -> None:
        """Force reload cameras_runtime.json from disk."""
        self._load_runtime()

# Add camera type identifier + call to update settings in <camera type>_controller.py based on config changes

# Module-level convenience functions
def get_camera_states() -> Dict[str, dict]:
    """Returns camera states using singleton instance."""
    return CameraController.get_instance().get_camera_states()


def get_camera(mac: str) -> Optional[Camera]:
    """Returns a camera by MAC address using singleton instance."""
    return CameraController.get_instance().get_camera(mac)
        
