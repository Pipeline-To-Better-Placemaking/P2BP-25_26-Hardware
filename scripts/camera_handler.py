import json
import os
import time
from typing import Dict, List, Optional

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

    def to_state_dict(self) -> dict:
        # In PascalCase to match the .NET backend
        return {
            "Mac": self.mac,
            "Ip": self.ip,
            "Resolution": self.resolution,
            "Enabled": self.enabled,
        }

# camera controller class (singleton)
class CameraController:
    _instance: Optional["CameraController"] = None

    def __init__(self, runtime_path: str = RUNTIME_PATH):
        self.runtime_path = runtime_path
        self.cameras: Dict[str, Camera] = {}
        self._last_refresh: float = 0
        self._load_runtime()

    @classmethod
    def get_instance(cls, runtime_path: str = RUNTIME_PATH) -> "CameraController":
        """Returns singleton instance, creating it if needed."""
        if cls._instance is None:
            cls._instance = cls(runtime_path)
        return cls._instance

    def _load_runtime(self) -> None:
        """Load cameras_runtime.json into Camera objects."""
        if not os.path.exists(self.runtime_path):
            return

        try:
            with open(self.runtime_path, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        if not isinstance(data, dict):
            return

        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    config_data = json.load(f)
                cameras_config = config_data.get("TrackingCameras", {})
                for mac, enabled in cameras_config.items():
                    if mac in data and isinstance(enabled, bool):
                        data[mac]["enabled"] = enabled
            except (OSError, json.JSONDecodeError):
                pass

        self.cameras.clear()
        for mac, info in data.items():
            if isinstance(info, dict):
                self.cameras[mac] = Camera(mac, info)

        self._last_refresh = time.time()

    def _maybe_refresh(self) -> None:
        """Refresh from disk if REFRESH_INTERVAL has passed."""
        if time.time() - self._last_refresh > REFRESH_INTERVAL:
            self._load_runtime()

    def get_camera(self, mac: str) -> Optional[Camera]:
        self._maybe_refresh()
        return self.cameras.get(mac)

    def get_camera_states(self) -> Dict[str, dict]:
        """Returns camera states for heartbeat payload."""
        self._maybe_refresh()
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
        
