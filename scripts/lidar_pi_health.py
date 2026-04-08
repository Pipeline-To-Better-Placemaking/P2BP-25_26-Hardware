"""Minimal lidar + Pi companion probes for device heartbeat (no scanner driver).

Optional JSON overlay: P2BP_LIDAR_STATE_PATH (default /opt/p2bp/camera/run/lidar_state.json).

Env:
  P2BP_LIDAR_SENSOR_ID   — key in Lidars map (default: primary)
  P2BP_LIDAR_DEVICE_GLOB — glob for serial devices (default: /dev/ttyUSB*)
  P2BP_LIDAR_STATE_PATH  — optional state file path
  P2BP_PI_HOST           — if set, ping this host once per heartbeat (Linux ping)
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import scripts.json_models.heartbeat_payload as hp

logger = logging.getLogger("p2bp.heartbeat")

_DEFAULT_STATE = "/opt/p2bp/camera/run/lidar_state.json"


def _state_path() -> Path:
    return Path(os.getenv("P2BP_LIDAR_STATE_PATH", _DEFAULT_STATE))


def _sensor_id() -> str:
    return (os.getenv("P2BP_LIDAR_SENSOR_ID") or "primary").strip() or "primary"


def _device_glob() -> str:
    g = (os.getenv("P2BP_LIDAR_DEVICE_GLOB") or "/dev/ttyUSB*").strip()
    return g or "/dev/ttyUSB*"


def _load_state_overlay() -> Optional[Dict[str, Any]]:
    try:
        p = _state_path()
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
    except Exception as e:
        logger.debug("lidar state file read failed: %s", e)
    return None


def _probe_serial_devices() -> tuple[bool, str]:
    paths = sorted(p for p in glob.glob(_device_glob()) if os.path.exists(p))
    if not paths:
        return False, ""
    return True, paths[0]


def _lsusb_hint() -> str:
    try:
        r = subprocess.run(
            ["lsusb"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if r.returncode != 0 or not r.stdout:
            return ""
        for line in r.stdout.splitlines():
            lower = line.lower()
            if any(x in lower for x in ("cp210", "ch340", "ch341", "0403:6001", "rplidar")):
                return line.strip()[:200]
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return ""


def collect_lidar_health() -> Dict[str, hp.LidarHealth]:
    connected, device_path = _probe_serial_devices()
    usb_hint = _lsusb_hint()
    last_error = "" if connected else "no_matching_serial_device"
    sensor_id = _sensor_id()

    overlay = _load_state_overlay()
    if overlay:
        sensor_id = str(overlay.get("SensorId") or overlay.get("sensorId") or sensor_id)
        if "Connected" in overlay or "connected" in overlay:
            connected = bool(overlay.get("Connected", overlay.get("connected")))
        dp = overlay.get("DevicePath") or overlay.get("devicePath")
        if dp:
            device_path = str(dp)
        uh = overlay.get("UsbHint") or overlay.get("usbHint")
        if uh:
            usb_hint = str(uh)[:200]
        if "LastError" in overlay or "lastError" in overlay:
            le = overlay.get("LastError", overlay.get("lastError"))
            last_error = "" if le is None else str(le)[:500]

    return {
        sensor_id: hp.LidarHealth(
            SensorId=sensor_id,
            Connected=connected,
            DevicePath=device_path,
            UsbHint=usb_hint,
            LastError=last_error,
        )
    }


def collect_pi_companion_health() -> hp.PiCompanionHealth:
    host = (os.getenv("P2BP_PI_HOST") or "").strip()
    if not host:
        return hp.PiCompanionHealth()

    try:
        r = subprocess.run(
            ["ping", "-c", "1", "-W", "1", host],
            capture_output=True,
            text=True,
            timeout=4,
        )
        ok = r.returncode == 0
        latency_ms = -1
        if ok and r.stdout:
            m = re.search(r"time[=<](\d+\.?\d*)", r.stdout)
            if m:
                try:
                    latency_ms = int(float(m.group(1)))
                except ValueError:
                    latency_ms = -1
        return hp.PiCompanionHealth(
            Configured=True,
            Host=host,
            Reachable=ok,
            LatencyMs=latency_ms,
            LastError="" if ok else "ping_failed",
        )
    except Exception as e:
        return hp.PiCompanionHealth(
            Configured=True,
            Host=host,
            Reachable=False,
            LatencyMs=-1,
            LastError=str(e)[:200],
        )
