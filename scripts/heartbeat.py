# This script puts together a health report and sends it to the server
# It expects a new config file in return and will update the local config file if there are any changes
# It will also send systemd signals to start/stop services based on the new config file

import os
import time
import json
import hashlib
import requests
import dotenv
import logging
from pathlib import Path
import scripts.json_models.heartbeat_payload as heartbeat_payload
import scripts.systemd_services as systemd_services
import scripts.system_stats as system_stats
import scripts.config_io as config_io
import scripts.signals as signals
import scripts.camera_handler as camera_handler
import scripts.lidar_pi_health as lidar_pi_health
from typing import Any, Dict, Optional


def _build_logger() -> logging.Logger:
    """Create a dedicated logger without configuring the root logger.

    Under systemd, stdout/stderr ends up in journald and can also be forwarded
    to rsyslog (/var/log/syslog). Logging large payloads every heartbeat can
    inflate syslog quickly, so we keep logs compact and rate-limited.
    """

    logger = logging.getLogger("p2bp.heartbeat")
    if logger.handlers:
        return logger

    level_name = os.getenv("P2BP_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)

    # Do not propagate to root to avoid other modules accidentally inheriting
    # a more verbose root configuration.
    logger.propagate = False
    return logger


logger = _build_logger()

CONFIG_PATH = "/opt/p2bp/camera/config/config.json"
SIGNAL_DIR = "/run/p2bp"
DEFAULT_HEARTBEAT_INTERVAL = 10  # seconds
MIN_HEARTBEAT_INTERVAL = 2  # seconds (prevents accidental busy loops)
FAST_HEARTBEAT_INTERVAL = 2.0   # seconds, used during active operations or after a config change
CONFIG_CHANGE_FAST_WINDOW_S = 15.0  # seconds to stay on fast interval after any config change

# Log a compact "heartbeat OK" line at most this often.
DEFAULT_LOG_EVERY_SECONDS = 300


def _normalize_endpoint(endpoint: str) -> str:
    e = (endpoint or "").strip()
    while e.endswith("/"):
        e = e[:-1]
    if e.lower().endswith("/api"):
        e = e[:-4]
        while e.endswith("/"):
            e = e[:-1]
    return e


def get_heartbeat_interval_seconds(config: Optional[Dict[str, Any]]) -> float:
    """Return heartbeat interval in seconds from config, with validation.

    Uses config["HeartbeatInterval"] when present; falls back to DEFAULT_HEARTBEAT_INTERVAL.
    """
    if not isinstance(config, dict):
        return float(DEFAULT_HEARTBEAT_INTERVAL)

    raw = config.get("HeartbeatInterval", DEFAULT_HEARTBEAT_INTERVAL)
    try:
        interval = float(raw)
    except (TypeError, ValueError):
        return float(DEFAULT_HEARTBEAT_INTERVAL)

    # Prevent busy loops / invalid values.
    if interval <= 0:
        return float(DEFAULT_HEARTBEAT_INTERVAL)

    if interval < MIN_HEARTBEAT_INTERVAL:
        return float(MIN_HEARTBEAT_INTERVAL)
    return interval


def _is_active_operation(config: Optional[Dict[str, Any]]) -> bool:
    """Return True when any operation flag that warrants a fast heartbeat is set."""
    if not isinstance(config, dict):
        return False
    charuco = config.get("CharucoBoard") or {}
    aruco = config.get("ArucoLock") or {}
    intrinsics = config.get("Intrinsics") or {}
    return (
        bool(charuco.get("BeginScanning"))
        or bool(aruco.get("BeginScanning"))
        or bool(intrinsics.get("BeginCalibration"))
    )


def sanitize_camera_state_for_heartbeat(state: Dict[str, Any]) -> Dict[str, Any]:
    """Only include fields that the backend/heartbeat model expects for now."""
    allowed = {"Mac", "Ip", "Resolution", "Enabled"}
    return {k: v for k, v in state.items() if k in allowed}


def _stable_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _short_hash(obj: Any) -> str:
    return hashlib.sha256(_stable_json_bytes(obj)).hexdigest()[:12]


def _payload_summary(payload: Dict[str, Any]) -> str:
    try:
        services = payload.get("Services") if isinstance(payload, dict) else None
        cameras = payload.get("Cameras") if isinstance(payload, dict) else None
        system = payload.get("System") if isinstance(payload, dict) else None

        svc_n = len(services) if isinstance(services, dict) else 0
        cam_n = len(cameras) if isinstance(cameras, dict) else 0
        lidars = payload.get("Lidars") if isinstance(payload, dict) else None
        lid_n = len(lidars) if isinstance(lidars, dict) else 0

        mem = system.get("Memory") if isinstance(system, dict) else None
        used_mb = mem.get("UsedMb") if isinstance(mem, dict) else None
        total_mb = mem.get("TotalMb") if isinstance(mem, dict) else None

        size_b = len(_stable_json_bytes(payload))
        mem_part = ""
        if isinstance(used_mb, int) and isinstance(total_mb, int) and total_mb > 0:
            mem_part = f" mem={used_mb}/{total_mb}MB"

        return f"services={svc_n} cameras={cam_n} lidars={lid_n} payload={size_b}B{mem_part}"
    except Exception:
        return "(summary unavailable)"

def load_env():
    #dotenv.load_dotenv("../config/agent.env") # for local testing
    dotenv.load_dotenv("/opt/p2bp/camera/config/agent.env")
    api_key = (os.getenv("API_KEY") or "").strip()
    endpoint = (os.getenv("ENDPOINT") or "").strip()

    if not api_key:
        raise RuntimeError("Missing API_KEY")
    elif not endpoint:
        raise RuntimeError("Missing ENDPOINT")

    endpoint_norm = _normalize_endpoint(endpoint)
    if not endpoint_norm:
        raise RuntimeError("ENDPOINT is invalid")
    return api_key, endpoint_norm

def send_heartbeat(api_key, endpoint, payload): # send a health report heartbeat the server. Expect a new config file in return
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    r = requests.post(f"{endpoint}/api/Device/heartbeat", headers=headers, json=payload, timeout=5)
    r.raise_for_status()

    try:
        return r.json()
    except ValueError:
        raise RuntimeError("Backend returned invalid JSON")

_DISK_STATE_PATH = "/opt/p2bp/camera/run/disk_state.json"
_INTRINSICS_STATE_PATH = "/opt/p2bp/camera/run/intrinsics_calibration_state.json"


def _load_disk_state() -> list:
    try:
        p = Path(_DISK_STATE_PATH)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _load_intrinsics_state() -> dict:
    try:
        p = Path(_INTRINSICS_STATE_PATH)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _system_stats_to_model(raw: Dict[str, Any]) -> heartbeat_payload.SystemStats:
    """Build SystemStats from get_system_stats() dict plus disk list."""
    g = raw.get("Gpu") or {}
    m = raw.get("Memory") or {}
    disk = raw.get("Disk")
    if not isinstance(disk, list):
        disk = []
    return heartbeat_payload.SystemStats(
        Gpu=heartbeat_payload.GpuStats(
            UtilizationPct=int(g.get("UtilizationPct", -1)),
            FrequencyMhz=int(g.get("FrequencyMhz", -1)),
        ),
        Memory=heartbeat_payload.MemoryStats(
            UsedMb=int(m.get("UsedMb", -1)),
            TotalMb=int(m.get("TotalMb", -1)),
        ),
        Disk=disk,
    )


def create_heartbeat_payload(): # create a payload for the heartbeat request
    services = systemd_services.get_all_service_states()
    raw_system = system_stats.get_system_stats()
    raw_system["Disk"] = _load_disk_state()
    system = _system_stats_to_model(raw_system)
    camera_dicts = camera_handler.get_camera_states()

    # Convert dicts to CameraState dataclasses
    cameras = {
        mac: heartbeat_payload.CameraState(**sanitize_camera_state_for_heartbeat(state))
        for mac, state in camera_dicts.items()
    }

    lidars = lidar_pi_health.collect_lidar_health()
    pi_companion = lidar_pi_health.collect_pi_companion_health()

    payload = heartbeat_payload.HeartbeatPayload.build(
        services=services,
        system=system,
        cameras=cameras,
        intrinsics_calibration=_load_intrinsics_state(),
        lidars=lidars,
        pi_companion=pi_companion,
    ).to_dict()

    return payload

def main():
    api_key, endpoint = load_env()

    last_interval: Optional[float] = None
    last_ok_log_ts = 0.0
    last_config_hash: Optional[str] = None
    last_config_change_ts = 0.0
    last_error_key: Optional[str] = None
    last_error_log_ts = 0.0

    log_every_s_raw = os.getenv("P2BP_HEARTBEAT_LOG_EVERY_S", str(DEFAULT_LOG_EVERY_SECONDS))
    try:
        log_every_s = max(30.0, float(log_every_s_raw))
    except (TypeError, ValueError):
        log_every_s = float(DEFAULT_LOG_EVERY_SECONDS)

    while True:
        new_config: Optional[Dict[str, Any]] = None
        try:
            old_config = config_io.load_local_config(CONFIG_PATH)

            payload = create_heartbeat_payload()
            # Never log the full payload at INFO; it can be large and repeated.
            logger.debug("Heartbeat payload: %s", payload)

            new_config = send_heartbeat(api_key, endpoint, payload)

            now = time.time()
            if now - last_ok_log_ts >= log_every_s:
                logger.info("Heartbeat OK (%s)", _payload_summary(payload))
                last_ok_log_ts = now

            if new_config:
                # As long as new_config is not None, send systemd signals.
                # this makes sure signals are sent even if the config file is not updated
                signals.send_systemd_signals(SIGNAL_DIR, new_config)

                # Only log config details when it actually changes.
                if isinstance(new_config, dict):
                    cfg_hash = _short_hash(new_config)
                    if cfg_hash != last_config_hash:
                        logger.info("Config received (hash=%s)", cfg_hash)
                        logger.debug("New config: %s", new_config)
                        last_config_hash = cfg_hash
                        last_config_change_ts = now

            if old_config != new_config:
                config_io.write_config_atomic(CONFIG_PATH, new_config)
                logger.info("Config updated on disk")

        except Exception as e:
            # Avoid spamming the same error every loop; log at most once per minute per error type.
            now = time.time()
            error_key = f"{type(e).__name__}:{str(e)[:200]}"
            if error_key != last_error_key or (now - last_error_log_ts) >= 60.0:
                logger.warning("Heartbeat error: %s", e)
                logger.debug("Heartbeat error detail", exc_info=True)
                last_error_key = error_key
                last_error_log_ts = now

        # Drive cadence from latest config when available.
        base_config = new_config if isinstance(new_config, dict) else old_config
        base_interval = get_heartbeat_interval_seconds(base_config)
        active = _is_active_operation(base_config)
        in_change_window = (time.time() - last_config_change_ts) < CONFIG_CHANGE_FAST_WINDOW_S
        fast = active or in_change_window
        interval = FAST_HEARTBEAT_INTERVAL if fast else base_interval
        if last_interval is None or interval != last_interval:
            if active:
                logger.info("Heartbeat interval: %ss (active operation)", interval)
            elif in_change_window:
                logger.info("Heartbeat interval: %ss (config changed)", interval)
            else:
                logger.info("Heartbeat interval: %ss", interval)
            last_interval = interval

        time.sleep(interval)

if __name__ == "__main__":
    main()
