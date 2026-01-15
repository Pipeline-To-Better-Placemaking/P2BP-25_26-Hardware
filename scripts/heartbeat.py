# This script puts together a health report and sends it to the server
# It expects a new config file in return and will update the local config file if there are any changes
# It will also send systemd signals to start/stop services based on the new config file

# first we load the .env file with the API key and endpoint
import os
import time
import json
import requests
import dotenv
import logging
import scripts.json_models.heartbeat_payload as heartbeat_payload
import scripts.systemd_services as systemd_services
import scripts.system_stats as system_stats
import scripts.config_io as config_io
import scripts.signals as signals
import scripts.camera_handler as camera_handler
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

CONFIG_PATH = "/opt/p2bp/camera/config/config.json"
SIGNAL_DIR = "/run/p2bp"
DEFAULT_HEARTBEAT_INTERVAL = 10  # seconds


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
    return interval


def sanitize_camera_state_for_heartbeat(state: Dict[str, Any]) -> Dict[str, Any]:
    """Only include fields that the backend/heartbeat model expects for now."""
    allowed = {"Mac", "Ip", "Resolution", "Enabled"}
    return {k: v for k, v in state.items() if k in allowed}

def load_env():
    #dotenv.load_dotenv("../config/agent.env") # for local testing
    dotenv.load_dotenv("/opt/p2bp/camera/config/agent.env")
    api_key = os.getenv("API_KEY")
    endpoint = os.getenv("ENDPOINT")

    if not api_key:
        raise RuntimeError("Missing API_KEY")
    elif not endpoint:
        raise RuntimeError("Missing ENDPOINT")

    return api_key, endpoint

def send_heartbeat(api_key, endpoint, payload): # send a health report heartbeat the server. Expect a new config file in return
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    r = requests.post(f"{endpoint}/Device/heartbeat", headers=headers, json=payload, timeout=5)
    r.raise_for_status()

    try:
        return r.json()
    except ValueError:
        raise RuntimeError("Backend returned invalid JSON")

def create_heartbeat_payload(): # create a payload for the heartbeat request
    services = systemd_services.get_all_service_states()
    system = system_stats.get_system_stats()
    camera_dicts = camera_handler.get_camera_states()

    # Convert dicts to CameraState dataclasses
    cameras = {
        mac: heartbeat_payload.CameraState(**sanitize_camera_state_for_heartbeat(state))
        for mac, state in camera_dicts.items()
    }

    payload = heartbeat_payload.HeartbeatPayload.build(
        services=services,
        system=system,
        cameras=cameras,
    ).to_dict()

    return payload

def main():
    api_key, endpoint = load_env()

    last_interval = None  # type: Optional[float]

    while True:
        try:
            old_config = config_io.load_local_config(CONFIG_PATH)

            payload = create_heartbeat_payload()
            logging.info(f"Sending heartbeat with payload:\n {payload}")

            new_config = send_heartbeat(api_key, endpoint, payload)

            if new_config:
                # as long as new_config is not None, send systemd signals
                # this makes sure signals are sent even if the config file is not updated
                signals.send_systemd_signals(SIGNAL_DIR, new_config)
                logging.info(f"New config:\n {new_config}")

            if old_config != new_config:
                config_io.write_config_atomic(CONFIG_PATH, new_config)
                logging.info("Config updated")

        except Exception as e:
            logging.error(f"Heartbeat error: {e}")

        # Drive cadence from latest config when available.
        interval = get_heartbeat_interval_seconds(new_config if isinstance(new_config, dict) else old_config)
        if last_interval is None or interval != last_interval:
            logging.info(f"Heartbeat interval: {interval}s")
            last_interval = interval

        time.sleep(interval)

if __name__ == "__main__":
    main()
