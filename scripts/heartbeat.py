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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

CONFIG_PATH = "/opt/p2bp/camera/config/config.json"
SIGNAL_DIR = "/run/p2bp"
HEARTBEAT_INTERVAL = 10 # in seconds

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
        mac: heartbeat_payload.CameraState(**state)
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

    while True:
        try:
            payload = create_heartbeat_payload()
            logging.info(f"Sending heartbeat with payload:\n {payload}")

            new_config = send_heartbeat(api_key, endpoint, payload)
            old_config = config_io.load_local_config(CONFIG_PATH)

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

        time.sleep(HEARTBEAT_INTERVAL)

if __name__ == "__main__":
    main()
