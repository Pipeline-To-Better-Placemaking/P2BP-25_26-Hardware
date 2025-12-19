# This file will constantly request the config file from the server (GET requests)
# This happens in a set time interval (not decided yet)
# It will replace the local config file if there are any changes
# If there are any changes, it will send the necessary systemd signals to start/stop services
# The heartbeat system will also monitor the status of the Jetson and report any issues (POST requests)
# The health report can actually be combined with the config file request to reduce the number of requests


# first we load the .env file with the API key and endpoint
import os
import time
import json
import requests
import dotenv
import subprocess
import logging
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

TEGRastats_RE_GPU = re.compile(r"GR3D_FREQ\s+(\d+)%@(\d+)")
TEGRastats_RE_RAM = re.compile(r"RAM\s+(\d+)/(\d+)MB")

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

    r = requests.post(f"{endpoint}/Heartbeat", headers=headers, json=payload, timeout=5)
    r.raise_for_status()

    try:
        return r.json()
    except ValueError:
        raise RuntimeError("Backend returned invalid JSON")

def load_local_config(): # load the local copy of the config file
    if not os.path.exists(CONFIG_PATH):
        return None
    try:
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None

def write_config_atomic(path, data): # replace the local copy of the config file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def get_service_state(name):
    try:
        result = subprocess.run(
            ["systemctl", "show", name, "-p", "ActiveState", "-p", "SubState"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        state = {
            "Active": "unknown",
            "Sub": "unknown"
        }

        for line in result.stdout.splitlines():
            if line.startswith("ActiveState="):
                state["Active"] = line.split("=", 1)[1]
            elif line.startswith("SubState="):
                state["Sub"] = line.split("=", 1)[1]

        return state
    except Exception as e:
        logging.error(f"Failed to get state for {name}: {e}")
        return {"Active": "unknown", "Sub": "unknown"}
    
def normalize_service_name(service_file):
    # "heartbeat.service" → "heartbeat"
    if service_file.endswith(".service"):
        return service_file[:-8]
    return service_file


def discover_services(services_dir="/opt/p2bp/camera/services"):
    services = []

    if not os.path.isdir(services_dir):
        return services

    for fname in os.listdir(services_dir):
        if fname.endswith(".service"):
            services.append(fname)

    return sorted(services)

def get_all_service_states(services_dir="/opt/p2bp/camera/services"):
    service_states = {}

    service_files = discover_services(services_dir)

    for service_file in service_files:
        unit_name = service_file
        display_name = normalize_service_name(service_file)

        service_states[display_name] = get_service_state(unit_name)

    return service_states


def get_gpu_and_memory_stats():
    try:
        # Run tegrastats once
        proc = subprocess.run(
            ["tegrastats", "--interval", "1000", "--count", "1"],
            capture_output=True,
            text=True,
            timeout=3,
        )

        output = proc.stdout.strip()
        if not output:
            return {}

        stats = {}

        for line in output.splitlines():
            gpu_match = TEGRastats_RE_GPU.search(line)
            if gpu_match:
                stats["Gpu"] = {
                    "UtilizationPct": int(gpu_match.group(1)),
                    "FrequencyMhz": int(gpu_match.group(2)),
                }

            ram_match = TEGRastats_RE_RAM.search(line)
            if ram_match:
                stats["Memory"] = {
                    "UsedMb": int(ram_match.group(1)),
                    "TotalMb": int(ram_match.group(2)),
                }


        return stats

    except Exception as e:
        logging.error(f"Failed to read GPU/memory stats: {e}")
        return {}
    
def get_system_stats():
    stats = {
        "Gpu": {
            "UtilizationPct": "unknown",
            "FrequencyMhz": "unknown",
        },
        "Memory": {
            "UsedMb": "unknown",
            "TotalMb": "unknown",
        },
    }

    gpu_mem = get_gpu_and_memory_stats()

    # merge results into one dict
    for key in gpu_mem:
        stats[key].update(gpu_mem[key])

    return stats

def create_heartbeat_payload(): # create a payload for the heartbeat request
    return {
        "Id": "0",
        "ProjectId": "0",
        "DeviceId": os.uname().nodename,
        "Timestamp": int(time.time()),
        "Services": get_all_service_states(),
        "System": get_system_stats(),
    }

def ensure_signal_dir_exists():
    os.makedirs(SIGNAL_DIR, exist_ok=True)

def set_signal(name, enabled):
    path = os.path.join(SIGNAL_DIR, name)

    if enabled:
        if not os.path.exists(path):
            open(path, "w").close()
            logging.info(f"Signal enabled: {name}")
    else:
        if os.path.exists(path):
            os.remove(path)
            logging.info(f"Signal disabled: {name}")

def send_systemd_signals(config):
    ensure_signal_dir_exists()

    # Tracking service
    tracking_cfg = config.get("tracking", {})
    tracking_enabled = tracking_cfg.get("enabled", False)

    set_signal("tracking.enabled", tracking_enabled)

def main():
    api_key, endpoint = load_env()

    while True:
        try:
            payload = create_heartbeat_payload()

            new_config = send_heartbeat(api_key, endpoint, payload)
            old_config = load_local_config()

            if new_config:
                # as long as new_config is not None, send systemd signals
                # this makes sure signals are sent even if the config file is not updated
                send_systemd_signals(new_config)

            if old_config != new_config:
                write_config_atomic(CONFIG_PATH, new_config)
                logging.info("Config updated")

        except Exception as e:
            logging.error(f"Heartbeat error: {e}")

        time.sleep(HEARTBEAT_INTERVAL)

if __name__ == "__main__":
    main()
