import subprocess
import logging
import os

logger = logging.getLogger(__name__)

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