import os
import logging

logger = logging.getLogger(__name__)

def ensure_signal_dir_exists(signal_dir):
    os.makedirs(signal_dir, exist_ok=True)

def set_signal(signal_dir, name, enabled):
    path = os.path.join(signal_dir, name)

    if enabled:
        if not os.path.exists(path):
            open(path, "w").close()
            logging.info(f"Signal enabled: {name}")
    else:
        if os.path.exists(path):
            os.remove(path)
            logging.info(f"Signal disabled: {name}")

def send_systemd_signals(signal_dir, config):
    ensure_signal_dir_exists(signal_dir)

    # Tracking service
    tracking_cfg = config.get("tracking", {})
    tracking_enabled = tracking_cfg.get("enabled", False)

    set_signal(signal_dir, "tracking.enabled", tracking_enabled)