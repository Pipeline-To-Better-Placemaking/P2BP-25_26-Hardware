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
    tracking_enabled = False
    try:
        # Preferred (current): {"Tracking": {"Enabled": true}}
        tracking_cfg = config.get("Tracking") if isinstance(config, dict) else None
        if isinstance(tracking_cfg, dict):
            tracking_enabled = bool(tracking_cfg.get("Enabled", tracking_cfg.get("enabled", False)))
        elif isinstance(tracking_cfg, bool):
            tracking_enabled = bool(tracking_cfg)
        elif isinstance(tracking_cfg, str):
            tracking_enabled = tracking_cfg.strip().lower() in {"enabled", "true", "1", "yes", "on"}
        else:
            # Back-compat: {"tracking": {"enabled": true}}
            tracking_cfg2 = config.get("tracking", {}) if isinstance(config, dict) else {}
            if isinstance(tracking_cfg2, dict):
                tracking_enabled = bool(tracking_cfg2.get("Enabled", tracking_cfg2.get("enabled", False)))
    except Exception as e:
        logger.warning(f"Failed to parse Tracking config; defaulting disabled: {e}")
        tracking_enabled = False

    set_signal(signal_dir, "tracking.enabled", tracking_enabled)