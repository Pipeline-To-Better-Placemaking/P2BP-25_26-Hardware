import os
import logging

logger = logging.getLogger(__name__)

def ensure_signal_dir_exists(signal_dir):
    os.makedirs(signal_dir, exist_ok=True)


def _parse_tracking_enabled(config):
    """Return desired Tracking.Enabled if explicitly present; otherwise None.

    Returning None means "leave current signal unchanged".
    """

    if not isinstance(config, dict):
        return None

    if "Tracking" in config:
        tracking_cfg = config.get("Tracking")
    elif "tracking" in config:
        tracking_cfg = config.get("tracking")
    else:
        return None

    if isinstance(tracking_cfg, dict):
        raw = tracking_cfg.get("Enabled", tracking_cfg.get("enabled"))
        if raw is None:
            return None
        return bool(raw)

    if isinstance(tracking_cfg, bool):
        return bool(tracking_cfg)

    if isinstance(tracking_cfg, str):
        return tracking_cfg.strip().lower() in {"enabled", "true", "1", "yes", "on"}

    return None

def set_signal(signal_dir, name, enabled):
    path = os.path.join(signal_dir, name)

    if enabled:
        if not os.path.exists(path):
            open(path, "w").close()
            logger.info("Signal enabled: %s", name)
    else:
        if os.path.exists(path):
            os.remove(path)
            logger.info("Signal disabled: %s", name)

def send_systemd_signals(signal_dir, config):
    ensure_signal_dir_exists(signal_dir)

    # Tracking service
    try:
        tracking_enabled = _parse_tracking_enabled(config)
    except Exception as e:
        logger.warning("Failed to parse Tracking config; leaving unchanged: %s", e)
        tracking_enabled = None

    if tracking_enabled is None:
        logger.debug("Tracking config absent/invalid; not updating tracking.enabled")
        return

    set_signal(signal_dir, "tracking.enabled", bool(tracking_enabled))