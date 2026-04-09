from __future__ import annotations

from typing import Optional


# Single source of truth for MAC prefix → camera model ID.
# Add a new entry here when a new camera model is introduced.
MAC_PREFIX_TO_MODEL: dict[str, str] = {
    "d0:3b:f4": "ANNKE",
}


def _normalize_mac(mac: str) -> str:
    return (mac or "").strip().lower()


def _detect_camera_type(mac: str) -> Optional[str]:
    mac_l = _normalize_mac(mac)
    for prefix, model in MAC_PREFIX_TO_MODEL.items():
        if mac_l.startswith(prefix):
            return model
    return None


def onboard_camera(cam_ip: str, cam_mac: str) -> bool:
    cam_type = _detect_camera_type(cam_mac)
    if cam_type == "ANNKE":
        from scripts.camera_controllers import annke_controller
        try:
            if bool(annke_controller.ensure_activated(cam_ip)):
                ok = bool(annke_controller.disable_osd_text(cam_ip))
                if not ok:
                    print(f"[ONBOARD] ANNKE disable_osd_text failed for {cam_ip}")
                #annke_controller.apply_defaults(cam_ip)
                return ok
        except Exception as e:
            print(f"[ONBOARD] ANNKE onboarding unavailable: {e}")

    return False