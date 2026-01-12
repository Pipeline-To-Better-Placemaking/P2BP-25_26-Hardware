import requests, time
from requests.auth import HTTPDigestAuth
from requests.exceptions import RequestException


ACTIVATION_PASSWORD = "Placemaking25"
ACTIVATION_ENDPOINT = "/ISAPI/System/activate"

def first_time_activation(ip: str, timeout: int = 5) -> bool:
    url = f"http://{ip}/ISAPI/System/activate"

    # payload could be json or xml
    for payload in (
        {"password": ACTIVATION_PASSWORD},
        {"Password": ACTIVATION_PASSWORD},
    ):
        try:
            r = requests.put(url, json=payload, timeout=timeout)
            if r.status_code in (200, 401):
                time.sleep(1)
                return True
        except RequestException:
            pass

    xml_payload = f"""<?xml version="1.0" encoding="UTF-8"?>
<ActivateInfo>
    <password>{ACTIVATION_PASSWORD}</password>
</ActivateInfo>
"""

    try:
        r = requests.put(
            url,
            data=xml_payload,
            headers={"Content-Type": "application/xml"},
            timeout=timeout,
        )
        if r.status_code in (200, 401):
            time.sleep(1)
            return True
    except RequestException as e:
        raise RuntimeError(f"Activation request failed: {e}")

    raise RuntimeError(
        f"Activation failed: HTTP {r.status_code} – {r.text}"
    )


def ensure_activated(ip: str, timeout: int = 5) -> bool:
    try:
        return first_time_activation(ip, timeout)
    except RuntimeError as e:
        print(f"Activation error for {ip}: {e}")
        return False

def disable_osd(ip: str, password: str, timeout: int = 5) -> bool:
    """
    Disable all on-screen display (OSD) text overlays on an ANNKE camera.
    """

    url = f"http://{ip}/ISAPI/System/Video/inputs/channels/1/overlays/text"

    xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
<TextOverlay>
    <enabled>false</enabled>
</TextOverlay>
"""

    try:
        response = requests.put(
            url,
            data=xml_payload,
            headers={"Content-Type": "application/xml"},
            auth=HTTPDigestAuth("admin", password),
            timeout=timeout
        )
    except RequestException as e:
        raise RuntimeError(f"OSD disable request failed: {e}")

    if response.status_code in (200, 204):
        return True

    if response.status_code == 401:
        raise RuntimeError("Unauthorized: invalid credentials")

    raise RuntimeError(
        f"Failed to disable OSD (status {response.status_code}): {response.text}"
    )

def apply_defaults(ip: str, timeout: int = 5) -> bool:
    """
    Apply default settings to an ANNKE camera after activation.
    Currently disables OSD overlays.
    """

    try:
        return disable_osd(ip, ACTIVATION_PASSWORD, timeout)
    except RuntimeError as e:
        print(f"Error applying defaults for {ip}: {e}")
        return False
