import time
import requests
from requests.exceptions import RequestException

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout


ACTIVATION_PASSWORD = "Placemaking25"
STATUS_ENDPOINT = "/ISAPI/System/status"

PLAYWRIGHT_ARGS = [
    "--no-sandbox",
    "--disable-gpu",
    "--disable-dev-shm-usage",
]


def _is_activated(ip: str, timeout: int = 5) -> bool:
    """
    Returns True if the camera is already activated.
    """
    url = f"http://{ip}{STATUS_ENDPOINT}"

    try:
        r = requests.get(url, timeout=timeout)
    except RequestException:
        return False

    # Activated cameras require auth
    if r.status_code == 401:
        return True

    if "notActivated" in r.text:
        return False

    return False


def _activate_via_browser(ip: str, headless: bool = True) -> None:
    """
    Activate the camera by automating the web UI.
    """
    url = f"http://{ip}/doc/page/login.asp"

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=PLAYWRIGHT_ARGS,
        )

        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded")

        # Wait for activation password field
        page.wait_for_selector("#activePassword", timeout=10_000)

        # Fill password + confirm password
        page.fill("#activePassword", ACTIVATION_PASSWORD)
        page.fill('input[ng-model="activePasswordConfirm"]', ACTIVATION_PASSWORD)

        # Click OK
        page.click('button:has-text("OK")')

        # Give firmware time to commit activation
        time.sleep(2)

        browser.close()

def ensure_activated(ip: str, headless: bool = True) -> bool:
    """
    Ensure the camera is activated.
    """
    if _is_activated(ip):
        return True

    try:
        _activate_via_browser(ip, headless=headless)
    except PlaywrightTimeout as e:
        print(f"[ANNKE] Activation UI timeout for {ip}: {e}")
        return False
    except Exception as e:
        print(f"[ANNKE] Activation failed for {ip}: {e}")
        return False

    return _is_activated(ip)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python annke_controller.py <camera_ip>")
        raise SystemExit(1)

    ip = sys.argv[1]
    success = ensure_activated(ip, headless=True)
    print(f"Activated: {success}")
