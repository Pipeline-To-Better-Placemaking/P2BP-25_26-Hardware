import time
import requests
from requests.exceptions import RequestException

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout


ACTIVATION_PASSWORD = "Placemaking25"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = ACTIVATION_PASSWORD
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


def disable_osd_text(ip: str, headless: bool = True, timeout_ms: int = 15_000) -> bool:
    """Disable OSD overlay text (Display Name + Display Date) via the camera web UI.

    Flow (matches UI): Login -> Configuration -> Image -> OSD Settings -> uncheck -> Save.
    Idempotent: leaves checkboxes unchecked if already unchecked.
    """

    if not _is_activated(ip):
        print(f"[ANNKE] Camera {ip} appears not activated; cannot disable OSD.")
        return False

    login_url = f"http://{ip}/doc/page/login.asp"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=headless,
                args=PLAYWRIGHT_ARGS,
            )

            page = browser.new_page()
            page.set_default_timeout(timeout_ms)
            page.goto(login_url, wait_until="domcontentloaded")

            if page.locator("#activePassword").is_visible():
                print(f"[ANNKE] Camera {ip} is showing activation UI; cannot disable OSD.")
                browser.close()
                return False

            # Login
            if page.locator("#username").is_visible():
                page.fill("#username", ADMIN_USERNAME)
                page.fill("#password", ADMIN_PASSWORD)

                login_btn = page.locator('button[ng-click="login()"]')
                if login_btn.count() > 0:
                    login_btn.click()
                else:
                    page.click('button:has-text("Login")')

                print("[ANNKE] Logged in to camera.")
                # Post-login is often SPA-driven; wait for the top nav to appear.
                page.wait_for_selector("ul#nav", timeout=timeout_ms)

            # Navigate: Configuration -> Image -> OSD Settings.
            page.wait_for_selector("ul#nav", timeout=timeout_ms)

            config_nav = page.locator('ul#nav a[ng-click*="jumpTo(\'config\')"]')
            if config_nav.count() > 0:
                config_nav.first.click()
            else:
                # Text fallback
                config_by_text = page.locator('ul#nav a:has-text("Configuration")')
                if config_by_text.count() > 0:
                    config_by_text.first.click()
                else:
                    try:
                        page.evaluate("typeof jumpTo === 'function' && jumpTo('config')")
                    except Exception:
                        pass

            # Wait until we're actually on the Configuration view.
            try:
                page.wait_for_selector("body#config", timeout=timeout_ms)
            except PlaywrightTimeout:
                page.wait_for_selector("#menu", timeout=timeout_ms)

            image_menu = page.locator('#menu div[name="image"] .menu-title, #menu div[name="image"], div[name="image"]')
            if image_menu.count() > 0:
                image_menu.click()

            # Click the OSD tab
            osd_tab = page.locator('#tabs li[module="osd"] a, #tabs a[href*="config/image/osd.asp"], #tabs a:has-text("OSD Settings")')
            if osd_tab.count() > 0:
                osd_tab.first.click()

            # Toggle checkboxes (selectors derived from saved HTML).
            display_name = page.locator('input[ng-model="oOsdParams.bDisplayName"]')
            display_date = page.locator('input[ng-model="oOsdParams.bDisplayDate"]')

            # Wait until OSD controls are ready (tab content is often injected dynamically).
            display_name.wait_for(state="visible")
            display_date.wait_for(state="visible")

            if display_name.is_checked():
                display_name.uncheck()

            if display_date.is_checked():
                display_date.uncheck()

            # Save.
            save_btn = page.locator('button.btn-save')
            if save_btn.count() == 0:
                save_btn = page.locator('button[ng-click="save()"]')
            save_btn.click()

            # Give UI time to apply.
            time.sleep(1)

            browser.close()

    except PlaywrightTimeout as e:
        print(f"[ANNKE] OSD UI timeout for {ip}: {e}")
        return False
    except Exception as e:
        print(f"[ANNKE] Failed to disable OSD for {ip}: {e}")
        return False

    return True

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
