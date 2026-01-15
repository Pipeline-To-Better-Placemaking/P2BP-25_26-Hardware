import time
import os
import requests
from requests.exceptions import RequestException
from typing import Any, Callable, Optional

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


DEFAULT_WINDOWS_CHROME_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _all_frames(page: Any) -> list[Any]:
    try:
        return list(page.frames)
    except Exception:
        return [page]


def _config_frames_for_root(page: Any, config_root_frame: Optional[Any]) -> list[Any]:
    if config_root_frame is None:
        return _all_frames(page)

    try:
        frames = [config_root_frame]
        for f in _all_frames(page):
            try:
                if f.parent_frame == config_root_frame:
                    frames.append(f)
            except Exception:
                continue
        return frames
    except Exception:
        return [config_root_frame]


def _click_first_in_config(
    page: Any,
    config_root_frame: Optional[Any],
    selector: str,
    label: str,
    log: Callable[[str], None],
) -> bool:
    for f in _config_frames_for_root(page, config_root_frame):
        try:
            loc = f.locator(selector)
            if loc.count() > 0 and loc.first.is_visible():
                log(f"Clicking {label} via selector: {selector}")
                loc.first.click()
                return True
        except Exception:
            continue
    log(f"Did not find visible {label} for selector: {selector}")
    return False


def _click_by_text_in_config(
    page: Any,
    config_root_frame: Optional[Any],
    text: str,
    label: str,
    log: Callable[[str], None],
) -> bool:
    selectors = [
        f'a:has-text("{text}")',
        f'button:has-text("{text}")',
        f'li:has-text("{text}")',
        f"div[role='tab']:has-text(\"{text}\")",
    ]
    for sel in selectors:
        if _click_first_in_config(page, config_root_frame, sel, label, log):
            return True
    return False


def _wait_for_nav(
    page: Any,
    timeout_ms: int,
    log: Callable[[str], None],
    dump_state: Callable[[str], None],
) -> bool:
    """Wait for the top nav that exists on Live View after login."""

    nav_locator = page.locator("ul#nav")
    deadline = time.time() + (timeout_ms / 1000.0)

    # On these firmwares, ul#nav is frequently present but never Playwright-visible.
    # Waiting for it to be "visible" adds consistent delay with no benefit.
    try:
        page.wait_for_selector("ul#nav", state="attached", timeout=min(timeout_ms, 5_000))
    except PlaywrightTimeout:
        last_log = 0.0
        while time.time() < deadline:
            # If the initial short wait timed out, keep polling until attached.
            # (Previously this loop would log but never succeed, even if ul#nav appeared later.)
            try:
                if nav_locator.count() > 0:
                    try:
                        visible = nav_locator.is_visible()
                        log(f"Top nav ul#nav is attached (visible={visible}).")
                    except Exception:
                        log("Top nav ul#nav is attached.")
                    return True
            except Exception:
                pass

            now = time.time()
            if now - last_log >= 1.0:
                last_log = now
                try:
                    url = page.url
                except Exception:
                    url = "<unknown>"

                try:
                    has_username = page.locator("#username").count() > 0
                    username_visible = page.locator("#username").is_visible() if has_username else False
                except Exception:
                    username_visible = False

                try:
                    nav_count = nav_locator.count()
                except Exception:
                    nav_count = -1

                log(
                    f"Waiting for ul#nav (attached)... url={url} nav_count={nav_count} "
                    f"login_field_visible={username_visible}"
                )
            page.wait_for_timeout(250)

        dump_state("nav_missing")
        return False

    try:
        visible = nav_locator.is_visible()
        log(f"Top nav ul#nav is attached (visible={visible}).")
    except Exception:
        log("Top nav ul#nav is attached.")
    return True


def login(
    page: Any,
    timeout_ms: int,
    log: Callable[[str], None],
    dump_state: Callable[[str], None],
    username: str = ADMIN_USERNAME,
    password: str = ADMIN_PASSWORD,
) -> bool:
    """Login if needed; returns True once Live View nav is present."""

    def _try_login_on_frame(frame) -> bool:
        try:
            user = frame.locator(
                "#username, input#username, input[name='username'], input[name='userName'], input[autocomplete='username']"
            )
            pwd = frame.locator(
                "#password, input#password, input[name='password'], input[autocomplete='current-password'], input[type='password']"
            )

            if user.count() == 0 or pwd.count() == 0:
                return False
            if not user.first.is_visible() or not pwd.first.is_visible():
                return False

            log("Login form detected; submitting credentials.")
            user.first.fill(username)
            pwd.first.fill(password)

            login_btn = frame.locator('button[ng-click="login()"]')
            if login_btn.count() > 0 and login_btn.first.is_visible():
                login_btn.first.click()
            else:
                login_fallback = frame.locator(
                    'button:has-text("Login"), button:has-text("Sign in"), input[value="Login"], input[type="submit"]'
                )
                if login_fallback.count() > 0 and login_fallback.first.is_visible():
                    login_fallback.first.click()
                else:
                    frame.keyboard.press("Enter")

            return True
        except Exception as e:
            log(f"Login attempt failed: {e}")
            return False

    # Fast-path: if nav is already present, we are already logged in.
    # Do not wait for ul#nav here: on the login page it will never appear and wastes time.
    try:
        if page.locator("ul#nav").count() > 0:
            return True
    except Exception:
        pass

    # The login page often finishes rendering *after* domcontentloaded.
    # Wait briefly for the username/password fields to become visible before giving up.
    submitted = False
    form_deadline = time.time() + min(6.0, timeout_ms / 1000.0)
    last_log = 0.0

    while time.time() < form_deadline and not submitted:
        submitted = _try_login_on_frame(page)
        if not submitted:
            try:
                for f in page.frames:
                    if f == page.main_frame:
                        continue
                    if _try_login_on_frame(f):
                        submitted = True
                        break
            except Exception:
                pass

        if submitted:
            break

        now = time.time()
        if now - last_log >= 1.0:
            last_log = now
            try:
                url = page.url
            except Exception:
                url = "<unknown>"
            log(f"Waiting for login form... url={url}")

        try:
            page.wait_for_timeout(250)
        except Exception:
            time.sleep(0.25)

    if not submitted:
        log("No visible login form found and ul#nav not present.")
        dump_state("no_login_form_no_nav")
        return False

    print("[ANNKE] Logged in to camera.")
    try:
        page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    except PlaywrightTimeout:
        pass
    try:
        page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 5_000))
    except PlaywrightTimeout:
        pass

    if not _wait_for_nav(page, timeout_ms, log, dump_state):
        log("Timed out waiting for ul#nav after login submission.")
        dump_state("after_login_nav_timeout")
        return False

    return True


def gotoConfig(
    page: Any,
    timeout_ms: int,
    log: Callable[[str], None],
    dump_state: Callable[[str], None],
) -> Optional[Any]:
    """Navigate to Configuration; returns the frame where config UI is ready."""

    if not _wait_for_nav(page, timeout_ms, log, dump_state):
        log("Timed out waiting for ul#nav before navigation.")
        dump_state("pre_nav_timeout")
        return None

    config_nav = page.locator('ul#nav a[ng-click*="jumpTo(\'config\')"]')
    if config_nav.count() > 0:
        log("Clicking Configuration (ng-click jumpTo('config')).")
        config_nav.first.click()
    else:
        config_by_text = page.locator('ul#nav a:has-text("Configuration")')
        if config_by_text.count() > 0:
            log("Clicking Configuration (text fallback).")
            config_by_text.first.click()
        else:
            try:
                log("Attempting jumpTo('config') via page.evaluate fallback.")
                page.evaluate("typeof jumpTo === 'function' && jumpTo('config')")
            except Exception:
                pass

    page.wait_for_selector(
        'body#config, #menu, #tabs, div[name="image"], input[ng-model="oOsdParams.bDisplayName"]',
        state="attached",
        timeout=timeout_ms,
    )

    config_root_frame: Optional[Any] = None
    deadline = time.time() + (timeout_ms / 1000.0)
    last_log = 0.0

    while time.time() < deadline:
        for f in _all_frames(page):
            try:
                menu = f.locator("#menu")
                tabs = f.locator("#tabs")
                menu_visible = menu.count() > 0 and menu.first.is_visible()
                tabs_visible = tabs.count() > 0 and tabs.first.is_visible()
                if menu_visible or tabs_visible:
                    config_root_frame = f
                    log(
                        "Config UI is ready in a frame: "
                        f"menu_visible={menu_visible} tabs_visible={tabs_visible}"
                    )
                    return config_root_frame
            except Exception:
                continue

        now = time.time()
        if now - last_log >= 1.0:
            last_log = now
            try:
                menu_cnt = page.locator("#menu").count()
                menu_vis = page.locator("#menu").first.is_visible() if menu_cnt > 0 else False
            except Exception:
                menu_cnt, menu_vis = -1, False
            try:
                tabs_cnt = page.locator("#tabs").count()
                tabs_vis = page.locator("#tabs").first.is_visible() if tabs_cnt > 0 else False
            except Exception:
                tabs_cnt, tabs_vis = -1, False

            log(
                "Waiting for config UI... "
                f"url={page.url} #menu(count={menu_cnt},visible={menu_vis}) "
                f"#tabs(count={tabs_cnt},visible={tabs_vis}) frames={len(_all_frames(page))}"
            )

        page.wait_for_timeout(250)

    dump_state("config_ui_not_ready")
    return None


def gotoImage(
    page: Any,
    config_root_frame: Optional[Any],
    timeout_ms: int,
    log: Callable[[str], None],
    dump_state: Callable[[str], None],
) -> bool:
    """Within Configuration, select the Image section and ensure tabs are present."""

    clicked_image = (
        _click_first_in_config(page, config_root_frame, '#menu div[name="image"] .menu-title', "Image menu", log)
        or _click_by_text_in_config(page, config_root_frame, "Image", "Image menu", log)
    )

    if not clicked_image:
        log("Image menu click did not succeed; UI structure differs from expected.")

    try:
        page.wait_for_selector("#tabs", state="attached", timeout=min(timeout_ms, 5_000))
    except PlaywrightTimeout:
        dump_state("image_tabs_missing")
        return False

    return True


def gotoVideoAudio(
    page: Any,
    config_root_frame: Optional[Any],
    timeout_ms: int,
    log: Callable[[str], None],
    dump_state: Callable[[str], None],
) -> bool:
    """Within Configuration, select Video/Audio and ensure the Video tab content is present."""

    clicked_menu = (
        _click_first_in_config(page, config_root_frame, '#menu div[name="videoAudio"] .menu-title', "Video/Audio menu", log)
        or _click_first_in_config(page, config_root_frame, '#menu div[name="videoAudio"]', "Video/Audio menu", log)
        or _click_by_text_in_config(page, config_root_frame, "Video/Audio", "Video/Audio menu", log)
    )

    if not clicked_menu:
        log("Video/Audio menu click did not succeed; UI structure differs from expected.")

    # Ensure tabs are present.
    try:
        page.wait_for_selector("#tabs", state="attached", timeout=min(timeout_ms, 5_000))
    except PlaywrightTimeout:
        dump_state("videoaudio_tabs_missing")
        return False

    # Prefer clicking the Video tab explicitly.
    _click_first_in_config(page, config_root_frame, '#tabs a:has-text("Video")', "Video tab", log) or _click_by_text_in_config(
        page, config_root_frame, "Video", "Video tab", log
    )

    # Wait for key controls in the Video tab.
    deadline = time.time() + min(8.0, timeout_ms / 1000.0)
    while time.time() < deadline:
        for f in _config_frames_for_root(page, config_root_frame):
            try:
                if (
                    f.locator('select[ng-model="oVideoParams.resolution"]').count() > 0
                    and f.locator('select[ng-model="oVideoParams.frameRate"], #frameRateSelect').count() > 0
                ):
                    return True
            except Exception:
                continue
        try:
            page.wait_for_timeout(250)
        except Exception:
            time.sleep(0.25)

    dump_state("video_tab_controls_missing")
    return False


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


def disable_osd_text(
    ip: str,
    headless: bool = True,
    timeout_ms: int = 15_000,
    debug: bool = False,
    debug_dump_dir: str = "tracks",
) -> bool:
    """Disable OSD overlay text (Display Name + Display Date) via the camera web UI.

    Flow (matches UI): Login -> Configuration -> Image -> OSD Settings -> uncheck -> Save.
    Idempotent: leaves checkboxes unchecked if already unchecked.
    """

    # Keep the main flow simple and robust: ensure activation first.
    if not ensure_activated(ip, headless=headless):
        print(f"[ANNKE] Camera {ip} appears not activated; cannot disable OSD.")
        return False

    login_url = f"http://{ip}/doc/page/login.asp"

    # Allow enabling richer dumps without changing call sites.
    debug = bool(debug or os.getenv("ANNKE_OSD_DEBUG") == "1")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=headless,
                args=PLAYWRIGHT_ARGS,
            )

            def _log(message: str) -> None:
                ts = time.strftime("%H:%M:%S")
                print(f"[ANNKE][OSD {ip} {ts}] {message}", flush=True)

            def _dump_state(tag: str) -> None:
                if not debug:
                    return

                try:
                    os.makedirs(debug_dump_dir, exist_ok=True)
                    safe_ip = ip.replace(":", "-").replace(".", "-")
                    base = os.path.join(debug_dump_dir, f"annke_osd_{safe_ip}_{int(time.time())}_{tag}")
                    try:
                        page.screenshot(path=f"{base}.png", full_page=True)
                    except Exception:
                        pass
                    try:
                        html = page.content()
                        with open(f"{base}.html", "w", encoding="utf-8") as f:
                            f.write(html)
                    except Exception:
                        pass
                    try:
                        _log(f"Dumped debug state to {base}.(png|html)")
                    except Exception:
                        pass
                except Exception:
                    # Never fail the main flow because debugging failed.
                    return

            # Keep selectors tight: current firmwares consistently use ng-model.
            OSD_DISPLAY_NAME_SELECTORS = [
                'input[ng-model="oOsdParams.bDisplayName"]',
                'input[ng-model*="bDisplayName"]',
            ]
            OSD_DISPLAY_DATE_SELECTORS = [
                'input[ng-model="oOsdParams.bDisplayDate"]',
                'input[ng-model*="bDisplayDate"]',
            ]

            # Some firmwares behave differently for HeadlessChrome UA or behind HTTP auth.
            # Use a realistic UA/viewport and also provide HTTP basic credentials if challenged.
            context = browser.new_context(
                user_agent=DEFAULT_WINDOWS_CHROME_UA,
                viewport={"width": 1280, "height": 720},
                http_credentials={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD},
            )

            page = context.new_page()
            page.set_default_timeout(timeout_ms)

            if debug:
                try:
                    page.on("console", lambda msg: _log(f"[console:{msg.type}] {msg.text}"))
                    page.on("pageerror", lambda exc: _log(f"[pageerror] {exc}"))

                    def _on_request_failed(req):
                        try:
                            failure = req.failure
                            err = failure.error_text if failure else ""
                        except Exception:
                            err = ""
                        _log(f"[requestfailed] {req.url} {err}")

                    page.on("requestfailed", _on_request_failed)
                except Exception:
                    pass

            resp = page.goto(login_url, wait_until="domcontentloaded")

            status = None
            try:
                status = resp.status if resp else None
            except Exception:
                status = None

            try:
                title = page.title()
            except Exception:
                title = "<unknown>"

            _log(f"Loaded login URL: {page.url} (status={status}, title={title})")

            _dump_state("login_loaded")

            if page.locator("#activePassword").is_visible():
                print(f"[ANNKE] Camera {ip} is showing activation UI; cannot disable OSD.")
                browser.close()
                return False

            # High-level navigation flow:
            # ensure_activated() already ran at function entry.
            # login() -> gotoConfig() -> gotoImage() -> OSD-specific logic below.
            if not login(page, timeout_ms, _log, _dump_state):
                context.close()
                browser.close()
                return False

            config_root_frame = gotoConfig(page, timeout_ms, _log, _dump_state)
            if config_root_frame is None:
                context.close()
                browser.close()
                return False

            if not gotoImage(page, config_root_frame, timeout_ms, _log, _dump_state):
                context.close()
                browser.close()
                return False

            def _config_frames():
                return _config_frames_for_root(page, config_root_frame)

            def _iter_descendant_frames(frames):
                """Yield frames plus any descendants (depth-first)."""
                seen = set()

                def _walk(fr):
                    try:
                        fid = id(fr)
                    except Exception:
                        fid = None
                    if fid is not None and fid in seen:
                        return
                    if fid is not None:
                        seen.add(fid)
                    yield fr

                    try:
                        children = list(fr.child_frames)
                    except Exception:
                        children = []
                    for ch in children:
                        yield from _walk(ch)

                for f in frames:
                    yield from _walk(f)

            def _frame_has_any(f, selectors) -> bool:
                for sel in selectors:
                    try:
                        if f.locator(sel).count() > 0:
                            return True
                    except Exception:
                        continue
                return False

            def _find_osd_frame():
                # Prefer config-related frames, but include their descendants and fall back to all frames.
                primary = list(_iter_descendant_frames(_config_frames()))
                for f in primary:
                    if _frame_has_any(f, OSD_DISPLAY_NAME_SELECTORS) or _frame_has_any(f, OSD_DISPLAY_DATE_SELECTORS):
                        return f

                fallback = list(_iter_descendant_frames(_all_frames(page)))
                for f in fallback:
                    if _frame_has_any(f, OSD_DISPLAY_NAME_SELECTORS) or _frame_has_any(f, OSD_DISPLAY_DATE_SELECTORS):
                        return f

                return None

            # Click the OSD tab.
            clicked_osd_tab = (
                _click_first_in_config(page, config_root_frame, '#tabs li[module="osd"] a', "OSD tab", _log)
                or _click_by_text_in_config(page, config_root_frame, "OSD Settings", "OSD tab", _log)
            )

            if not clicked_osd_tab:
                _log("OSD tab click did not succeed; dumping state for selector grounding.")
                _dump_state("osd_tab_not_found")

            # Give the firmware UI a moment to finish injecting OSD tab content.
            # This mirrors what you observed manually (the first render can be partial).
            try:
                entry_sleep_s = float(os.getenv("ANNKE_OSD_ENTRY_SLEEP", "1.5"))
            except Exception:
                entry_sleep_s = 1.5
            if entry_sleep_s > 0:
                _log(f"Settling after OSD tab click for {entry_sleep_s:.1f}s")
                try:
                    page.wait_for_timeout(int(entry_sleep_s * 1000))
                except Exception:
                    time.sleep(entry_sleep_s)

            # Best-effort: allow tab navigation and dynamic iframe injection to settle.
            try:
                page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 5_000))
            except PlaywrightTimeout:
                pass

            def _wait_for_osd_ui_ready():
                """Wait until the OSD tab content is fully usable.

                Some firmwares inject the tab content/iframe incrementally; selectors can exist
                while the page is still rendering/loading, which leads to early scrolling/clicks.
                """

                # Common loading overlays/spinners across firmwares (best-effort).
                loading_selectors = [
                    "#loading",
                    ".loading",
                    ".loading-mask",
                    ".loadingMask",
                    ".ui-widget-overlay",
                    ".blockUI",
                    ".mask",
                    "div[aria-busy='true']",
                ]

                def _any_loading_visible() -> bool:
                    for sel in loading_selectors:
                        try:
                            loc = page.locator(sel)
                            if loc.count() > 0 and loc.first.is_visible():
                                return True
                        except Exception:
                            continue
                    return False

                def _labels_visible(fr) -> bool:
                    # The checkboxes can exist in the DOM before the OSD panel fully renders.
                    # Waiting for the human-visible labels prevents acting on a half-rendered page.
                    try:
                        dn = fr.locator("text=Display Name")
                        dd = fr.locator("text=Display Date")
                        return (
                            dn.count() > 0
                            and dd.count() > 0
                            and dn.first.is_visible()
                            and dd.first.is_visible()
                        )
                    except Exception:
                        return False

                deadline = time.time() + min(12.0, timeout_ms / 1000.0)
                last_log = 0.0

                while time.time() < deadline:
                    osd_fr = _find_osd_frame()

                    # Require the OSD controls exist and a Save button exists somewhere.
                    ready = False
                    if osd_fr is not None:
                        try:
                            has_dn = _frame_has_any(osd_fr, OSD_DISPLAY_NAME_SELECTORS)
                            has_dd = _frame_has_any(osd_fr, OSD_DISPLAY_DATE_SELECTORS)
                        except Exception:
                            has_dn, has_dd = False, False

                        try:
                            save_in_frame = osd_fr.locator('button.btn-save, button[ng-click="save()"]').count() > 0
                        except Exception:
                            save_in_frame = False

                        try:
                            save_in_page = page.locator('button.btn-save, button[ng-click="save()"]').count() > 0
                        except Exception:
                            save_in_page = False

                        if (
                            has_dn
                            and has_dd
                            and _labels_visible(osd_fr)
                            and (save_in_frame or save_in_page)
                            and not _any_loading_visible()
                        ):
                            ready = True

                    if ready and osd_fr is not None:
                        return osd_fr

                    now = time.time()
                    if now - last_log >= 1.0:
                        last_log = now
                        try:
                            url = page.url
                        except Exception:
                            url = "<unknown>"
                        _log(f"Waiting for OSD UI... url={url} loading_visible={_any_loading_visible()}")

                    try:
                        page.wait_for_timeout(250)
                    except Exception:
                        time.sleep(0.25)

                return None

            # Locate OSD controls (may be injected into an iframe and can take a moment).
            osd_frame = _wait_for_osd_ui_ready()

            # If the first render is incomplete, a refresh often makes the OSD UI fully load.
            # Retry exactly once: reload -> wait for config UI -> re-click Image/OSD -> wait again.
            if osd_frame is None:
                _log("OSD UI not ready; reloading and retrying OSD navigation once.")
                try:
                    page.reload(wait_until="domcontentloaded")
                except Exception:
                    try:
                        page.goto(page.url, wait_until="domcontentloaded")
                    except Exception:
                        pass

                # Re-detect config UI after reload and re-navigate to Image/OSD.
                config_root_frame = gotoConfig(page, timeout_ms, _log, _dump_state)
                if config_root_frame is None:
                    _log("After reload, config UI did not become ready.")
                    _dump_state("config_ui_not_ready_after_reload")
                else:
                    gotoImage(page, config_root_frame, timeout_ms, _log, _dump_state)
                    _click_first_in_config(page, config_root_frame, '#tabs li[module="osd"] a', "OSD tab", _log) or _click_by_text_in_config(page, config_root_frame, "OSD", "OSD tab", _log)

                    if entry_sleep_s > 0:
                        try:
                            page.wait_for_timeout(int(entry_sleep_s * 1000))
                        except Exception:
                            time.sleep(entry_sleep_s)

                    try:
                        page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 5_000))
                    except PlaywrightTimeout:
                        pass

                    osd_frame = _wait_for_osd_ui_ready()

            if osd_frame is None:
                _log("OSD controls still not ready after navigation.")
                _dump_state("osd_controls_not_found")
                context.close()
                browser.close()
                return False

            try:
                _log(f"OSD controls found in frame: name={getattr(osd_frame, 'name', '')!r} url={getattr(osd_frame, 'url', '')}")
            except Exception:
                pass

            def _first_locator(frame, selectors):
                for sel in selectors:
                    try:
                        loc = frame.locator(sel)
                        if loc.count() > 0:
                            return loc.first
                    except Exception:
                        continue
                # Fall back to the original selector if nothing matched; this keeps behavior stable.
                return frame.locator(selectors[0]).first

            display_name = _first_locator(osd_frame, OSD_DISPLAY_NAME_SELECTORS)
            display_date = _first_locator(osd_frame, OSD_DISPLAY_DATE_SELECTORS)

            # Wait until OSD controls are ready (tab content is often injected dynamically).
            try:
                # Do NOT require visible: many firmwares hide the actual <input type=checkbox>
                # and render a styled widget. We only need it to exist in DOM.
                display_name.wait_for(state="attached", timeout=timeout_ms)
                display_date.wait_for(state="attached", timeout=timeout_ms)
            except PlaywrightTimeout as e:
                try:
                    dn_count = display_name.count()
                    dd_count = display_date.count()
                except Exception:
                    dn_count, dd_count = -1, -1
                _log(
                    f"Timed out waiting for OSD controls. url={page.url} "
                    f"display_name_count={dn_count} display_date_count={dd_count} err={e}"
                )
                _dump_state("osd_controls_timeout")
                context.close()
                browser.close()
                return False

            def _ensure_unchecked(loc, label: str) -> None:
                # Prefer direct checkbox uncheck; fall back to label click for styled controls.
                try:
                    t = (loc.get_attribute("type") or "").lower()
                except Exception:
                    t = ""

                if t == "checkbox" or t == "":
                    try:
                        # force=True handles hidden inputs.
                        loc.uncheck(force=True)
                        return
                    except Exception as e:
                        _log(f"Failed to uncheck {label} directly (type={t!r}): {e}")

                try:
                    elem_id = loc.get_attribute("id")
                except Exception:
                    elem_id = None

                # Try clicking the associated label.
                if elem_id:
                    try:
                        lab = osd_frame.locator(f'label[for="{elem_id}"]')
                        if lab.count() > 0:
                            lab.first.click(force=True)
                            return
                    except Exception:
                        pass

                # Last resort: click the input itself.
                try:
                    loc.click(force=True)
                except Exception as e:
                    _log(f"Failed to toggle {label} via click fallback: {e}")

            def _try_select_channel_if_present() -> None:
                """Some firmwares require selecting a channel before Save is enabled."""

                selectors = [
                    'select[ng-model="iChannelId"]',
                    'select[ng-model*="ChannelId"]',
                    'select[id*="channel"]',
                    'select[name*="channel"]',
                ]

                def _set_first_nonzero(select_loc, where: str) -> bool:
                    try:
                        if select_loc.count() == 0:
                            return False
                        sel = select_loc.first
                        # Collect option values.
                        opts = sel.locator("option")
                        values = []
                        for i in range(min(opts.count(), 32)):
                            try:
                                v = opts.nth(i).get_attribute("value")
                            except Exception:
                                v = None
                            if v is None:
                                continue
                            values.append(v)

                        # Prefer first non-zero numeric option.
                        chosen = None
                        for v in values:
                            try:
                                if int(v) != 0:
                                    chosen = v
                                    break
                            except Exception:
                                continue

                        if chosen is None:
                            return False

                        try:
                            current = sel.input_value()
                        except Exception:
                            current = None

                        if current == chosen:
                            return False

                        sel.select_option(chosen)
                        _log(f"Selected channel {chosen} via {where}.")
                        return True
                    except Exception as e:
                        _log(f"Channel selection attempt failed via {where}: {e}")
                        return False

                # Try inside the OSD frame first, then the outer page.
                for s in selectors:
                    if _set_first_nonzero(osd_frame.locator(s), f"osd_frame:{s}"):
                        return
                for s in selectors:
                    if _set_first_nonzero(page.locator(s), f"page:{s}"):
                        return

            def _bool_attr(locator, attr: str):
                try:
                    v = locator.get_attribute(attr)
                except Exception:
                    v = None
                if v is None:
                    return False
                return str(v).lower() in ("", "1", "true", "disabled")

            def _is_effectively_checked(locator) -> bool:
                try:
                    return locator.is_checked()
                except Exception:
                    try:
                        return bool(locator.evaluate("el => !!el.checked"))
                    except Exception:
                        return False

            _ensure_unchecked(display_name, "Display Name")
            _ensure_unchecked(display_date, "Display Date")

            # Verify the UI state actually changed before saving.
            try:
                dn_checked = _is_effectively_checked(display_name)
                dd_checked = _is_effectively_checked(display_date)
                _log(f"OSD checkbox states pre-save: display_name={dn_checked} display_date={dd_checked}")
            except Exception:
                pass

            # Save. (Button can be in the OSD frame or in the outer config shell.)
            save_btn = osd_frame.locator('button.btn-save')
            if save_btn.count() == 0:
                save_btn = osd_frame.locator('button[ng-click="save()"]')
            if save_btn.count() == 0:
                save_btn = page.locator('button.btn-save')
            if save_btn.count() == 0:
                save_btn = page.locator('button[ng-click="save()"]')

            # Ensure Save is enabled (some firmwares disable it until a channel is selected).
            try:
                btn = save_btn.first
                try:
                    btn.scroll_into_view_if_needed(timeout=1_000)
                except Exception:
                    pass

                enabled = True
                try:
                    enabled = btn.is_enabled()
                except Exception:
                    # If Playwright can't determine, fall back to disabled attribute.
                    enabled = not _bool_attr(btn, "disabled")

                if not enabled:
                    _log("Save button appears disabled; attempting to select a channel and retry.")
                    _try_select_channel_if_present()
                    try:
                        page.wait_for_timeout(500)
                    except Exception:
                        time.sleep(0.5)

                # Re-check enabled state.
                try:
                    enabled = btn.is_enabled()
                except Exception:
                    enabled = not _bool_attr(btn, "disabled")

                if not enabled:
                    _log("Save button still disabled; cannot apply OSD changes.")
                    _dump_state("save_disabled")
                    context.close()
                    browser.close()
                    return False

                btn.click()
            except Exception as e:
                _log(f"Failed to click Save button: {e}")
                _dump_state("save_click_failed")
                context.close()
                browser.close()
                return False

            # Give UI time to apply.
            try:
                page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 5_000))
            except PlaywrightTimeout:
                pass
            time.sleep(1)

            # Confirm persistence by reloading and re-checking the checkbox state.
            try:
                page.reload(wait_until="domcontentloaded")
                # Re-find the OSD frame and locators after reload.
                osd_frame2 = _find_osd_frame() or osd_frame
                dn2 = _first_locator(osd_frame2, OSD_DISPLAY_NAME_SELECTORS)
                dd2 = _first_locator(osd_frame2, OSD_DISPLAY_DATE_SELECTORS)
                dn2.wait_for(state="attached", timeout=timeout_ms)
                dd2.wait_for(state="attached", timeout=timeout_ms)

                dn2_checked = _is_effectively_checked(dn2)
                dd2_checked = _is_effectively_checked(dd2)
                _log(f"OSD checkbox states post-save reload: display_name={dn2_checked} display_date={dd2_checked}")

                if dn2_checked or dd2_checked:
                    _log("OSD settings did not persist after Save/reload; treating as failure.")
                    _dump_state("osd_not_persisted")
                    context.close()
                    browser.close()
                    return False
            except Exception as e:
                _log(f"Post-save verification failed (continuing anyway): {e}")

            _log("OSD disable flow completed (saved settings).")

            context.close()
            browser.close()

    except PlaywrightTimeout as e:
        print(f"[ANNKE] OSD UI timeout for {ip}: {e}")
        return False
    except Exception as e:
        print(f"[ANNKE] Failed to disable OSD for {ip}: {e}")
        return False

    return True


def _set_video_audio_params(
    ip: str,
    *,
    resolution: Optional[str] = None,
    frame_rate: Optional[int | str] = None,
    headless: bool = True,
    timeout_ms: int = 15_000,
    debug: bool = False,
    debug_dump_dir: str = "tracks",
) -> bool:
    """Internal helper: set video parameters in Configuration -> Video/Audio -> Video and Save."""

    if not ensure_activated(ip, headless=headless):
        print(f"[ANNKE] Camera {ip} appears not activated; cannot change Video/Audio settings.")
        return False

    login_url = f"http://{ip}/doc/page/login.asp"
    debug = bool(debug or os.getenv("ANNKE_OSD_DEBUG") == "1")

    # Resolution options for this firmware (from DevTools):
    # 0=1280*720P, 1=1920*1080P, 2=3072*1728
    allowed_resolutions = {
        "1280*720p": "1280*720P",
        "1280x720p": "1280*720P",
        "1280x720": "1280*720P",
        "1280*720": "1280*720P",
        "1920*1080p": "1920*1080P",
        "1920x1080p": "1920*1080P",
        "1920x1080": "1920*1080P",
        "1920*1080": "1920*1080P",
        "3072*1728": "3072*1728",
        "3072x1728": "3072*1728",
        "3072x1728p": "3072*1728",
        "3072*1728p": "3072*1728",
    }

    if resolution is not None:
        norm = str(resolution).strip().lower().replace(" ", "")
        if norm not in allowed_resolutions:
            raise ValueError(
                "Unsupported resolution. Allowed: 1280*720P, 1920*1080P, 3072*1728. "
                f"Got: {resolution!r}"
            )
        desired_resolution_text = allowed_resolutions[norm]
    else:
        desired_resolution_text = None

    if frame_rate is not None:
        # Strict whitelist based on the camera dropdown.
        # (Includes fractional fps entries like "1/16".)
        allowed_frame_rate_labels = [
            "1/16",
            "1/8",
            "1/4",
            "1/2",
            "1",
            "2",
            "4",
            "6",
            "8",
            "10",
            "12",
            "15",
            "16",
            "18",
            "20",
            "22",
            "24",
        ]

        allowed_frame_rates = {k: k for k in allowed_frame_rate_labels}
        # Accept variants with spaces or a "fps" suffix.
        raw = str(frame_rate).strip().lower()
        raw = raw.replace("fps", "").strip()
        raw = raw.replace(" ", "")

        if raw not in allowed_frame_rates:
            raise ValueError(
                "Unsupported frame rate. Allowed: "
                + ", ".join(allowed_frame_rate_labels)
                + f". Got: {frame_rate!r}"
            )

        desired_fps_label = allowed_frame_rates[raw]
    else:
        desired_fps_label = None

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless, args=PLAYWRIGHT_ARGS)

            def _log(message: str) -> None:
                ts = time.strftime("%H:%M:%S")
                print(f"[ANNKE][VID {ip} {ts}] {message}", flush=True)

            def _dump_state(tag: str) -> None:
                if not debug:
                    return
                try:
                    os.makedirs(debug_dump_dir, exist_ok=True)
                    safe_ip = ip.replace(":", "-").replace(".", "-")
                    base = os.path.join(debug_dump_dir, f"annke_vid_{safe_ip}_{int(time.time())}_{tag}")
                    try:
                        page.screenshot(path=f"{base}.png", full_page=True)
                    except Exception:
                        pass
                    try:
                        html = page.content()
                        with open(f"{base}.html", "w", encoding="utf-8") as f:
                            f.write(html)
                    except Exception:
                        pass
                except Exception:
                    return

            context = browser.new_context(
                user_agent=DEFAULT_WINDOWS_CHROME_UA,
                viewport={"width": 1280, "height": 720},
                http_credentials={"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD},
            )
            page = context.new_page()
            page.set_default_timeout(timeout_ms)

            page.goto(login_url, wait_until="domcontentloaded")

            if not login(page, timeout_ms, _log, _dump_state):
                context.close()
                browser.close()
                return False

            config_root_frame = gotoConfig(page, timeout_ms, _log, _dump_state)
            if config_root_frame is None:
                context.close()
                browser.close()
                return False

            if not gotoVideoAudio(page, config_root_frame, timeout_ms, _log, _dump_state):
                context.close()
                browser.close()
                return False

            def _find_select(selector: str) -> Optional[Any]:
                for f in _config_frames_for_root(page, config_root_frame):
                    try:
                        loc = f.locator(selector)
                        if loc.count() > 0:
                            return loc.first
                    except Exception:
                        continue
                try:
                    loc = page.locator(selector)
                    if loc.count() > 0:
                        return loc.first
                except Exception:
                    pass
                return None

            if desired_resolution_text is not None:
                res_sel = _find_select('select[ng-model="oVideoParams.resolution"]')
                if res_sel is None:
                    _log("Resolution select not found.")
                    _dump_state("resolution_select_missing")
                    context.close()
                    browser.close()
                    return False

                try:
                    res_sel.select_option(label=desired_resolution_text)
                    _log(f"Set resolution to {desired_resolution_text}.")
                except Exception as e:
                    _log(f"Failed selecting resolution {desired_resolution_text}: {e}")
                    _dump_state("resolution_select_failed")
                    context.close()
                    browser.close()
                    return False

            if desired_fps_label is not None:
                fps_sel = _find_select('select[ng-model="oVideoParams.frameRate"], #frameRateSelect')
                if fps_sel is None:
                    _log("Frame rate select not found.")
                    _dump_state("framerate_select_missing")
                    context.close()
                    browser.close()
                    return False

                # Determine the correct option by inspecting available options at runtime.
                try:
                    options = fps_sel.locator("option")
                    chosen_value: Optional[str] = None
                    chosen_label: Optional[str] = None

                    def _norm_rate_label(s: str) -> str:
                        s = (s or "").strip().lower()
                        s = s.replace("fps", "").strip()
                        s = s.replace(" ", "")
                        return s

                    desired_norm = _norm_rate_label(desired_fps_label)

                    for i in range(min(options.count(), 64)):
                        opt = options.nth(i)
                        try:
                            text = (opt.text_content() or "").strip()
                        except Exception:
                            text = ""
                        try:
                            value = opt.get_attribute("value")
                        except Exception:
                            value = None

                        # Match exact label (supports fractional labels like "1/16")
                        if _norm_rate_label(text) == desired_norm:
                            chosen_value = value
                            chosen_label = text
                            break

                    if chosen_label is None:
                        available = []
                        for i in range(min(options.count(), 64)):
                            try:
                                available.append((options.nth(i).text_content() or "").strip())
                            except Exception:
                                pass
                        raise ValueError(
                            f"Frame rate option {desired_fps_label!r} not found in camera dropdown. Available: {available}"
                        )

                    if chosen_value is not None:
                        fps_sel.select_option(value=chosen_value)
                    else:
                        # Fallback if value is missing.
                        fps_sel.select_option(label=chosen_label or desired_fps_label)

                    _log(f"Set frame rate to {desired_fps_label}.")
                except Exception as e:
                    _log(f"Failed selecting frame rate {desired_fps_label}: {e}")
                    _dump_state("framerate_select_failed")
                    context.close()
                    browser.close()
                    return False

            # Save.
            save_btn = None
            for f in _config_frames_for_root(page, config_root_frame):
                try:
                    btn = f.locator('button.btn-save, button[ng-click="save()"]')
                    if btn.count() > 0 and btn.first.is_visible():
                        save_btn = btn.first
                        break
                except Exception:
                    continue
            if save_btn is None:
                try:
                    btn = page.locator('button.btn-save, button[ng-click="save()"]')
                    if btn.count() > 0:
                        save_btn = btn.first
                except Exception:
                    save_btn = None

            if save_btn is None:
                _log("Save button not found on Video/Audio page.")
                _dump_state("video_save_missing")
                context.close()
                browser.close()
                return False

            try:
                save_btn.click()
                _log("Clicked Save.")
            except Exception as e:
                _log(f"Failed to click Save: {e}")
                _dump_state("video_save_click_failed")
                context.close()
                browser.close()
                return False

            # Basic persistence check: reload and confirm selected values.
            try:
                page.wait_for_load_state("networkidle", timeout=min(timeout_ms, 5_000))
            except PlaywrightTimeout:
                pass
            try:
                page.reload(wait_until="domcontentloaded")
            except Exception:
                pass

            def _find_select_any(selector: str) -> Optional[Any]:
                for f in _all_frames(page):
                    try:
                        loc = f.locator(selector)
                        if loc.count() > 0:
                            return loc.first
                    except Exception:
                        continue
                try:
                    loc = page.locator(selector)
                    if loc.count() > 0:
                        return loc.first
                except Exception:
                    pass
                return None

            def _checked_text(select_loc: Any) -> str:
                try:
                    return ((select_loc.locator("option:checked").first.text_content() or "").strip())
                except Exception:
                    return ""

            # After reload, the page is often still on config.asp and the Video/Audio controls may already exist.
            # Try to read them directly first; only re-navigate if needed.
            res_sel2 = (
                _find_select_any('select[ng-model="oVideoParams.resolution"]')
                if desired_resolution_text is not None
                else None
            )
            fps_sel2 = (
                _find_select_any('select[ng-model="oVideoParams.frameRate"], #frameRateSelect')
                if desired_fps_label is not None
                else None
            )

            needs_nav = (desired_resolution_text is not None and res_sel2 is None) or (
                desired_fps_label is not None and fps_sel2 is None
            )

            if needs_nav:
                config_root_frame = gotoConfig(page, timeout_ms, _log, _dump_state)
                if config_root_frame is not None and gotoVideoAudio(page, config_root_frame, timeout_ms, _log, _dump_state):
                    if desired_resolution_text is not None and res_sel2 is None:
                        res_sel2 = _find_select('select[ng-model="oVideoParams.resolution"]')
                    if desired_fps_label is not None and fps_sel2 is None:
                        fps_sel2 = _find_select('select[ng-model="oVideoParams.frameRate"], #frameRateSelect')

            if desired_resolution_text is not None and res_sel2 is not None:
                _log(f"Resolution after reload: {_checked_text(res_sel2)}")
            if desired_fps_label is not None and fps_sel2 is not None:
                _log(f"Frame rate after reload: {_checked_text(fps_sel2)}")

            context.close()
            browser.close()
            return True

    except PlaywrightTimeout as e:
        print(f"[ANNKE] Video/Audio UI timeout for {ip}: {e}")
        return False
    except Exception as e:
        print(f"[ANNKE] Failed to change Video/Audio settings for {ip}: {e}")
        return False


def set_video_resolution(
    ip: str,
    resolution: str,
    headless: bool = True,
    timeout_ms: int = 15_000,
    debug: bool = False,
    debug_dump_dir: str = "tracks",
) -> bool:
    """Set the Video/Audio -> Video -> Resolution.

    Allowed values (exactly these three): 1280*720P, 1920*1080P, 3072*1728
    """

    return _set_video_audio_params(
        ip,
        resolution=resolution,
        frame_rate=None,
        headless=headless,
        timeout_ms=timeout_ms,
        debug=debug,
        debug_dump_dir=debug_dump_dir,
    )


def set_video_framerate(
    ip: str,
    frame_rate: int | str,
    headless: bool = True,
    timeout_ms: int = 15_000,
    debug: bool = False,
    debug_dump_dir: str = "tracks",
) -> bool:
    """Set the Video/Audio -> Video -> Frame Rate.

    Only allows the exact dropdown options:
    1/16, 1/8, 1/4, 1/2, 1, 2, 4, 6, 8, 10, 12, 15, 16, 18, 20, 22, 24.
    """

    return _set_video_audio_params(
        ip,
        resolution=None,
        frame_rate=frame_rate,
        headless=headless,
        timeout_ms=timeout_ms,
        debug=debug,
        debug_dump_dir=debug_dump_dir,
    )

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
