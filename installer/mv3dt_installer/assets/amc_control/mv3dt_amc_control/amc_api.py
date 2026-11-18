"""Client for AutoMagicCalib's REST API"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any

import httpx


class AmcApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def _detail(r: httpx.Response) -> str:
    try:
        body = r.json()
        if isinstance(body, dict):
            return str(body.get("detail") or body.get("message") or body)[:500]
    except ValueError:
        pass
    return r.text[:500]


def _reply_state(r: httpx.Response, key: str = "rectification_state") -> str:
    """The state a request's reply carries under `key`, upper-cased, or ""."""
    try:
        body = r.json()
    except ValueError:
        return ""
    state = body.get(key) if isinstance(body, dict) else None
    return state.upper() if isinstance(state, str) else ""


class AmcApi:
    def __init__(self, base_url: str, *, timeout: float = 60, upload_timeout: float = 600,
                 client: httpx.Client | None = None):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.upload_timeout = upload_timeout
        self._http = client or httpx.Client()

    def close(self) -> None:
        self._http.close()

    def _url(self, path: str) -> str:
        return f"{self.base}/{path.lstrip('/')}"

    def _check(self, r: httpx.Response, label: str) -> httpx.Response:
        if r.status_code >= 400:
            raise AmcApiError(f"{label} failed: HTTP {r.status_code}: {_detail(r)}", r.status_code)
        return r

    def _json(self, r: httpx.Response, label: str) -> dict:
        try:
            body = r.json()
        except ValueError as e:
            raise AmcApiError(f"{label} returned non-JSON: {r.text[:200]}") from e
        if not isinstance(body, dict):
            raise AmcApiError(f"{label} returned {type(body).__name__}, expected an object")
        return body

    def request(self, method: str, path: str, *, json: Any = None, label: str | None = None) -> dict:
        r = self._http.request(method, self._url(path), json=json, timeout=self.timeout)
        label = label or f"{method} {path}"
        return self._json(self._check(r, label), label)

    def ready(self) -> bool:
        try:
            r = self._http.get(self._url("ready"), timeout=5)
        except httpx.HTTPError:
            return False
        if r.status_code != 200:
            return False
        try:
            return r.json().get("code") in (None, 0, "0")
        except (ValueError, AttributeError):
            return False

    def create_project(self, name: str) -> str:
        r = self._check(self._http.post(self._url("create_project"), data={"project_name": name},
                                        timeout=self.timeout), "create_project")
        pid = self._json(r, "create_project").get("project_id")
        if not isinstance(pid, str) or not pid:
            raise AmcApiError("create_project reply has no project_id")
        return pid

    def get_project_info(self, project_id: str) -> dict:
        r = self._check(self._http.get(self._url(f"get_project_info/{project_id}"), timeout=self.timeout),
                        "get_project_info")
        info = self._json(r, "get_project_info").get("project_info")
        if not isinstance(info, dict):
            raise AmcApiError("get_project_info reply has no project_info object")
        return info

    def calibration_states(self, project_id: str) -> tuple[str, str]:
        """(project_state, amc_state), upper-cased, "" for a missing field.

        AMC 3.3.0 reports a calibration through both: RUNNING in either while it runs, then
        project_state back to READY and amc_state COMPLETED (or ERROR) when it ends.
        """
        info = self.get_project_info(project_id)
        return tuple(v.upper() if isinstance(v := info.get(k), str) else ""
                     for k in ("project_state", "amc_state"))

    def upload_videos(self, project_id: str, videos: list[tuple[str, Path]]) -> None:
        """videos: (name AMC should see, local path) in camera order."""
        with ExitStack() as stack:
            files = [("files", (name, stack.enter_context(open(path, "rb")), "video/mp4"))
                     for name, path in videos]
            r = self._http.post(self._url(f"upload_video_files/{project_id}"), files=files,
                                timeout=self.upload_timeout)
        self._check(r, "upload_video_files")

    def _config_params(self, path: str, label: str, bare_ok: bool = False) -> dict:
        r = self._check(self._http.get(self._url(path), timeout=self.timeout), label)
        body = self._json(r, label)
        params = body.get("config_params", body if bare_ok else None)
        if not isinstance(params, dict):
            raise AmcApiError(f"{label} reply has no config_params object")
        return params

    def get_config(self) -> dict:
        """AMC's one current configuration (ConfigParams); AMC 3.3.0 keeps no per-project copy."""
        return self._config_params("config", "get config")

    def get_default_config(self) -> dict:
        """AMC's factory configuration (layout_px_per_m 24.22 in AMC 3.3.0)."""
        return self._config_params("config/defaults", "get default config", bare_ok=True)

    def post_config(self, project_id: str, settings: dict) -> None:
        """Replace AMC's current configuration. ConfigParams allows no extra keys, so
        `settings` must be a whole object: build it on get_config()."""
        self._check(self._http.post(self._url(f"config/{project_id}"), json=settings, timeout=self.timeout),
                    "config")

    def upload_layout(self, project_id: str, png: bytes) -> None:
        files = {"layout_file": ("layout.png", png, "image/png")}
        self._check(self._http.post(self._url(f"upload_layout/{project_id}"), files=files,
                                    timeout=self.timeout), "upload_layout")

    def upload_alignment(self, project_id: str, path: Path, coord_space: str) -> None:
        with open(path, "rb") as fh:
            files = {"alignment_file": ("alignment_data.json", fh, "application/json")}
            r = self._http.post(self._url(f"upload_alignment/{project_id}"), files=files,
                                params={"coord_space": coord_space}, timeout=self.timeout)
        self._check(r, "upload_alignment")

    def upload_focal_lengths(self, project_id: str, focal_lengths: list[float]) -> None:
        data = {"focal_length": [str(f) for f in focal_lengths]}
        self._check(self._http.post(self._url(f"upload_focal_length/{project_id}"), data=data,
                                    timeout=self.timeout), "upload_focal_length")

    def verify_project(self, project_id: str) -> str:
        r = self._check(self._http.post(self._url(f"verify_project/{project_id}"), timeout=self.timeout),
                        "verify_project")
        state = self._json(r, "verify_project").get("project_state")
        return state.upper() if isinstance(state, str) else ""

    def calibrate(self, project_id: str, detector_type: str) -> str:
        """Start the calibration; returns the reply's project_state, or "" when it has none."""
        return _reply_state(self._check(self._http.post(self._url(f"calibrate/{project_id}"),
                                                        json={"detector_type": detector_type},
                                                        timeout=self.timeout), "calibrate"),
                            "project_state")

    def start_postprocess(self, project_id: str) -> str:
        """Start AMC's Layout Post-processing, which builds the result exports.

        Returns the reply's postprocess_state, or "" when it has none (AMC 3.3.0 replies
        without one, so the caller's lag window always opens).
        """
        return _reply_state(self._check(self._http.post(self._url(f"postprocess/{project_id}"),
                                                        timeout=self.timeout), "postprocess"),
                            "postprocess_state")

    def postprocess_state(self, project_id: str, result_type: str) -> tuple[str, bool]:
        """(postprocess_state upper-cased or "", whether the <result_type> export is ready).

        AMC 3.3.0: READY before post-processing, RUNNING while it runs, COMPLETED with
        amc_export_ready true after it, ERROR or FAILED when it fails.
        """
        info = self.get_project_info(project_id)
        state = info.get("postprocess_state")
        return (state.upper() if isinstance(state, str) else "",
                info.get(f"{result_type}_export_ready") is True)

    def stop_calibration(self, project_id: str) -> None:
        self._check(self._http.post(self._url(f"stop_calibration/{project_id}"), timeout=self.timeout),
                    "stop_calibration")

    def delete_project(self, project_id: str) -> bool:
        """Delete the AMC project. False when AMC has no such project (its 404), which counts as deleted."""
        r = self._http.delete(self._url(f"delete_project/{project_id}"), timeout=self.timeout)
        if r.status_code == 404:
            return False
        self._check(r, "delete_project")
        return True

    def calibration_log(self, project_id: str, tail_lines: int = 20) -> str:
        try:
            r = self._http.get(self._url(f"amc/calibrate/{project_id}/log"), timeout=self.timeout)
        except httpx.HTTPError as e:
            return f"(could not fetch the calibration log: {e})"
        if r.status_code >= 400:
            return f"(could not fetch the calibration log: HTTP {r.status_code})"
        return "\n".join(r.text.splitlines()[-tail_lines:])

    def rectification_state(self, project_id: str) -> str:
        r = self._check(self._http.get(self._url(f"rectification/{project_id}"), timeout=self.timeout),
                        "rectification state")
        state = self._json(r, "rectification state").get("rectification_state")
        if not isinstance(state, str) or not state:
            raise AmcApiError("rectification state reply has no rectification_state")
        return state.upper()

    def start_rectification(self, project_id: str) -> str:
        """Start the estimate; returns the reply's rectification_state, or "" when it has none."""
        return _reply_state(self._check(self._http.post(self._url(f"rectification/{project_id}"),
                                                        timeout=self.timeout), "start rectification"))

    def rectification_cameras(self, project_id: str) -> list[dict]:
        r = self._check(self._http.get(self._url(f"rectification/{project_id}/cameras"), timeout=self.timeout),
                        "rectification cameras")
        cameras = self._json(r, "rectification cameras").get("cameras")
        if not isinstance(cameras, list):
            raise AmcApiError("rectification cameras reply has no cameras list")
        return [c for c in cameras if isinstance(c, dict)]

    def commit_rectification(self, project_id: str, cameras: dict[str, dict]) -> str:
        """cameras: {camera_id: {"model", "k1", "k2"}}; AMC then generates the rectified videos.

        Returns the reply's rectification_state, or "" when it has none.
        """
        return _reply_state(self._check(self._http.post(self._url(f"rectification/{project_id}/manual"),
                                                        json={"cameras": cameras}, timeout=self.timeout),
                                        "commit rectification"))

    def stage_linear_media(self, project_id: str) -> str:
        """Returns the reply's rectification_state, or "" when it has none."""
        return _reply_state(self._check(self._http.post(self._url(f"linear_media/{project_id}"),
                                                        timeout=self.timeout), "stage linear media"))

    def rectification_log(self, project_id: str, tail_lines: int = 10) -> str:
        try:
            r = self._http.get(self._url(f"rectification/{project_id}/log"), timeout=self.timeout)
        except httpx.HTTPError as e:
            return f"(could not fetch the rectification log: {e})"
        if r.status_code >= 400:
            return f"(could not fetch the rectification log: HTTP {r.status_code})"
        return "\n".join(r.text.splitlines()[-tail_lines:])

    def evaluation_statistics(self, project_id: str) -> dict | None:
        r = self._http.get(self._url(f"result/{project_id}/evaluation_statistics"), timeout=self.timeout)
        if r.status_code != 200:
            return None
        try:
            body = r.json()
        except ValueError:
            return None
        return body if isinstance(body, dict) else None

    def mv3dt_result(self, project_id: str, result_type: str) -> bytes:
        r = self._http.get(self._url(f"result/{project_id}/mv3dt_result"), params={"result_type": result_type},
                           timeout=self.upload_timeout)
        return self._check(r, "mv3dt_result").content

    def openapi(self) -> str:
        root = self.base.rsplit("/v1", 1)[0]
        r = self._http.get(f"{root}/openapi.yaml", timeout=self.timeout)
        return self._check(r, "openapi.yaml").text


def default_base_url(ms_port: int | None) -> str:
    if not ms_port:
        raise AmcApiError("AUTO_MAGIC_CALIB_MS_PORT is not set in installer.conf")
    return f"http://127.0.0.1:{ms_port}/v1"

