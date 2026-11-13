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

    def project_state(self, project_id: str) -> str:
        state = self.get_project_info(project_id).get("project_state")
        if not isinstance(state, str):
            raise AmcApiError("project_info.project_state is missing")
        return state.upper()

    def upload_videos(self, project_id: str, videos: list[tuple[str, Path]]) -> None:
        """videos: (name AMC should see, local path) in camera order."""
        with ExitStack() as stack:
            files = [("files", (name, stack.enter_context(open(path, "rb")), "video/mp4"))
                     for name, path in videos]
            r = self._http.post(self._url(f"upload_video_files/{project_id}"), files=files,
                                timeout=self.upload_timeout)
        self._check(r, "upload_video_files")

    def post_config(self, project_id: str, settings: dict) -> None:
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

    def calibrate(self, project_id: str, detector_type: str) -> None:
        self._check(self._http.post(self._url(f"calibrate/{project_id}"), json={"detector_type": detector_type},
                                    timeout=self.timeout), "calibrate")

    def calibration_log(self, project_id: str, tail_lines: int = 20) -> str:
        try:
            r = self._http.get(self._url(f"amc/calibrate/{project_id}/log"), timeout=self.timeout)
        except httpx.HTTPError as e:
            return f"(could not fetch the calibration log: {e})"
        if r.status_code >= 400:
            return f"(could not fetch the calibration log: HTTP {r.status_code})"
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

