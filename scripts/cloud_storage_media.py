from __future__ import annotations

import os
import posixpath
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import dotenv
import requests

from scripts.json_models.cloud_storage import (
    ConfirmUploadedMediaDto,
    MediaRecordResponseDto,
    RequestUploadUrlDto,
    UploadUrlResponseDto,
)


DEFAULT_ENV_PATH = "/opt/p2bp/camera/config/agent.env"


def _normalize_endpoint(endpoint: str) -> str:
    e = (endpoint or "").strip()
    while e.endswith("/"):
        e = e[:-1]
    # Support legacy/pasted endpoints like https://host/api
    if e.lower().endswith("/api"):
        e = e[:-4]
        while e.endswith("/"):
            e = e[:-1]
    return e


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _sleep_backoff_seconds(attempt: int, base: float = 0.5, cap: float = 30.0) -> float:
    # Exponential backoff with jitter.
    exp = min(cap, base * (2 ** max(0, attempt - 1)))
    jitter = random.uniform(0.0, exp * 0.25)
    return exp + jitter


def _join_url(base: str, path: str) -> str:
    base = (base or "").rstrip("/")
    path = (path or "")
    if not path.startswith("/"):
        path = "/" + path
    return base + path


def load_env(env_path: str = DEFAULT_ENV_PATH) -> Tuple[str, str]:
    dotenv.load_dotenv(env_path)
    api_key = os.getenv("API_KEY")
    endpoint = os.getenv("ENDPOINT")

    if not api_key:
        raise RuntimeError("Missing API_KEY")
    if not endpoint:
        raise RuntimeError("Missing ENDPOINT")
    endpoint_norm = _normalize_endpoint(endpoint)
    if not endpoint_norm:
        raise RuntimeError("ENDPOINT is invalid")
    return api_key, endpoint_norm


def _parse_remote_path(remote_path: str) -> Tuple[str, str, str]:
    """Return (path_from_root_dir, file_name_no_ext, extension_no_dot)."""
    if not remote_path:
        raise ValueError("remote_path is required")

    normalized = remote_path.replace("\\", "/").strip()
    if not normalized.startswith("/"):
        normalized = "/" + normalized

    dir_path, filename = posixpath.split(normalized)
    base, ext = os.path.splitext(filename)
    ext = ext.lstrip(".")

    if not dir_path:
        dir_path = "/"
    if not base or not ext:
        raise ValueError(f"remote_path must include filename and extension: {remote_path!r}")

    return dir_path, base, ext


@dataclass(frozen=True)
class UploadResult:
    local_path: str
    remote_path: str
    media: Optional[MediaRecordResponseDto]


def _request_json_with_retries(
    method: str,
    url: str,
    headers: Dict[str, str],
    json_body: Dict[str, Any],
    timeout_s: float,
    max_attempts: int,
) -> Dict[str, Any]:
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.request(method, url, headers=headers, json=json_body, timeout=timeout_s)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            last_exc = e
            if attempt >= max_attempts:
                break
            time.sleep(_sleep_backoff_seconds(attempt))
    raise RuntimeError(f"Request failed after {max_attempts} attempts: {url}") from last_exc


def request_upload_url(
    api_key: str,
    endpoint: str,
    remote_path: str,
    size_bytes: int,
    timeout_s: float = 10.0,
) -> UploadUrlResponseDto:
    path_from_root, file_name, extension = _parse_remote_path(remote_path)

    dto = RequestUploadUrlDto(
        PathFromRoot=path_from_root,
        FileName=file_name,
        Extension=extension,
        SizeBytes=int(size_bytes),
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    url = _join_url(endpoint, "/api/files/request-upload")
    max_attempts = max(1, _env_int("P2BP_CLOUDSTORAGE_REQUEST_UPLOAD_RETRIES", 5))
    data = _request_json_with_retries(
        method="POST",
        url=url,
        headers=headers,
        json_body=dto.to_dict(),
        timeout_s=timeout_s,
        max_attempts=max_attempts,
    )
    resp = UploadUrlResponseDto.from_dict(data if isinstance(data, dict) else {})
    if not resp.SignedUrl:
        raise RuntimeError("Backend returned empty SignedUrl")
    return resp


def upload_to_signed_url(
    local_path: str,
    signed_url: str,
    size_bytes: int,
    timeout_connect_s: float = 10.0,
    timeout_read_s: float = 300.0,
) -> None:
    with open(local_path, "rb") as f:
        headers = {
            "Content-Length": str(int(size_bytes)),
            # Signed URLs typically require no auth; keep headers minimal.
            # curl "-T" defaults to application/octet-stream.
            "Content-Type": "application/octet-stream",
        }
        r = requests.put(signed_url, data=f, headers=headers, timeout=(timeout_connect_s, timeout_read_s))
        r.raise_for_status()


def confirm_upload(
    api_key: str,
    endpoint: str,
    remote_path: str,
    timeout_s: float = 10.0,
) -> MediaRecordResponseDto:
    path_from_root, file_name, extension = _parse_remote_path(remote_path)
    dto = ConfirmUploadedMediaDto(PathFromRoot=path_from_root, FileName=file_name, Extension=extension)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = _join_url(endpoint, "/api/files/confirm-upload")

    max_retries = _env_int("P2BP_CLOUDSTORAGE_CONFIRM_MAX_RETRIES", 0)  # 0 == infinite
    attempt = 0
    last_exc: Optional[BaseException] = None
    while True:
        attempt += 1
        try:
            r = requests.post(url, headers=headers, json=dto.to_dict(), timeout=timeout_s)
            # Only retry on transient-ish failures.
            if r.status_code in {429} or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            data = r.json()
            media = MediaRecordResponseDto.from_dict(data if isinstance(data, dict) else {})
            return media
        except (requests.RequestException, ValueError) as e:
            last_exc = e

            # If it's a non-retryable HTTP code, fail fast.
            if isinstance(e, requests.HTTPError) and getattr(e, "response", None) is not None:
                status = e.response.status_code
                if status < 500 and status != 429:
                    raise RuntimeError(f"Confirm upload failed (HTTP {status})") from e

            if max_retries and attempt > max_retries:
                break

            time.sleep(_sleep_backoff_seconds(attempt, base=1.0, cap=60.0))

    raise RuntimeError("Confirm upload failed after retries") from last_exc


def upload(
    local_path: str,
    remote_path: str,
    *,
    api_key: Optional[str] = None,
    endpoint: Optional[str] = None,
    env_path: str = DEFAULT_ENV_PATH,
) -> UploadResult:
    """Upload a local file to Cloud Storage via backend-signed URL.

    Flow:
      - POST /api/files/request-upload
      - PUT to SignedUrl
      - POST /api/files/confirm-upload (with retries)
    """

    if not os.path.isfile(local_path):
        raise FileNotFoundError(local_path)

    size_bytes = os.path.getsize(local_path)
    if size_bytes <= 0:
        raise RuntimeError(f"Refusing to upload empty file: {local_path}")

    if api_key is None or endpoint is None:
        api_key_loaded, endpoint_loaded = load_env(env_path)
        api_key = api_key or api_key_loaded
        endpoint = endpoint or endpoint_loaded

    upload_url = request_upload_url(api_key, endpoint, remote_path, size_bytes)
    upload_to_signed_url(local_path, upload_url.SignedUrl, size_bytes)
    media = confirm_upload(api_key, endpoint, remote_path)

    return UploadResult(local_path=local_path, remote_path=remote_path, media=media)
