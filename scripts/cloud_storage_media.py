from __future__ import annotations

import os
import json
import posixpath
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import dotenv
import logging
import requests

from scripts.json_models.cloud_storage import (
    ConfirmUploadedMediaDto,
    DownloadUrlResponseDto,
    MediaRecordResponseDto,
    RequestDownloadUrlDto,
    RequestUploadUrlDto,
    UploadUrlResponseDto,
)


DEFAULT_ENV_PATH = "/opt/p2bp/camera/config/agent.env"


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("p2bp.cloud_storage")
    if logger.handlers:
        return logger

    level_name = os.getenv("P2BP_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = _build_logger()


def _redact_url(url: str) -> str:
    # Signed URLs include secrets in query params; never log those.
    if not url:
        return ""
    q = url.find("?")
    return url if q < 0 else url[:q] + "?<redacted>"


def _truncate(text: str, limit: int = 2000) -> str:
    if text is None:
        return "<empty>"
    if not str(text).strip():
        return "<empty>"
    if len(text) <= limit:
        return text
    return text[:limit] + f"… <truncated {len(text) - limit} chars>"


def _log_http_failure(prefix: str, method: str, url: str, response: Optional[requests.Response]) -> None:
    if response is None:
        logger.warning("%s failed: %s %s (no response)", prefix, method, _redact_url(url))
        return

    body: str = ""
    try:
        body = response.text or ""
    except Exception:
        body = ""

    logger.warning(
        "%s failed: %s %s -> HTTP %s; body=%s",
        prefix,
        method,
        _redact_url(url),
        response.status_code,
        _truncate(body),
    )


def _log_request_context(prefix: str, method: str, url: str, json_body: Optional[Dict[str, Any]], attempt: Optional[int]) -> None:
    attempt_part = f" attempt={attempt}" if attempt is not None else ""
    if json_body is None:
        payload_part = "<none>"
    else:
        try:
            # Use real JSON encoding so logs show double quotes.
            payload_part = _truncate(
                json.dumps(json_body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
        except Exception:
            payload_part = _truncate(str(json_body))
    logger.warning(
        "%s context:%s %s %s payload=%s",
        prefix,
        attempt_part,
        method,
        _redact_url(url),
        payload_part,
    )


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
                _log_request_context("Backend request", method, url, json_body, attempt)
                _log_http_failure("Backend request", method, url, r)
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            try:
                r.raise_for_status()
            except requests.HTTPError:
                _log_request_context("Backend request", method, url, json_body, attempt)
                _log_http_failure("Backend request", method, url, r)
                raise

            try:
                return r.json()
            except ValueError:
                _log_request_context("Backend JSON parse", method, url, json_body, attempt)
                _log_http_failure("Backend JSON parse", method, url, r)
                raise
        except (requests.RequestException, ValueError) as e:
            last_exc = e

            # Best-effort logging for errors that didn't produce a response.
            resp = getattr(e, "response", None)
            if isinstance(resp, requests.Response):
                _log_request_context("Backend request", method, url, json_body, attempt)
                _log_http_failure("Backend request", method, url, resp)
            else:
                logger.warning(
                    "Backend request exception: %s %s (%s)",
                    method,
                    _redact_url(url),
                    str(e)[:500],
                )

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
        logger.warning("Backend returned empty SignedUrl for %s: %s", remote_path, _truncate(str(data)))
        raise RuntimeError("Backend returned empty SignedUrl")
    return resp


def request_download_url(
    api_key: str,
    endpoint: str,
    object_path: str,
    timeout_s: float = 10.0,
) -> DownloadUrlResponseDto:
    """POST /api/files/request-download — full GCS object path (no leading slash), same as after upload."""
    normalized = object_path.replace("\\", "/").strip().strip("/")
    if not normalized:
        raise ValueError("object_path is required for request-download")

    dto = RequestDownloadUrlDto(PathFromRoot=normalized)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = _join_url(endpoint, "/api/files/request-download")
    max_attempts = max(1, _env_int("P2BP_CLOUDSTORAGE_REQUEST_DOWNLOAD_RETRIES", 5))
    data = _request_json_with_retries(
        method="POST",
        url=url,
        headers=headers,
        json_body=dto.to_dict(),
        timeout_s=timeout_s,
        max_attempts=max_attempts,
    )
    resp = DownloadUrlResponseDto.from_dict(data if isinstance(data, dict) else {})
    if not resp.SignedUrl:
        logger.warning(
            "Backend returned empty download SignedUrl for %s: %s",
            normalized,
            _truncate(str(data)),
        )
        raise RuntimeError("Backend returned empty download SignedUrl")
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
        try:
            r.raise_for_status()
        except requests.HTTPError:
            _log_http_failure("Signed upload", "PUT", signed_url, r)
            raise


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

    max_retries = _env_int("P2BP_CLOUDSTORAGE_CONFIRM_MAX_RETRIES", 8)
    attempt = 0
    last_exc: Optional[BaseException] = None
    while True:
        attempt += 1
        try:
            r = requests.post(url, headers=headers, json=dto.to_dict(), timeout=timeout_s)
            # Only retry on transient-ish failures.
            if r.status_code in {429} or r.status_code >= 500:
                _log_request_context("Confirm upload", "POST", url, dto.to_dict(), attempt)
                _log_http_failure("Confirm upload", "POST", url, r)
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)

            try:
                r.raise_for_status()
            except requests.HTTPError:
                _log_request_context("Confirm upload", "POST", url, dto.to_dict(), attempt)
                _log_http_failure("Confirm upload", "POST", url, r)
                raise

            try:
                data = r.json()
            except ValueError:
                _log_request_context("Confirm upload JSON parse", "POST", url, dto.to_dict(), attempt)
                _log_http_failure("Confirm upload JSON parse", "POST", url, r)
                raise
            media = MediaRecordResponseDto.from_dict(data if isinstance(data, dict) else {})
            return media
        except (requests.RequestException, ValueError) as e:
            last_exc = e

            # If it's a non-retryable HTTP code, fail fast.
            if isinstance(e, requests.HTTPError) and getattr(e, "response", None) is not None:
                status = e.response.status_code
                if status < 500 and status != 429:
                    raise RuntimeError(f"Confirm upload failed (HTTP {status})") from e

            if attempt >= max_retries:
                break

            time.sleep(_sleep_backoff_seconds(attempt, base=1.0, cap=60.0))

    raise RuntimeError(f"Confirm upload failed after {max_retries} attempts") from last_exc


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
