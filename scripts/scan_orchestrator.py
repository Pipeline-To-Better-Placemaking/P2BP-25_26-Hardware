"""Poll Web for pending lidar scans, run stub/real scan command, upload .xyz via cloud_storage_media.

Uses API_KEY and ENDPOINT from /opt/p2bp/camera/config/agent.env (same as heartbeat).

Env:
  P2BP_SCAN_POLL_INTERVAL_S — seconds between polls when idle (default 10)
  P2BP_SCAN_CMD — shell command to run before upload (default: python3 -u scripts/LidarScanV1.py)
  P2BP_SCAN_OUTPUT_XYZ — passed to the child: exact .xyz path to write (orchestrator sets per job)
  P2BP_LIDAR_SCAN_NONINTERACTIVE — set to 1 for LidarScanV1 (default 1 when child env is built)

Pi + Jetson bridge (scan on Pi, upload from Jetson): set P2BP_SCAN_CMD to scripts/run_lidar_on_pi.sh
  (installed path /opt/p2bp/camera/scripts/run_lidar_on_pi.sh). run_lidar_on_pi.sh defaults to
  pi@192.168.28.2; set P2BP_PI_SSH to override. Optional: P2BP_PI_REMOTE_SCRIPT, P2BP_PI_REMOTE_XYZ.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import dotenv
import requests

import scripts.cloud_storage_media as cloud_storage_media

DEFAULT_ENV_PATH = "/opt/p2bp/camera/config/agent.env"
DEFAULT_POLL_S = 10.0


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("p2bp.scan_orchestrator")
    if logger.handlers:
        return logger
    level_name = os.getenv("P2BP_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)
    logger.propagate = False
    return logger


logger = _build_logger()


def _poll_interval() -> float:
    try:
        return max(3.0, float(os.getenv("P2BP_SCAN_POLL_INTERVAL_S", str(DEFAULT_POLL_S))))
    except (TypeError, ValueError):
        return DEFAULT_POLL_S


def _load_api() -> Tuple[str, str]:
    return cloud_storage_media.load_env(DEFAULT_ENV_PATH)


def _get_next_pending(api_key: str, endpoint: str) -> Optional[Dict[str, Any]]:
    url = cloud_storage_media._join_url(endpoint, "/api/scan/device/next-pending")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    r = requests.get(url, headers=headers, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, dict) else None


def _patch_status(
    api_key: str,
    endpoint: str,
    scan_id: str,
    *,
    status: str,
    obj_url: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    url = cloud_storage_media._join_url(endpoint, f"/api/scan/device/{scan_id}/status")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Web API uses PropertyNamingPolicy null (PascalCase JSON).
    body: Dict[str, Any] = {"Status": status}
    if obj_url is not None:
        body["ObjUrl"] = obj_url
    if error is not None:
        body["Error"] = error
    r = requests.patch(url, headers=headers, json=body, timeout=30)
    r.raise_for_status()


def _ensure_xyz_file(path: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.is_file():
        p.write_text("0 0 0\n", encoding="utf-8")
        logger.info("Created stub XYZ at %s", path)
    return str(p.resolve())


def _gcs_object_path_from_remote(remote_path: str) -> str:
    """Full object key for /api/files/request-download (matches upload remote_path, no leading slash)."""
    return remote_path.replace("\\", "/").strip().strip("/")


def _run_scan_cmd(cmd: str, *, child_env: dict[str, str], timeout_s: float = 3600.0) -> None:
    if not cmd.strip():
        return
    logger.info("Running P2BP_SCAN_CMD")
    r = subprocess.run(
        cmd,
        shell=True,
        env=child_env,
        timeout=timeout_s,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "")[:2000]
        raise RuntimeError(f"scan command failed ({r.returncode}): {err}")
    logger.info("P2BP_SCAN_CMD finished OK")


def process_one_scan(api_key: str, endpoint: str, job: Dict[str, Any]) -> None:
    scan_id = job.get("scanId") or job.get("ScanId")
    project_id = job.get("projectId") or job.get("ProjectId")
    device_id = job.get("deviceId") or job.get("DeviceId")
    if not scan_id or not project_id or not device_id:
        raise RuntimeError(f"Invalid next-pending payload: {job!r}")

    logger.info("Claiming scan %s (project=%s device=%s)", scan_id, project_id, device_id)
    _patch_status(api_key, endpoint, scan_id, status="running")

    cmd = os.getenv("P2BP_SCAN_CMD", "python3 -u scripts/LidarScanV1.py")
    xyz_path = os.getenv(
        "P2BP_SCAN_OUTPUT_XYZ",
        f"/opt/p2bp/camera/run/scan_{scan_id}.xyz",
    )

    child_env = os.environ.copy()
    child_env["P2BP_SCAN_OUTPUT_XYZ"] = xyz_path
    child_env.setdefault("P2BP_SCAN_WORKDIR", str(Path(xyz_path).resolve().parent))
    if "P2BP_LIDAR_SCAN_NONINTERACTIVE" not in child_env:
        child_env["P2BP_LIDAR_SCAN_NONINTERACTIVE"] = "1"

    try:
        _run_scan_cmd(cmd, child_env=child_env)
        local_xyz = _ensure_xyz_file(xyz_path)
        remote = os.getenv(
            "P2BP_SCAN_REMOTE_PATH",
            f"/vision/lidar-scans/{project_id}/{device_id}/{scan_id}.xyz",
        )
        logger.info("Uploading %s -> %s", local_xyz, remote)
        cloud_storage_media.upload(local_xyz, remote, api_key=api_key, endpoint=endpoint)
        object_path = _gcs_object_path_from_remote(remote)
        dl = cloud_storage_media.request_download_url(api_key, endpoint, object_path)
        obj_url = dl.SignedUrl
        _patch_status(api_key, endpoint, scan_id, status="complete", obj_url=obj_url)
        logger.info("Scan %s complete, ObjUrl (https signed) obtained for host parse", scan_id)
    except Exception as e:
        logger.exception("Scan %s failed", scan_id)
        try:
            _patch_status(api_key, endpoint, scan_id, status="error", error=str(e)[:2000])
        except Exception:
            logger.exception("Failed to PATCH error status")


def main() -> None:
    dotenv.load_dotenv(DEFAULT_ENV_PATH)
    api_key, endpoint = _load_api()
    interval = _poll_interval()
    logger.info("Scan orchestrator started (poll=%ss)", interval)

    while True:
        try:
            job = _get_next_pending(api_key, endpoint)
            if job:
                process_one_scan(api_key, endpoint, job)
            else:
                time.sleep(interval)
        except requests.HTTPError as e:
            logger.warning("HTTP error: %s", e)
            time.sleep(interval)
        except Exception as e:
            logger.exception("Orchestrator loop error: %s", e)
            time.sleep(interval)


if __name__ == "__main__":
    main()
