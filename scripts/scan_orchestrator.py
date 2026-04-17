"""Poll Web for pending lidar scans, run stub/real scan command, upload .xyz via cloud_storage_media.

Uses API_KEY and ENDPOINT from /opt/p2bp/camera/config/agent.env (same as heartbeat).

Reads optional lidar settings from config.json (default /opt/p2bp/camera/config/config.json;
override with P2BP_CONFIG_PATH). Section LidarScan:
  Enabled, BeginScanning, PollIntervalSeconds, ScanCmd, OutputXyzTemplate, RemotePathTemplate
Placeholders for templates: {scan_id}, {project_id}, {device_id}.
Tunables come from config only (no P2BP_SCAN_* env overrides).

Optional: systemd may set P2BP_LIDAR_SCAN_NONINTERACTIVE=1 for LidarScanV1 (passed through child env).

Default ScanCmd is /opt/p2bp/camera/scripts/run_lidar_on_pi.sh (Pi bridge).
Default BeginScanning is false; orchestrator sits idle until server sets it true.

Service runs continuously under systemd and triggers scans from config.json LidarScan flags.
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
import scripts.config_io as config_io

DEFAULT_ENV_PATH = "/opt/p2bp/camera/config/agent.env"
DEFAULT_CONFIG_PATH = "/opt/p2bp/camera/config/config.json"
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


def _config_path() -> str:
    return (os.getenv("P2BP_CONFIG_PATH") or DEFAULT_CONFIG_PATH).strip() or DEFAULT_CONFIG_PATH


def _load_json_config() -> Optional[Dict[str, Any]]:
    return config_io.load_local_config(_config_path())


def _lidar_section(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        return {}
    ls = cfg.get("LidarScan")
    if ls is None:
        ls = cfg.get("lidarScan")
    return ls if isinstance(ls, dict) else {}


def _merge_poll_interval_s(cfg: Optional[Dict[str, Any]]) -> float:
    ls = _lidar_section(cfg)
    raw = ls.get("PollIntervalSeconds", ls.get("pollIntervalSeconds"))
    if raw is not None:
        try:
            return max(3.0, float(raw))
        except (TypeError, ValueError):
            pass
    return DEFAULT_POLL_S


def _as_bool(raw: Any, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return raw != 0
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in {"1", "true", "yes", "on", "enabled"}:
            return True
        if s in {"0", "false", "no", "off", "disabled"}:
            return False
    return default


def _lidar_enabled(ls: Dict[str, Any]) -> bool:
    return _as_bool(ls.get("Enabled", ls.get("enabled")), True)


def _lidar_begin_scanning(ls: Dict[str, Any]) -> bool:
    return _as_bool(ls.get("BeginScanning", ls.get("beginScanning")), False)


def _lidar_str(ls: Dict[str, Any], config_keys: Tuple[str, ...], default: str) -> str:
    """Config-only string (no env); used for path templates."""
    for ck in config_keys:
        raw = ls.get(ck)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return default


def _format_path_template(template: str, *, scan_id: str, project_id: str, device_id: str) -> str:
    try:
        return template.format(scan_id=scan_id, project_id=project_id, device_id=device_id)
    except Exception:
        return template


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

    cfg = _load_json_config()
    ls = _lidar_section(cfg)

    cmd = _lidar_str(
        ls,
        ("ScanCmd", "scanCmd"),
        "/opt/p2bp/camera/scripts/run_lidar_on_pi.sh",
    )
    out_tmpl = _lidar_str(
        ls,
        ("OutputXyzTemplate", "outputXyzTemplate"),
        "/opt/p2bp/camera/run/scan_{scan_id}.xyz",
    )
    xyz_path = _format_path_template(
        out_tmpl, scan_id=scan_id, project_id=project_id, device_id=device_id
    )

    child_env = os.environ.copy()
    child_env["P2BP_SCAN_OUTPUT_XYZ"] = xyz_path
    child_env.setdefault("P2BP_SCAN_WORKDIR", str(Path(xyz_path).resolve().parent))
    if "P2BP_LIDAR_SCAN_NONINTERACTIVE" not in child_env:
        child_env["P2BP_LIDAR_SCAN_NONINTERACTIVE"] = "1"

    try:
        _run_scan_cmd(cmd, child_env=child_env)
        local_xyz = _ensure_xyz_file(xyz_path)
        remote_tmpl = _lidar_str(
            ls,
            ("RemotePathTemplate", "remotePathTemplate"),
            "/vision/lidar-scans/{project_id}/{device_id}/{scan_id}.xyz",
        )
        remote = _format_path_template(
            remote_tmpl, scan_id=scan_id, project_id=project_id, device_id=device_id
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
    interval = _merge_poll_interval_s(_load_json_config())
    cfg_path = Path(_config_path())
    logger.info("Scan orchestrator started (poll=%ss config=%s)", interval, cfg_path)

    last_begin_scanning: Optional[bool] = None
    last_cfg_mtime: Optional[float] = None
    while True:
        try:
            cfg = _load_json_config()
            ls = _lidar_section(cfg)
            interval = _merge_poll_interval_s(cfg)

            if not _lidar_enabled(ls):
                last_begin_scanning = False
                time.sleep(interval)
                continue

            begin = _lidar_begin_scanning(ls)
            mtime: Optional[float] = None
            if cfg_path.exists():
                try:
                    mtime = cfg_path.stat().st_mtime
                except OSError:
                    mtime = None

            should_trigger = False
            if begin:
                if last_begin_scanning is False or last_begin_scanning is None:
                    should_trigger = True
                elif mtime is not None and last_cfg_mtime is not None and mtime != last_cfg_mtime:
                    should_trigger = True

            last_begin_scanning = begin
            if mtime is not None:
                last_cfg_mtime = mtime

            if should_trigger:
                job = _get_next_pending(api_key, endpoint)
                if job:
                    process_one_scan(api_key, endpoint, job)
                else:
                    logger.info("BeginScanning true but no pending scan for this device")

            time.sleep(interval)
        except requests.HTTPError as e:
            logger.warning("HTTP error: %s", e)
            time.sleep(interval)
        except Exception as e:
            logger.exception("Orchestrator loop error: %s", e)
            time.sleep(interval)


if __name__ == "__main__":
    main()
