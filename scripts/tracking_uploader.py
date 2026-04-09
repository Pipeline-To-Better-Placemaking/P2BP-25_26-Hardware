from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from scripts.cloud_storage_media import upload

import logging


_CONFIG_PATH = os.getenv("P2BP_CONFIG_PATH", "/opt/p2bp/camera/config/config.json")


def _load_project_id() -> Optional[str]:
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f).get("ProjectId") or None
    except Exception:
        return None


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("p2bp.tracking_uploader")
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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _write_state_atomic(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _should_upload(file_path: str, min_age_s: float) -> bool:
    try:
        st = os.stat(file_path)
    except FileNotFoundError:
        return False
    if st.st_size <= 0:
        return False
    age = time.time() - st.st_mtime
    return age >= min_age_s


def _file_fingerprint(file_path: str) -> Optional[Dict[str, float]]:
    try:
        st = os.stat(file_path)
    except FileNotFoundError:
        return None
    if st.st_size <= 0:
        return None
    return {
        "size": float(st.st_size),
        "mtime": float(st.st_mtime),
    }


def _state_entry(state: Dict[str, Any], fname: str) -> Optional[Dict[str, Any]]:
    raw = state.get(fname)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (int, float)):
        # Back-compat: old format stored just an uploaded_at timestamp.
        return {"uploaded_at": float(raw)}
    return None


def _remote_path(remote_dir: str, filename: str) -> str:
    remote_dir_norm = remote_dir.replace("\\", "/").rstrip("/")
    if not remote_dir_norm.startswith("/"):
        remote_dir_norm = "/" + remote_dir_norm
    return f"{remote_dir_norm}/{filename}"


def _in_cooldown(entry: Optional[Dict[str, Any]], max_attempts: int, cooldown_s: float) -> bool:
    """Return True if this file has hit the failure threshold and is still within its cooldown window."""
    if entry is None:
        return False
    fails = int(entry.get("failed_attempts") or 0)
    if fails < max_attempts:
        return False
    last_failed = entry.get("last_failed_at")
    if last_failed is None:
        return False
    return (time.time() - float(last_failed)) < cooldown_s


def main() -> None:
    upload_dir = os.getenv("P2BP_TRACKING_UPLOAD_DIR", "/opt/p2bp/camera/tracks")
    project_id = _load_project_id()
    if project_id is None:
        logger.warning("ProjectId missing from config — tracking uploads will use fallback path /vision/tracks-raw")
    _default_remote = f"/vision/tracks-raw/{project_id}" if project_id else "/vision/tracks-raw"
    remote_dir = os.getenv("P2BP_TRACKING_UPLOAD_REMOTE_DIR", _default_remote)
    state_path = os.getenv("P2BP_TRACKING_UPLOAD_STATE_PATH", os.path.join(upload_dir, ".uploaded_state.json"))

    scan_interval_s = max(5.0, _env_float("P2BP_TRACKING_UPLOAD_SCAN_INTERVAL_S", 240.0))
    min_age_s = max(0.0, _env_float("P2BP_TRACKING_UPLOAD_MIN_AGE_S", 0.0))
    max_attempts = max(1, int(_env_float("P2BP_TRACKING_UPLOAD_MAX_ATTEMPTS", 5)))
    cooldown_s = max(0.0, _env_float("P2BP_TRACKING_UPLOAD_COOLDOWN_S", 1800.0))

    logger.info("Tracking uploader started (dir=%s remote=%s)", upload_dir, remote_dir)

    # State file format:
    #   {
    #     "file.jsonl": {
    #       "size": <bytes>, "mtime": <unix>, "uploaded_at": <unix>,
    #       "failed_attempts": <int>, "last_failed_at": <unix|null>
    #     }
    #   }
    # Back-compat: older versions stored { "file.jsonl": <uploaded_at> }.
    state: Dict[str, Any] = {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            state = data
    except FileNotFoundError:
        state = {}
    except Exception:
        logger.warning("Failed to read uploader state; starting fresh")
        state = {}

    while True:
        try:
            if not os.path.isdir(upload_dir):
                logger.debug("Upload dir missing: %s", upload_dir)
                time.sleep(scan_interval_s)
                continue

            files = [
                f
                for f in os.listdir(upload_dir)
                if f.endswith(".jsonl") and os.path.isfile(os.path.join(upload_dir, f))
            ]

            files.sort()

            uploaded_any = False
            for fname in files:
                local_path = os.path.join(upload_dir, fname)
                if not _should_upload(local_path, min_age_s):
                    continue

                fp = _file_fingerprint(local_path)
                if fp is None:
                    continue

                entry = _state_entry(state, fname)

                if entry is not None and "size" not in entry and "mtime" not in entry:
                    # Hydrate old-format entry; don't force re-upload.
                    entry = {
                        **entry,
                        "size": fp["size"],
                        "mtime": fp["mtime"],
                    }
                    state[fname] = entry
                    _write_state_atomic(state_path, state)
                    continue

                if _in_cooldown(entry, max_attempts, cooldown_s):
                    last_failed = float((entry or {}).get("last_failed_at") or 0)
                    retry_at = time.strftime("%H:%M:%S", time.localtime(last_failed + cooldown_s))
                    logger.debug("Skipping %s — in cooldown until %s", fname, retry_at)
                    continue

                needs_upload = entry is None
                if entry is not None:
                    try:
                        prev_size = float(entry.get("size"))
                        prev_mtime = float(entry.get("mtime"))
                        # Re-upload if the file content likely changed (grew or was rewritten).
                        if fp["size"] != prev_size or fp["mtime"] > prev_mtime + 1e-6:
                            needs_upload = True
                    except (TypeError, ValueError):
                        needs_upload = True

                if not needs_upload:
                    continue

                remote_path = _remote_path(remote_dir, fname)
                logger.info("Uploading %s -> %s", local_path, remote_path)

                try:
                    result = upload(local_path, remote_path)
                    state[fname] = {
                        "size": fp["size"],
                        "mtime": fp["mtime"],
                        "uploaded_at": time.time(),
                        "failed_attempts": 0,
                        "last_failed_at": None,
                    }
                    _write_state_atomic(state_path, state)
                    logger.info("Upload confirmed: %s (media_id=%s)", fname, result.media.Id if result.media else "")
                    uploaded_any = True
                except Exception as e:
                    fails = int((entry or {}).get("failed_attempts") or 0) + 1
                    state[fname] = {
                        **({} if entry is None else entry),
                        "size": fp["size"],
                        "mtime": fp["mtime"],
                        "failed_attempts": fails,
                        "last_failed_at": time.time(),
                    }
                    _write_state_atomic(state_path, state)
                    if fails >= max_attempts:
                        retry_at = time.strftime("%H:%M:%S", time.localtime(time.time() + cooldown_s))
                        logger.warning(
                            "Upload failed %d/%d times for %s; cooling down until %s. Error: %s",
                            fails, max_attempts, fname, retry_at, e,
                        )
                    else:
                        logger.warning("Upload failed for %s (attempt %d/%d): %s", fname, fails, max_attempts, e)

            if not uploaded_any:
                logger.debug("No eligible tracking files to upload")

        except Exception as e:
            logger.warning("Uploader error: %s", e)
            logger.debug("Uploader error detail", exc_info=True)

        time.sleep(scan_interval_s)


if __name__ == "__main__":
    main()
