#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

IP_MAC_RE = re.compile(
    r"^(?P<ip>\d{1,3}(?:\.\d{1,3}){3})\s+(?P<mac>(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2})\b"
)

def log(level: str, msg: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"{ts} [{level}] {msg}", flush=True)

def require_root() -> None:
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        raise PermissionError("This script is intended to run as root (e.g., via systemd User=root).")

def which_or_fail(name: str, extra_paths: Optional[List[str]] = None) -> str:
    path = shutil.which(name)
    if path:
        return path
    if extra_paths:
        for p in extra_paths:
            if os.path.exists(p) and os.access(p, os.X_OK):
                return p
    raise FileNotFoundError(f"Required binary not found in PATH: {name}")

def run_cmd(cmd: List[str], timeout: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

def iface_is_up(interface: str) -> bool:
    oper = f"/sys/class/net/{interface}/operstate"
    try:
        with open(oper, "r") as f:
            state = f.read().strip().lower()
        return state == "up"
    except OSError:
        return True

def scan_arp(arp_scan_bin: str, interface: str) -> List[Tuple[str, str]]:
    """
    Returns list of (mac_lower, ip) discovered by arp-scan.
    Requires root.
    """
    proc = run_cmd(
        [arp_scan_bin, f"--interface={interface}", "--localnet"],
        timeout=300.0,
    )

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"arp-scan failed: {err}")

    results: List[Tuple[str, str]] = []
    for line in proc.stdout.splitlines():
        m = IP_MAC_RE.match(line.strip())
        if not m:
            continue
        ip = m.group("ip")
        mac = m.group("mac").lower()
        results.append((mac, ip))
    return results

def scan_arp_with_retry(
    arp_scan_bin: str,
    interface: str,
    timeout_sec: float,
    sleep_sec: float,
) -> List[Tuple[str, str]]:
    """
    Boot-friendly ARP scan:
      - waits for iface to be up
      - retries arp-scan until timeout
    """
    start = time.time()
    last_err: Optional[str] = None

    while True:
        if not iface_is_up(interface):
            last_err = f"Interface {interface} not up yet."
        else:
            try:
                return scan_arp(arp_scan_bin, interface)
            except subprocess.TimeoutExpired:
                last_err = "arp-scan timed out"
            except Exception as e:
                last_err = str(e)

        if time.time() - start > timeout_sec:
            raise TimeoutError(f"ARP scan did not succeed within {timeout_sec}s. Last error: {last_err}")
        time.sleep(sleep_sec)

def ffprobe_resolution(
    ffprobe_bin: str,
    ip: str,
    user: str,
    password: str,
    port: int,
    path: str
) -> Optional[Tuple[int, int]]:
    """
    Returns (width,height) if RTSP stream is valid, else None.
    """
    url = f"rtsp://{user}:{password}@{ip}:{port}{path}"
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-rtsp_transport", "tcp",
        "-stimeout", "2000000",  # microseconds (2s)
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        url,
    ]
    try:
        proc = run_cmd(cmd, timeout=6.5)
    except subprocess.TimeoutExpired:
        return None

    if proc.returncode != 0:
        return None

    out = (proc.stdout or "").strip()
    if not out:
        return None

    first_line = out.splitlines()[0].strip()
    if "," not in first_line:
        return None
    w_str, h_str = first_line.split(",", 1)
    try:
        return int(w_str), int(h_str)
    except ValueError:
        return None

def load_registry(path: str) -> Dict:
    """
    Boot-safe: if file missing or corrupted, returns empty registry shape.
    """
    if not os.path.exists(path):
        return {"cameras": {}}
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        log("WARN", f"Registry unreadable/corrupted; starting fresh: {path}")
        return {"cameras": {}}

    if not isinstance(data, dict):
        return {"cameras": {}}
    cams = data.get("cameras")
    if not isinstance(cams, dict):
        data["cameras"] = {}
    return data

def save_json_atomic(path: str, data: Dict) -> None:
    out_path = os.path.abspath(path)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out_path)

def auto_enroll_registry(
    arp_scan_bin: str,
    ffprobe_bin: str,
    interface: str,
    user: str,
    password: str,
    port: int,
    rtsp_path: str,
    timeout_sec: float,
    sleep_sec: float,
) -> Dict:
    """
    Auto-enroll cameras deterministically:
      - scan LAN
      - test RTSP on fixed path
      - collect MACs that work
      - sort by MAC
      - assign cam0..camN
    """
    start = time.time()

    while True:
        seen = scan_arp_with_retry(arp_scan_bin, interface, timeout_sec=timeout_sec, sleep_sec=sleep_sec)


        working_macs: List[str] = []
        for mac, ip in seen:
            res = ffprobe_resolution(ffprobe_bin, ip, user, password, port, rtsp_path)
            if res is not None:
                working_macs.append(mac)

        working_macs = sorted(set(working_macs))
        if working_macs:
            cameras = {}
            for idx, mac in enumerate(working_macs):
                cameras[f"cam{idx}"] = {"mac": mac, "rtsp_path": rtsp_path}
            return {"cameras": cameras}

        if time.time() - start > timeout_sec:
            raise TimeoutError(f"No RTSP-ready cameras found within {timeout_sec}s.")
        time.sleep(sleep_sec)

def build_runtime(
    arp_scan_bin: str,
    ffprobe_bin: str,
    registry: Dict,
    interface: str,
    user: str,
    password: str,
    port: int,
    scan_timeout_sec: float,
    sleep_sec: float,
) -> Dict:
    """
    Builds cameras_runtime.json by:
      - scanning ARP to map MAC->IP
      - for each registered cam, find its current IP
      - probe RTSP once (readiness + resolution)
    """
    seen = scan_arp_with_retry(arp_scan_bin, interface, timeout_sec=scan_timeout_sec, sleep_sec=sleep_sec)

    mac_to_ip: Dict[str, str] = {}
    for mac, ip in seen:
        # detect MAC address duplicates
        if mac in mac_to_ip and mac_to_ip[mac] != ip:
            log("WARN", f"Duplicate MAC seen with different IPs: {mac} -> {mac_to_ip[mac]} and {ip} (keeping latest)")
        mac_to_ip[mac] = ip

    runtime: Dict[str, Dict] = {}
    for cam_name, cam_info in registry.get("cameras", {}).items():
        mac = str(cam_info.get("mac", "")).lower().strip()
        path = str(cam_info.get("rtsp_path", "")).strip()
        if not mac or not path:
            continue

        ip = mac_to_ip.get(mac)
        if not ip:
            continue

        res = ffprobe_resolution(ffprobe_bin, ip, user, password, port, path)
        if res is None:
            continue

        w, h = res
        runtime[cam_name] = {
            "mac": mac,
            "ip": ip,
            "rtsp": f"rtsp://{user}:{password}@{ip}:{port}{path}",
            "resolution": [w, h],
        }

    return runtime

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface",          default="eno1")
    ap.add_argument("--user",           default="admin")
    ap.add_argument("--password",       default="Placemaking25")
    ap.add_argument("--port",           type=int, default=554)
    ap.add_argument("--rtsp-path",      default="/Streaming/Channels/101")
    ap.add_argument("--registry",       default="camera_registry.json")
    ap.add_argument("--runtime",        default="cameras_runtime.json")
    ap.add_argument("--timeout",        type=float, default=300.0, help="Overall timeout for initial enrollment.")
    ap.add_argument("--sleep",          type=float, default=2.0)
    ap.add_argument("--scan-timeout",   type=float, default=300.0, help="Timeout for ARP scan phase when building runtime.")
    args = ap.parse_args()

    require_root()

    # make sure arp-scan and ffprobe are installed
    arp_scan_bin = which_or_fail("arp-scan", extra_paths=["/usr/sbin/arp-scan", "/sbin/arp-scan"])
    ffprobe_bin = which_or_fail("ffprobe", extra_paths=["/usr/bin/ffprobe", "/bin/ffprobe"])

    log("INFO", f"Using iface={args.iface}")
    log("INFO", f"Registry={os.path.abspath(args.registry)} Runtime={os.path.abspath(args.runtime)}")

    # 1) Load/create registry
    registry = load_registry(args.registry)
    if not registry.get("cameras"):
        log("INFO", "No cameras in registry; performing first-run auto-enroll...")
        registry = auto_enroll_registry(
            arp_scan_bin=arp_scan_bin,
            ffprobe_bin=ffprobe_bin,
            interface=args.iface,
            user=args.user,
            password=args.password,
            port=args.port,
            rtsp_path=args.rtsp_path,
            timeout_sec=args.timeout,
            sleep_sec=args.sleep,
        )
        save_json_atomic(args.registry, registry)
        log("OK", f"Auto-enrolled and wrote registry: {args.registry} (count={len(registry.get('cameras', {}))})")

    # 2) Build runtime (always, since IPs can change)
    runtime = build_runtime(
        arp_scan_bin=arp_scan_bin,
        ffprobe_bin=ffprobe_bin,
        registry=registry,
        interface=args.iface,
        user=args.user,
        password=args.password,
        port=args.port,
        scan_timeout_sec=args.scan_timeout,
        sleep_sec=args.sleep,
    )
    save_json_atomic(args.runtime, runtime)
    log("OK", f"Wrote runtime config: {args.runtime} (ready={len(runtime)}/{len(registry.get('cameras', {}))})")

    if len(runtime) == 0:
        log("WARN", "No cameras are RTSP-ready right now.")
        return 2
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as e:
        log("ERROR", str(e))
        raise SystemExit(1)
