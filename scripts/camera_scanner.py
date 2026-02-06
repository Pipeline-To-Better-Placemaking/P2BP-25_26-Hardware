#!/usr/bin/env python3
"""
camera_scanner.py

Scans the local network for RTSP-capable cameras and writes a runtime JSON file
with each camera keyed by its MAC address.

Output format (cameras_runtime.json):
{
  "aa:bb:cc:dd:ee:ff": {
    "mac": "aa:bb:cc:dd:ee:ff",
    "ip": "192.168.1.100",
    "rtsp": "rtsp://user:pass@192.168.1.100:554/path",
    "resolution": [1920, 1080]
  },
  ...
}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
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
        raise PermissionError(
            "This script is intended to run as root (e.g., via systemd User=root)."
        )

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

def _iface_has_mac(interface: str) -> bool:
    return os.path.exists(f"/sys/class/net/{interface}/address")

def _iface_is_virtual_or_non_eth(interface: str) -> bool:
    # Skip loopback and common non-physical interfaces
    if interface == "lo":
        return True
    for prefix in (
        "docker",
        "br-",
        "veth",
        "virbr",
        "vmnet",
        "zt",
        "tailscale",
        "wg",
        "tun",
        "tap",
        "wlan",
        "wl",
    ):
        if interface.startswith(prefix):
            return True
    return False

def _iface_is_usb_ethernet(interface: str) -> bool:
    # USB ethernet adapters are commonly named enx<MAC>
    return interface.startswith("enx")

def _iface_is_pci(interface: str) -> bool:
    # Prefer interfaces whose device subsystem is PCI
    subsystem_path = f"/sys/class/net/{interface}/device/subsystem"
    if not os.path.exists(subsystem_path):
        return False
    try:
        real = os.path.realpath(subsystem_path)
        return os.path.basename(real) == "pci"
    except OSError:
        return False

def select_pci_eth_iface(preferred: Optional[List[str]] = None) -> Optional[str]:
    """Select the most likely PCI Ethernet interface.

    Priority:
      1) Preferred names if present (e.g., eno1, enP8p1s0)
      2) Any interface backed by PCI subsystem

    Excludes USB ethernet (enx*) and common virtual/non-physical interfaces.
    """
    preferred = preferred or []

    for candidate in preferred:
        if _iface_has_mac(candidate):
            return candidate

    try:
        ifaces = sorted(os.listdir("/sys/class/net"))
    except OSError:
        return None

    for iface in ifaces:
        if _iface_is_virtual_or_non_eth(iface) or _iface_is_usb_ethernet(iface):
            continue
        if not _iface_has_mac(iface):
            continue
        if _iface_is_pci(iface):
            return iface

    return None

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
            raise TimeoutError(
                f"ARP scan did not succeed within {timeout_sec}s. Last error: {last_err}"
            )
        time.sleep(sleep_sec)

def ffprobe_resolution(
    ffprobe_bin: str,
    ip: str,
    user: str,
    password: str,
    port: int,
    path: str,
) -> Optional[Tuple[int, int]]:
    """
    Returns (width, height) if RTSP stream is valid, else None.
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

def scan_cameras(
    arp_scan_bin: str,
    ffprobe_bin: str,
    interface: str,
    user: str,
    password: str,
    port: int,
    rtsp_path: str,
    timeout_sec: float,
    sleep_sec: float,
) -> Dict[str, Dict]:
    """
    Scans the network and returns a dict of RTSP-ready cameras keyed by MAC address.

    Returns:
        {
            "aa:bb:cc:dd:ee:ff": {
                "mac": "aa:bb:cc:dd:ee:ff",
                "ip": "192.168.1.100",
                "rtsp": "rtsp://...",
                "resolution": [w, h]
            },
            ...
        }
    """
    seen = scan_arp_with_retry(
        arp_scan_bin, interface, timeout_sec=timeout_sec, sleep_sec=sleep_sec
    )

    # Build MAC -> IP mapping (warn on duplicates)
    mac_to_ip: Dict[str, str] = {}
    for mac, ip in seen:
        if mac in mac_to_ip and mac_to_ip[mac] != ip:
            log(
                "WARN",
                f"Duplicate MAC with different IPs: {mac} -> {mac_to_ip[mac]} and {ip} (keeping latest)",
            )
        mac_to_ip[mac] = ip

    log("INFO", f"ARP scan found {len(mac_to_ip)} unique MAC addresses")

    # Probe each device for RTSP
    cameras: Dict[str, Dict] = {}
    for mac, ip in sorted(mac_to_ip.items()):
        res = ffprobe_resolution(ffprobe_bin, ip, user, password, port, rtsp_path)
        if res is None:
            # If RTSP isn't ready yet, try onboarding/activation once, then retry.
            try:
                from scripts.camera_onboard import onboard_camera

                if onboard_camera(ip, mac):
                    log("INFO", f"Onboarded camera, retrying RTSP: {mac} @ {ip}")
                else:
                    log("INFO", f"Onboard not applicable/failed, retrying RTSP anyway: {mac} @ {ip}")
            except Exception as e:
                log("WARN", f"Onboard attempt failed for {mac} @ {ip}: {e}")

            res = ffprobe_resolution(ffprobe_bin, ip, user, password, port, rtsp_path)
            if res is None:
                continue

        w, h = res
        cameras[mac] = {
            "mac": mac,
            "ip": ip,
            "rtsp": f"rtsp://{user}:{password}@{ip}:{port}{rtsp_path}",
            "resolution": [w, h],
        }
        log("OK", f"Found camera: {mac} @ {ip} ({w}x{h})")

    return cameras

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Scan network for RTSP cameras and output runtime config keyed by MAC address."
    )
    ap.add_argument(
        "--iface",
        default="auto",
        help="Network interface to scan (default: auto-detect PCI ethernet; try eno1/enP8p1s0 first)",
    )
    ap.add_argument("--user", default="admin", help="RTSP username")
    ap.add_argument("--password", default="Placemaking25", help="RTSP password")
    ap.add_argument("--port", type=int, default=554, help="RTSP port")
    ap.add_argument("--rtsp-path", default="/Streaming/Channels/101", help="RTSP path")
    ap.add_argument(
        "--output",
        default="config/cameras_runtime.json",
        help="Output runtime JSON file",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Timeout for ARP scan (seconds)",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Sleep between retries (seconds)",
    )
    args = ap.parse_args()

    require_root()

    iface = args.iface
    if iface.strip().lower() == "auto":
        iface = select_pci_eth_iface(preferred=["eno1", "enP8p1s0"]) or ""
    if not iface or not os.path.exists(f"/sys/class/net/{iface}"):
        raise RuntimeError(
            "No suitable ethernet interface found. Specify one explicitly with --iface <name>."
        )

    arp_scan_bin = which_or_fail(
        "arp-scan", extra_paths=["/usr/sbin/arp-scan", "/sbin/arp-scan"]
    )
    ffprobe_bin = which_or_fail(
        "ffprobe", extra_paths=["/usr/bin/ffprobe", "/bin/ffprobe"]
    )

    log("INFO", f"Interface: {iface}")
    log("INFO", f"Output: {os.path.abspath(args.output)}")

    cameras = scan_cameras(
        arp_scan_bin=arp_scan_bin,
        ffprobe_bin=ffprobe_bin,
        interface=iface,
        user=args.user,
        password=args.password,
        port=args.port,
        rtsp_path=args.rtsp_path,
        timeout_sec=args.timeout,
        sleep_sec=args.sleep,
    )

    save_json_atomic(args.output, cameras)
    log("OK", f"Wrote {args.output} ({len(cameras)} cameras)")

    if len(cameras) == 0:
        log("WARN", "No RTSP-capable cameras found.")
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
