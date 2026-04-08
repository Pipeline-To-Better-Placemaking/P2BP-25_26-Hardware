#!/usr/bin/env bash
# Run LidarScanV1 on a Raspberry Pi over SSH, copy the resulting .xyz back to this host
# (Jetson) so scan_orchestrator can upload it to Web.
#
# Required env (set by orchestrator or systemd):
#   P2BP_SCAN_OUTPUT_XYZ — local path where the .xyz must appear (set per job by scan_orchestrator)
#
# Optional env (default SSH target is pi@192.168.28.2 — permanent Pi address on this rig):
#   P2BP_PI_SSH — override, e.g. otheruser@192.168.28.2
#
# Optional:
#   P2BP_PI_REMOTE_SCRIPT — path on the Pi (default: /opt/p2bp/lidar/scripts/LidarScanV1.py)
#   P2BP_PI_REMOTE_XYZ    — path on the Pi written by LidarScanV1 (default: /tmp/p2bp_lidar_scan.xyz)
#
# Prerequisites: passwordless SSH from this host to the Pi; same paths exist on the Pi.

set -euo pipefail

: "${P2BP_SCAN_OUTPUT_XYZ:?P2BP_SCAN_OUTPUT_XYZ must be set (scan_orchestrator sets this)}"

# Permanent Pi address on camera network (override with P2BP_PI_SSH if username/host changes).
P2BP_PI_SSH="${P2BP_PI_SSH:-pi@192.168.28.2}"

REMOTE_SCRIPT="${P2BP_PI_REMOTE_SCRIPT:-/opt/p2bp/lidar/scripts/LidarScanV1.py}"
REMOTE_XYZ="${P2BP_PI_REMOTE_XYZ:-/tmp/p2bp_lidar_scan.xyz}"
REMOTE_WDIR="$(dirname "${REMOTE_XYZ}")"

mkdir -p "$(dirname "${P2BP_SCAN_OUTPUT_XYZ}")"

SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30)
SCP=(scp -o BatchMode=yes -o ConnectTimeout=30)

echo "[run_lidar_on_pi] SSH ${P2BP_PI_SSH} -> python3 ${REMOTE_SCRIPT}"
"${SSH[@]}" "${P2BP_PI_SSH}" \
  "P2BP_LIDAR_SCAN_NONINTERACTIVE=1 P2BP_SCAN_OUTPUT_XYZ='${REMOTE_XYZ}' P2BP_SCAN_WORKDIR='${REMOTE_WDIR}' python3 -u '${REMOTE_SCRIPT}'"

echo "[run_lidar_on_pi] scp ${P2BP_PI_SSH}:${REMOTE_XYZ} -> ${P2BP_SCAN_OUTPUT_XYZ}"
"${SCP[@]}" "${P2BP_PI_SSH}:${REMOTE_XYZ}" "${P2BP_SCAN_OUTPUT_XYZ}"
