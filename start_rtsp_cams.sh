#!/usr/bin/env bash
# You need ffmpeg installed to use this script
set -euo pipefail

echo "=== RTSP multi-cam from files (MediaMTX + FFmpeg) ==="

read -rp "How many cameras? [3]: " N
N="${N:-3}"

read -rp "Output resolution WxH [1280x720]: " RES
RES="${RES:-1280x720}"

read -rp "Output FPS [30]: " FPS
FPS="${FPS:-30}"

echo "Starting MediaMTX on rtsp://127.0.0.1:8554 ..."
sudo docker rm -f mediamtx >/dev/null 2>&1 || true
sudo docker run -d --name mediamtx --network host bluenviron/mediamtx:latest >/dev/null
sleep 1
sudo docker logs --tail=10 mediamtx || true

declare -a FILES OFFS
for ((i=0; i<N; i++)); do
  default="4p-c${i}.avi"
  read -rp "Path for cam${i} file [${default}]: " f
  f="${f:-$default}"
  if [[ ! -e "$f" ]]; then
    echo "  File not found. Enter a valid path (or Ctrl+C to abort)."
    read -rp "  Path for cam${i} file: " f
  fi
  FILES[$i]="$(realpath "$f")"

  read -rp "Offset (seconds) for cam${i} [0]: " o
  OFFS[$i]="${o:-0}"
done

mkdir -p /tmp/rtsp_cam_logs
: > /tmp/rtsp_ffmpeg_pids.txt

echo
for ((i=0; i<N; i++)); do
  log="/tmp/rtsp_cam_logs/cam${i}.log"
  uri="rtsp://127.0.0.1:8554/cam${i}"
  echo "Launching cam${i}: ${FILES[$i]} (offset ${OFFS[$i]}s) -> ${uri}"
  ffmpeg -hide_banner -loglevel info -nostdin \
    -re -stream_loop -1 \
    -itsoffset "${OFFS[$i]}" -i "${FILES[$i]}" \
    -vf "scale=${RES},fps=${FPS}" -pix_fmt yuv420p \
    -c:v libx264 -preset veryfast -tune zerolatency -g "${FPS}" -an \
    -f rtsp -rtsp_transport tcp "${uri}" \
    >"$log" 2>&1 &
  echo $! >> /tmp/rtsp_ffmpeg_pids.txt
done

echo
echo "== Streams (copy/paste to view) =="
for ((i=0; i<N; i++)); do
  echo "  ffplay -rtsp_transport tcp rtsp://127.0.0.1:8554/cam${i}"
done

echo
echo "Tail a log:   tail -f /tmp/rtsp_cam_logs/cam0.log"
echo "Stop senders: xargs -r kill < /tmp/rtsp_ffmpeg_pids.txt"
echo "Stop server:  sudo docker rm -f mediamtx"
