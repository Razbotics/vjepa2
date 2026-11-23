#!/usr/bin/env bash
# Live view of the desktop's webcam on this laptop, for aiming the camera.
# Starts local/camera_preview.py on the desktop and tunnels it over SSH, so nothing is exposed on the network.
#
#   local/remote/camera_preview.sh          # then open http://localhost:8765
#   PORT=9000 local/remote/camera_preview.sh --no-overlay
#
# Ctrl-C stops the preview and releases the camera. Stop it before collecting data: only one program
# can hold the camera.
set -euo pipefail

REMOTE="${REMOTE:-aorus}"
REMOTE_DIR="${REMOTE_DIR:-projects/vjepa2}"
PORT="${PORT:-8765}"
PY="${PY:-\$HOME/projects/lerobot/.venv/bin/python}"

extra=""
if [ "$#" -gt 0 ]; then extra=$(printf '%q ' "$@"); fi

echo "[camera_preview] open http://localhost:${PORT}  (Ctrl-C to stop)"
# -tt gives the remote python a terminal, so Ctrl-C or a dropped connection stops it and frees the camera.
exec ssh -tt -o ExitOnForwardFailure=yes -L "${PORT}:127.0.0.1:${PORT}" "$REMOTE" \
  "cd $REMOTE_DIR && exec $PY -m local.camera_preview --port ${PORT} $extra"
