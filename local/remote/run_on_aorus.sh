#!/usr/bin/env bash
# Sync this checkout to the GPU desktop and run a command there inside its venv.
#
#   local/remote/run_on_aorus.sh nvidia-smi
#   local/remote/run_on_aorus.sh python -m pytest tests -q
#   local/remote/run_on_aorus.sh python -m local.smoke_inference --model vjepa2_1_vit_large_384 --frames 64 --size 384
#   local/remote/run_on_aorus.sh python -m app.main --fname local/configs/vits_mac_smoke.yaml --devices cuda:0
#
# Env overrides: REMOTE (ssh host, default aorus), REMOTE_DIR (default ~/projects/vjepa2),
# NO_SYNC=1 to skip the rsync. Output streams back live; exit code is the remote command's.
# Source, configs and local/ are synced; .venv, checkpoints, runs and data are not
# (checkpoints are synced once by hand, runs stay on the desktop).
set -euo pipefail

REMOTE="${REMOTE:-aorus}"
REMOTE_DIR="${REMOTE_DIR:-projects/vjepa2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ "${NO_SYNC:-0}" != "1" ]; then
    rsync -az --delete \
        --exclude .venv --exclude checkpoints --exclude 'local/runs' --exclude 'local/data' \
        --exclude '__pycache__' --exclude '*.egg-info' \
        "$HERE/" "$REMOTE:$REMOTE_DIR/"
fi

if [ $# -eq 0 ]; then
    exec ssh -t "$REMOTE" "cd $REMOTE_DIR && exec \$SHELL -l"
fi

# Quote each argument for the remote shell.
cmd=$(printf '%q ' "$@")
exec ssh -t "$REMOTE" "cd $REMOTE_DIR && source .venv/bin/activate && $cmd"
