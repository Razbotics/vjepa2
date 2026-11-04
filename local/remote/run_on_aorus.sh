#!/usr/bin/env bash
# Sync this checkout to the GPU desktop and run a command there inside its venv.
#
#   local/remote/run_on_aorus.sh nvidia-smi
#   local/remote/run_on_aorus.sh python -m pytest tests -q
#   local/remote/run_on_aorus.sh python -m local.smoke_inference --model vjepa2_1_vit_large_384 --frames 64 --size 384
#   local/remote/run_on_aorus.sh python -m app.main --fname local/configs/vits_mac_smoke.yaml --devices cuda:0
#
# Sync modes (SYNC=git is the default):
#   git    push the current branch to origin (the Razbotics/vjepa2 fork), then on the
#          desktop fetch and hard-reset that branch to origin/<branch>. Only committed
#          work travels; the script warns if the tree is dirty.
#   rsync  copy the working tree as-is (uncommitted edits included), excluding .venv,
#          checkpoints, local/runs, local/data. Use for quick experiments before a commit.
#   NO_SYNC=1 skips syncing entirely.
# Other env: REMOTE (ssh host, default aorus), REMOTE_DIR (default ~/projects/vjepa2).
# No args -> interactive shell in the remote repo. Output streams back; exit code is the
# remote command's. Checkpoints are copied once by hand; runs stay on the desktop.
set -euo pipefail

REMOTE="${REMOTE:-aorus}"
REMOTE_DIR="${REMOTE_DIR:-projects/vjepa2}"
SYNC="${SYNC:-git}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ "${NO_SYNC:-0}" = "1" ]; then
    :
elif [ "$SYNC" = "git" ]; then
    branch=$(git -C "$HERE" rev-parse --abbrev-ref HEAD)
    if [ -n "$(git -C "$HERE" status --porcelain --untracked-files=normal)" ]; then
        echo "[run_on_aorus] WARNING: uncommitted changes will not be synced (commit, or use SYNC=rsync):" >&2
        git -C "$HERE" status --short >&2
    fi
    git -C "$HERE" push -q origin "$branch"
    ssh "$REMOTE" "cd $REMOTE_DIR && git fetch -q origin && git checkout -q -B $branch origin/$branch && git reset -q --hard origin/$branch && echo \"[run_on_aorus] $REMOTE at \$(git log --oneline -1)\""
elif [ "$SYNC" = "rsync" ]; then
    rsync -az --delete \
        --exclude .venv --exclude checkpoints --exclude 'local/runs' --exclude 'local/data' \
        --exclude '__pycache__' --exclude '*.egg-info' --exclude .git \
        "$HERE/" "$REMOTE:$REMOTE_DIR/"
else
    echo "unknown SYNC=$SYNC (git | rsync)" >&2
    exit 2
fi

if [ $# -eq 0 ]; then
    exec ssh -t "$REMOTE" "cd $REMOTE_DIR && exec \$SHELL -l"
fi

# Quote each argument for the remote shell.
cmd=$(printf '%q ' "$@")
exec ssh -t "$REMOTE" "cd $REMOTE_DIR && source .venv/bin/activate && $cmd"
