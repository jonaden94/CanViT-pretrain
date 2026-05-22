#!/bin/bash
# See slurm_nhr/tmux_resubmit/README.md for usage.

SCRIPT="$(dirname "$0")/check_and_submit.sh"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Watcher started. Checking every 60s."

while true; do
    bash "$SCRIPT"
    sleep 60
done
