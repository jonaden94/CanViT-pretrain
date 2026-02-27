#!/bin/bash
# No F-IID rollout (R-IID only).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
sbatch --array=0-40%1 -J abl-no-fiid "$SCRIPT_DIR/train.sbatch" \
    --warmup-steps 20000 --run-name abl-no-fiid-200k \
    --n-full-start-branches 0
