#!/bin/bash
# No VPE (viewpoint position encoding disabled).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
sbatch --array=0-40%1 -J abl-no-vpe "$SCRIPT_DIR/train.sbatch" \
    --warmup-steps 20000 --run-name abl-no-vpe-200k \
    --model.no-enable-vpe
