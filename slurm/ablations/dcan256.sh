#!/bin/bash
# D_can=256 (n_h=2, d_h=128), asymmetric. Quarter canvas width, no QKVO.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
sbatch --array=0-40%1 -J abl-dcan256 "$SCRIPT_DIR/train.sbatch" \
    --warmup-steps 20000 --run-name abl-dcan256-200k \
    --model.canvas-num-heads 2
