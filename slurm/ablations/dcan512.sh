#!/bin/bash
# D_can=512 (n_h=4, d_h=128), asymmetric. Halved canvas width, no QKVO.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
sbatch --array=0-40%1 -J abl-dcan512 "$SCRIPT_DIR/train.sbatch" \
    --warmup-steps 20000 --run-name abl-dcan512-200k \
    --model.canvas-num-heads 4
