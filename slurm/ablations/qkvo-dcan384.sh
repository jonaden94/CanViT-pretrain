#!/bin/bash
# +QKVO, D_can=384 (n_h=3, d_h=128). Slightly above FLOP-matched.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
sbatch --array=0-40%1 -J abl-qkvo-384 "$SCRIPT_DIR/train.sbatch" \
    --warmup-steps 20000 --run-name abl-qkvo-dcan384-200k \
    --model.canvas-proj-mode full --model.canvas-num-heads 3
