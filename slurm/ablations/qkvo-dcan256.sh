#!/bin/bash
# +QKVO, D_can=256 (n_h=2, d_h=128). FLOP-matched to baseline.
# Core claim: does asymmetric identity beat full projections at matched FLOPs?
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
sbatch --array=0-40%1 -J abl-qkvo-256 "$SCRIPT_DIR/train.sbatch" \
    --warmup-steps 20000 --run-name abl-qkvo-dcan256-200k \
    --model.canvas-proj-mode full --model.canvas-num-heads 2
