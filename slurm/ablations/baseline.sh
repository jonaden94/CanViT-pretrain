#!/bin/bash
# Baseline: asymmetric (identity on canvas side), D_can=1024, VPE, additive.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
sbatch --array=0-40%1 -J abl-baseline "$SCRIPT_DIR/train.sbatch" \
    --warmup-steps 20000 --run-name abl-baseline-200k
