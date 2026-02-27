#!/bin/bash
# No canvas reads (write-only canvas).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
sbatch --array=0-40%1 -J abl-no-reads "$SCRIPT_DIR/train.sbatch" \
    --warmup-steps 20000 --run-name abl-no-reads-200k \
    --model.no-enable-reads
