#!/bin/bash
# Interactive training for V100 GPUs — uses .venv-v100 (cu124 torch).
# Run _setup_v100/setup.sh once before using this script.
#
# Usage:
#   bash slurm_nhr/train-v100.sh [extra args passed to canvit_pretrain.train]
#
# Examples:
#   bash slurm_nhr/train-v100.sh
#   bash slurm_nhr/train-v100.sh --run-name my-run
#   bash slurm_nhr/train-v100.sh --run-name my-run --steps-per-job 50000

cd /user/henrich1/u25995/jonathan/repos/CanViT-pretrain

set -eu

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== CanViT Training (interactive, V100) ==="
log "Host: $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm_nhr/env.sh

VENV=".venv-v100"
if [ ! -d "$VENV" ]; then
    log "ERROR: $VENV not found. Run: bash _setup_v100/setup.sh"
    exit 1
fi

mkdir -p logs

RUN_NAME="train-$(date '+%Y%m%d-%H%M%S')"
LOG_FILE="logs/${RUN_NAME}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

log "Starting training (run: $RUN_NAME, log: $LOG_FILE)..."
exec "$VENV/bin/python" -m canvit_pretrain.train \
    --run-name "$RUN_NAME" \
    --webdataset-dir "$WEBDATASET_DIR" \
    --ckpt-dir "$CHECKPOINTS_DIR" \
    --wandb-project "${WANDB_PROJECT:-canvit-pretrain}" \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
    ${WANDB_DIR:+--wandb-dir "$WANDB_DIR"} \
    --batch-size-per-gpu 64 \
    --steps-per-job 256 \
    "$@"
