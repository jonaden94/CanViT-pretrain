#!/bin/bash
# Interactive training — run directly on a machine with a GPU (no SLURM needed).
#
# Usage:
#   bash slurm_jonathan/train.sh [extra args passed to canvit_pretrain.train]
#
# Examples:
#   bash slurm_jonathan/train.sh                              # full run, auto run-name
#   bash slurm_jonathan/train.sh --steps-per-job 100         # smoke test
#   bash slurm_jonathan/train.sh --run-name my-run           # explicit run name (for resume)
#   bash slurm_jonathan/train.sh --run-name my-run --steps-per-job 50000

cd /user/henrich1/u25995/jonathan/repos/CanViT-pretrain

set -eu

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== CanViT Training (interactive) ==="
log "Host: $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm_jonathan/env.sh

log "Installing/syncing dependencies..."
uv sync

mkdir -p logs

RUN_NAME="train-$(date '+%Y%m%d-%H%M%S')"
LOG_FILE="logs/${RUN_NAME}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

log "Starting training (run: $RUN_NAME, log: $LOG_FILE)..."
exec uv run python -m canvit_pretrain.train \
    --run-name "$RUN_NAME" \
    --webdataset-dir "$WEBDATASET_DIR" \
    --ckpt-dir "$CHECKPOINTS_DIR" \
    --wandb-project "${WANDB_PROJECT:-canvit-pretrain}" \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
    ${WANDB_DIR:+--wandb-dir "$WANDB_DIR"} \
    --batch-size 16 \
    --steps-per-job 256 \
    "$@"
