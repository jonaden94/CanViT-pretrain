#!/bin/bash
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -t 0-01:00:00
#SBATCH -A nib00021              # TODO: verify account name
#SBATCH -C inet                  # required for HuggingFace Hub access
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --delay-boot=1
#SBATCH -o logs/train-%A_%a.out
#SBATCH -e logs/train-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jonathan.henrich@uni-goettingen.de
#SBATCH --array=0-39%1           # 40 jobs × 50000 steps = 2M steps total (%1 = sequential)

# CanViT pretraining on Grete — array job pattern:
#   each job does steps_per_job steps, saves checkpoint, exits.
#   next job in array resumes from that checkpoint automatically.
#
# Default run (CanViT-B):
#   sbatch slurm_nhr/train.sbatch
#
# Resume existing run:
#   sbatch --array=0-20%1 slurm_nhr/train.sbatch --run-name train-XXXXXXX
#
# Quick smoke test (2 jobs, 10 steps each):
#   sbatch --array=0-1%1 --time=0-00:30:00 slurm_nhr/train.sbatch --steps-per-job 10
#
# Tune steps_per_job: start with --steps-per-job 10, check how long it takes,
# then scale up to fill ~20h of a 24h slot (leave buffer for startup/checkpoint).
cd /user/henrich1/u25995/jonathan/repos/CanViT-pretrain
source /user/henrich1/u25995/.bashrc

set -eu

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== CanViT Training ==="
log "Array Job: ${SLURM_ARRAY_JOB_ID:-none}, Task: ${SLURM_ARRAY_TASK_ID:-none}, Host: $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm_nhr/env.sh

log "Installing/syncing dependencies..."
uv sync

mkdir -p logs

RUN_NAME="train-${SLURM_ARRAY_JOB_ID}"

log "Starting training (run: $RUN_NAME)..."
exec uv run python -m canvit_pretrain.train \
    --run-name "$RUN_NAME" \
    --feature-base-dir "$FEATURES_DIR/in21k" \
    --feature-image-root "$IN21K_IMAGE_DIR" \
    --val-dir "$IN1K_VAL_IMAGE_DIR" \
    --train-index-dir "$INDEX_DIR" \
    --ckpt-dir "$CHECKPOINTS_DIR" \
    --wandb-project "${WANDB_PROJECT:-canvit-pretrain}" \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
    ${WANDB_DIR:+--wandb-dir "$WANDB_DIR"} \
    --steps-per-job 50000 \
    "$@"
