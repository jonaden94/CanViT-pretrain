#!/bin/bash
#SBATCH -p jupyter
#SBATCH -G V100:1
#SBATCH -t 0-01:00:00
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub access
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --delay-boot=1
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=jonathan.henrich@uni-goettingen.de

# See slurm_nhr/tmux_resubmit/README.md for usage.

cd /user/henrich1/u25995/jonathan/repos/CanViT-pretrain
source /user/henrich1/u25995/.bashrc

set -eu

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== CanViT Training (V100, tmux-resubmit) ==="
log "Job: ${SLURM_JOB_ID:-none}, Host: $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm_nhr/env.sh

VENV=".venv-v100"
if [ ! -d "$VENV" ]; then
    log "ERROR: $VENV not found. Run: bash _setup_v100/setup.sh"
    exit 1
fi

mkdir -p logs
mkdir -p "$LOG_DIR"

log "Starting training..."
exec "$VENV/bin/python" -m canvit_pretrain.train \
    --webdataset-dir "$WEBDATASET_DIR" \
    --ckpt-dir "$CHECKPOINTS_DIR" \
    --wandb-project "${WANDB_PROJECT:-canvit-pretrain}" \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
    ${WANDB_DIR:+--wandb-dir "$WANDB_DIR"} \
    --batch-size-per-gpu 64 \
    --steps-per-job 128 \
    --val-every 256 \
    "$@"
