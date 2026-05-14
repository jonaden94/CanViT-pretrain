#!/bin/bash
#SBATCH -p jupyter
#SBATCH -G V100:1
#SBATCH -t 0-01:00:00
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub access
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --delay-boot=1
#SBATCH -o logs/train-%A_%a.out
#SBATCH -e logs/train-%A_%a.err
#SBATCH --array=0-0%1           # N jobs × steps_per_job; %1 = sequential
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=jonathan.henrich@uni-goettingen.de

# CanViT pretraining on V100 (jupyter partition) — array job pattern:
#   each job does steps_per_job steps, saves checkpoint, exits.
#   next job in array resumes from that checkpoint automatically.
#
# Requires .venv-v100 to be set up first:
#   bash _setup_v100/setup.sh
#
# Default run:
#   sbatch slurm_nhr/train_v100.sbatch
#
# Resume existing run:
#   sbatch --array=0-50%1 slurm_nhr/train_v100.sbatch --run-name train-XXXXXXX
#
# Tune steps_per_job: start small, check how long one job takes,
# then scale to fill ~10h of a 12h slot.

cd /user/henrich1/u25995/jonathan/repos/CanViT-pretrain
source /user/henrich1/u25995/.bashrc

set -eu

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== CanViT Training (V100) ==="
log "Array Job: ${SLURM_ARRAY_JOB_ID:-none}, Task: ${SLURM_ARRAY_TASK_ID:-none}, Host: $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm_nhr/env.sh

VENV=".venv-v100"
if [ ! -d "$VENV" ]; then
    log "ERROR: $VENV not found. Run: bash _setup_v100/setup.sh"
    exit 1
fi

mkdir -p logs

RUN_NAME="train-${SLURM_ARRAY_JOB_ID}"

log "Starting training (run: $RUN_NAME)..."
exec "$VENV/bin/python" -m canvit_pretrain.train \
    --run-name "$RUN_NAME" \
    --webdataset-dir "$WEBDATASET_DIR" \
    --ckpt-dir "$CHECKPOINTS_DIR" \
    --wandb-project "${WANDB_PROJECT:-canvit-pretrain}" \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
    ${WANDB_DIR:+--wandb-dir "$WANDB_DIR"} \
    --batch-size-per-gpu 64 \
    --steps-per-job 256 \
    --val-every 64 \
    "$@"
