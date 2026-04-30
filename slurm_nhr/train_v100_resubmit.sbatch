#!/bin/bash
#SBATCH -p jupyter
#SBATCH -G V100:1
#SBATCH -t 0-01:00:00
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub access
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --delay-boot=1
#SBATCH -o logs/train-%j.out
#SBATCH -e logs/train-%j.err
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=jonathan.henrich@uni-goettingen.de

# CanViT pretraining on V100 (jupyter partition) — self-resubmitting pattern.
# Submits only 1 job at a time to stay within the partition's job limit.
# Each job trains for steps_per_job steps, saves a checkpoint, then resubmits
# itself with REMAINING_JOBS decremented by 1.
#
# All logs for a run are collected in one folder: logs/train-{first_job_id}/
#
# Requires .venv-v100 to be set up first:
#   bash _setup_v100/setup.sh
#
# First submission (100 jobs total):
#   sbatch slurm_nhr/train_v100_resubmit.sbatch
#
# Custom number of jobs:
#   sbatch --export=ALL,REMAINING_JOBS=50 slurm_nhr/train_v100_resubmit.sbatch
#
# Resume existing run:
#   sbatch --export=ALL,REMAINING_JOBS=50,LOG_DIR=logs/train-XXXXXXX \
#       slurm_nhr/train_v100_resubmit.sbatch --run-name train-XXXXXXX

cd /user/henrich1/u25995/jonathan/repos/CanViT-pretrain
source /user/henrich1/u25995/.bashrc

set -eu

log() { echo "[$(date '+%H:%M:%S')] $*"; }

REMAINING_JOBS=${REMAINING_JOBS:-10}

# First job: create the shared log dir named after this job's ID.
# Resubmits: LOG_DIR is inherited via --export, keeping all logs together.
LOG_DIR=${LOG_DIR:-"logs/train-${SLURM_JOB_ID}"}
mkdir -p "$LOG_DIR"

# Redirect all further output into the shared log dir (tee keeps SLURM's own
# logs/train-%j.out as a fallback for scheduler-level errors).
exec > >(tee -a "$LOG_DIR/job-${SLURM_JOB_ID}.log") 2>&1

log "=== CanViT Training (V100, self-resubmit) ==="
log "Job: ${SLURM_JOB_ID:-none}, Host: $(hostname), remaining_jobs=$REMAINING_JOBS"
log "Log dir: $LOG_DIR"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm_nhr/env.sh

VENV=".venv-v100"
if [ ! -d "$VENV" ]; then
    log "ERROR: $VENV not found. Run: bash _setup_v100/setup.sh"
    exit 1
fi

# On first job, generate run name from job ID. On resubmits, --run-name is
# passed explicitly so the checkpoint directory stays the same.
RUN_NAME="train-${SLURM_JOB_ID}"

log "Starting training (run: $RUN_NAME)..."
"$VENV/bin/python" -m canvit_pretrain.train \
    --run-name "$RUN_NAME" \
    --webdataset-dir "$WEBDATASET_DIR" \
    --ckpt-dir "$CHECKPOINTS_DIR" \
    --wandb-project "${WANDB_PROJECT:-canvit-pretrain}" \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
    ${WANDB_DIR:+--wandb-dir "$WANDB_DIR"} \
    --batch-size-per-gpu 64 \
    --steps-per-job 256 \
    --val-every 128 \
    "$@"

# Resubmit if more jobs remain
if [ "$REMAINING_JOBS" -gt 1 ]; then
    log "Resubmitting (remaining: $((REMAINING_JOBS - 1)))..."
    sbatch \
        --export=ALL,REMAINING_JOBS=$((REMAINING_JOBS - 1)),LOG_DIR="$LOG_DIR" \
        slurm_nhr/train_v100_resubmit.sbatch \
        --run-name "$RUN_NAME" \
        "$@"
else
    log "All jobs completed."
fi
