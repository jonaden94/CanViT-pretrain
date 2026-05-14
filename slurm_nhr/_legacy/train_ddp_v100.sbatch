#!/bin/bash
#SBATCH -p jupyter
#SBATCH --nodes=1
#SBATCH --gpus-per-node=V100:2
#SBATCH --ntasks-per-node=2      # MUST equal --gpus-per-node (one task per GPU)
#SBATCH --cpus-per-task=16
#SBATCH -t 0-01:00:00
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub access
#SBATCH --delay-boot=1
#SBATCH --mem=64G
#SBATCH -o logs/train-ddp-%A_%a.out
#SBATCH -e logs/train-ddp-%A_%a.err
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=jonathan.henrich@uni-goettingen.de
#SBATCH --array=0-99%1           # 100 jobs × steps_per_job; %1 = sequential

# CanViT pretraining (WebDataset path) with 2× V100 DDP on jupyter partition.
#
# Requires .venv-v100 to be set up first:
#   bash _setup_v100/setup.sh
#
# Each job processes shards_per_job = shards_per_gpu * world_size contiguous shards,
# saves a checkpoint with job_index, and exits.
# The next job in the array reads job_index from the checkpoint and consumes
# the next slice of shards.
#
# Default run:
#   sbatch slurm_nhr/train_ddp_v100.sbatch
#
# Resume existing run:
#   sbatch --array=0-50%1 slurm_nhr/train_ddp_v100.sbatch --run-name train-XXXXXXX

cd /user/henrich1/u25995/jonathan/repos/CanViT-pretrain
source /user/henrich1/u25995/.bashrc

set -eu

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== CanViT DDP Training (V100) ==="
log "Array Job: ${SLURM_ARRAY_JOB_ID:-none}, Task: ${SLURM_ARRAY_TASK_ID:-none}, Host: $(hostname)"
log "Nodes: ${SLURM_NNODES:-?}, Tasks/node: ${SLURM_NTASKS_PER_NODE:-?}, GPUs/node: ${SLURM_GPUS_ON_NODE:-?}"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm_nhr/env.sh

VENV=".venv-v100"
if [ ! -d "$VENV" ]; then
    log "ERROR: $VENV not found. Run: bash _setup_v100/setup.sh"
    exit 1
fi

mkdir -p logs

# WEBDATASET_DIR is expected from .envrc.grete; warn loudly if missing.
if [ -z "${WEBDATASET_DIR:-}" ]; then
    log "ERROR: WEBDATASET_DIR is not set. Add it to .envrc.grete (parent of train-shuffled/ and val/)."
    exit 1
fi

RUN_NAME="train-${SLURM_ARRAY_JOB_ID}"

# DDP environment — one rank per GPU.
export MASTER_PORT=29400
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

log "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  WORLD_SIZE=$WORLD_SIZE"

log "Starting training (run: $RUN_NAME)..."
exec srun "$VENV/bin/python" -m canvit_pretrain.train \
    --run-name "$RUN_NAME" \
    --webdataset-dir "$WEBDATASET_DIR" \
    --ckpt-dir "$CHECKPOINTS_DIR" \
    --wandb-project "${WANDB_PROJECT:-canvit-pretrain}" \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
    ${WANDB_DIR:+--wandb-dir "$WANDB_DIR"} \
    --batch-size-per-gpu 64 \
    --steps-per-job 256 \
    "$@"
