#!/bin/bash
#SBATCH -p grete
#SBATCH --nodes=1                # set to N for multi-node training
#SBATCH --gpus-per-node=A100:2   # set to total GPUs per node
#SBATCH --ntasks-per-node=2      # MUST equal --gpus-per-node (one task per GPU)
#SBATCH --cpus-per-task=16
#SBATCH -t 0-24:00:00
#SBATCH -A nib00021              # TODO: verify account name
#SBATCH -C inet                  # required for HuggingFace Hub access
#SBATCH --mem=64G
#SBATCH -o logs/train-ddp-%A_%a.out
#SBATCH -e logs/train-ddp-%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jonathan.henrich@uni-goettingen.de
#SBATCH --array=0-19%1           # 20 jobs × steps_per_job = total steps; %1 = sequential

# CanViT pretraining (WebDataset path) with multi-GPU / multi-node DDP.
#
# Each job processes shards_per_job = shards_per_gpu * n_gpus contiguous shards,
# saves a checkpoint with job_index, and exits.
# The next job in the array reads job_index from the checkpoint and consumes
# the next slice of shards.
#
# Default run (1 node × 2 GPUs):
#   sbatch slurm_jonathan/train_ddp.sbatch
#
# Multi-node (e.g. 2 nodes × 2 GPUs = 4 ranks):
#   sbatch --nodes=2 --ntasks-per-node=2 --gpus-per-node=A100:2 \
#          slurm_jonathan/train_ddp.sbatch
#
# Resume existing run:
#   sbatch --array=0-9%1 slurm_jonathan/train_ddp.sbatch --run-name train-XXXXXXX
#
cd /user/henrich1/u25995/jonathan/repos/CanViT-pretrain
source /user/henrich1/u25995/.bashrc

set -eu

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== CanViT DDP Training ==="
log "Array Job: ${SLURM_ARRAY_JOB_ID:-none}, Task: ${SLURM_ARRAY_TASK_ID:-none}, Host: $(hostname)"
log "Nodes: ${SLURM_NNODES:-?}, Tasks/node: ${SLURM_NTASKS_PER_NODE:-?}, GPUs/node: ${SLURM_GPUS_ON_NODE:-?}"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm_jonathan/env.sh

log "Installing/syncing dependencies..."
uv sync

mkdir -p logs

# WEBDATASET_DIR is expected from .envrc.grete; warn loudly if missing.
if [ -z "${WEBDATASET_DIR:-}" ]; then
    log "ERROR: WEBDATASET_DIR is not set. Add it to .envrc.grete (parent of train-shuffled/ and val/)."
    exit 1
fi

RUN_NAME="train-${SLURM_ARRAY_JOB_ID}"

# DDP environment — see plan_dataloading/ddp/minimal_distributed_training.sh.
# WORLD_SIZE = total ranks across all nodes; one rank per GPU.
export MASTER_PORT=29400
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

log "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  WORLD_SIZE=$WORLD_SIZE"

log "Starting training (run: $RUN_NAME)..."
exec srun uv run python -m canvit_pretrain.train \
    --run-name "$RUN_NAME" \
    --webdataset-dir "$WEBDATASET_DIR" \
    --ckpt-dir "$CHECKPOINTS_DIR" \
    --wandb-project "${WANDB_PROJECT:-canvit-pretrain}" \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
    ${WANDB_DIR:+--wandb-dir "$WANDB_DIR"} \
    --steps-per-job 50000 \
    "$@"
