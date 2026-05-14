#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:2
#SBATCH --ntasks-per-node=2      # MUST equal --gpus-per-node (one task per GPU)
#SBATCH --cpus-per-task=16
#SBATCH -t 0-00:45:00
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub access
#SBATCH --delay-boot=1
#SBATCH --mem=256G
#SBATCH --exclude=ggpu131,ggpu132,ggpu134,ggpu105
#SBATCH -o logs/exp17-2gpu-match1gpu-post-ddp-fix/job-%A_%a.log
#SBATCH -e logs/exp17-2gpu-match1gpu-post-ddp-fix/job-%A_%a.log
#SBATCH --array=0-99%1           # 100 jobs × 4096 steps each; %1 = sequential
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=jonathan.henrich@uni-goettingen.de

# CanViT pretraining — exp17 scaling study, 2× A100 DDP, replicating the
# 1-GPU optimization dynamics (global batch 64, peak_lr 4e-4) so this run is
# a direct wall-clock speedup of train_1gpu.sbatch, NOT a larger-batch run.
#
#   batch-size-per-gpu = 32   ->  global batch = 2 × 32 = 64  (== 1-GPU)
#   peak_lr           = 4e-4  ->  identical to 1-GPU (no linear LR scaling)
#
# 100 sequential jobs × 4096 steps; resume mechanic carries wandb run +
# checkpoint forward across the array tasks.
#
# Submit with:
#   sbatch slurm_nhr/exp17/train_2gpu_match1gpu.sbatch

cd /user/henrich1/u25995/jonathan/repos/CanViT-pretrain
source /user/henrich1/u25995/.bashrc

set -eu

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== exp17 2-GPU DDP training, match-1gpu (A100, grete:shared) ==="
log "Array Job: ${SLURM_ARRAY_JOB_ID:-none}, Task: ${SLURM_ARRAY_TASK_ID:-none}, Host: $(hostname)"
log "Nodes: ${SLURM_NNODES:-?}, Tasks/node: ${SLURM_NTASKS_PER_NODE:-?}, GPUs/node: ${SLURM_GPUS_ON_NODE:-?}"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm_nhr/env.sh

# Some grete nodes have /local saturated by other users' leftover data
# (SLURM only auto-cleans /local/jobs/<jobid>/, not /local/<uid>/). When that
# happens, our compile cache (~300 MB measured) won't fit and the job wedges.
# Fall back to project NFS: slower than NVMe but always has space.
TMPDIR_NEED_GB=2
ERSATZ_BASE=/mnt/vast-nhr/projects/nib00021/jonathan/ersatz_local
tmpdir_avail_gb=$(df -BG --output=avail "$TMPDIR" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ "${tmpdir_avail_gb:-0}" -lt "$TMPDIR_NEED_GB" ]; then
    log "WARN: $TMPDIR on $(hostname) has ${tmpdir_avail_gb:-?}G free (<${TMPDIR_NEED_GB}G); falling back to NFS"
    export TMPDIR="$ERSATZ_BASE/u25995_${SLURM_JOB_ID}"
    mkdir -p "$TMPDIR"
fi
log "TMPDIR=$TMPDIR"

VENV=".venv-v100"
if [ ! -d "$VENV" ]; then
    log "ERROR: $VENV not found. Run: bash _setup_v100/setup.sh"
    exit 1
fi

if [ -z "${WEBDATASET_DIR:-}" ]; then
    log "ERROR: WEBDATASET_DIR is not set. Add it to .envrc.grete (parent of train-shuffled/ and val/)."
    exit 1
fi

RUN_NAME="exp17-2gpu-match1gpu-post-ddp-fix"

# Per-job MASTER_PORT keeps this run from colliding with concurrent runs on
# the same node (grete:shared can pack multiple jobs onto one node).
export MASTER_PORT=$((20000 + (SLURM_JOB_ID % 10000)))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

log "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  WORLD_SIZE=$WORLD_SIZE"

# Wipe stale Triton/torch.compile/CUDA caches on the assigned compute node.
# Cancelled prior jobs leave half-written kernels, KMP shm entries, and
# NVIDIA module-cache state that can poison subsequent runs.
log "Clearing per-node caches (Triton, Inductor, NVRTC, NVIDIA Compute, KMP shm, .nv)..."
srun --ntasks-per-node=1 --ntasks=$SLURM_NNODES bash -c '
    rm -rf /tmp/triton_cache_rank* /tmp/torchinductor_* /tmp/nvrtc_cache_* /tmp/.nv 2>/dev/null || true
    rm -rf $HOME/.triton/cache 2>/dev/null || true
    rm -rf $HOME/.nv/ComputeCache 2>/dev/null || true
    find /dev/shm -maxdepth 1 -user $USER \( -name "__KMP_REGISTERED_LIB_*" -o -name "torch_*" -o -name "python_*" -o -name "pymp-*" \) -delete 2>/dev/null || true
    echo "[cache-clear] $(hostname): /tmp + ~/.triton + ~/.nv/ComputeCache + /dev/shm wiped"
' || true

log "Starting training (run: $RUN_NAME)..."
# `env TMPDIR=...` is required because SLURM's TaskProlog clears TMPDIR
# inside every srun task; setting it in the outer shell does not propagate.
exec srun env TMPDIR="$TMPDIR" "$VENV/bin/python" -m canvit_pretrain.train \
    --run-name "$RUN_NAME" \
    --webdataset-dir "$WEBDATASET_DIR" \
    --ckpt-dir "$CHECKPOINTS_DIR" \
    --wandb-project "jon_exp17_canvit_first_tryouts" \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
    ${WANDB_DIR:+--wandb-dir "$WANDB_DIR"} \
    --batch-size-per-gpu 32 \
    --steps-per-job 4096 \
    --val-every 4096 \
    --log-every 512 \
    --num-workers 4 \
    --peak-lr 0.0004 \
    "$@"
