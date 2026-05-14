#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 0-00:30:00
#SBATCH -A nib00021
#SBATCH -C inet
#SBATCH --delay-boot=1
#SBATCH --mem=128G
#SBATCH --exclude=ggpu132,ggpu211
#SBATCH -o logs/exp_workers/w2-job-%A_%a.log
#SBATCH -e logs/exp_workers/w2-job-%A_%a.log
#SBATCH --array=0-4

# DataLoader workers ablation: 1024 training steps with --num-workers 0.
# 5 reps via the SLURM array; reps run in parallel.
# Wandb disabled, validation disabled, checkpoints written to per-job
# scratch so they auto-clean.

WORKERS=2

cd /user/henrich1/u25995/jonathan/repos/CanViT-pretrain
source /user/henrich1/u25995/.bashrc

set -eu

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== exp_workers — workers=$WORKERS rep=$SLURM_ARRAY_TASK_ID ==="
log "Host: $(hostname)  Job: ${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?}"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

source slurm_nhr/env.sh

# Same TMPDIR fallback as exp17 — handles nodes whose /local is saturated.
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
    log "ERROR: WEBDATASET_DIR is not set."
    exit 1
fi

RUN_NAME="exp_workers-w${WORKERS}-r${SLURM_ARRAY_TASK_ID}"
CKPT_DIR="$TMPDIR/exp_workers_ckpt"

export MASTER_PORT=$((20000 + (SLURM_JOB_ID % 10000)))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

# Wipe stale Triton/Inductor/NVRTC/etc caches to keep compile cost
# comparable across reps and across conditions.
log "Clearing per-node caches..."
srun --ntasks-per-node=1 --ntasks=$SLURM_NNODES bash -c '
    rm -rf /tmp/triton_cache_rank* /tmp/torchinductor_* /tmp/nvrtc_cache_* /tmp/.nv 2>/dev/null || true
    rm -rf $HOME/.triton/cache 2>/dev/null || true
    rm -rf $HOME/.nv/ComputeCache 2>/dev/null || true
    find /dev/shm -maxdepth 1 -user $USER \( -name "__KMP_REGISTERED_LIB_*" -o -name "torch_*" -o -name "python_*" -o -name "pymp-*" \) -delete 2>/dev/null || true
    echo "[cache-clear] $(hostname): caches wiped"
' || true

log "Starting training (run: $RUN_NAME)..."
exec srun env TMPDIR="$TMPDIR" "$VENV/bin/python" -m canvit_pretrain.train \
    --run-name "$RUN_NAME" \
    --webdataset-dir "$WEBDATASET_DIR" \
    --ckpt-dir "$CKPT_DIR" \
    --tracker none \
    --batch-size-per-gpu 64 \
    --steps-per-job 1024 \
    --val-every 100000 \
    --log-every 128 \
    --num-workers "$WORKERS" \
    --peak-lr 0.0004 \
    "$@"
