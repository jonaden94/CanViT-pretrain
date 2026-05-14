#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:2
#SBATCH --ntasks-per-node=2      # MUST equal --gpus-per-node (one task per GPU)
#SBATCH --cpus-per-task=16
#SBATCH -t 0-00:30:00
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub access
#SBATCH --delay-boot=1
#SBATCH --mem=256G
#SBATCH --exclude=ggpu131,ggpu132,ggpu134,ggpu105
#SBATCH -o logs/test-ddp-cleanup-fixes/job-%A_%a.log
#SBATCH -e logs/test-ddp-cleanup-fixes/job-%A_%a.log
#SBATCH --array=0                # single-task test
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=jonathan.henrich@uni-goettingen.de

# Quick sanity test for the two dist.py fixes:
#   1) atexit.register(destroy_process_group) — silences "destroy_process_group()
#      was not called before program exit" warning.
#   2) init_process_group(..., device_id=_device) — silences "Guessing device ID
#      based on global rank" warning.
#
# Run completes 100 training steps (well within time limit) so the Python process
# exits normally, atexit handlers fire, and we can verify both warnings are gone
# from the log. No wandb logging. Same DDP config as train_2gpu_match1gpu.sbatch.
#
# Submit with:
#   sbatch slurm_nhr/exp17/test_ddp_cleanup.sbatch

cd /user/henrich1/u25995/jonathan/repos/CanViT-pretrain
source /user/henrich1/u25995/.bashrc

set -eu

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== TEST: DDP cleanup + device_id fixes (2× A100, grete:shared) ==="
log "Array Job: ${SLURM_ARRAY_JOB_ID:-none}, Task: ${SLURM_ARRAY_TASK_ID:-none}, Host: $(hostname)"
log "Nodes: ${SLURM_NNODES:-?}, Tasks/node: ${SLURM_NTASKS_PER_NODE:-?}, GPUs/node: ${SLURM_GPUS_ON_NODE:-?}"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm_nhr/env.sh

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

RUN_NAME="test-ddp-cleanup-fixes"

export MASTER_PORT=$((20000 + (SLURM_JOB_ID % 10000)))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

log "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  WORLD_SIZE=$WORLD_SIZE"

log "Clearing per-node caches (Triton, Inductor, NVRTC, NVIDIA Compute, KMP shm, .nv)..."
srun --ntasks-per-node=1 --ntasks=$SLURM_NNODES bash -c '
    rm -rf /tmp/triton_cache_rank* /tmp/torchinductor_* /tmp/nvrtc_cache_* /tmp/.nv 2>/dev/null || true
    rm -rf $HOME/.triton/cache 2>/dev/null || true
    rm -rf $HOME/.nv/ComputeCache 2>/dev/null || true
    find /dev/shm -maxdepth 1 -user $USER \( -name "__KMP_REGISTERED_LIB_*" -o -name "torch_*" -o -name "python_*" -o -name "pymp-*" \) -delete 2>/dev/null || true
    echo "[cache-clear] $(hostname): /tmp + ~/.triton + ~/.nv/ComputeCache + /dev/shm wiped"
' || true

log "Starting test training (run: $RUN_NAME)..."
exec srun env TMPDIR="$TMPDIR" "$VENV/bin/python" -m canvit_pretrain.train \
    --run-name "$RUN_NAME" \
    --webdataset-dir "$WEBDATASET_DIR" \
    --ckpt-dir "$CHECKPOINTS_DIR" \
    --tracker none \
    --batch-size-per-gpu 32 \
    --steps-per-job 128 \
    --val-every 256 \
    --log-every 32 \
    --num-workers 4 \
    --peak-lr 0.0004 \
    "$@"
