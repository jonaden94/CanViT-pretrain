#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub access
#SBATCH --delay-boot=1
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=jonathan.henrich@uni-goettingen.de
#
# Generic CanViT pretraining base. ALL variation comes from:
#   - sbatch CLI flags (--gpus-per-node, --ntasks-per-node, --mem, --time,
#     --array, --exclude, --output, --error)  — supplied by submit.sh
#   - env vars (RUN_GROUP, RUN_NAME, NGPU, PEAK_LR, BATCH_SIZE, EXTRA_ARGS)
#     — supplied by submit.sh via --export
#
# Launch via: slurm_nhr/submit.sh --group GROUP --name NAME --ngpu N [...]

cd /user/henrich1/u25995/jonathan/repos/CanViT-pretrain
source /user/henrich1/u25995/.bashrc

set -eu

# Defense-in-depth: only the essentials must be set.
# Optional vars use `${VAR:-default}` below — empty/unset → default applies.
: "${RUN_GROUP:?RUN_GROUP must be set — launch via slurm_nhr/submit.sh}"
: "${RUN_NAME:?RUN_NAME must be set — launch via slurm_nhr/submit.sh}"
: "${NGPU:?NGPU must be set — launch via slurm_nhr/submit.sh}"
EXTRA_ARGS=${EXTRA_ARGS:-}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== ${RUN_GROUP}/${RUN_NAME} (${NGPU}× A100, grete:shared) ==="
log "Array Job: ${SLURM_ARRAY_JOB_ID:-none}, Task: ${SLURM_ARRAY_TASK_ID:-none}, Host: $(hostname)"
log "Nodes: ${SLURM_NNODES:-?}, Tasks/node: ${SLURM_NTASKS_PER_NODE:-?}, GPUs/node: ${SLURM_GPUS_ON_NODE:-?}"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

source slurm_nhr/env.sh

: "${LOGS_DIR:?LOGS_DIR must be set in .envrc.grete}"

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

# CFG_* fallbacks for vars whose Python default would fail at runtime.
# Anything not listed here falls through to canvit_pretrain/train/config.py.
CFG_WANDB_PROJECT="${CFG_WANDB_PROJECT:-jon_exp_canvit_default}"

# Auto-build `--flag value` pairs from every CFG_* env var (CFG_FOO_BAR → --foo-bar).
_CFG_ARGS=()
for name in $(compgen -v); do
    [[ "$name" == CFG_* ]] || continue
    val="${!name:-}"
    [ -z "$val" ] && continue
    flag="--$(echo "${name#CFG_}" | tr '[:upper:]_' '[:lower:]-')"
    _CFG_ARGS+=("$flag" "$val")
done

log "CFG overrides: $(printenv | grep -E '^CFG_' | sort | tr '\n' ' ')"
if [ -n "$EXTRA_ARGS" ]; then log "EXTRA_ARGS: $EXTRA_ARGS"; fi
log "Starting training (group=$RUN_GROUP, name=$RUN_NAME)..."
# `env TMPDIR=...` is required because SLURM's TaskProlog clears TMPDIR
# inside every srun task; setting it in the outer shell does not propagate.
exec srun env TMPDIR="$TMPDIR" "$VENV/bin/python" -m canvit_pretrain.train \
    --run-group "$RUN_GROUP" \
    --run-name "$RUN_NAME" \
    --logs-dir "$LOGS_DIR" \
    --webdataset-dir "$WEBDATASET_DIR" \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
    ${WANDB_DIR:+--wandb-dir "$WANDB_DIR"} \
    "${_CFG_ARGS[@]}" \
    $EXTRA_ARGS \
    "$@"
