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

# If PRETRAIN_COMMIT / PYTORCH_COMMIT / FOVI_COMMIT are exported by the run
# script, extract those exact commits into TMPDIR and prepend them to PYTHONPATH
# so they win over the venv's editable install. All three vars are optional and
# independent. `git archive <commit>` reads the LOCAL clone's object store only
# (no clone/fetch/pull), so no network or SSH auth is involved even though these
# are private repos — the commit just has to exist in the local clone. fovi is a
# pretraining dependency too (foveated/square patchers import it), so pin it as
# well for fully reproducible runs.
_REPO_BASE=/user/henrich1/u25995/jonathan/repos
_CODE_DIR="$TMPDIR/canvit_src"
if [ -n "${PRETRAIN_COMMIT:-}" ] || [ -n "${PYTORCH_COMMIT:-}" ] || [ -n "${FOVI_COMMIT:-}" ]; then
    mkdir -p "$_CODE_DIR"
    if [ -n "${PRETRAIN_COMMIT:-}" ]; then
        mkdir -p "$_CODE_DIR/CanViT-pretrain"
        git -C "$_REPO_BASE/CanViT-pretrain" archive "$PRETRAIN_COMMIT" \
            | tar -x -C "$_CODE_DIR/CanViT-pretrain"
        export PYTHONPATH="$_CODE_DIR/CanViT-pretrain${PYTHONPATH:+:$PYTHONPATH}"
        log "Pinned canvit_pretrain → $PRETRAIN_COMMIT"
    fi
    if [ -n "${PYTORCH_COMMIT:-}" ]; then
        mkdir -p "$_CODE_DIR/CanViT-PyTorch"
        git -C "$_REPO_BASE/CanViT-PyTorch" archive "$PYTORCH_COMMIT" \
            | tar -x -C "$_CODE_DIR/CanViT-PyTorch"
        export PYTHONPATH="$_CODE_DIR/CanViT-PyTorch${PYTHONPATH:+:$PYTHONPATH}"
        log "Pinned canvit_pytorch → $PYTORCH_COMMIT"
    fi
    if [ -n "${FOVI_COMMIT:-}" ]; then
        mkdir -p "$_CODE_DIR/fovi"
        git -C "$_REPO_BASE/fovi" archive "$FOVI_COMMIT" \
            | tar -x -C "$_CODE_DIR/fovi"
        export PYTHONPATH="$_CODE_DIR/fovi${PYTHONPATH:+:$PYTHONPATH}"
        log "Pinned fovi → $FOVI_COMMIT"
    fi
fi

# Per-JOB, node-local compile caches (Inductor + Triton + NVIDIA JIT) under TMPDIR.
# TMPDIR is /local/jobs/<jobid> (per-job NVMe, auto-cleaned by SLURM at job end),
# or the per-job NFS ersatz set above. Putting all three here gives each job a
# private, fast cache with ZERO cross-job interference — no shared ~/.triton or
# ~/.nv for ~20 concurrent jobs to race on, and nothing to wipe (SLURM reclaims it).
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export CUDA_CACHE_PATH="$TMPDIR/nv"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
log "Compile caches (node-local, per-job): inductor+triton+cuda under $TMPDIR"

VENV=".venv-v100"
if [ ! -d "$VENV" ]; then
    log "ERROR: $VENV not found. Run: bash _setup_v100/setup.sh"
    exit 1
fi

if [ -z "${WEBDATASET_DIR:-}" ]; then
    log "ERROR: WEBDATASET_DIR is not set. Add it to .envrc.grete (parent of train-shuffled/)."
    exit 1
fi

if [ -z "${VAL_DIR:-}" ]; then
    log "ERROR: VAL_DIR is not set. Add it to .envrc.grete (raw ImageNet-1k val ImageFolder, synset-named class subfolders)."
    exit 1
fi

# Per-job MASTER_PORT keeps this run from colliding with concurrent runs on
# the same node (grete:shared can pack multiple jobs onto one node).
export MASTER_PORT=$((20000 + (SLURM_JOB_ID % 10000)))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

log "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  WORLD_SIZE=$WORLD_SIZE"

# (Removed) Per-node cache wipe. Compile caches now live in the per-job TMPDIR
# (set above), which SLURM auto-cleans at job end — so there are no stale shared
# caches to clear. The old wipe also raced across the ~20 concurrent jobs that
# shared ~/.triton and /dev/shm (one job deleting another's live kernels/shm
# segments), which is exactly the interference we're eliminating here.

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
#
# PYTHONSAFEPATH=1: `python -m` otherwise prepends the cwd (this repo root) to
# sys.path[0], which shadows the pinned canvit_pretrain snapshot on PYTHONPATH
# (canvit_pytorch/fovi aren't under cwd so they were unaffected). With safe-path
# on, PYTHONPATH is authoritative — all three pinned packages win; when unpinned,
# imports fall back to the venv's editable install as before.
exec srun env TMPDIR="$TMPDIR" \
    PYTHONSAFEPATH=1 \
    ${PYTHONPATH:+PYTHONPATH="$PYTHONPATH"} \
    TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" \
    TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    CUDA_CACHE_PATH="$CUDA_CACHE_PATH" \
    TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
    "$VENV/bin/python" -m canvit_pretrain.train \
    --run-group "$RUN_GROUP" \
    --run-name "$RUN_NAME" \
    --logs-dir "$LOGS_DIR" \
    --webdataset-dir "$WEBDATASET_DIR" \
    --val-dir "$VAL_DIR" \
    ${VAL_INDEX_DIR:+--val-index-dir "$VAL_INDEX_DIR"} \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
    ${WANDB_DIR:+--wandb-dir "$WANDB_DIR"} \
    "${_CFG_ARGS[@]}" \
    $EXTRA_ARGS \
    "$@"
