#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub access
#SBATCH --delay-boot=1
#
# Unified-harness training launcher: the `python -m canvit_pretrain.harness.run`
# counterpart of base_train.sbatch. ONE script for all three tasks — the task is
# chosen by the TASK env var (distill | ade20k | in1k), which is the whole point
# of the harness.
#
# This sits ALONGSIDE the existing per-task launchers (base_train.sbatch,
# ade20k/train_ade20k.sbatch, in1k/train_in1k.sbatch), which are unchanged and
# still work. Nothing is cut over.
#
# ALL variation comes from:
#   - sbatch CLI flags (--gpus-per-node, --ntasks-per-node, --mem, --time,
#     --array, --exclude, --output, --error)
#   - env vars:
#       TASK        distill (default) | ade20k | in1k
#       RUN_GROUP / RUN_NAME / NGPU     (RUN_* required for distill only)
#       CFG_FOO_BAR=value  → --cfg.foo-bar value      (task config)
#       OPT_FOO_BAR=value  → --opts.foo-bar value     (harness knobs, e.g. OPT_N_STEPS)
#       EXTRA_ARGS         verbatim extra flags, the only way to reach NESTED
#                          config trees (`--cfg.model.patcher-name foveated`),
#                          since FOO_BAR→foo-bar cannot encode a dot.
#
# Launch via: SBATCH_SCRIPT=slurm_nhr/harness_train.sbatch bash slurm_nhr/submit.sh
# or directly: TASK=ade20k sbatch --export=ALL ... slurm_nhr/harness_train.sbatch

cd /user/henrich1/u25995/jonathan/repos/CanViT-pretrain
source /user/henrich1/u25995/.bashrc

set -eu

TASK="${TASK:-distill}"
NGPU="${NGPU:-1}"
EXTRA_ARGS=${EXTRA_ARGS:-}

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== harness/${TASK} (${NGPU}× A100, grete:shared) ==="
log "Array Job: ${SLURM_ARRAY_JOB_ID:-none}, Task: ${SLURM_ARRAY_TASK_ID:-none}, Host: $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

source slurm_nhr/env.sh

# TMPDIR fallback + commit pinning + per-job compile caches: identical to
# base_train.sbatch (see its comments for why each is needed).
TMPDIR_NEED_GB=2
ERSATZ_BASE=/mnt/vast-nhr/projects/nib00021/jonathan/ersatz_local
tmpdir_avail_gb=$(df -BG --output=avail "$TMPDIR" 2>/dev/null | tail -1 | tr -dc '0-9')
if [ "${tmpdir_avail_gb:-0}" -lt "$TMPDIR_NEED_GB" ]; then
    log "WARN: $TMPDIR on $(hostname) has ${tmpdir_avail_gb:-?}G free (<${TMPDIR_NEED_GB}G); falling back to NFS"
    export TMPDIR="$ERSATZ_BASE/u25995_${SLURM_JOB_ID}"
    mkdir -p "$TMPDIR"
fi
log "TMPDIR=$TMPDIR"

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
        git -C "$_REPO_BASE/fovi" archive "$FOVI_COMMIT" | tar -x -C "$_CODE_DIR/fovi"
        export PYTHONPATH="$_CODE_DIR/fovi${PYTHONPATH:+:$PYTHONPATH}"
        log "Pinned fovi → $FOVI_COMMIT"
    fi
fi

export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export CUDA_CACHE_PATH="$TMPDIR/nv"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

VENV=".venv-cu126"
if [ ! -d "$VENV" ]; then
    log "ERROR: $VENV not found. Run: UV_PROJECT_ENVIRONMENT=.venv-cu126 uv sync --no-group cuda --group cu126"
    exit 1
fi

export MASTER_PORT=$((20000 + (SLURM_JOB_ID % 10000)))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
log "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  WORLD_SIZE=$WORLD_SIZE"

# CFG_FOO_BAR → --cfg.foo-bar ; OPT_FOO_BAR → --opts.foo-bar
_ARGS=()
for name in $(compgen -v); do
    case "$name" in
        CFG_*) prefix="--cfg."; stem="${name#CFG_}" ;;
        OPT_*) prefix="--opts."; stem="${name#OPT_}" ;;
        *) continue ;;
    esac
    val="${!name:-}"
    [ -z "$val" ] && continue
    _ARGS+=("${prefix}$(echo "$stem" | tr '[:upper:]_' '[:lower:]-')" "$val")
done

# Per-task required flags. Only distill takes the run-group/logs-dir/data trio;
# ade20k and in1k read their data roots from ADE20K_ROOT / IN1K_* in env.sh and
# have no run_group concept, so passing those flags there would be a hard error.
_TASK_ARGS=()
case "$TASK" in
    distill)
        : "${RUN_GROUP:?RUN_GROUP must be set for TASK=distill}"
        : "${RUN_NAME:?RUN_NAME must be set for TASK=distill}"
        : "${LOGS_DIR:?LOGS_DIR must be set in .envrc.grete}"
        : "${WEBDATASET_DIR:?WEBDATASET_DIR must be set in .envrc.grete}"
        : "${VAL_DIR:?VAL_DIR must be set in .envrc.grete}"
        _TASK_ARGS+=(--cfg.run-group "$RUN_GROUP" --cfg.run-name "$RUN_NAME"
                     --cfg.logs-dir "$LOGS_DIR" --cfg.webdataset-dir "$WEBDATASET_DIR"
                     --cfg.val-dir "$VAL_DIR")
        # if-statements rather than `[ x ] && arr+=(...)`: the &&-form yields status 1
        # when the test is false, which is harmless mid-script under `set -e` but bites
        # if such a line ever ends up last in a script or function.
        if [ -n "${VAL_INDEX_DIR:-}" ]; then _TASK_ARGS+=(--cfg.val-index-dir "$VAL_INDEX_DIR"); fi
        ;;
    ade20k|in1k)
        # These two have no run_group in their configs, so they cannot derive a run dir
        # the way distill does — without one there is nowhere to put visualization/ and
        # the segmentation figures silently stay off. Give them the same
        # logs/GROUP/NAME convention when the launcher exports those (all of them do).
        # Checkpoints are unaffected: their ckpt_dir is always set (cfg.probe_ckpt_dir /
        # cfg.clf_ckpt_dir default non-None), so it keeps winning over run_dir.
        if [ -n "${RUN_GROUP:-}" ] && [ -n "${RUN_NAME:-}" ]; then
            _TASK_ARGS+=(--opts.run-dir "logs/$RUN_GROUP/$RUN_NAME")
        fi
        ;;
    *) log "ERROR: TASK must be distill|ade20k|in1k, got '$TASK'"; exit 1 ;;
esac
if [ -n "${WANDB_ENTITY:-}" ]; then _TASK_ARGS+=(--cfg.wandb-entity "$WANDB_ENTITY"); fi
if [ -n "${WANDB_DIR:-}" ]; then _TASK_ARGS+=(--cfg.wandb-dir "$WANDB_DIR"); fi

log "CFG/OPT overrides: ${_ARGS[*]:-none}"
if [ -n "$EXTRA_ARGS" ]; then log "EXTRA_ARGS: $EXTRA_ARGS"; fi
log "Starting harness task '$TASK'..."
exec srun env TMPDIR="$TMPDIR" \
    PYTHONSAFEPATH=1 \
    ${PYTHONPATH:+PYTHONPATH="$PYTHONPATH"} \
    TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" \
    TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
    CUDA_CACHE_PATH="$CUDA_CACHE_PATH" \
    TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
    "$VENV/bin/python" -m canvit_pretrain.harness.run "$TASK" \
    "${_TASK_ARGS[@]}" \
    "${_ARGS[@]}" \
    $EXTRA_ARGS \
    "$@"
