#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH -A nib00021
#SBATCH -C inet                  # required for HuggingFace Hub access
#SBATCH --delay-boot=1
#
# Unified-harness training launcher: the `python -m canvit_train.harness.run`
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

cd /user/henrich1/u25995/jonathan/repos/CanViT-train
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
        mkdir -p "$_CODE_DIR/CanViT-train"
        git -C "$_REPO_BASE/CanViT-train" archive "$PRETRAIN_COMMIT" \
            | tar -x -C "$_CODE_DIR/CanViT-train"
        export PYTHONPATH="$_CODE_DIR/CanViT-train${PYTHONPATH:+:$PYTHONPATH}"
        log "Pinned canvit_train → $PRETRAIN_COMMIT"
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

# The package was renamed canvit_pretrain -> canvit_train on 2026-07-31. This launcher is
# shared between NEW runs (post-rename code, package canvit_train) and the exp23/exp27
# arms, which pin PRE-rename commits whose git-archive snapshot still contains
# canvit_pretrain/. Hardcoding either name silently breaks half of them, so derive it from
# what the snapshot actually holds. No pin => no snapshot => the venv's editable install,
# which is always the current name.
_PKG=canvit_train
if [ -d "$_CODE_DIR/CanViT-train/canvit_pretrain" ]; then
    _PKG=canvit_pretrain
    log "Pinned commit predates the canvit_train rename → entry point is $_PKG.harness.run"
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

# Run identity is now the SAME for all three tasks (--cfg.run-group / --cfg.run-name /
# --cfg.logs-dir), so each derives `LOGS_DIR/GROUP/NAME/{checkpoints,visualization}` and
# its own wandb run name. ade20k/in1k used to get neither: their checkpoints went to one
# flat shared dir and every ade20k run was named "ade20k" in wandb, which is why the
# launchers had to pass OPT_CKPT_DIR by hand. Those still work — an explicit CFG_/OPT_
# override lands in _ARGS, after these, and tyro takes the last flag.
: "${RUN_GROUP:?RUN_GROUP must be set (all tasks)}"
: "${RUN_NAME:?RUN_NAME must be set (all tasks)}"
: "${LOGS_DIR:?LOGS_DIR must be set in .envrc.grete}"
_TASK_ARGS=(--cfg.run-group "$RUN_GROUP" --cfg.run-name "$RUN_NAME" --cfg.logs-dir "$LOGS_DIR")

# Per-task required flags: only distill takes the data-path trio (ade20k and in1k read
# their data roots from ADE20K_ROOT / IN1K_* in env.sh), so passing those there would be
# a hard error.
case "$TASK" in
    distill)
        : "${WEBDATASET_DIR:?WEBDATASET_DIR must be set in .envrc.grete}"
        : "${VAL_DIR:?VAL_DIR must be set in .envrc.grete}"
        _TASK_ARGS+=(--cfg.webdataset-dir "$WEBDATASET_DIR" --cfg.val-dir "$VAL_DIR")
        # if-statements rather than `[ x ] && arr+=(...)`: the &&-form yields status 1
        # when the test is false, which is harmless mid-script under `set -e` but bites
        # if such a line ever ends up last in a script or function.
        if [ -n "${VAL_INDEX_DIR:-}" ]; then _TASK_ARGS+=(--cfg.val-index-dir "$VAL_INDEX_DIR"); fi
        ;;
    ade20k|in1k)
        # Nothing task-specific left: identity is set above, data roots come from env.sh.
        # ade20k is SINGLE-GPU only — its loader takes no world_size/rank, so ranks would
        # draw overlapping samples. The task asserts this too (caps().supports_ddp and
        # build_loaders); catching it here kills the job instantly instead of after the
        # model load, and names the variable that is actually wrong.
        if [ "$TASK" = ade20k ] && [ "${NGPU:-1}" -gt 1 ]; then
            log "ERROR: TASK=ade20k does not support DDP (NGPU=$NGPU) — use NGPU=1"
            exit 1
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
    "$VENV/bin/python" -m "$_PKG.harness.run" "$TASK" \
    "${_TASK_ARGS[@]}" \
    "${_ARGS[@]}" \
    $EXTRA_ARGS \
    "$@"
