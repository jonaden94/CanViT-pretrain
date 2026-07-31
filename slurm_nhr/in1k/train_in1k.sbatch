#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-02:00:00
#SBATCH -A nib00021
#SBATCH -C inet                  # HuggingFace Hub access (remote model repos)
#SBATCH --mem=64G
#SBATCH -o logs/in1k/job-%j.log
#SBATCH -e logs/in1k/job-%j.log
#SBATCH --job-name=in1k-clf

# IN1k classification training in the UNIFIED repo (python -m canvit_pretrain.in1k
# — unification master plan P5). Frozen-backbone linear probe (default) or finetune
# over a glimpse rollout; WebDataset train shards + ImageFolder val.
#
# COMMIT-PINNED like train_ade20k.sbatch: export PRETRAIN_COMMIT / PYTORCH_COMMIT /
# FOVI_COMMIT for a frozen snapshot immune to later edits.
#
# Env:
#   MODEL_REPO    (required) HF repo id OR local HF-format PRETRAINING checkpoint dir
#   IN1K_TRAIN_DIR / IN1K_VAL_DIR  (optional) override the config defaults
#   PRETRAIN_COMMIT / PYTORCH_COMMIT / FOVI_COMMIT  (optional) pin commits
# Extra CLI args are forwarded to the trainer.
#
# Single-GPU SMOKE example (validates GPU rollout / amp / data / eval cheaply):
#   export MODEL_REPO=/user/henrich1/u25995/jonathan/repos/CanViT-train/logs/jon_exp22_full_runs/exp22-uniform16/checkpoints/step-1515520-hf
#   sbatch --export=ALL slurm_nhr/in1k/train_in1k.sbatch \
#       --mode frozen --max-steps 20 --val-every 20 --n-timesteps 3 --batch-size 16 \
#       --limit-val-batches 10 --tracker none

cd /user/henrich1/u25995/jonathan/repos/CanViT-train
source /user/henrich1/u25995/.bashrc

set -eu  # NOT -x: would trace secret exports into logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

: "${MODEL_REPO:?MODEL_REPO must be set (HF repo id or local HF-format pretraining checkpoint dir)}"

log "=== IN1k classification (unified) :: $MODEL_REPO ==="
log "Job: ${SLURM_JOB_ID:-none}, Host: $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm_nhr/env.sh
mkdir -p logs/in1k

# Commit pinning (offline git archive → PYTHONPATH, PYTHONSAFEPATH=1 authoritative).
_REPO_BASE=/user/henrich1/u25995/jonathan/repos
_CODE_DIR="$TMPDIR/canvit_src"
if [ -n "${PRETRAIN_COMMIT:-}" ] || [ -n "${PYTORCH_COMMIT:-}" ] || [ -n "${FOVI_COMMIT:-}" ]; then
    mkdir -p "$_CODE_DIR"
    if [ -n "${PRETRAIN_COMMIT:-}" ]; then
        mkdir -p "$_CODE_DIR/CanViT-train"
        git -C "$_REPO_BASE/CanViT-train" archive "$PRETRAIN_COMMIT" | tar -x -C "$_CODE_DIR/CanViT-train"
        export PYTHONPATH="$_CODE_DIR/CanViT-train${PYTHONPATH:+:$PYTHONPATH}"
        log "Pinned canvit_train → $PRETRAIN_COMMIT"
    fi
    if [ -n "${PYTORCH_COMMIT:-}" ]; then
        mkdir -p "$_CODE_DIR/CanViT-PyTorch"
        git -C "$_REPO_BASE/CanViT-PyTorch" archive "$PYTORCH_COMMIT" | tar -x -C "$_CODE_DIR/CanViT-PyTorch"
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

VENV=".venv-cu126"
[ -d "$VENV" ] || { log "ERROR: $VENV not found."; exit 1; }

if [ -z "${MODEL_REPO##/*}" ]; then
    [ -d "$MODEL_REPO" ] || { log "ERROR: local HF checkpoint missing: $MODEL_REPO"; exit 1; }
fi

log "Starting IN1k classification training..."
exec env TMPDIR="$TMPDIR" \
    PYTHONSAFEPATH=1 \
    ${PYTHONPATH:+PYTHONPATH="$PYTHONPATH"} \
    ${IN1K_TRAIN_DIR:+IN1K_TRAIN_DIR="$IN1K_TRAIN_DIR"} \
    ${IN1K_VAL_DIR:+IN1K_VAL_DIR="$IN1K_VAL_DIR"} \
    "$VENV/bin/python" -m canvit_pretrain.in1k \
    --model-repo "$MODEL_REPO" \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
    ${WANDB_DIR:+--wandb-dir "$WANDB_DIR"} \
    "$@"
