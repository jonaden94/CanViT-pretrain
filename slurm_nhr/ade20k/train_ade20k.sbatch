#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-08:00:00
#SBATCH -A nib00021
#SBATCH -C inet                  # HuggingFace Hub access (remote model repos)
#SBATCH --mem=64G
#SBATCH -o logs/ade20k/job-%j.log
#SBATCH -e logs/ade20k/job-%j.log
#SBATCH --job-name=ade20k-probe

# ADE20K frozen-backbone probe training in the UNIFIED repo
# (python -m canvit_pretrain.ade20k — unification master plan P2).
#
# Unlike the retired specialize launcher this one is COMMIT-PINNED (closes
# unification-status §5.6): export PRETRAIN_COMMIT / PYTORCH_COMMIT /
# FOVI_COMMIT to run a frozen snapshot immune to later edits of the clones.
#
# Env:
#   MODEL_REPO   (required) HF repo id OR local HF-format checkpoint dir
#   ADE20K_ROOT  (required) ADEChallengeData2016 root
#   PRETRAIN_COMMIT / PYTORCH_COMMIT / FOVI_COMMIT  (optional) pin commits
# Extra CLI args are forwarded to the trainer (e.g. --max-steps 20000).
#
# Submit example (gate run, same checkpoint as the specialize reference):
#   export MODEL_REPO=/user/henrich1/u25995/jonathan/repos/CanViT-pretrain/logs/jon_exp22_full_runs/exp22-uniform16/checkpoints/step-1515520-hf
#   export ADE20K_ROOT=/user/henrich1/u25995/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016
#   sbatch --export=ALL slurm_nhr/ade20k/train_ade20k.sbatch

cd /user/henrich1/u25995/jonathan/repos/CanViT-pretrain
source /user/henrich1/u25995/.bashrc

set -eu  # NOT -x: would trace secret exports into logs

log() { echo "[$(date '+%H:%M:%S')] $*"; }

: "${MODEL_REPO:?MODEL_REPO must be set (HF repo id or local HF-format checkpoint dir)}"
: "${ADE20K_ROOT:?ADE20K_ROOT must be set (ADEChallengeData2016 root)}"

log "=== ADE20K frozen-probe (unified) :: $MODEL_REPO ==="
log "Job: ${SLURM_JOB_ID:-none}, Host: $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm_nhr/env.sh
mkdir -p logs/ade20k

# Commit pinning (same mechanism as base_train.sbatch): offline `git archive`
# of the local clones into TMPDIR, prepended to PYTHONPATH; PYTHONSAFEPATH=1
# makes it authoritative over cwd and the venv's editable installs.
_REPO_BASE=/user/henrich1/u25995/jonathan/repos
_CODE_DIR="$TMPDIR/canvit_src"
if [ -n "${PRETRAIN_COMMIT:-}" ] || [ -n "${PYTORCH_COMMIT:-}" ] || [ -n "${FOVI_COMMIT:-}" ]; then
    mkdir -p "$_CODE_DIR"
    if [ -n "${PRETRAIN_COMMIT:-}" ]; then
        mkdir -p "$_CODE_DIR/CanViT-pretrain"
        git -C "$_REPO_BASE/CanViT-pretrain" archive "$PRETRAIN_COMMIT" | tar -x -C "$_CODE_DIR/CanViT-pretrain"
        export PYTHONPATH="$_CODE_DIR/CanViT-pretrain${PYTHONPATH:+:$PYTHONPATH}"
        log "Pinned canvit_pretrain → $PRETRAIN_COMMIT"
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
if [ ! -d "$VENV" ]; then
    log "ERROR: $VENV not found. Run: UV_PROJECT_ENVIRONMENT=.venv-cu126 uv sync --no-group cuda --group cu126"
    exit 1
fi

if [ -n "${MODEL_REPO##/*}" ]; then
    log "MODEL_REPO is a hub id"
else
    [ -d "$MODEL_REPO" ] || { log "ERROR: local HF checkpoint missing: $MODEL_REPO"; exit 1; }
fi

log "Starting ADE20K probe training..."
exec env TMPDIR="$TMPDIR" \
    PYTHONSAFEPATH=1 \
    ${PYTHONPATH:+PYTHONPATH="$PYTHONPATH"} \
    ADE20K_ROOT="$ADE20K_ROOT" \
    "$VENV/bin/python" -m canvit_pretrain.ade20k \
    --model-repo "$MODEL_REPO" \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
    ${WANDB_DIR:+--wandb-dir "$WANDB_DIR"} \
    "$@"
