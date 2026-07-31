#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-12:00:00
#SBATCH -A nib00021
#SBATCH -C inet                  # HF Hub: backbone + probe (+ policy publishing)
#SBATCH --mem=64G
#SBATCH -o logs/ade20k/policy-job-%j.log
#SBATCH -e logs/ade20k/policy-job-%j.log
#SBATCH --job-name=ade20k-policy

# Frozen-model viewing-policy training in the UNIFIED repo (P3 gate run):
#   python -m canvit_pretrain.ade20k.rl_train
# Default = the RL repo's canonical QReg recipe (640k glimpse-forwards -> 8000
# steps). Gate: mean val CE t1-t4 inside the documented 8-seed qband band
# (CanViT-PyTorch-RL docs/qband_results.md).
#
# Env:
#   ADE20K_ROOT  (required)
#   PRETRAIN_COMMIT / PYTORCH_COMMIT / FOVI_COMMIT  (optional) commit pinning
# Extra CLI args forwarded (e.g. --seed 0 --run-name qband-port).
#
# Submit example (the P3 gate):
#   export ADE20K_ROOT=/user/henrich1/u25995/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016
#   sbatch --export=ALL slurm/archive/ade20k/train_policy.sbatch --run-name qband-port --seed 0

cd /user/henrich1/u25995/jonathan/repos/CanViT-train
source /user/henrich1/u25995/.bashrc

set -eu

log() { echo "[$(date '+%H:%M:%S')] $*"; }

: "${ADE20K_ROOT:?ADE20K_ROOT must be set (ADEChallengeData2016 root)}"

log "=== ADE20K viewing-policy training (unified, P3) ==="
log "Job: ${SLURM_JOB_ID:-none}, Host: $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

source slurm/env.sh
mkdir -p logs/ade20k

# Commit pinning (same mechanism as base_train.sbatch / train_ade20k.sbatch)
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
if [ ! -d "$VENV" ]; then
    log "ERROR: $VENV not found. Run: UV_PROJECT_ENVIRONMENT=.venv-cu126 uv sync --no-group cuda --group cu126"
    exit 1
fi

log "Starting policy training..."
exec env TMPDIR="$TMPDIR" \
    PYTHONSAFEPATH=1 \
    ${PYTHONPATH:+PYTHONPATH="$PYTHONPATH"} \
    ADE20K_ROOT="$ADE20K_ROOT" \
    "$VENV/bin/python" -m canvit_pretrain.ade20k.rl_train \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
    ${WANDB_DIR:+--wandb-dir "$WANDB_DIR"} \
    "$@"
