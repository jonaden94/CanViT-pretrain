#!/bin/bash
# One-time setup for the V100 virtual environment (.venv-v100).
#
# Must be run from the repo root:
#   bash _setup_v100/setup.sh
#
# See _setup_v100/README.md for full usage and update instructions.

set -eu
cd "$(dirname "$0")/.."

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== V100 venv setup ==="
log "Host: $(hostname)"

# Proxy needed for package downloads on Grete
source slurm_nhr/env.sh

VENV=".venv-v100"

# Step 1: create the venv (skip if already exists)
if [ ! -d "$VENV" ]; then
    log "Creating $VENV..."
    uv venv "$VENV"
else
    log "$VENV already exists, skipping creation"
fi

# Step 2: install the locked V100 (cu126) resolution. torch==2.11.0 /
# torchvision==0.26.0 are pinned in the `v100` dependency group in
# pyproject.toml and resolved from the cu126 index in uv.lock, so this is fully
# reproducible — no unpinned post-hoc `uv pip install --reinstall`.
log "Syncing V100 (cu126) deps from lock file..."
UV_PROJECT_ENVIRONMENT="$VENV" uv sync --no-group cuda --group v100

log "=== Done ==="
log "Torch version in $VENV:"
"$VENV/bin/python" -c "import torch; print(' ', torch.__version__); print('  CUDA available:', torch.cuda.is_available())"
log "Run training with: bash slurm_nhr/train-v100.sh"
