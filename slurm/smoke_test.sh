#!/bin/bash
# Smoke test for Nibi. Run with:
#   srun --account=def-skrishna --mem=32G --cpus-per-task=4 --time=0:30:00 --pty bash slurm/smoke_test.sh
# Or from login node:
#   bash slurm/smoke_test.sh --srun   # spawns srun, runs checks, drops to interactive shell

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

ok() { echo -e "${GREEN}✓${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; exit 1; }

run_checks() {
    echo "=== Smoke Test ==="
    echo ""

    # Env setup
    cd ~/scratch/avp-vit
    source slurm/env.sh

    echo "--- Environment ---"
    echo "  SLURM_TMPDIR:           $SLURM_TMPDIR"
    echo "  UV_CACHE_DIR:           $UV_CACHE_DIR"
    echo "  UV_PROJECT_ENVIRONMENT: $UV_PROJECT_ENVIRONMENT"
    echo "  HF_HOME:                $HF_HOME"
    echo "  TORCH_HOME:             $TORCH_HOME"
    echo "  SCRATCH:                $SCRATCH"
    echo "  AVP_TRAIN_DIR:          $AVP_TRAIN_DIR"
    echo "  AVP_INDEX_DIR:          $AVP_INDEX_DIR"
    echo ""

    # 1. Paths (uses AVP_* from env.sh)
    echo "--- Paths ---"
    [ -f "$AVP_TEACHER_CKPT" ] && ok "Teacher checkpoint" || fail "Teacher checkpoint missing: $AVP_TEACHER_CKPT"
    [ -d "$AVP_TRAIN_DIR" ] && ok "IN21k train dir" || fail "IN21k train dir missing: $AVP_TRAIN_DIR"
    [ -d "$AVP_VAL_DIR" ] && ok "IN1k val dir" || fail "IN1k val dir missing: $AVP_VAL_DIR"
    [ -f ~/comet_api_key.txt ] && ok "Comet API key" || echo "   (optional) Comet API key missing"
    echo ""

    # 2. uv sync
    echo "--- uv sync ---"
    uv sync && ok "uv sync" || fail "uv sync failed (SSH key issue?)"
    echo ""

    # 3. Imports
    echo "--- Imports ---"
    uv run python -c "import canvit" && ok "canvit" || fail "canvit import"
    uv run python -c "import dinov3_probes" && ok "dinov3_probes" || fail "dinov3_probes import"
    uv run python -c "import drac_imagenet" && ok "drac_imagenet" || fail "drac_imagenet import"
    uv run python -c "import avp_vit" && ok "avp_vit" || fail "avp_vit import"
    echo ""

    # 4. Teacher load
    echo "--- Teacher ---"
    uv run python -c "
from canvit import create_backbone
import os
b = create_backbone('dinov3_vitb16', weights=os.path.expanduser(os.environ['AVP_TEACHER_CKPT']))
print(f'  {b.embed_dim}d, {b.n_blocks} blocks')
" && ok "Teacher loads" || fail "Teacher load failed"
    echo ""

    # 5. HF probe
    echo "--- HF Probe ---"
    uv run python -c "
from dinov3_probes import DINOv3LinearClassificationHead
p = DINOv3LinearClassificationHead.from_pretrained('yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe')
print('  loaded')
" && ok "Probe loads" || fail "Probe load failed"
    echo ""

    # 6. Dataset index
    echo "--- Dataset Index ---"
    if [ -f "$AVP_INDEX_DIR/winter21_whole.parquet" ]; then
        ok "IN21k index cached"
    else
        echo "   Not cached: $AVP_INDEX_DIR/winter21_whole.parquet"
        echo "   Will be built on first train run"
    fi
    echo ""

    echo "=== All checks passed ==="
}

if [ "$1" = "--srun" ]; then
    exec srun --account=def-skrishna --mem=32G --cpus-per-task=4 --time=0:30:00 --pty bash -c "
        cd ~/scratch/avp-vit
        bash slurm/smoke_test.sh
        echo ''
        echo '--- Dropping to interactive shell ---'
        exec bash
    "
else
    run_checks
fi
