#!/bin/bash
#SBATCH --job-name=ade20k-probe
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/ade20k-probe-%j.out
#SBATCH --error=logs/ade20k-probe-%j.err
#SBATCH --mail-user=me@yberreby.com
#SBATCH --mail-type=BEGIN,END,FAIL

# ADE20k probe training on frozen/finetuned AVP features
#
# Usage:
#   sbatch slurm/ade20k_probe.sbatch --avp-ckpt <path>
#
# Examples:
#   sbatch slurm/ade20k_probe.sbatch --avp-ckpt $CHECKPOINTS_DIR/avp_vit/train-12345/latest.pt
#   sbatch slurm/ade20k_probe.sbatch --avp-ckpt $CHECKPOINTS_DIR/avp_vit/reference.pt --max-steps 1000

set -eux

log() { echo "[$(date '+%H:%M:%S')] $*"; }
timed() { local start=$SECONDS; "$@"; log "  ↳ $((SECONDS - start))s"; }

log "=== ADE20k Probe Training ==="
log "Job: $SLURM_JOB_ID on $(hostname)"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "Args: $*"

cd ~/scratch/avp-vit
source slurm/env.sh

log "Installing dependencies..."
uv sync

mkdir -p logs

# Extract dataset to fast local SSD
log "Extracting ADE20k to SLURM_TMPDIR..."
timed unzip -q "$ADE20K_ZIP" -d "$SLURM_TMPDIR"
ADE20K_DIR="$SLURM_TMPDIR/ADEChallengeData2016"

PROBE_CKPT_DIR="$CHECKPOINTS_DIR/ade20k_probes/$SLURM_JOB_ID"
mkdir -p "$PROBE_CKPT_DIR"

log "Starting training..."
log "Probe checkpoint dir: $PROBE_CKPT_DIR"
exec uv run python scripts/train_ade20k_probe.py \
    --ade20k-root "$ADE20K_DIR" \
    --teacher-ckpt "$DINOV3_VITB16_CKPT" \
    --probe-ckpt-dir "$PROBE_CKPT_DIR" \
    "$@"
