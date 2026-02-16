#!/bin/bash
#SBATCH --job-name=ade20k-dinov3-eval
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/ade20k-dinov3-eval-%j.out
#SBATCH --error=logs/ade20k-dinov3-eval-%j.err

# DINOv3 baseline probe evaluation (deterministic, single run)
# Usage: sbatch slurm/eval_ade20k_dinov3_probe.sbatch /path/to/best.pt

set -eux

PROBE_CKPT="${1:?Usage: sbatch eval_ade20k_dinov3_probe.sbatch <probe_ckpt>}"

cd ~/scratch/CanViT-train
source slurm/env.sh
mkdir -p logs

echo "=== DINOv3 Probe Evaluation ==="
echo "Probe: $PROBE_CKPT"
echo "Job: $SLURM_JOB_ID on $(hostname)"

# Extract ADE20K
export ADE20K_ROOT="$SLURM_TMPDIR/ADEChallengeData2016"
[ ! -d "$ADE20K_ROOT" ] && unzip -q "$ADE20K_ZIP" -d "$SLURM_TMPDIR"

exec uv run python -m canvit_eval.ade20k eval-dinov3-probe --probe-ckpt "$PROBE_CKPT"
