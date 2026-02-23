#!/bin/bash
#SBATCH --job-name=bench-dataloader
#SBATCH --account=def-skrishna
#SBATCH --time=0:15:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/bench_dataloader_%j.out
#SBATCH --error=logs/bench_dataloader_%j.err

# CPU-only benchmark of SA-1B dataloader.
# Tests throughput at different image sizes and worker counts.

set -euo pipefail
source slurm/env.sh
mkdir -p logs

SHARD_DIR="$SA1B_FEATURES_DIR/sa1b/dinov3_vitb16/1024/shards"
SHARD="$SHARD_DIR/sa_000020.pt"
TAR="$SA1B_TAR_DIR/sa_000020.tar"

echo "========================================"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "CPUs:         $SLURM_CPUS_PER_TASK"
echo "Date:         $(date -Iseconds)"
echo "========================================"

[[ -f "$TAR" ]]   || { echo "FATAL: Tar not found: $TAR" >&2; exit 1; }
[[ -f "$SHARD" ]] || { echo "FATAL: Shard not found: $SHARD" >&2; exit 1; }

time uv run python sa1b/bench_dataloader.py \
    --tar "$TAR" \
    --shard "$SHARD" \
    --image-sizes 1024 1500 \
    --workers 0 1 2 4 8 \
    --n-images 200
