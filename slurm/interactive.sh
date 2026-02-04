#!/bin/bash
# Interactive GPU session on Nibi
# Usage: bash ~/scratch/CanViT-train/slurm/interactive.sh [time]
#   bash ~/scratch/CanViT-train/slurm/interactive.sh         # 1h
#   bash ~/scratch/CanViT-train/slurm/interactive.sh 2:00:00 # 2h

TIME="${1:-0:20:00}"

exec srun --account=rrg-skrishna_gpu --gres=gpu:1 --mem=16G --cpus-per-task=16 --time="$TIME" --pty bash -c "
cd ~/scratch/CanViT-train
source slurm/env.sh
uv sync
exec bash
"
