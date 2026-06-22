#!/bin/bash
# exp21-uniform16-L plus a wider canvas stream: canvas_num_heads 8->10 (x head_dim
# 128 = canvas_dim 1280, up from 1024). canvas_head_dim stays 128 so the canvas
# RoPE structure is unchanged; only the canvas width grows. The scene_patches head
# then projects 1280 -> 1024 (teacher_dim). Everything else identical to
# exp21-uniform16-L (DINOv3 ViT-L target, on-the-fly).
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp21_modulation
RUN_NAME=exp21-uniform16-L-canvas1280
ARRAY=0-48%1                                   # 49 jobs x 4096 = 200704 steps (full run)
TIME=0-01:30:00
MEM=128G
NGPU=1

# === OPTIONAL ===
CFG_WANDB_PROJECT=jon_exp21_modulation
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=4096
CFG_VAL_EVERY=4096  # validate once per job
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
# DINOv3 ViT-L distillation target (vs the default ViT-B). teacher_dim is
# auto-detected from the loaded teacher (1024); do NOT set --model.teacher-dim.
CFG_TEACHER_REPO_ID=facebook/dinov3-vitl16-pretrain-lvd1689m
CFG_TEACHER_NAME=dinov3_vitl16
EXTRA_ARGS="--model.patcher-name uniform --backbone-name vitb16 --model.canvas-num-heads 10 --webdataset-dir /mnt/lustre-rzg/workspaces/ws/nib00021/u25995-canvit-data-no-features/webdataset-imagenet-1k-no-features"
# =================

# Pin all pretraining code to exact commits. base_train.sbatch extracts these
# via offline `git archive` from the local clones (no network/SSH), snapshotting
# the run against any future `git pull` on the originals while the array is in flight.
PRETRAIN_COMMIT=bc2db02
PYTORCH_COMMIT=d864b83
FOVI_COMMIT=763bf7a

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-pretrain
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* ]] && export "$v"; done

sbatch     --gpus-per-node=A100:$NGPU     --ntasks-per-node=$NGPU     --mem=$MEM     --time=$TIME     --array="$ARRAY"     --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --export=ALL     slurm_nhr/base_train.sbatch
