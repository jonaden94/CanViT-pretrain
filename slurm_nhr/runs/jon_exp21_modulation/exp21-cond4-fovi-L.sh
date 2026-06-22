#!/bin/bash
# Same as exp21-cond4-fovi-film-pos-sigma4-repro (foveated res64/cart8/doubleres,
# film fourier 256 sigma 4) but distilling from DINOv3 ViT-L (teacher_dim 1024,
# auto-detected -> prediction heads resize to 1024) instead of ViT-B. Requires
# on-the-fly features (precomputed shards are B-only), so it reads the no-features
# dataset. Student/canvas/registers/patcher unchanged.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp21_modulation
RUN_NAME=exp21-cond4-fovi-L
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
EXTRA_ARGS="--model.patcher-name foveated --model.foveated-patcher.resolution 64 --model.foveated-patcher.cart-patch-size 8 --model.foveated-patcher.arch-flag doubleres --model.foveated-patcher.no-force-patches-less-than-matched --model.foveated-patcher.conditioning.mode film --model.foveated-patcher.conditioning.film.fourier.num-features 256 --model.foveated-patcher.conditioning.film.fourier.sigma 4 --webdataset-dir /mnt/lustre-rzg/workspaces/ws/nib00021/u25995-canvit-data-no-features/webdataset-imagenet-1k-no-features"
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
