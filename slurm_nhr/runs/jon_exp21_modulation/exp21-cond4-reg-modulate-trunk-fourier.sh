#!/bin/bash
# cond4-reg + ViT trunk adaLN modulation (fourier encoding)
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp21_modulation
RUN_NAME=exp21-cond4-reg-modulate-trunk-fourier
ARRAY=0-26%1                                   # 27 remaining of 49 (resume from step 90112); validation OFF (diagnostic)
TIME=0-01:30:00
MEM=128G
NGPU=1

# === OPTIONAL ===
CFG_WANDB_PROJECT=jon_exp21_modulation
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=4096
CFG_VAL_EVERY=999999999  # validation disabled: never a multiple of this within a job
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--model.patcher-name square --model.square-patcher.method fovi_regularized --model.square-patcher.resolution 64 --model.square-patcher.cart-patch-size 8 --model.square-patcher.no-force-patches-less-than-matched --model.square-patcher.m-override 10 --model.square-patcher.strict-nest-when-possible --backbone-name vitb16_modulate --model.vit-modulation.enabled"
# =================

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-pretrain
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export RUN_GROUP RUN_NAME NGPU EXTRA_ARGS
for v in $(compgen -v); do [[ "$v" == CFG_* ]] && export "$v"; done

sbatch     --gpus-per-node=A100:$NGPU     --ntasks-per-node=$NGPU     --mem=$MEM     --time=$TIME     --array="$ARRAY"     --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --export=ALL     slurm_nhr/base_train.sbatch
