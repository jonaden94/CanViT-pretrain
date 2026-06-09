#!/bin/bash
# TOPUP exp20-cond4-reg: resume 42->49 jobs (exclude slow nodes ggpu109,ggpu124,ggpu132)
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=patcher_compare
RUN_NAME=exp20-cond4-reg
ARRAY=0-6%1                                   # TOPUP: 42 jobs done -> 7 more -> 49 total (200,704)
TIME=0-00:45:00
MEM=128G
NGPU=1

# === OPTIONAL ===
CFG_WANDB_PROJECT=jon_exp20_canvit_compare_patchers
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=4096
CFG_VAL_EVERY=4096
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--model.patcher-name square --model.square-patcher.method fovi_regularized --model.square-patcher.resolution 64 --model.square-patcher.cart-patch-size 8 --model.square-patcher.no-force-patches-less-than-matched --model.square-patcher.m-override 10 --model.square-patcher.strict-nest-when-possible"
# =================

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-pretrain
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export RUN_GROUP RUN_NAME NGPU EXTRA_ARGS
for v in $(compgen -v); do [[ "$v" == CFG_* ]] && export "$v"; done

sbatch \
    --gpus-per-node=A100:$NGPU \
    --ntasks-per-node=$NGPU \
    --mem=$MEM \
    --time=$TIME \
    --array="$ARRAY" \
    --exclude=ggpu109,ggpu124,ggpu132 \
    --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --export=ALL \
    slurm_nhr/base_train.sbatch
