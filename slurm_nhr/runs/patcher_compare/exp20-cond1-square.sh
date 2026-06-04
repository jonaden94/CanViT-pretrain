#!/bin/bash
# condition 1 fovi-square (auto m): res49 cart6 cortexF
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=patcher_compare
RUN_NAME=exp20-cond1-square
ARRAY=0-48%1                                   # 49 tasks x 4096 steps = 200,704 steps (~200k)
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
EXTRA_ARGS="--model.patcher-name square --model.square-patcher.method fovi --model.square-patcher.resolution 49 --model.square-patcher.cart-patch-size 6 --model.square-patcher.sample-cortex False --model.square-patcher.no-force-patches-less-than-matched"
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
    --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --export=ALL \
    slurm_nhr/base_train.sbatch
