#!/bin/bash
# TOPUP exp20-cond4-fovi-coordconv: 41 done -> 8 more -> 49 total (200,704 steps)
# +5 min time limit; exclude known slow nodes ggpu106, ggpu117, ggpu128, ggpu136
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=patcher_compare
RUN_NAME=exp20-cond4-fovi-coordconv
ARRAY=0-7%1                                   # TOPUP: 8 tasks x 4096 steps = 32,768 remaining
TIME=0-00:50:00
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
EXTRA_ARGS="--model.patcher-name foveated --model.foveated-patcher.resolution 64 --model.foveated-patcher.cart-patch-size 8 --model.foveated-patcher.arch-flag doubleres --model.foveated-patcher.no-force-patches-less-than-matched --model.foveated-patcher.conditioning.mode coordconv"
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
