#!/bin/bash
# (2) res36 cart6 doubleres (baseline but resolution=36, cart_patch_size=6)
# Baseline = exp17 foveated res64/cart8/doubleres; ~200k steps; W&B jon_exp19_canvit_fovi_add_ons.
set -euo pipefail

# === ESSENTIALS (ALWAYS NEED TO BE SPECIFIED) ===
RUN_GROUP=foveated_add_ons
RUN_NAME=exp19-res36-cart6-doubleres
ARRAY=0-48%1                                   # 49 tasks x 4096 steps = 200,704 steps (~200k)
TIME=0-00:45:00
MEM=128G
NGPU=1

# === OPTIONAL ===
CFG_WANDB_PROJECT=jon_exp19_canvit_fovi_add_ons
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=4096
CFG_VAL_EVERY=4096
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--model.patcher-name foveated --model.foveated-patcher.resolution 36 --model.foveated-patcher.cart-patch-size 6 --model.foveated-patcher.arch-flag doubleres"
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
