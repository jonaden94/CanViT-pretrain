#!/bin/bash
# (12) FiLM(position, sigma=4) + MLP head [768]
# Baseline = exp17 foveated res64/cart8/doubleres; ~200k steps; W&B jon_exp19_canvit_fovi_add_ons.
set -euo pipefail

# === ESSENTIALS (ALWAYS NEED TO BE SPECIFIED) ===
RUN_GROUP=foveated_add_ons
RUN_NAME=exp19-film-pos-sigma4-mlp768
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
EXTRA_ARGS="--model.patcher-name foveated --model.foveated-patcher.resolution 64 --model.foveated-patcher.cart-patch-size 8 --model.foveated-patcher.arch-flag doubleres --model.foveated-patcher.conditioning.mode film --model.foveated-patcher.conditioning.film.input position --model.foveated-patcher.conditioning.film.fourier.num-features 256 --model.foveated-patcher.conditioning.film.fourier.sigma 4 --model.foveated-patcher.hidden-dims-patch-embed 768"
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
