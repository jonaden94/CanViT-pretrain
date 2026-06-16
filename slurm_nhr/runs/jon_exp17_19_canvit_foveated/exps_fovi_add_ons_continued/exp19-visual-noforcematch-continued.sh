#!/bin/bash
# baseline + visual-space KNN (sample_cortex=False) and no patch-count match
# constraint (force_patches_less_than_matched=False). Otherwise identical to
# exp19-baseline-res64-doubleres. ~200k steps; W&B jon_exp19_canvit_fovi_add_ons.
set -euo pipefail

# === ESSENTIALS (ALWAYS NEED TO BE SPECIFIED) ===
RUN_GROUP=foveated_add_ons
RUN_NAME=exp19-visual-noforcematch
ARRAY=0-25%1                                   # CONTINUE: 47 jobs done (step 192512) -> 26 remaining -> 73 total (299,008 steps, ~300k)
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
EXTRA_ARGS="--model.patcher-name foveated --model.foveated-patcher.resolution 64 --model.foveated-patcher.cart-patch-size 8 --model.foveated-patcher.arch-flag doubleres --model.foveated-patcher.sample-cortex False --model.foveated-patcher.no-force-patches-less-than-matched"
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
    --exclude=ggpu129 \
    --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --export=ALL \
    slurm_nhr/base_train.sbatch
