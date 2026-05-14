#!/bin/bash
# Smoke test: 4-GPU DDP, single-job, 128 iterations.
set -euo pipefail

# === ESSENTIALS (ALWAYS NEED TO BE SPECIFIED) ===
RUN_GROUP=test
RUN_NAME=test4_gpu
EXCLUDE=ggpu131,ggpu132,ggpu134,ggpu105
ARRAY=0-0%1
TIME=0-00:20:00
MEM=512G
NGPU=4

# === OPTIONAL ===
CFG_WANDB_PROJECT=jon_exp17_canvit_first_tryouts
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=128
CFG_VAL_EVERY=128
CFG_LOG_EVERY=16
CFG_NUM_WORKERS=4
CFG_VIZ_EVERY_N_VALS=1
CFG_CURVE_EVERY_N_VALS=1
EXTRA_ARGS=
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
    --exclude="$EXCLUDE" \
    --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --export=ALL \
    slurm_nhr/base_train.sbatch
