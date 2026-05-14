#!/bin/bash
# Continuation of exp17-1gpu-foveated.
# Auto-resumes from logs/foveated/exp17-1gpu-foveated/checkpoints/latest.pt
# (currently step-241664.pt) and runs 41 more array tasks × 4096 steps to
# complete the original 100-task / 409,600-step plan.
# Original legacy script: slurm_nhr/_legacy/exp17/train_1gpu_foveated.sbatch
set -euo pipefail

# === ESSENTIALS (ALWAYS NEED TO BE SPECIFIED) ===
RUN_GROUP=foveated
RUN_NAME=exp17-1gpu-foveated
EXCLUDE=ggpu131,ggpu132,ggpu105
ARRAY=0-39%1                                   # 40 remaining tasks to reach 100-task target (1 already done in earlier resume)
TIME=0-00:45:00
MEM=128G
NGPU=1

# === OPTIONAL ===
CFG_WANDB_PROJECT=jon_exp17_canvit_first_tryouts
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=4096
CFG_VAL_EVERY=4096
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--model.patcher-name foveated --model.foveated-patcher.fixation-size 128"
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
