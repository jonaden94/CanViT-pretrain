#!/bin/bash
# Continuation of exp17-4gpu-post-ddp-fix.
# Auto-resumes from logs/default/exp17-4gpu-post-ddp-fix/checkpoints/latest.pt
# (currently step-135168.pt) and runs 67 more array tasks × 4096 steps to
# complete the original 100-task / 409,600-step plan.
# Original legacy script: slurm_nhr/_legacy/exp17/train_4gpu.sbatch
set -euo pipefail

# === ESSENTIALS (ALWAYS NEED TO BE SPECIFIED) ===
RUN_GROUP=default
RUN_NAME=exp17-4gpu-post-ddp-fix
EXCLUDE=ggpu131,ggpu132,ggpu134,ggpu105
ARRAY=0-65%1                                   # 66 remaining tasks to reach 100-task target (1 already done in earlier resume)
TIME=0-00:45:00
MEM=512G
NGPU=4

# === OPTIONAL ===
CFG_WANDB_PROJECT=jon_exp17_canvit_first_tryouts
CFG_PEAK_LR=0.0016                             # 4x base LR (linear scaling for 4 GPUs)
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=4096
CFG_VAL_EVERY=4096
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
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
