#!/bin/bash
# Fresh 1-GPU run, single-GPU counterpart of exp17-4gpu-post-ddp-fix.
# Same training config, but: 1 GPU, batch size 64 (no 4x), base LR (no 4x
# linear scaling), and starts from scratch (full 100-task / 409,600-step
# plan rather than a mid-run continuation).
#
# RUN_NAME is pinned to the CanViT-pretrain commit it was created against
# (9649b9e) for reproducibility — bump it if you re-create this run on a
# newer commit.
#
# Note: effective batch per step is 64 here vs 64x4=256 for the 4gpu run,
# so at the same step count this run sees 1/4 the samples — expected for a
# 1-GPU baseline.
set -euo pipefail

# === ESSENTIALS (ALWAYS NEED TO BE SPECIFIED) ===
RUN_GROUP=default
RUN_NAME=exp17-1gpu-reproduce-9649b9e
EXCLUDE=ggpu131,ggpu132,ggpu134,ggpu105
ARRAY=0-99%1                                   # 100 tasks × 4096 steps = 409,600 steps total
TIME=0-00:45:00
MEM=128G                                       # 1 GPU — 512G (the 4gpu value) would be wasteful
NGPU=1

# === OPTIONAL ===
CFG_WANDB_PROJECT=jon_exp17_canvit_first_tryouts
CFG_PEAK_LR=0.0004                             # base LR (the 4gpu run uses 4x = 0.0016)
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
