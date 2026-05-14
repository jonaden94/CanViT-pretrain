#!/bin/bash
# Minimal SLURM launcher. Edit values below, then: bash slurm_nhr/submit.sh
set -euo pipefail

# === ESSENTIALS (ALWAYS NEED TO BE SPECIFIED) ===
RUN_GROUP=test
RUN_NAME=test1_gpu
EXCLUDE=ggpu131,ggpu132,ggpu134,ggpu105
ARRAY=0-0%1                                    # e.g. 0-99%1 for 100 sequential tasks
TIME=0-00:20:00
MEM=128G                                       # 128G/256G/512G for 1/2/4 GPU
NGPU=1

# === OPTIONAL (TRAINING CONFIGS THAT CAN BE MODIFIED BUT DO NOT HAVE TO) ====================================================
# Add new tunables here as CFG_FOO_BAR=value  — base_train.sbatch auto-emits
# them as `--foo-bar $value` (uppercase/underscores → lowercase/hyphens).
# Empty value → falls back to Python's own default in canvit_pretrain/train/config.py.
CFG_WANDB_PROJECT=jon_exp17_canvit_first_tryouts
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=4096
CFG_VAL_EVERY=4096
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
# one-off Python flags (e.g. --model.patcher-name foveated)
EXTRA_ARGS=
# ========================================================================

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
