#!/bin/bash
# Fresh run: foveated with full-image foveation (notebook-aligned defaults).
# Replaces the older `exp17-1gpu-foveated` run, whose checkpoint is incompatible
# (different foveation semantics: foveation-of-crop vs full-image foveation).
# Defaults from FoveatedPatcherConfig: fov=180, cmf_a=0.5, resolution=36,
# cart_patch_size=6, fixation_size=512, sampler='pooling', sample_cortex=True.
set -euo pipefail

# === ESSENTIALS (ALWAYS NEED TO BE SPECIFIED) ===
RUN_GROUP=foveated
RUN_NAME=exp17-1gpu-foveated-fullimg
EXCLUDE=ggpu131,ggpu132,ggpu105
ARRAY=0-99%1                                   # 100 tasks × 4096 steps = 409,600 steps total
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
EXTRA_ARGS="--model.patcher-name foveated"
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
