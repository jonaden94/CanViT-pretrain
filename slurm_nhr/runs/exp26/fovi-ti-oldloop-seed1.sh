#!/bin/bash
# exp26 — RUN-TO-RUN NOISE FLOOR for the foveated recipe (old-loop side, seed 1).
#
# The control for fovi-ti-harness-normfix.sh. exp23 compared exactly ONE harness run
# against ONE old-loop run, so a 4-5 point spread had no baseline to be judged against.
# The compiled/bf16 A/A gradient noise floor is ~4e-3 relative PER STEP, which means
# even two identical-code runs diverge chaotically over 100k+ steps — so "the harness is
# worse" and "the harness is merely different" were not separable from exp23 alone.
#
# This is exp23-fovi-ti-oldloop with ONE change: seed 0 -> 1. The seed drives weight
# init, viewpoint sampling AND the WebDataset shard schedule/shuffle, so the gap between
# this and exp23-fovi-ti-oldloop (seed 0, already on disk) IS the run-to-run spread of
# the recipe. Same 8-job window as the harness side so the comparison is step-matched.
#
# Interpretation:
#   |harness-normfix - oldloop(seed0)|  <=  |oldloop(seed1) - oldloop(seed0)|
#       => the harness is within the recipe's own noise; parity, normalizer was the cause.
#   harness-normfix still ~4-5 points low, well outside that spread
#       => a real difference remains; go instrument the compiled path on GPU.
# NOTHING IS SUBMITTED by writing this file.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=exp26
RUN_NAME=exp26-fovi-ti-oldloop-seed1
ARRAY=0-7%1                  # 8 jobs x 8192 = 65,536 steps (step-matched to the harness side)
TIME=0-02:00:00
MEM=128G
NGPU=1

# === config (identical to exp23-fovi-ti-oldloop EXCEPT CFG_SEED) ===
CFG_WANDB_PROJECT=exp26
CFG_SEED=1                   # <-- the only difference; this is what makes it a noise floor
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=8192
CFG_VAL_EVERY=8192
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--model.patcher-name foveated --model.foveated-patcher.fov 35 --model.foveated-patcher.resolution 64 --model.foveated-patcher.cmf-a 0.5 --model.foveated-patcher.cart-patch-size 5 --model.foveated-patcher.arch-flag doubleres --model.foveated-patcher.conditioning.mode film --model.foveated-patcher.conditioning.film.fourier.num-features 256 --model.foveated-patcher.conditioning.film.fourier.sigma 4 --foveated-scale.fixed-scale 2.0 --init-backbone-from-teacher --webdataset-dir /mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/webdataset-imagenet-21k-with-features"
# =================

# exp22/exp23's exact OLD non-unified stack — unchanged, so this really is a seed repeat.
PRETRAIN_COMMIT=fe24aa1
PYTORCH_COMMIT=3277048
FOVI_COMMIT=c399d3b

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
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
