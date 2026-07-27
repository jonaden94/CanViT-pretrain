#!/bin/bash
# exp23 — CUTOVER FIDELITY A/B, OLD-LOOP side, foveated + teacher-init.
#
# BASELINE half: exp22-fovi-teacherinit re-run FRESH from step 0 on exp22's EXACT stack
# (fe24aa1 / 3277048 / c399d3b, old python -m canvit_pretrain.train). Overlays 1:1 against
# fovi-ti-harness.sh. Fovi config provenance (fov=35, res=64, cmf_a=0.5, cart_patch_size=5,
# doubleres, FiLM sigma-4, fixed-scale 2.0 => 140 patches) is copied verbatim from
# exp22-fovi-teacherinit.sh. NOTHING IS SUBMITTED by writing this file.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=exp23
RUN_NAME=exp23-fovi-ti-oldloop
ARRAY=0-24%1                 # 25 jobs x 8192 = 204,800 steps (100k warmup + ~105k plateau)
TIME=0-02:00:00
MEM=128G
NGPU=1

# === config (identical to exp22-fovi-teacherinit) ===
CFG_WANDB_PROJECT=exp23
CFG_SEED=0
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=8192
CFG_VAL_EVERY=8192
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--model.patcher-name foveated --model.foveated-patcher.fov 35 --model.foveated-patcher.resolution 64 --model.foveated-patcher.cmf-a 0.5 --model.foveated-patcher.cart-patch-size 5 --model.foveated-patcher.arch-flag doubleres --model.foveated-patcher.conditioning.mode film --model.foveated-patcher.conditioning.film.fourier.num-features 256 --model.foveated-patcher.conditioning.film.fourier.sigma 4 --foveated-scale.fixed-scale 2.0 --init-backbone-from-teacher --webdataset-dir /mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/webdataset-imagenet-21k-with-features"
# =================

# exp22's exact pins — the OLD non-unified stack.
PRETRAIN_COMMIT=fe24aa1
PYTORCH_COMMIT=3277048
FOVI_COMMIT=c399d3b

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-pretrain
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
