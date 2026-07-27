#!/bin/bash
# exp23 — CUTOVER FIDELITY A/B, HARNESS side, foveated + teacher-init.
#
# CURRENT-STACK half: exp22-fovi-teacherinit through the UNIFIED harness on current code
# (pretrain 24a8500 = HEAD w/ the compile-correctness fixes; pytorch 017ce9b = HEAD;
# fovi c399d3b = unchanged). Overlay against fovi-ti-oldloop.sh. Config identical to the
# old-loop side; only the training CODE + entry point differ (flags gain `--cfg.`).
# webdataset-dir injected from .envrc.grete (= exp22's path). NOTHING IS SUBMITTED.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=exp23
RUN_NAME=exp23-fovi-ti-harness
ARRAY=0-24%1                 # 25 jobs x 8192 = 204,800 steps (matches the old-loop side)
TIME=0-02:00:00
MEM=128G
NGPU=1
TASK=distill

# === config (identical to fovi-ti-oldloop.sh) ===
CFG_WANDB_PROJECT=exp23
CFG_SEED=0
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=8192
CFG_VAL_EVERY=8192
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--cfg.model.patcher-name foveated --cfg.model.foveated-patcher.fov 35 --cfg.model.foveated-patcher.resolution 64 --cfg.model.foveated-patcher.cmf-a 0.5 --cfg.model.foveated-patcher.cart-patch-size 5 --cfg.model.foveated-patcher.arch-flag doubleres --cfg.model.foveated-patcher.conditioning.mode film --cfg.model.foveated-patcher.conditioning.film.fourier.num-features 256 --cfg.model.foveated-patcher.conditioning.film.fourier.sigma 4 --cfg.foveated-scale.fixed-scale 2.0 --cfg.init-backbone-from-teacher"
# =================

# CURRENT unified stack. 24a8500 = pretrain HEAD (includes the compile-correctness fixes).
PRETRAIN_COMMIT=24a8500
PYTORCH_COMMIT=017ce9b
FOVI_COMMIT=c399d3b

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-pretrain
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export TASK RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
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
    slurm_nhr/harness_train.sbatch
