#!/bin/bash
# exp23 — CUTOVER FIDELITY A/B, OLD-LOOP side, uniform16 + teacher-init.
#
# This is the BASELINE half of the exp23 pair: exp22-uniform16-teacherinit re-run
# FRESH from step 0 on exp22's EXACT stack (python -m canvit_pretrain.train, pins
# fe24aa1 / 3277048 / c399d3b), so it overlays 1:1 against uniform16-ti-harness.sh
# (the CURRENT unified stack through the harness). Same seed (0), same data, same
# 200k window => any curve gap between the two is the current code vs the old code.
#
# Why a fresh re-run and not the historical exp22 curves: exp22 was a 2M-step run
# with LR drops; this is a clean 0->~205k constant-schedule window, and re-running
# both sides pins seed/schedule/logging identically. NOTHING IS SUBMITTED by writing
# this file.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=exp23
RUN_NAME=exp23-uniform16-ti-oldloop
ARRAY=0-24%1                 # 25 jobs x 8192 = 204,800 steps (100k warmup + ~105k plateau)
TIME=0-02:00:00
MEM=128G
NGPU=1

# === config (identical to exp22-uniform16-teacherinit) ===
CFG_WANDB_PROJECT=exp23
CFG_SEED=0
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=8192
CFG_VAL_EVERY=8192           # validate once per job
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--model.patcher-name uniform --init-backbone-from-teacher --webdataset-dir /mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/webdataset-imagenet-21k-with-features"
# =================

# exp22's exact pins — the OLD non-unified stack.
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
    slurm/archive/base_train.sbatch
