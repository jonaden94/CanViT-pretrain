#!/bin/bash
# exp24 — ADE20K frozen probe through the HARNESS, on the best-scene_cos_norm_t9 checkpoint of
# exp22-uniform16-teacherinit-lrdrop2-803k (step-16384, converted to HF).
#
# Recipe = the original specialize ade20k probe (harness ade20k defaults reproduce it; mIoU gate
# passed): FROZEN backbone (default preset = TrainSpec.probe), 40k steps, random-view training,
# n_timesteps=10, scene 512, canvas_grid 32, val resize = SQUISH (the specialize reference). The
# ONLY non-essential difference from the original is the entry point (harness vs canvit_specialize).
# NOTHING IS SUBMITTED by writing this file.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=exp24
RUN_NAME=ade20k-uni16ti-803k
ARRAY=0-0%1                  # single job (frozen probe, no cross-job resume)
TIME=0-08:00:00
MEM=64G
NGPU=1
TASK=ade20k

# === config ===
CFG_WANDB_PROJECT=exp24
CFG_MODEL_REPO=/user/henrich1/u25995/jonathan/repos/CanViT-train/logs/jon_exp22_full_runs/exp22-uniform16-teacherinit-lrdrop2-803k/checkpoints/step-16384-hf
CFG_RESIZE_MODE=squish
OPT_CKPT_DIR=logs/exp24/ade20k-uni16ti-803k/checkpoints
# =================

export ADE20K_ROOT=/user/henrich1/u25995/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016

PRETRAIN_COMMIT=24a8500
PYTORCH_COMMIT=017ce9b
FOVI_COMMIT=c399d3b

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export TASK RUN_GROUP RUN_NAME NGPU ADE20K_ROOT PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* || "$v" == OPT_* ]] && export "$v"; done

sbatch \
    --gpus-per-node=A100:$NGPU \
    --ntasks-per-node=$NGPU \
    --mem=$MEM \
    --time=$TIME \
    --array="$ARRAY" \
    --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --export=ALL \
    slurm/harness_train.sbatch
