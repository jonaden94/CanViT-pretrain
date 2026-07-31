#!/bin/bash
# exp24 — ADE20K frozen probe through the HARNESS, on the best-scene_cos_norm_t9 checkpoint of
# exp22-uniform16-lrdrop-1516k (step-319488, converted to HF).
# See ade20k-uni16ti-803k.sh for the full recipe rationale (frozen probe, 40k, squish resize).
# NOTHING IS SUBMITTED by writing this file.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=exp24
RUN_NAME=ade20k-uni16-1516k
ARRAY=0-0%1
TIME=0-08:00:00
MEM=64G
NGPU=1
TASK=ade20k

# === config ===
CFG_WANDB_PROJECT=exp24
CFG_MODEL_REPO=/mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train/logs/jon_exp22_full_runs/exp22-uniform16-lrdrop-1516k/checkpoints/step-319488-hf
CFG_RESIZE_MODE=squish
OPT_CKPT_DIR=logs/exp24/ade20k-uni16-1516k/checkpoints
# =================

export ADE20K_ROOT=/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016

PRETRAIN_COMMIT=24a8500
PYTORCH_COMMIT=017ce9b
FOVI_COMMIT=c399d3b

# Repo root, derived from this script's own location (slurm/runs/<group>/<run>.sh),
# so the run submits from YOUR clone rather than one hardcoded checkout.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
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
