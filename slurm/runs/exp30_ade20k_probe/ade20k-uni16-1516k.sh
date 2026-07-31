#!/bin/bash
# exp30 — ADE20K frozen-PROBE training through the harness at the CURRENT code base.
# Source: exp22-uniform16-lrdrop-1516k step-319488 (best val/scene_cos_norm_t9; the source exp24/exp25 used)
#
# Recipe is exp24's = the original canvit_specialize ade20k probe, reproduced by the harness
# ade20k defaults (mIoU gate passed vs the specialize-derived standalone): frozen backbone
# (default preset = TrainSpec.probe), 40k steps, random-view training, n_timesteps 10,
# scene 512, canvas_grid 32, val resize squish.
#
# resize_mode=squish for ALL arms incl. foveated -- the protocol every earlier CanViT /
# specialize number was measured under. It distorts aspect ratio, so it is not the choice for
# a human-viewing comparison (center_crop preserves the geometry foveated sampling assumes),
# but it is the one that makes these numbers comparable to exp24 and to the published values.
#
# vs exp24 the ONLY changes are the pins and, for ade20k-fovi-1901k, a source exp24 never had.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=exp30_ade20k_probe
RUN_NAME=ade20k-uni16-1516k
ARRAY=0-0%1
TIME=0-08:00:00
MEM=64G
NGPU=1                       # ade20k has supports_ddp=False
TASK=ade20k

# === config (exp24 recipe) ===
CFG_WANDB_PROJECT=exp30_ade20k_probe
CFG_MODEL_REPO=/user/henrich1/u25995/jonathan/repos/CanViT-train/logs/jon_exp22_full_runs/exp22-uniform16-lrdrop-1516k/checkpoints/step-319488-hf
CFG_RESIZE_MODE=squish
EXTRA_ARGS=""
# =================

PRETRAIN_COMMIT=455bdae
PYTORCH_COMMIT=1f5121b
FOVI_COMMIT=c399d3b

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export ADE20K_ROOT=/user/henrich1/u25995/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016
export TASK RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* || "$v" == OPT_* ]] && export "$v"; done

sbatch \
    --gpus-per-node=A100:$NGPU --ntasks-per-node=$NGPU --mem=$MEM --time=$TIME \
    --array="$ARRAY" \
    --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --export=ALL slurm/harness_train.sbatch
