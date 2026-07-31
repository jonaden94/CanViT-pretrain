#!/bin/bash
# ADE20K viewpoint POLICY (QReg) on top of the FOVEATED exp22 backbone + the probe trained
# on it in exp30. Starting point for training policies on that model.
#
# THE TWO HALVES. A policy run needs both, and neither alone is enough:
#   --cfg.model-repo   the frozen pretrained backbone, HF layout
#   --cfg.probe-repo   the trained segmentation head, HF layout -- it IS the reward model
#                      (the reward is the fraction of the probe's CE a glimpse removes, so
#                      a random head makes the reward pure noise)
# Together they rebuild exp30's best.pt EXACTLY (verified: every head and backbone tensor
# bit-identical, no missing keys). The probe half was exported from best.pt with
#   python -m canvit_train.checkpoint.probe_to_hf \
#       --pt-path <exp30 run>/checkpoints/best.pt --out-dir <same>/best-probe-hf
# because `probe_repo` is read by SegmentationProbe.from_pretrained, i.e. an HF directory --
# a harness .pt cannot be passed here, and `to_hf` deliberately refuses ade20k payloads.
#
# CANVAS_GRID 32 IS NOT OPTIONAL. The probe was trained at canvas_grid 32 (exp30), so a
# policy run at another grid feeds it a canvas resolution it never saw and the reward
# degrades silently. exp31's policy recipe used 64 because ITS probe
# (canvit/probe-ade20k-40k-s512-c64-in21k) was trained at 64. Match the probe, not exp31.
#
# FOVEATED SCALE 2.0 IS NOT OPTIONAL EITHER. A foveated backbone derives its fixation
# window as fix_size = scale * H, so a rollout at a scale it never saw makes EVERY glimpse
# out of distribution. It does not crash -- mIoU just falls as glimpses accumulate.
#
# NOT YET RUN. The recipe below is exp31's `lossfix` arm (the harness path after the one
# real policy-gradient divergence was fixed) with exp30's backbone+probe swapped in and the
# grid matched to that probe. exp31 ran against the PUBLISHED backbone, which is uniform;
# this is foveated, so the policy's action space becomes the fixation grid rather than the
# safe-box grid. The config is validated (tyro parse + resolve_spec) but no training step
# has been taken with it, so treat the first run as a smoke test: check that eval/miou_t0
# looks sane (it is the pre-policy full-image glimpse, so it should match a probe-only
# eval) before trusting later timesteps.
#
# 10 seeds: for s in 0 1 2 3 4 5 6 7 8 9; do SEED=$s bash slurm/runs/policy_on_exp30_fovi_ti/policy-qreg-fovi-ti-s0.sh; done
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=policy_on_exp30_fovi_ti
SEED="${SEED:-0}"
RUN_NAME=policy-qreg-fovi-ti-s$SEED
ARRAY=0-0%1                  # single job: 8000 steps fits inside the walltime
TIME="${TIME:-0-08:00:00}"
MEM=64G
NGPU=1                       # ade20k is single-GPU only (supports_ddp=False)
TASK=ade20k

# === the two halves of the model ===
_EXP30=/mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train/logs/exp30_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints
CFG_MODEL_REPO=/mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648-hf
CFG_PROBE_REPO=$_EXP30/best-probe-hf

# === config (exp31 lossfix recipe; only the grid follows the probe) ===
CFG_WANDB_PROJECT=policy_on_exp30_fovi_ti
CFG_SEED=$SEED
CFG_MAX_STEPS=8000
CFG_N_TIMESTEPS=5
CFG_BATCH_SIZE=16
CFG_CANVAS_GRID=32           # matches the exp30 probe -- see header
CFG_EVAL_POLICY=policy
CFG_RESIZE_MODE=squish       # the measurement contract every earlier CanViT number used
CFG_VAL_EVERY=1000
CFG_LOG_EVERY=50
CFG_NUM_WORKERS=4
EXTRA_ARGS="--preset policy_only --cfg.no-augment --cfg.foveated-scale.fixed-scale 2.0"
# =================

PRETRAIN_COMMIT=""           # set to a commit hash to pin the code for a long run
PYTORCH_COMMIT=""
FOVI_COMMIT=""

# Repo root, derived from this script's own location (slurm/runs/<group>/<run>.sh),
# so the run submits from YOUR clone rather than one hardcoded checkout.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export ADE20K_ROOT="${ADE20K_ROOT:-/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016}"
export TASK RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* || "$v" == OPT_* ]] && export "$v"; done

sbatch \
    --gpus-per-node=A100:$NGPU --ntasks-per-node=$NGPU --mem=$MEM --time=$TIME \
    --array="$ARRAY" \
    --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --export=ALL slurm/harness_train.sbatch
