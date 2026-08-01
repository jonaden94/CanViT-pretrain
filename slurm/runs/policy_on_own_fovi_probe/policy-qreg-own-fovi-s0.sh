#!/bin/bash
# ADE20K viewpoint POLICY (QReg) on our OWN foveated model: the exp22 foveated-teacherinit
# backbone plus the ADE20K probe trained on it. The counterpart of the exp35 policy runs,
# which use the published backbone and probe instead.
#
# THE TWO HALVES. A policy run needs both, and neither alone is enough:
#   --cfg.model-repo   the frozen pretrained backbone
#   --cfg.probe-repo   the trained segmentation head -- it IS the reward model (the reward
#                      is the fraction of the probe`s CE a glimpse removes, so a random head
#                      makes the reward pure noise)
# Both flags take a training .pt, a local HF directory, or a Hub id, so the probe is passed
# as the checkpoint the probe run wrote. No conversion step.
#
# Both halves are the training checkpoints themselves. Nothing is exported first; the HF
# layout is for publishing, not for feeding one of our own runs into another.
#
# CANVAS_GRID 32 IS NOT OPTIONAL. The probe was trained at canvas_grid 32, so a policy run
# at another grid feeds the reward model a canvas resolution it never saw and the reward
# degrades silently. The exp35 recipe uses 64 because ITS probe was trained at 64. Match the
# probe you are actually using.
#
# FOVEATED SCALE 2.0 IS NOT OPTIONAL EITHER. A foveated backbone derives its fixation window
# as fix_size = scale * H, so a rollout at a scale it never saw makes EVERY glimpse out of
# distribution. It does not crash -- mIoU just falls as glimpses accumulate.
#
# NOT YET RUN. The recipe is the exp35 policy arm with this backbone+probe swapped in and the
# grid matched to the probe. That arm runs against the published UNIFORM backbone; this one
# is foveated, so the policy`s action space becomes the fixation grid rather than the
# safe-box grid. The config is validated (tyro parse + spec resolution) but no training step
# has been taken with it. Treat the first run as a smoke test: eval/miou_t0 is the pre-policy
# full-image glimpse, so it should match a probe-only eval before later timesteps are
# trusted.
#
# 10 seeds: for s in 0 1 2 3 4 5 6 7 8 9; do SEED=$s bash slurm/runs/policy_on_own_fovi_probe/policy-qreg-own-fovi-s0.sh; done
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=policy_on_own_fovi_probe
SEED="${SEED:-0}"
RUN_NAME=policy-qreg-own-fovi-s$SEED
ARRAY=0-0%1                  # single job: 8000 steps fits inside the walltime
TIME="${TIME:-0-08:00:00}"
MEM=64G
NGPU=1                       # ade20k is single-GPU only (supports_ddp=False)
TASK=ade20k

# === the two halves of the model ===
_PROBE_RUN=/mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train/logs/exp34_ade20k_probe/ade20k-fovi-ti-1196k/checkpoints
CFG_MODEL_REPO=/mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648.pt
CFG_PROBE_REPO=$_PROBE_RUN/best.pt   # the training checkpoint itself; no conversion

# === config (exp31 lossfix recipe; only the grid follows the probe) ===
CFG_WANDB_PROJECT=policy_on_own_fovi_probe
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

TRAIN_COMMIT=""              # set to a commit hash to pin the code for a long run
PYTORCH_COMMIT=""
FOVI_COMMIT=""

# Repo root, derived from this script's own location (slurm/runs/<group>/<run>.sh),
# so the run submits from YOUR clone rather than one hardcoded checkout.
# The probe must come from a FINISHED probe run. `best.pt` is not a completion marker -- it
# appears at the first evaluation (step 500) and is rewritten whenever the metric improves,
# so testing for it would happily hand the policy a probe a few hundred steps old. Gate on
# the final step checkpoint instead, the same way the pretrain LR-drop launchers gate on
# their drop-step file.
_PROBE_DONE=$_PROBE_RUN/step-40000.pt          # ade20k max_steps
if [ ! -f "$_PROBE_DONE" ]; then
    echo "REFUSING: the probe run has not finished ($_PROBE_DONE is missing)."
    echo "  best.pt may already exist, but it would be an early checkpoint."
    echo "  Wait for exp34's ade20k-fovi-ti-1196k arm, or point _PROBE_RUN at a finished"
    echo "  ade20k run -- only its head.* weights are read, so any finished run works if"
    echo "  --cfg.canvas-grid matches the grid that probe was trained at."
    exit 1
fi

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export ADE20K_ROOT="${ADE20K_ROOT:-/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016}"
export TASK RUN_GROUP RUN_NAME NGPU EXTRA_ARGS TRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* || "$v" == OPT_* ]] && export "$v"; done

sbatch \
    --gpus-per-node=A100:$NGPU --ntasks-per-node=$NGPU --mem=$MEM --time=$TIME \
    --array="$ARRAY" \
    --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log" \
    --export=ALL slurm/harness_train.sbatch
