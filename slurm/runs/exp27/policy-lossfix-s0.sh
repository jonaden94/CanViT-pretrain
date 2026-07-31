#!/bin/bash
# exp27 ARM D — arm B (the unified harness) with the POLICY-LOSS SCALE FIX.
#
#   python -m canvit_train.harness.run ade20k --preset policy_only
#
# WHY THIS ARM EXISTS. Arm B (5 seeds, 15100922-26) landed at mean(t1-t4) CE
# 0.6880 +- 0.0010 / mIoU t4 44.77 +- 0.13, while arm C (`rl_train`, the ported
# reference, 2 seeds) landed at 0.6864 +- 0.0008 / 44.91 +- 0.03 — the band's own
# last-step row being 0.6863 / 44.91. Both metrics favoured the ported trainer, so the
# gap looked real rather than noise, but at n=2 vs n=5 the exact permutation floor is
# p=0.095 and significance was unreachable. Two things were done about that:
#
#   1. rl_train seeds 2/3/4 (jobs 15103016-18) take arm C to n=5, dropping the
#      permutation floor to 1/C(10,5) = 0.004.
#   2. a differential read of the two training paths found a CONCRETE defect, fixed in
#      the commit this arm pins.
#
# THE DEFECT. `harness/rollout.py` accumulated the per-glimpse QReg policy loss into
# `chunk_loss`, which is then divided by `n_glimpses` — but only `n_glimpses-1` of the
# glimpses are POLICY glimpses (t0 is the full-scene anchor and carries no policy loss).
# The reference instead cats every depth into one [horizon*B, A] tensor and takes a
# single `F.mse_loss`, i.e. one mean over horizon*B. Net effect at horizon 4: the
# harness's scorer gradient was EXACTLY 0.8x the reference's — a 20% smaller effective
# policy LR at the same nominal `policy_lr`, so the scorer was systematically
# under-trained at a fixed 8000-step budget. Under-training is the right DIRECTION for
# the observed deficit, which is what makes this the leading candidate.
#
# Measured, not argued: `harness/tests/test_policy_loss_scale.py` pins the fixed
# gradient bit-identical to the reference's, and pins the unfixed one at exactly 0.8x.
# The VPG path already compensated for the same division (`* n_glimpses` in the
# deferred-credit branch); only the inline QReg/PG path did not.
#
# WHAT TO EXPECT. If this is the cause, arm D reproduces arm C / the band:
# CE -> ~0.6864, mIoU t4 -> ~44.9. If arm D still sits at ~0.688, the 0.8x was real
# but not the cause, and the next suspects are the two remaining divergences recorded
# in doc 15 SS A5 (the encoder recomputing probe logits OUTSIDE amp_ctx at t>=1, where
# rl_train passes bf16 logits it already has; and the extra probe-head forward that
# implies).
#
# RUN_NAME is DELIBERATELY not arm B's. Reusing it would overwrite arm B's
# checkpoints — that has already happened once in this experiment (see the README).
#
# Everything else is arm B verbatim; see policy-harness-s0.sh for the full
# rl_train -> harness config mapping and the resize-mode / EXTRA_ARGS notes.
#
# NOTHING IS SUBMITTED by writing this file.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=exp27
SEED="${SEED:-0}"            # SEED=1 bash <this> for a second seed
RUN_NAME=exp27-policy-lossfix-s$SEED
ARRAY=0-0%1                  # single job: 8000 steps fits well inside the walltime
TIME="${TIME:-0-04:00:00}"    # override for slow nodes: TIME=0-06:00:00 SEED=8 bash <this>
MEM=64G
NGPU=1                       # ade20k has supports_ddp=False
TASK=ade20k

# === config === (identical to arm B)
CFG_WANDB_PROJECT=exp27
CFG_SEED=$SEED
CFG_MAX_STEPS=8000
CFG_N_TIMESTEPS=5
CFG_BATCH_SIZE=16
CFG_CANVAS_GRID=64
CFG_PROBE_REPO=canvit/probe-ade20k-40k-s512-c64-in21k
CFG_EVAL_POLICY=policy
CFG_RESIZE_MODE=squish       # THE MEASUREMENT CONTRACT. Not optional.
CFG_VAL_EVERY=1000           # rl_train evaluates on the same cadence (9 evals/run)
CFG_LOG_EVERY=50
CFG_NUM_WORKERS=4
EXTRA_ARGS="--preset policy_only --cfg.no-augment"   # mode (b) + score_res 128 are DEFAULTS
# =================

# THE ONLY DIFFERENCE FROM ARM B: this pin carries the policy-loss scale fix.
PRETRAIN_COMMIT=bc0b16b
PYTORCH_COMMIT=1f5121b
FOVI_COMMIT=c399d3b

# NOT in env.sh — every exp24 ade20k script exports it explicitly.
export ADE20K_ROOT=/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016

# Repo root, derived from this script's own location (slurm/runs/<group>/<run>.sh),
# so the run submits from YOUR clone rather than one hardcoded checkout.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export TASK RUN_GROUP RUN_NAME NGPU EXTRA_ARGS ADE20K_ROOT PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
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
    slurm/harness_train.sbatch
