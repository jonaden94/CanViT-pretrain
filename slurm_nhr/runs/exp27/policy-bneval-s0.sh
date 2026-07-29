#!/bin/bash
# exp27 ARM C — arm A + BN mode (b): glimpse SELECTION under eval-mode BN.
#
# Tests the leading candidate for the residual gap to the qband band. Judged against
# ARM A (exp27-policy-oldloop-s0/s1, 44.76/44.82 mIoU t4) and against the band
# RE-MEASURED under our own eval by running all 8 published qband policies through it:
#
#     CE 0.6855 +- 0.0007   mIoU t4 44.94 +- 0.09
#
# (our eval reproduces their published per-seed numbers to +0.0002 CE / -0.04 mIoU, so
# the eval is validated and any remaining difference is training-side).
#
# WHY: p3-notes delta #1 — the in-graph rollout merged glimpse selection into the
# training forward, so frontend.bn selects on BATCH stats; the RL repo selects on
# RUNNING stats. Measured: the two modes disagree on 45.7% of chosen glimpses.
#
# WHAT TO EXPECT: arm A already matches the band on CE (0.6857) and misses on mIoU t4
# by ~0.15. If mode (b) is the cause, mIoU t4 should move toward 44.94 while CE stays
# put. If BOTH move, the story is wrong and mode (b) is just a different optimum.
#
# ~9% slower than arm A (one extra scorer forward per depth) -> walltime raised to 4h.
#
# NOTHING IS SUBMITTED by writing this file.
#
# --- inherited header from arm A (policy-oldloop-s0.sh) ---
# the gate-validated RL trainer, at TODAY's code.
#
# `python -m canvit_pretrain.ade20k.rl_train`, canonical QReg recipe, seed 0.
# This is the reference the harness arm is judged against. Two reasons it exists
# rather than just citing the published band:
#
#   1. The band (0.6853 +- 0.0007 mean t1-t4 val CE, 8 seeds) was measured on the
#      RL repo's machine. A LOCAL reference removes the hardware/stack question
#      from the comparison entirely.
#   2. The 2026-07-23 P3 gate runs (15025279 / 15025337) predate commit 845e401,
#      which added per-timestep mIoU to the deploy eval — so they logged CE only.
#      The harness arm reports BOTH, and this arm must too or half the comparison
#      has no counterpart.
#
# Expected: mean(t1-t4) val CE inside 0.6845 ... 0.6865 (the band's own per-seed
# spread). ~65 min on one A100 (job 15025279 took 01:04:46).
#
# NOTHING IS SUBMITTED by writing this file.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=exp27
SEED="${SEED:-0}"            # SEED=1 bash <this> for a second seed
RUN_NAME=exp27-policy-bneval-s$SEED
TIME=0-04:00:00
MEM=64G

export ADE20K_ROOT=/user/henrich1/u25995/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016
export WANDB_PROJECT=exp27

# Pin the reference implementation. 4db7c3f restores rl_train's resize_mode default
# to squish, so THIS ARM DOES DEPEND ON IT — do not roll the pin back to cea4dee or
# earlier, where the default was center_crop and the run is not band-comparable.
PRETRAIN_COMMIT=d3c32d7
PYTORCH_COMMIT=017ce9b
FOVI_COMMIT=c399d3b

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-pretrain
mkdir -p "logs/$RUN_GROUP"
export PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT

# Recipe = rl_train's own defaults (lr 2e-4, wd 1e-2, betas .9/.95, clip 1.0,
# 640k glimpse-forwards -> 8000 steps, warmup 12.5% then hold, batch 16,
# train_horizon 4, score_res 128, NO augmentation, c64 probe). Only the run
# identity and seed are passed; everything else must come from the defaults, or
# this stops being the reference.
#
# --resize-mode squish is passed EXPLICITLY even though it is now the default
# again. It is the measurement contract the qband band is defined by, and it has
# already silently regressed once: commit 1a0b452 lifted rl_train's hardcoded
# "squish" into a knob defaulting to center_crop, and the first exp27 arm A
# (job 15093707) landed at 0.6693 -- 0.016 "better" than the band, ~20x its
# 0.0007 seed spread -- purely from the protocol change. Pin it here so the
# reference cannot drift out from under this launcher again.
sbatch \
    --job-name="$RUN_NAME" \
    --time=$TIME \
    --mem=$MEM \
    --output="logs/$RUN_GROUP/${RUN_NAME}-%j.log" \
    --error="logs/$RUN_GROUP/${RUN_NAME}-%j.log" \
    --export=ALL \
    slurm_nhr/ade20k/train_policy.sbatch \
    --run-name "$RUN_NAME" \
    --seed "$SEED" \
    --resize-mode squish \
    --select-bn-eval \
    --wandb-project exp27
