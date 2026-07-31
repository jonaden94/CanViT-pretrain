# exp27 ARM E — arm C + POOLED policy loss: the original repo's rollout ARCHITECTURE.
#
# ONE knob vs arm C (`--pooled-policy-loss`); everything else is arm C verbatim.
#
# WHY. Read from CanViT-PyTorch-RL's own source 2026-07-30, this is the last substantive
# training-side difference from the original. Their `training/rollout.py` collects features
# per depth under no_grad, then `train.py:222` runs ONE grad-bearing train-mode
# `net(cat(feats))` over B*H = 64 samples POOLED across depths t0..t3 -> ONE BatchNorm
# running-stat update per step, from depth-pooled statistics. Our in-graph default instead
# does 4 per-depth train-mode forwards over 16 samples each -> 4 BN updates per step from
# single-depth statistics. Verified by counting train-mode BN forwards:
#
#     pooled=False  4 forwards, batch sizes [16,16,16,16]   (our default)
#     pooled=True   1 forward,  batch size  [64]            (matches the original)
#
# WHY IT IS THE SUSPECT. Deployment selects under EVAL-mode BN, i.e. off the running stats.
# This changes exactly those stats (their source AND their update rate) while leaving the
# objective's expectation about intact — which matches the measured signature: our policies
# hit the published CE band dead on (0.6855/0.6856 vs 0.6853 +- 0.0007) yet sit 0.116 mIoU
# t4 below the published policies scored through the SAME eval (p=0.018, n=9 vs 8).
#
# `pooled_policy_loss` had never been executed before this arm — it was written as the
# other half of the p3-notes delta #1 and left unvalidated. Smoke-tested locally first
# (trains, reward_frac positive, BN structure as above) rather than trusted.
#
# WHAT TO EXPECT. If this is the cause, mIoU t4 -> ~44.96 with CE unchanged. If t4 stays at
# ~44.86, the residual is not the rollout architecture and the next surface is the core
# library revisions (their band was trained against their own sibling clones at
# bcb9742f->007f7173 on a 4090; ours against these clones on A100s) — see doc 15 §A5.
#
# NOTHING IS SUBMITTED by writing this file.
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
RUN_NAME=exp27-policy-pooled-s$SEED
TIME="${TIME:-0-06:00:00}"   # 6h: a slow node killed a 4h job at step ~7100 once
MEM=64G

export ADE20K_ROOT=/user/henrich1/u25995/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016
export WANDB_PROJECT=exp27

# Pin the reference implementation. 4db7c3f restores rl_train's resize_mode default
# to squish, so THIS ARM DOES DEPEND ON IT — do not roll the pin back to cea4dee or
# earlier, where the default was center_crop and the run is not band-comparable.
# PIN DIFFERS FROM ARM C ON PURPOSE: arm C pinned d3c32d7, which PREDATES
# `pooled_policy_loss` (added in 4428e34) — so this arm cannot use it. The intervening
# rl_train changes are both no-ops for this arm: `select_bn_eval`'s default flipped to True
# (we pass --select-bn-eval explicitly either way) and `ce_from_logits` became a delegation
# to the shared `reward_ce` (pinned bit-identical by test_reward_ce_shared.py). Verified
# locally by gradient comparison at pooled=False before submitting.
PRETRAIN_COMMIT=4bf8bf8
PYTORCH_COMMIT=017ce9b
FOVI_COMMIT=c399d3b

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train
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
    --pooled-policy-loss \
    --wandb-project exp27
