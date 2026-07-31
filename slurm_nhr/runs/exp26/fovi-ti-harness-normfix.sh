#!/bin/bash
# exp26 — DOES THE NORMALIZER FIX CLOSE THE exp23 FOVEATED GAP? (harness side)
#
# exp23 ran the fovi A/B and the HARNESS side came out ~4-5 points low on
# val/scene_cos_norm_t9 (0.794 vs 0.834 @ 180k) while uniform16 matched to 0.002.
# Root cause found: `tasks/distill/task.py` passed `normalizer_max_samples or 512`,
# clobbering the documented `0` = "use the whole shard" sentinel, so the harness
# standardized its DINOv3 targets from 512 samples where the old loop used 4096.
# Fixed in 2c47a69 (verified: max_samples=0 reproduces the old-loop checkpoint's
# standardizer buffers to 4.8e-7).
#
# This is the SAME config as exp23-fovi-ti-harness, seed 0, only the pin moves
# (24a8500 -> 2c47a69). Overlay the curve directly on exp23-fovi-ti-oldloop, which
# is already on disk — no need to re-run that side.
# 8 jobs, not 25: in exp23 the gap was already unmistakable by step ~49k
# (val cos 0.654 vs 0.738, train loss +0.21), so 65,536 steps settles it in hours.
#
# READ WITH fovi-ti-oldloop-seed1.sh: the rollout engine is byte-exact but the
# compiled/bf16 A/A gradient noise floor is ~4e-3 per step, so two identical-code
# runs still diverge chaotically over 100k+ steps. That companion run measures the
# run-to-run spread of THIS recipe, which we have never had. Verdict rule:
#   harness-normfix inside the oldloop seed spread => parity, normalizer was it;
#   still ~4-5 points low                          => something else remains.
# NOTHING IS SUBMITTED by writing this file.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=exp26
RUN_NAME=exp26-fovi-ti-harness-normfix
ARRAY=0-7%1                  # 8 jobs x 8192 = 65,536 steps
TIME=0-02:00:00
MEM=128G
NGPU=1
TASK=distill

# === config (byte-identical to exp23-fovi-ti-harness) ===
CFG_WANDB_PROJECT=exp26
CFG_SEED=0
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=8192
CFG_VAL_EVERY=8192
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--cfg.model.patcher-name foveated --cfg.model.foveated-patcher.fov 35 --cfg.model.foveated-patcher.resolution 64 --cfg.model.foveated-patcher.cmf-a 0.5 --cfg.model.foveated-patcher.cart-patch-size 5 --cfg.model.foveated-patcher.arch-flag doubleres --cfg.model.foveated-patcher.conditioning.mode film --cfg.model.foveated-patcher.conditioning.film.fourier.num-features 256 --cfg.model.foveated-patcher.conditioning.film.fourier.sigma 4 --cfg.foveated-scale.fixed-scale 2.0 --cfg.init-backbone-from-teacher"
# =================

# 2c47a69 = normalizer sample-cap fix + patcher grad-norm detail. Everything else
# unchanged from exp23's harness pin.
PRETRAIN_COMMIT=2c47a69
PYTORCH_COMMIT=017ce9b
FOVI_COMMIT=c399d3b

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train
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
