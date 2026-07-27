#!/bin/bash
# LR-DROP branch of exp22-uniform16-teacherinit: seed from its BEST-reconstruction
# checkpoint (step-638976.pt, val scene_cos_norm 0.8828 = the max over the run;
# picked 2026-07-16) and continue training with a 10x LOWER constant LR
# (4e-4 -> 4e-5), NO warmup. Otherwise identical to the parent run (uniform
# patcher, in21k with-features, batch 64, 8192 steps/job @ 2h, same pins).
#
# Mechanics (all verified in loop.py/scheduler.py/tracker.py):
#   - CFG_SEED_CKPT -> SEED mode: loads model weights + standardizers + model
#     config from the .pt; FRESH optimizer/scheduler/step=0; NEW wandb run
#     (parent run untouched). Parent step k <-> this run's step k-638976.
#   - CFG_WARMUP_STEPS=0 -> pure ConstantLR: LR = peak_lr = 4e-5 from step 0.
#   - --init-backbone-from-teacher is DROPPED: the seed checkpoint already
#     carries the trained weights (the flag would only re-init the backbone and
#     immediately be overwritten by the seed load).
#   - Fresh Adam moments are the one unavoidable difference vs "continuing" the
#     parent (SEED mode does not carry optimizer state) — standard for LR-drop
#     branch experiments.
#   - Data: fresh run -> shard schedule restarts at job 0 (seed 0). The parent
#     had consumed schedule jobs 0-77 (~0.4 epoch of its ~9.8-epoch budget) at
#     the seed point, so early shards repeat in the same order — harmless.
#
# ARRAY 0-166%1: 167 x 8192 = 1,368,064 steps; 638,976 + 1,368,064 = 2,007,040
# = exactly the parent's total budget, so parent-vs-branch compare 1:1 at equal
# total steps seen.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp22_full_runs
RUN_NAME=exp22-uniform16-teacherinit-lrdrop-639k
ARRAY=0-166%1                                  # 167 jobs x 8192 = 1,368,064 steps (ends at parent-equivalent 2,007,040)
TIME=0-02:00:00
MEM=128G
NGPU=1

# === OPTIONAL ===
CFG_WANDB_PROJECT=jon_exp22_full_runs
CFG_PEAK_LR=0.00004
CFG_WARMUP_STEPS=0
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=8192
CFG_VAL_EVERY=8192  # validate once per job (= steps_per_job)
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
CFG_SEED_CKPT=/user/henrich1/u25995/jonathan/repos/CanViT-pretrain/logs/jon_exp22_full_runs/exp22-uniform16-teacherinit/checkpoints/step-638976.pt
EXTRA_ARGS="--model.patcher-name uniform --webdataset-dir /mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/webdataset-imagenet-21k-with-features"
# =================

# Pin all pretraining code to exact commits. base_train.sbatch extracts these
# via offline `git archive` from the local clones (no network/SSH), snapshotting
# the run against any future `git pull` on the originals while the array is in flight.
PRETRAIN_COMMIT=fe24aa1
PYTORCH_COMMIT=3277048
FOVI_COMMIT=c399d3b

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-pretrain
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* ]] && export "$v"; done

sbatch     --gpus-per-node=A100:$NGPU     --ntasks-per-node=$NGPU     --mem=$MEM     --time=$TIME     --array="$ARRAY"     --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --export=ALL     slurm_nhr/base_train.sbatch
