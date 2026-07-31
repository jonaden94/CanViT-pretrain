#!/bin/bash
# LR STAIRCASE, second drop: CONTINUE the 10x LR-drop branch
# (exp22-uniform16-teacherinit-lrdrop-639k, LR 4e-5, cancelled 2026-07-17 after
# 20 completed jobs) from its LAST checkpoint and drop the LR by another 0.1x:
# 4e-5 -> 4e-6. Lineage: parent 4e-4 until step 638,976 -> 4e-5 until
# parent-equivalent 802,816 (= branch step 163,840) -> 4e-6 from here.
#
# Seed = the 10x branch's step-163840.pt (its latest; written by task 19).
# SEED mode again: weights + standardizers from the ckpt, fresh optimizer/
# scheduler/step=0, NEW wandb run. Step accounting for comparisons:
#   this run's step k <-> parent-equivalent step k + 802,816.
# CFG_WARMUP_STEPS=0 -> pure ConstantLR at 4e-6 from step 0.
# --init-backbone-from-teacher stays DROPPED (seed weights supersede it).
#
# ARRAY 0-146%1: 147 x 8192 = 1,204,224 steps; 802,816 + 1,204,224 = 2,007,040
# = exactly the original parent's total budget, for 1:1 endpoint comparison.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp22_full_runs
RUN_NAME=exp22-uniform16-teacherinit-lrdrop2-803k
ARRAY=0-146%1                                  # 147 jobs x 8192 = 1,204,224 steps (ends at parent-equivalent 2,007,040)
TIME=0-02:00:00
MEM=128G
NGPU=1

# === OPTIONAL ===
CFG_WANDB_PROJECT=jon_exp22_full_runs
CFG_PEAK_LR=0.000004
CFG_WARMUP_STEPS=0
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=8192
CFG_VAL_EVERY=8192  # validate once per job (= steps_per_job)
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
CFG_SEED_CKPT=/user/henrich1/u25995/jonathan/repos/CanViT-train/logs/jon_exp22_full_runs/exp22-uniform16-teacherinit-lrdrop-639k/checkpoints/step-163840.pt
EXTRA_ARGS="--model.patcher-name uniform --webdataset-dir /mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/webdataset-imagenet-21k-with-features"
# =================

# Pin all pretraining code to exact commits. base_train.sbatch extracts these
# via offline `git archive` from the local clones (no network/SSH), snapshotting
# the run against any future `git pull` on the originals while the array is in flight.
PRETRAIN_COMMIT=fe24aa1
PYTORCH_COMMIT=3277048
FOVI_COMMIT=c399d3b

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* ]] && export "$v"; done

sbatch     --gpus-per-node=A100:$NGPU     --ntasks-per-node=$NGPU     --mem=$MEM     --time=$TIME     --array="$ARRAY"     --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --export=ALL     slurm/archive/base_train.sbatch
