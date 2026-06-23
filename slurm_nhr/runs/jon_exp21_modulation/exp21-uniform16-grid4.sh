#!/bin/bash
# Uniform, vitb16 = 16px patches, 4x4 grid, non-overlapping -> 64px glimpse window,
# 16 patches. Third member of the same-pixels triplet: it samples the SAME 64px
# window (identical extracted pixels, verified maxdiff 0) as exp21-uniform8 (8px
# patches, 64 tiles) and exp21-uniform-overlap-p16-grid7-stride8 (16px patches,
# stride 8, 49 overlapping patches). Here the 16px patches are disjoint (stride 16),
# so each pixel is read exactly once, like uniform8 but at 16px granularity.
# 16 patches + 53 backbone registers = 69 tokens (matched to the uniform family).
# Precomputed features.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp21_modulation
RUN_NAME=exp21-uniform16-grid4
ARRAY=0-48%1                                   # 49 jobs x 4096 = 200704 steps (full run)
TIME=0-01:00:00
MEM=128G
NGPU=1

# === OPTIONAL ===
CFG_WANDB_PROJECT=jon_exp21_modulation
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=4096
CFG_VAL_EVERY=4096  # validate once per job
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--model.patcher-name uniform --backbone-name vitb16 --glimpse-grid-size 4 --model.n-backbone-registers 53"
# =================

# Pin all pretraining code to exact commits. base_train.sbatch extracts these
# via offline `git archive` from the local clones (no network/SSH), snapshotting
# the run against any future `git pull` on the originals while the array is in flight.
PRETRAIN_COMMIT=c2927a5
PYTORCH_COMMIT=d864b83
FOVI_COMMIT=763bf7a

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-pretrain
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* ]] && export "$v"; done

sbatch     --gpus-per-node=A100:$NGPU     --ntasks-per-node=$NGPU     --mem=$MEM     --time=$TIME     --array="$ARRAY"     --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --export=ALL     slurm_nhr/base_train.sbatch
