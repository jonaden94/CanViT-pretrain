#!/bin/bash
# Identical to exp21-strided-gf2 EXCEPT the 4 corner patches are NOT removed
# (no --drop-corners). Strided square patcher, grid_size_fovea=2, edge=[2,6,9,18].
# 44 patches (the full strided pattern, corners kept) + 31 backbone registers = 75
# backbone tokens. Registers are held identical to exp21-strided-gf2 (31), so the
# ONLY difference vs gf2 is the 4 corner patches (gf2 = 40 patches + 31 reg = 71).
# Same pinned commits as gf2 so the corner flag is the sole code/config difference.
# Precomputed features.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp21_modulation
RUN_NAME=exp21-strided-gf2-keepcorners
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
EXTRA_ARGS="--model.patcher-name square --model.square-patcher.method strided --model.square-patcher.grid-size-fovea 2 --model.square-patcher.patch-size 8 --model.square-patcher.edge-length-multipliers 2 6 9 18 --model.n-backbone-registers 31"
# =================

# Pin all pretraining code to exact commits. base_train.sbatch extracts these
# via offline `git archive` from the local clones (no network/SSH), snapshotting
# the run against any future `git pull` on the originals while the array is in flight.
PRETRAIN_COMMIT=bc2db02
PYTORCH_COMMIT=d864b83
FOVI_COMMIT=763bf7a

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-pretrain
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* ]] && export "$v"; done

sbatch     --gpus-per-node=A100:$NGPU     --ntasks-per-node=$NGPU     --mem=$MEM     --time=$TIME     --array="$ARRAY"     --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --export=ALL     slurm_nhr/base_train.sbatch
