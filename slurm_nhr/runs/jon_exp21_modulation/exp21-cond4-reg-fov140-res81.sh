#!/bin/bash
# Regularized (fovi_regularized square) counterpart of exp21-cond4-fovi-fov140-res81.
# Same fovi geometry (fov=140, res=81, cart=10), m_override=12, force=False -> 71 patches
# (matches the foveated version exactly). 71 patches + 0 backbone registers = 71. Precomputed features.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp21_modulation
RUN_NAME=exp21-cond4-reg-fov140-res81
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
EXTRA_ARGS="--model.patcher-name square --model.square-patcher.method fovi_regularized --model.square-patcher.fov 140 --model.square-patcher.resolution 81 --model.square-patcher.cart-patch-size 10 --model.square-patcher.no-force-patches-less-than-matched --model.square-patcher.m-override 12 --model.square-patcher.strict-nest-when-possible --model.square-patcher.conditioning.mode film --model.square-patcher.conditioning.film.fourier.num-features 256 --model.square-patcher.conditioning.film.fourier.sigma 4 --model.n-backbone-registers 0"
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
