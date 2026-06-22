#!/bin/bash
# cond4-reg + ViT trunk adaLN modulation (sinusoidal encoding, num_freqs=6)
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp21_modulation
RUN_NAME=exp21-cond4-reg-modulate-trunk-sinusoidal
ARRAY=0-27%1                                   # 28 remaining of 49 (resume from step 86016); on-the-fly features + validation ON
TIME=0-01:30:00
MEM=128G
NGPU=1

# === OPTIONAL ===
CFG_WANDB_PROJECT=jon_exp21_modulation
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=4096
CFG_VAL_EVERY=4096  # validate once per job (= steps_per_job)
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--model.patcher-name square --model.square-patcher.method fovi_regularized --model.square-patcher.resolution 64 --model.square-patcher.cart-patch-size 8 --model.square-patcher.no-force-patches-less-than-matched --model.square-patcher.m-override 10 --model.square-patcher.strict-nest-when-possible --backbone-name vitb16_modulate --model.vit-modulation.enabled --model.vit-modulation.encoding sinusoidal --webdataset-dir /mnt/lustre-rzg/workspaces/ws/nib00021/u25995-canvit-data-no-features/webdataset-imagenet-1k-no-features"
# =================

# Pin all pretraining code to exact commits. base_train.sbatch extracts these
# via offline `git archive` from the local clones (no network/SSH), snapshotting
# the run against any future `git pull` on the originals while the array is in flight.
PRETRAIN_COMMIT=4544bb8
PYTORCH_COMMIT=96c35fb
FOVI_COMMIT=763bf7a

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-pretrain
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* ]] && export "$v"; done

sbatch     --gpus-per-node=A100:$NGPU     --ntasks-per-node=$NGPU     --mem=$MEM     --time=$TIME     --array="$ARRAY"     --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --export=ALL     slurm_nhr/base_train.sbatch
