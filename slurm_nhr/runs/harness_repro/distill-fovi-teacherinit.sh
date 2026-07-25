#!/bin/bash
# HARNESS REPRODUCTION run — exp22-fovi-teacherinit (foveated + backbone INITIALIZED FROM
# THE DINOv3 TEACHER) through the unified harness. A/B vs the old-loop
# exp22-fovi-teacherinit run. The ONLY difference from distill-fovi.sh is
# --cfg.init-backbone-from-teacher, so the fovi / fovi-teacherinit pair is a clean
# teacher-init-vs-random A/B under the harness (mirrors the old-loop pair).
# NOTHING IS SUBMITTED by writing this file.
set -euo pipefail

RUN_GROUP=harness_repro
RUN_NAME=distill-fovi-teacherinit
ARRAY=0-12%1                 # ~100k steps
TIME=0-02:00:00
MEM=128G
NGPU=1
TASK=distill

CFG_WANDB_PROJECT=harness_repro
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=8192
CFG_VAL_EVERY=8192
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--cfg.model.patcher-name foveated --cfg.model.foveated-patcher.fov 35 --cfg.model.foveated-patcher.resolution 64 --cfg.model.foveated-patcher.cmf-a 0.5 --cfg.model.foveated-patcher.cart-patch-size 5 --cfg.model.foveated-patcher.arch-flag doubleres --cfg.model.foveated-patcher.conditioning.mode film --cfg.model.foveated-patcher.conditioning.film.fourier.num-features 256 --cfg.model.foveated-patcher.conditioning.film.fourier.sigma 4 --cfg.foveated-scale.fixed-scale 2.0 --cfg.init-backbone-from-teacher"

# See distill-uniform16.sh for the PYTORCH_COMMIT A/B caveat.
PRETRAIN_COMMIT=bc63eee
PYTORCH_COMMIT=017ce9b
FOVI_COMMIT=c399d3b

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-pretrain
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
