#!/bin/bash
# exp22-fovi but with the student backbone INITIALIZED FROM THE DINOv3 TEACHER
# (--init-backbone-from-teacher) -- the ONLY difference; see exp22-fovi.sh for the
# fovi config provenance (notebook free-form cell: fov=35, res=64, cmf_a=0.5,
# fixation 1024px -> fixed-scale 2.0; cart_patch_size=5, force=True (default),
# doubleres, FiLM sigma-4 conditioning; 140 patches) and exp22-uniform16.sh for
# the full-run design (2M steps, 8192 steps/job @ 2h default QOS, in21k with-features).
#
# For the FOVEATED patcher only the transformer trunk transfers from the teacher
# (verified in canvit_pytorch/backbone/dinov3_init.py: it touches backbone.* only);
# the KNN patch embedding + FiLM conditioning stay random-init. The backbone's
# unused uniform patch_embed also gets the teacher conv weights -- harmless, the
# foveated path never calls it. Same code pins as exp22-fovi, so the pair is a
# clean teacher-init-vs-random A/B.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp22_full_runs
RUN_NAME=exp22-fovi-teacherinit
ARRAY=0-227%1                                  # 228 remaining of 245 (resume from step 139264; 17 jobs done)
TIME=0-02:00:00
MEM=128G
NGPU=1

# === OPTIONAL ===
CFG_WANDB_PROJECT=jon_exp22_full_runs
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=8192
CFG_VAL_EVERY=8192  # validate once per job (= steps_per_job)
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--model.patcher-name foveated --model.foveated-patcher.fov 35 --model.foveated-patcher.resolution 64 --model.foveated-patcher.cmf-a 0.5 --model.foveated-patcher.cart-patch-size 5 --model.foveated-patcher.arch-flag doubleres --model.foveated-patcher.conditioning.mode film --model.foveated-patcher.conditioning.film.fourier.num-features 256 --model.foveated-patcher.conditioning.film.fourier.sigma 4 --foveated-scale.fixed-scale 2.0 --init-backbone-from-teacher --webdataset-dir /mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/webdataset-imagenet-21k-with-features"
# =================

# Pin all pretraining code to exact commits. base_train.sbatch extracts these
# via offline `git archive` from the local clones (no network/SSH), snapshotting
# the run against any future `git pull` on the originals while the array is in flight.
PRETRAIN_COMMIT=fe24aa1  # repin: includes d2f7b50 (foveated validation at the training scale; no-op for uniform)
PYTORCH_COMMIT=3277048
FOVI_COMMIT=c399d3b

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* ]] && export "$v"; done

sbatch     --gpus-per-node=A100:$NGPU     --ntasks-per-node=$NGPU     --mem=$MEM     --time=$TIME     --array="$ARRAY"     --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --export=ALL     slurm_nhr/base_train.sbatch
