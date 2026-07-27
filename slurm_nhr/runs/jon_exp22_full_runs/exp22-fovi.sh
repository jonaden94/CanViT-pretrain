#!/bin/bash
# exp22 FULL RUN: foveated counterpart of exp22-uniform16 (same 2M-step / 2h-walltime /
# in21k-with-features design; see exp22-uniform16.sh for that rationale).
#
# The fovi geometry + patching come from
# fovi/notebooks/fovi_square_patches/visualize_hyperparams_fovi.ipynb:
#   - "Free-form" oversampling cell (freeform_configs[0]):
#       fov=35, resolution=64, cmf_a=0.5, fixation_size=1024px (on a 512px image)
#   - "Free-form patches" cell:
#       cart_patch_size=5, arch_flag='doubleres', sample_cortex=True (the config
#       default) -- but force_patches_less_than_matched=True (config default),
#       DEVIATING from the notebook's False by user request: True quantizes the
#       ring grid DOWN, giving 140 patches instead of 168 (both verified by
#       instantiating FoveatedPatcher).
# fixation_size is NOT a patcher field -- the fixation window is scale*H per
# forward -- so fixation 1024px on the 512px scene maps to
# --foveated-scale.fixed-scale 2.0 (mode 'fixed' is the default: every glimpse,
# incl. the FULL start glimpse, foveates over a 2x-scene window; peripheral
# samples fall outside the image). 140 patches is deliberately NOT token-matched
# to uniform16's 64 or exp21-fovi-repro's 66.
#
# Patch-embed conditioning follows the exp21 fovi standard ("cond4"):
# FiLM position conditioning, fourier num-features 256, sigma 4
# (as in exp21-cond4-fovi-film-pos-sigma4-*).
#
# Timing: exp21 fovi jobs (65 patches, precomputed features) took ~36 min / 4096
# steps; at 140 patches, 8192 steps may land close to the 2h limit. If a task
# times out it writes no checkpoint and the next array task simply redoes it
# (array then ends one job short of 2M -- top up with extra tasks); watch task 0's
# elapsed time and halve CFG_STEPS_PER_JOB if needed.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp22_full_runs
RUN_NAME=exp22-fovi
ARRAY=0-224%1                                  # 225 remaining of 245 (resume from step 163840; 20 jobs done)
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
EXTRA_ARGS="--model.patcher-name foveated --model.foveated-patcher.fov 35 --model.foveated-patcher.resolution 64 --model.foveated-patcher.cmf-a 0.5 --model.foveated-patcher.cart-patch-size 5 --model.foveated-patcher.arch-flag doubleres --model.foveated-patcher.conditioning.mode film --model.foveated-patcher.conditioning.film.fourier.num-features 256 --model.foveated-patcher.conditioning.film.fourier.sigma 4 --foveated-scale.fixed-scale 2.0 --webdataset-dir /mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/webdataset-imagenet-21k-with-features"
# =================

# Pin all pretraining code to exact commits. base_train.sbatch extracts these
# via offline `git archive` from the local clones (no network/SSH), snapshotting
# the run against any future `git pull` on the originals while the array is in flight.
PRETRAIN_COMMIT=fe24aa1  # repin: includes d2f7b50 (foveated validation at the training scale; no-op for uniform)
PYTORCH_COMMIT=3277048
FOVI_COMMIT=c399d3b

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-pretrain
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* ]] && export "$v"; done

sbatch     --gpus-per-node=A100:$NGPU     --ntasks-per-node=$NGPU     --mem=$MEM     --time=$TIME     --array="$ARRAY"     --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --export=ALL     slurm_nhr/base_train.sbatch
