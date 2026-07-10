#!/bin/bash
# Extreme-foveation ("near-uniform" CMF) counterpart of exp21-cond4-fovi-film-pos-sigma4-repro.
# EXACTLY the repro config except cmf_a 0.5 -> 1000: with a=1000 the CMF 1/(r+a) is
# almost flat over the eccentricity range, so the fovi sensor samples nearly uniformly
# (little foveal magnification). This is the pattern in Plot 6 (cps=8 panel) of
# fovi/notebooks/fovi_square_patches/visualize_hyperparams_fovi.ipynb.
#
# NB the notebook's Plot 6 *markdown title* says cmf_a=100, but its code cell uses
# cmf_a=1000 -- and 65 patches only occurs at 1000 (at cmf_a=100 the ring grid quantizes
# to 79 patches at cps=8 / 52 at cps=9, never 65). Verified patch counts (res=64, fov=180,
# doubleres, no-force):
#   repro   cmf_a=0.5,  cart_patch_size=8 -> 66 patches   (the token-count target)
#   THIS    cmf_a=1000, cart_patch_size=8 -> 65 patches   (token-matched, off by 1;
#           65 is as close as the quantized ring grid gets to 66)
# So this holds cart_patch_size=8 fixed (same as the repro) and changes ONLY cmf_a,
# giving the extreme-foveation pattern at ~the same token count. cmf_a is a long-standing
# FoveatedPatcherConfig field, so no code change is needed.
#
# Differences vs exp21-cond4-fovi-film-pos-sigma4-repro:
#   1. EXTRA_ARGS adds --model.foveated-patcher.cmf-a 1000
#   2. ARRAY 0-19%1 (a resume of the repro) -> 0-48%1 (this is a fresh full run)
#   3. validation ON (CFG_VAL_EVERY=4096, "validate once per job") -- the repro had it
#      OFF as a diagnostic; here we want normal per-job validation.
#   4. code pinned to CURRENT HEAD commits (the repro was unpinned). base_train.sbatch
#      extracts these via offline `git archive` from the local clones, snapshotting the
#      run against any future `git pull` on the originals while the array is in flight.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp21_modulation
RUN_NAME=exp21-cond4-fovi-film-pos-sigma4-cmfa1000
ARRAY=0-48%1                                   # 49 jobs x 4096 = 200704 steps (fresh full run)
TIME=0-00:45:00
MEM=128G
NGPU=1

# === OPTIONAL ===
CFG_WANDB_PROJECT=jon_exp21_modulation
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=4096
CFG_VAL_EVERY=4096  # validate once per job (normal validation)
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--model.patcher-name foveated --model.foveated-patcher.resolution 64 --model.foveated-patcher.cart-patch-size 8 --model.foveated-patcher.cmf-a 1000 --model.foveated-patcher.arch-flag doubleres --model.foveated-patcher.no-force-patches-less-than-matched --model.foveated-patcher.conditioning.mode film --model.foveated-patcher.conditioning.film.fourier.num-features 256 --model.foveated-patcher.conditioning.film.fourier.sigma 4"
# =================

# Pin all pretraining code to exact CURRENT HEAD commits (main). base_train.sbatch
# extracts these via offline `git archive` from the local clones (no network/SSH).
# PYTORCH 2d6a807 includes the canvas-self-attn feature but it is a no-op here
# (n_canvas_self_attn_blocks defaults to 0), so this run is unaffected by it.
PRETRAIN_COMMIT=ab578c4
PYTORCH_COMMIT=2d6a807
FOVI_COMMIT=070526a

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-pretrain
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* ]] && export "$v"; done

sbatch     --gpus-per-node=A100:$NGPU     --ntasks-per-node=$NGPU     --mem=$MEM     --time=$TIME     --array="$ARRAY"     --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --export=ALL     slurm_nhr/base_train.sbatch
