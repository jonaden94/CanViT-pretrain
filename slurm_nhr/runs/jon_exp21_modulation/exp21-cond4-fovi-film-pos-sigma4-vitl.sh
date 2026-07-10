#!/bin/bash
# exp21-cond4-fovi-film-pos-sigma4-repro (foveated res64/cart8/doubleres, film
# fourier 256 sigma 4), but with a ViT-L STUDENT *and* ViT-L TEACHER (both
# 1024-dim) instead of ViT-B. Differs from that repro only by:
#   - --backbone-name vitl16            (student 768 -> 1024; 24 blocks, 16 heads)
#   - CFG_TEACHER_* = DINOv3 ViT-L      (teacher_dim auto-detected -> 1024)
#   - on-the-fly features (precomputed shards are B-only) -> no-features dataset
#   - validation ON + the heavier-model resource profile (matches exp21-cond4-fovi-L)
# Everything else (patcher, conditioning/FiLM, canvas, registers, VPE) is unchanged
# and dimension-generic.
#
# Runs DDP on 2 GPUs (ViT-L is infeasible single-GPU). batch_size_per_gpu=32 x
# world_size=2 = global batch 64 -- IDENTICAL to the single-GPU bs-64 global
# batch, so ALL hyperparameters stay the same (PEAK_LR is NOT scaled). srun
# launches one rank per GPU; the WebDataset loader shards across ranks (64
# distinct samples/step) and DDP all-reduces the per-rank mean losses -> global
# mean. NB checkpoints record (world_size, batch_size_per_gpu); this fresh run
# must stay at 2-GPU/bs-32 for all 49 array tasks (no mixed-world-size resume).
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp21_modulation
RUN_NAME=exp21-cond4-fovi-film-pos-sigma4-vitl
ARRAY=0-20%1                                   # RESUME: 21 remaining chunks (28/49 = 114688 steps already done) -> reaches 200,704; auto-resumes from latest checkpoint. validation ON
TIME=0-02:00:00                                # bumped from 1:30 -> wall margin (vitl task runs ~1:10) so a timeout can't leave the run short again
MEM=256G                                       # ~2x single-GPU: DDP replicates the full ViT-L per GPU
NGPU=2                                         # DDP across 2 GPUs (one srun rank each)

# === OPTIONAL ===
CFG_WANDB_PROJECT=jon_exp21_modulation
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=32  # x world_size 2 = global batch 64 (unchanged) -> PEAK_LR stays 0.0004
CFG_STEPS_PER_JOB=4096
CFG_VAL_EVERY=4096  # validate once per job (= steps_per_job)
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
# DINOv3 ViT-L distillation target (vs the default ViT-B). teacher_dim is
# auto-detected from the loaded teacher (1024); do NOT set --model.teacher-dim.
CFG_TEACHER_REPO_ID=facebook/dinov3-vitl16-pretrain-lvd1689m
CFG_TEACHER_NAME=dinov3_vitl16
EXTRA_ARGS="--backbone-name vitl16 --model.patcher-name foveated --model.foveated-patcher.resolution 64 --model.foveated-patcher.cart-patch-size 8 --model.foveated-patcher.arch-flag doubleres --model.foveated-patcher.no-force-patches-less-than-matched --model.foveated-patcher.conditioning.mode film --model.foveated-patcher.conditioning.film.fourier.num-features 256 --model.foveated-patcher.conditioning.film.fourier.sigma 4 --webdataset-dir /mnt/lustre-rzg/workspaces/ws/nib00021/u25995-canvit-data-no-features/webdataset-imagenet-1k-no-features"
# =================

# Pin all pretraining code to exact commits. base_train.sbatch extracts these
# via offline `git archive` from the local clones (no network/SSH), snapshotting
# the run against any future `git pull` on the originals while the array is in flight.
PRETRAIN_COMMIT=ab578c4
PYTORCH_COMMIT=f853eca
FOVI_COMMIT=763bf7a

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-pretrain
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* ]] && export "$v"; done

sbatch     --gpus-per-node=A100:$NGPU     --ntasks-per-node=$NGPU     --mem=$MEM     --time=$TIME     --array="$ARRAY"     --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --export=ALL     slurm_nhr/base_train.sbatch
