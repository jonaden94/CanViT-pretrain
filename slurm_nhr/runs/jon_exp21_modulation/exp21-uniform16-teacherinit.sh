#!/bin/bash
# exp21-uniform16-repro but with the student backbone INITIALIZED FROM THE DINOv3
# TEACHER instead of random init (--init-backbone-from-teacher). Uniform patcher +
# vitb16 is the best case: both the transformer trunk AND patch_embed transfer 1:1
# from the DINOv3 ViT-B/16 teacher (same design; teacher q/k/v fused into qkv,
# K-bias zero-filled). Everything else (data, on-the-fly features, LR, batch,
# validation) matches exp21-uniform16-repro.
#
# Differences vs exp21-uniform16-repro:
#   1. EXTRA_ARGS adds --init-backbone-from-teacher
#   2. ARRAY 0-26%1 (a resume of the repro) -> 0-48%1 (fresh full run, step 0)
#   3. code pinned to the commits that ADD this feature (repro predates it):
#        PYTORCH  96c35fb -> ed4f5aa  (adds dinov3_init backbone loader)
#        PRETRAIN 4544bb8 -> 669c05d  (adds init_backbone_from_teacher flag)
#      NB this is newer code than the repro's pinned commits, so a strictly clean
#      A/B needs a random-init CONTROL run on these SAME commits (i.e. this script
#      without --init-backbone-from-teacher), not the original repro run.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp21_modulation
RUN_NAME=exp21-uniform16-teacherinit
ARRAY=0-48%1                                   # 49 jobs x 4096 = 200704 steps (fresh full run)
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
EXTRA_ARGS="--model.patcher-name uniform --init-backbone-from-teacher --webdataset-dir /mnt/lustre-rzg/workspaces/ws/nib00021/u25995-canvit-data-no-features/webdataset-imagenet-1k-no-features"
# =================

# Pin all pretraining code to exact commits. base_train.sbatch extracts these
# via offline `git archive` from the local clones (no network/SSH), snapshotting
# the run against any future `git pull` on the originals while the array is in flight.
# PYTORCH ed4f5aa / PRETRAIN 669c05d are the commits that introduce the teacher-init
# feature (the repro's older pins predate it). FOVI unchanged (uniform doesn't use fovi).
PRETRAIN_COMMIT=669c05d
PYTORCH_COMMIT=ed4f5aa
FOVI_COMMIT=763bf7a

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* ]] && export "$v"; done

sbatch     --gpus-per-node=A100:$NGPU     --ntasks-per-node=$NGPU     --mem=$MEM     --time=$TIME     --array="$ARRAY"     --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --export=ALL     slurm_nhr/base_train.sbatch
