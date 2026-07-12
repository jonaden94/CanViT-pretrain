#!/bin/bash
# exp22-uniform16 but with the student backbone INITIALIZED FROM THE DINOv3 TEACHER
# (--init-backbone-from-teacher) -- the ONLY difference; see exp22-uniform16.sh for
# the full-run design (2M steps, 8192 steps/job @ 2h default QOS, in21k with-features).
# Uniform patcher + vitb16 is the best case for teacher init: both the transformer
# trunk AND patch_embed transfer 1:1 from the DINOv3 ViT-B/16 teacher (teacher q/k/v
# fused into qkv, K-bias zero-filled). Same code pins as exp22-uniform16, so the
# pair is a clean teacher-init-vs-random A/B.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp22_full_runs
RUN_NAME=exp22-uniform16-teacherinit
ARRAY=0-244%1                                  # 245 jobs x 8192 = 2,007,040 steps (~2M, fresh full run)
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
EXTRA_ARGS="--model.patcher-name uniform --init-backbone-from-teacher --webdataset-dir /mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/webdataset-imagenet-21k-with-features"
# =================

# Pin all pretraining code to exact commits. base_train.sbatch extracts these
# via offline `git archive` from the local clones (no network/SSH), snapshotting
# the run against any future `git pull` on the originals while the array is in flight.
PRETRAIN_COMMIT=66a12c4
PYTORCH_COMMIT=3277048
FOVI_COMMIT=c399d3b

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-pretrain
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* ]] && export "$v"; done

sbatch     --gpus-per-node=A100:$NGPU     --ntasks-per-node=$NGPU     --mem=$MEM     --time=$TIME     --array="$ARRAY"     --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --export=ALL     slurm_nhr/base_train.sbatch
