#!/bin/bash
# exp22 FULL RUN: default uniform16 model (= exp21-uniform16-repro config) trained
# 10x longer -- 2M steps -- on the NEW ImageNet-21k with-features webdataset.
#
# Differences vs exp21-uniform16-repro:
#   1. 2M steps: ARRAY 0-244%1, CFG_STEPS_PER_JOB 4096 -> 8192
#      (245 jobs x 8192 = 2,007,040 steps; exp21 was 49 x 4096 = 200,704)
#   2. each job spans 2x the steps and requests 2h walltime under the DEFAULT QOS.
#      (--qos=2h was tried but is unusable for arrays: MaxSubmitPU=10 counts every
#      pending array task, so a 245-task array is rejected outright. Measured
#      exp21 default-QOS gaps between consecutive %1 tasks were only ~2-45 min,
#      so the loss is minor.)
#   3. data: in21k with PRECOMPUTED dinov3_vitb16 features (13.15M imgs, 3212 shards,
#      512px; the loader auto-detects features from info.json "keys"), instead of
#      in1k no-features/on-the-fly. Passed explicitly (not via .envrc.grete's
#      WEBDATASET_DIR default) so a future .envrc change can't switch the dataset
#      mid-array -- same rationale as commit pinning. 245 jobs x 8192 x 64 imgs
#      ~= 9.8 epochs over in21k.
#   4. code pinned to current HEADs (newer than the exp21 pins; includes the
#      teacher-init feature so this run is a clean random-init CONTROL for
#      exp22-uniform16-teacherinit on identical code).
# LR schedule unchanged: 100k warmup -> constant 4e-4 (defaults; now 5% of the run).
# Timing headroom: exp21 uniform 4096-step jobs took 42-50 min ON-THE-FLY; with
# precomputed features 8192 steps should fit well within 2h.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp22_full_runs
RUN_NAME=exp22-uniform16
ARRAY=0-227%1                                  # 228 remaining of 245 (resume from step 139264: 10 qos-test + 7 continuation jobs done)
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
EXTRA_ARGS="--model.patcher-name uniform --webdataset-dir /mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/webdataset-imagenet-21k-with-features"
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
