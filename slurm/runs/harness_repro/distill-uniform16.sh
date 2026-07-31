#!/bin/bash
# HARNESS REPRODUCTION run — exp22-uniform16 through the UNIFIED harness
# (python -m canvit_train.harness.run distill) instead of the old train/loop.py.
#
# Purpose: production-scale fidelity A/B before the big-bang cutover. Overlay this run's
# loss + val-recon curves against the EXISTING old-loop exp22-uniform16 run (you already
# have those results) to confirm the harness reproduces distill training. NOT the full
# 2M-step run — ~100k steps is enough to see the curves agree.
#
# Config is IDENTICAL to slurm/archive/runs/jon_exp22_full_runs/exp22-uniform16.sh; only the
# training CODE differs (harness vs old loop). Flags gain the `--cfg.` prefix because the
# harness nests the task config under `cfg`. webdataset-dir is NOT repeated — .envrc.grete
# already points WEBDATASET_DIR at the in21k-with-features set, which harness_train.sbatch
# injects as --cfg.webdataset-dir.  NOTHING IS SUBMITTED by writing this file.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=harness_repro
RUN_NAME=distill-uniform16
ARRAY=0-12%1                 # 13 jobs x 8192 = 106,496 steps (~100k) for a curve comparison
TIME=0-02:00:00
MEM=128G
NGPU=1
TASK=distill

# === config (identical to exp22-uniform16) ===
CFG_WANDB_PROJECT=harness_repro
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=8192
CFG_VAL_EVERY=8192           # validate once per job
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--cfg.model.patcher-name uniform"

# === commit pins ===
# pretrain = the unified harness (bc63eee); fovi unchanged from exp22 (c399d3b).
# canvit_pytorch pinned to its CURRENT head (017ce9b) — what bc63eee was tested against.
# exp22 used PYTORCH 3277048, so for a STRICT loop-only A/B either re-pin to 3277048 (if
# bc63eee is API-compatible with it) or treat the 3277048->017ce9b drift as a second
# variable when reading curve differences.
PRETRAIN_COMMIT=bc63eee
PYTORCH_COMMIT=017ce9b
FOVI_COMMIT=c399d3b

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train
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
    slurm/harness_train.sbatch
