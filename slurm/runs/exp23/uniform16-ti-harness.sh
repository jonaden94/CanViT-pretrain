#!/bin/bash
# exp23 — CUTOVER FIDELITY A/B, HARNESS side, uniform16 + teacher-init.
#
# This is the CURRENT-STACK half of the exp23 pair: exp22-uniform16-teacherinit run
# through the UNIFIED harness (python -m canvit_train.harness.run distill) on the
# CURRENT code. Overlay against uniform16-ti-oldloop.sh (exp22's exact old stack).
#
# Pins: pretrain 24a8500 = current HEAD (NOT bc63eee — 24a8500 adds the matplotlib-DDP
# and backward_pass_autocast="off" fixes a COMPILED run needs; compile is on by default).
# pytorch 017ce9b = current HEAD; fovi c399d3b = unchanged from exp22. So this bundles the
# FULL current stack vs the FULL exp22 stack in the old-loop sibling — the real "does the
# code I want to ship reproduce exp22" question.
#
# Config identical to uniform16-ti-oldloop.sh; only the training CODE + entry point differ.
# Flags gain the `--cfg.` prefix (harness nests the task config under `cfg`). webdataset-dir
# is injected from .envrc.grete's WEBDATASET_DIR (= exp22's exact path), same as the old
# side. NOTHING IS SUBMITTED by writing this file.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=exp23
RUN_NAME=exp23-uniform16-ti-harness
ARRAY=0-24%1                 # 25 jobs x 8192 = 204,800 steps (matches the old-loop side)
TIME=0-02:00:00
MEM=128G
NGPU=1
TASK=distill

# === config (identical to uniform16-ti-oldloop.sh) ===
CFG_WANDB_PROJECT=exp23
CFG_SEED=0
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=8192
CFG_VAL_EVERY=8192
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--cfg.model.patcher-name uniform --cfg.init-backbone-from-teacher"
# =================

# CURRENT unified stack. 24a8500 = pretrain HEAD (includes the compile-correctness fixes).
PRETRAIN_COMMIT=24a8500
PYTORCH_COMMIT=017ce9b
FOVI_COMMIT=c399d3b

# Repo root, derived from this script's own location (slurm/runs/<group>/<run>.sh),
# so the run submits from YOUR clone rather than one hardcoded checkout.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
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
