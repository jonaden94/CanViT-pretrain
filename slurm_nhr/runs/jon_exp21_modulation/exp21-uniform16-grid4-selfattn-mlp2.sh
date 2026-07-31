#!/bin/bash
# Exactly like exp21-uniform16-grid4 (uniform, vitb16 = 16px patches, 4x4 grid,
# 16 patches + 53 backbone registers = 69 tokens) but with ONE canvas self-attn
# block added after each glimpse's writes (mlp_ratio 2). This is the first test
# of the new per-glimpse canvas (memory) self-attention: the 1040 canvas tokens
# (16 registers + 32^2 spatial) self-attend once per glimpse so memory tokens
# exchange information between glimpses, then the next glimpse reads/writes the
# self-attended canvas. The block is identity at init (zero-init output), so this
# starts from the exact uniform16-grid4 behavior and learns the self-attn as a delta.
#
# ONLY differences vs exp21-uniform16-grid4:
#   1. EXTRA_ARGS adds --model.n-canvas-self-attn-blocks 1 --model.canvas-self-attn-mlp-ratios 2.0
#   2. PYTORCH_COMMIT pinned to 2d6a807 (the commit that adds canvas self-attn)
#   3. TIME 1:00 -> 2:00 (one canvas self-attn block ~ +45-50% step time; margin
#      for torch.compile making the relative share higher — a timeout wastes the
#      whole 4096-step chunk, so over-allocate rather than risk it).
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp21_modulation
RUN_NAME=exp21-uniform16-grid4-selfattn-mlp2
ARRAY=0-48%1                                   # 49 jobs x 4096 = 200704 steps (full run)
TIME=0-02:00:00
MEM=128G
NGPU=1
CONSTRAINT=80gb_vram                           # canvas self-attn adds BPTT activation memory -> pin to 80GB GPU

# === OPTIONAL ===
CFG_WANDB_PROJECT=jon_exp21_modulation
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=4096
CFG_VAL_EVERY=4096  # validate once per job
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--model.patcher-name uniform --backbone-name vitb16 --glimpse-grid-size 4 --model.n-backbone-registers 53 --model.n-canvas-self-attn-blocks 1 --model.canvas-self-attn-mlp-ratios 2.0"
# =================

# Pin all pretraining code to exact commits. base_train.sbatch extracts these
# via offline `git archive` from the local clones (no network/SSH), snapshotting
# the run against any future `git pull` on the originals while the array is in flight.
# PYTORCH pinned to 2d6a807 (adds the canvas self-attn feature); PRETRAIN/FOVI same
# as exp21-uniform16-grid4 so the run differs ONLY by the self-attn block.
PRETRAIN_COMMIT=c2927a5
PYTORCH_COMMIT=2d6a807
FOVI_COMMIT=763bf7a

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* ]] && export "$v"; done

sbatch     --gpus-per-node=A100:$NGPU     --ntasks-per-node=$NGPU     --mem=$MEM     --time=$TIME     --constraint="$CONSTRAINT"     --array="$ARRAY"     --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --export=ALL     slurm_nhr/base_train.sbatch
