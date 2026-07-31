#!/bin/bash
# Uniform OVERLAPPING patches, vitb8 = 8px patches: the 8px-patch overlap
# counterpart to exp21-uniform8. 15x15 grid, stride 4 (50% overlap) -> window
# (15-1)*4 + 8 = 64px, 225 patches. SAME 64px window and SAME 8px patches as
# exp21-uniform8, but densely overlapping (stride 4 < patch 8) instead of disjoint.
#
# NB this is NOT token-matched to the uniform family (69 tokens): 225 patches +
# 5 default registers = 230 tokens. Holding patch size (8px) AND window (64px)
# fixed while overlapping necessarily multiplies the patch count, so the
# uniform8-vs-this comparison confounds "overlap" with "~3.5x more tokens" -- by
# design (the p16-grid7-stride8 run is the token-matched overlap variant; this is
# the 8px-patch-fixed variant). ~3.5x tokens / ~12x per-glimpse attention vs
# uniform8 -> TIME bumped to 2:00 and pinned to an 80GB-VRAM node.
#
# patch-stride feature lives in PYTORCH_COMMIT d864b83 (same commit uniform8 pins).
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=jon_exp21_modulation
RUN_NAME=exp21-uniform-overlap-p8-grid15-stride4
ARRAY=0-47%1                                   # RESUME on c2927a5: chunk 0 (step-4096) already done under bc2db02; 48 more -> 200704 total
TIME=0-02:00:00
MEM=128G
NGPU=1
CONSTRAINT=80gb_vram                           # 225 tokens/glimpse -> need an 80GB-VRAM GPU

# === OPTIONAL ===
CFG_WANDB_PROJECT=jon_exp21_modulation
CFG_PEAK_LR=0.0004
CFG_BATCH_SIZE_PER_GPU=64
CFG_STEPS_PER_JOB=4096
CFG_VAL_EVERY=4096  # validate once per job
CFG_LOG_EVERY=512
CFG_NUM_WORKERS=4
EXTRA_ARGS="--model.patcher-name uniform --backbone-name vitb8 --glimpse-grid-size 15 --patch-stride 4"
# =================

# Pin all pretraining code to exact commits. base_train.sbatch extracts these
# via offline `git archive` from the local clones (no network/SSH), snapshotting
# the run against any future `git pull` on the originals while the array is in flight.
# PRETRAIN pinned to c2927a5 (NOT uniform8's bc2db02): c2927a5 is the descendant
# that fixes the overlapping-uniform local-stream PCA viz (bc2db02 mis-derives the
# glimpse grid as glimpse_px//patch and crashes pca_val for stride<patch). The
# bc2db02->c2927a5 diff is viz + square-patcher only and does NOT affect uniform
# training, so this stays comparable to uniform8 (which is why the p16 overlap run
# also ended on c2927a5). PYTORCH d864b83 still supplies the --patch-stride feature.
PRETRAIN_COMMIT=c2927a5
PYTORCH_COMMIT=d864b83
FOVI_COMMIT=763bf7a

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* ]] && export "$v"; done

sbatch     --gpus-per-node=A100:$NGPU     --ntasks-per-node=$NGPU     --mem=$MEM     --time=$TIME     --constraint="$CONSTRAINT"     --array="$ARRAY"     --output="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --error="logs/$RUN_GROUP/$RUN_NAME/log/job-%A_%a.log"     --export=ALL     slurm_nhr/base_train.sbatch
