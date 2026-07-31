#!/bin/bash
# exp25 — ImageNet-1k FULL-MODEL finetune through the HARNESS, on the best-scene_cos_norm_t9
# checkpoint of exp22-uniform16-lrdrop-1516k (step-319488, converted to HF).
# See in1k-uni16ti-803k.sh for the full recipe rationale (TPU in1k finetune, batch-adapted:
# batch 64 / lr 6.25e-6 / warmup 100k / 401,408 steps; everything else byte-identical to TPU).
# NOTHING IS SUBMITTED by writing this file.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=exp25
RUN_NAME=in1k-uni16-1516k
ARRAY=0-48%1                 # 49 jobs x 8192 = 401,408 steps (~20 epochs @ batch 64)
TIME=0-12:00:00
MEM=128G
NGPU=1
TASK=in1k

# === config (TPU recipe, batch-adapted) ===
CFG_WANDB_PROJECT=exp25
CFG_RUN_NAME=in1k-uni16-1516k      # wandb run name = the finetune BASE model (owner request)
CFG_MODEL_REPO=/user/henrich1/u25995/jonathan/repos/CanViT-train/logs/jon_exp22_full_runs/exp22-uniform16-lrdrop-1516k/checkpoints/step-319488-hf
CFG_PROBE_REPO=canvit/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe  # fused into the head (TPU parity; was a RANDOM head before the fix)
CFG_MODE=finetune
CFG_BATCH_SIZE=64
CFG_PEAK_LR=6.25e-6
CFG_WEIGHT_DECAY=1e-4
CFG_WARMUP_STEPS=100000
CFG_MAX_STEPS=401408
CFG_STEPS_PER_JOB=8192
CFG_N_TIMESTEPS=4
CFG_GRAD_CLIP=1.0
CFG_LABEL_SMOOTHING=0.1
CFG_MIN_VP_SCALE=0.05
CFG_EVAL_POLICY=coarse_to_fine
CFG_VAL_EVERY=10000                # ~half-epoch (1 epoch @ b64 = 20,018 steps); full 50k val each
OPT_CKPT_DIR=logs/exp25/in1k-uni16-1516k/checkpoints
OPT_RESUME=True
EXTRA_ARGS="--cfg.train-start-full"   # t0 = full scene (bare flag: bools have no CFG_ form)
# =================

PRETRAIN_COMMIT=8f780ba
PYTORCH_COMMIT=017ce9b
FOVI_COMMIT=c399d3b

cd /mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train
mkdir -p "logs/$RUN_GROUP/$RUN_NAME/log"
export TASK RUN_GROUP RUN_NAME NGPU EXTRA_ARGS PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* || "$v" == OPT_* ]] && export "$v"; done

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
