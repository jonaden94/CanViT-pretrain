#!/bin/bash
# exp25 — ImageNet-1k FULL-MODEL finetune through the HARNESS, on the best-scene_cos_norm_t9
# checkpoint of exp22-uniform16-teacherinit-lrdrop2-803k (step-16384, converted to HF).
#
# Recipe = the ORIGINAL canvit_specialize TPU in1k finetune
# (canvit_specialize/training/gcp_in1k_clf_ft/train_imagenet.py), adapted for ONE A100 by the
# recipe's OWN sanctioned rule ("Scale LR linearly with batch size if changed"):
#   batch 256 -> 64 (/4) ; peak_lr 2.5e-5 -> 6.25e-6 (x64/256) ;
#   warmup 25000 -> 100000 (x256/64, holds the ~5-epoch warmup) ;
#   total 20 epochs = 100,080 steps @256 -> 401,408 steps @64 (held at ~20 epochs).
# Everything else BYTE-IDENTICAL to the TPU recipe: wd 1e-4, grad_clip 1.0, label_smoothing 0.1,
#   n_glimpses(=n_timesteps) 4, min_vp_scale 0.05, t0 = full scene (--cfg.train-start-full),
#   glimpse 128 / canvas 32 / scene 512 (derived from the g128px-s512px model), AdamW +
#   linear-warmup->cosine-to-0 (harness reproduces make_lr_lambda).
# NOT identical to the original (batch/lr/steps are batch-adapted) -> comparable UP TO that diff.
# Eval: coarse-to-fine (owner's deliberate choice; the TPU default was random -- training-
#   independent, only affects reported val top-1 / best.pt selection).
# NOTHING IS SUBMITTED by writing this file.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=exp25
RUN_NAME=in1k-uni16ti-803k
ARRAY=0-48%1                 # 49 jobs x 8192 = 401,408 steps (~20 epochs @ batch 64)
TIME=0-12:00:00              # 8192 in1k-ft steps ~= 2-4h (>= distill's 8192@2h, heavier); 12h = wide
                             # margin (a mid-job timeout would break the shard-aligned resume)
MEM=128G
NGPU=1
TASK=in1k

# === config (TPU recipe, batch-adapted) ===
CFG_WANDB_PROJECT=exp25
CFG_RUN_NAME=in1k-uni16ti-803k     # wandb run name = the finetune BASE model (owner request)
CFG_MODEL_REPO=/user/henrich1/u25995/jonathan/repos/CanViT-train/logs/jon_exp22_full_runs/exp22-uniform16-teacherinit-lrdrop2-803k/checkpoints/step-16384-hf
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
OPT_CKPT_DIR=logs/exp25/in1k-uni16ti-803k/checkpoints
OPT_RESUME=True                    # 49-job array must resume across tasks (in1k default is False)
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
    slurm_nhr/harness_train.sbatch
