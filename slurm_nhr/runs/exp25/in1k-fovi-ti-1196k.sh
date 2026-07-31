#!/bin/bash
# exp25 — ImageNet-1k FULL-MODEL finetune through the HARNESS, on the best-scene_cos_norm_t9
# checkpoint of exp22-fovi-teacherinit-lrdrop-1196k (step-155648, converted to HF).
#
# FOVEATED source. Recipe = the same TPU in1k finetune, batch-adapted (see in1k-uni16ti-803k.sh),
# plus two foveated-specific settings:
#   1. --cfg.foveated-scale.fixed-scale 2.0 — the rollout MUST view at the pretrain scale.
#      Mode defaults to 'fixed', so EVERY glimpse AND the full-scene t0 are at scale 2.0
#      (in-distribution; selector.py:258-264). This foveated backbone has NO original TPU
#      in1k counterpart -- it's the uniform recipe applied to a foveated model.
#   2. --cfg.eval-policy random — coarse-to-fine is uniform-only / OOD for a fixed-scale
#      foveated backbone (mirrors the exp24 foveated ade20k probe).
# NOTHING IS SUBMITTED by writing this file.
set -euo pipefail

# === ESSENTIALS ===
RUN_GROUP=exp25
RUN_NAME=in1k-fovi-ti-1196k
ARRAY=0-48%1                 # 49 jobs x 8192 = 401,408 steps (~20 epochs @ batch 64)
TIME=0-12:00:00
MEM=128G
NGPU=1
TASK=in1k

# === config (TPU recipe, batch-adapted; foveated) ===
CFG_WANDB_PROJECT=exp25
CFG_RUN_NAME=in1k-fovi-ti-1196k    # wandb run name = the finetune BASE model (owner request)
CFG_MODEL_REPO=/user/henrich1/u25995/jonathan/repos/CanViT-train/logs/jon_exp22_full_runs/exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648-hf
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
CFG_EVAL_POLICY=random             # c2f is uniform-only / OOD for fixed-scale foveated
CFG_VAL_EVERY=10000                # ~half-epoch (1 epoch @ b64 = 20,018 steps); full 50k val each
OPT_CKPT_DIR=logs/exp25/in1k-fovi-ti-1196k/checkpoints
OPT_RESUME=True
EXTRA_ARGS="--cfg.train-start-full --cfg.foveated-scale.fixed-scale 2.0"
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
