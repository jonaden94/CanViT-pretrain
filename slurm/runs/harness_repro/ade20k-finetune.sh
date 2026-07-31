#!/bin/bash
# HARNESS REPRODUCTION — ADE20K FINETUNE (backbone trained end to end, not the frozen probe)
# through the harness, to A/B vs a CanViT-specialize ade20k finetune. The frozen PROBE is
# already gated (docs 10/11 — within ~0.007 mIoU); this exercises the train_backbone path.
#
# TEMPLATE — set two things to match the SPECIFIC specialize finetune you compare against:
#   1. CFG_MAX_STEPS (the finetune length used there),
#   2. the pretrained base via  --cfg.model-repo <hf id | local ...-hf dir>  in EXTRA_ARGS
#      (default is the published in21k CanViT; fine if that's what specialize used too).
# NOTHING IS SUBMITTED by writing this file.
set -euo pipefail

RUN_NAME=ade20k-finetune
TIME=0-08:00:00
MEM=64G
NGPU=1
TASK=ade20k

# ade20k reads its data root from ADE20K_ROOT (Ade20kConfig._default_ade20k_root); it has
# no run_group (harness_train.sbatch requires RUN_GROUP for distill only).
export ADE20K_ROOT=/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016

CFG_WANDB_PROJECT=harness_repro
CFG_MAX_STEPS=40000          # TODO: match the specialize finetune length you A/B against
CFG_VAL_EVERY=2000
# `--preset finetune` trains backbone+head (the DEFAULT preset is the frozen probe). It is a
# TOP-LEVEL CLI flag (not under cfg/opts), so it must go in EXTRA_ARGS.
EXTRA_ARGS="--preset finetune"

PRETRAIN_COMMIT=bc63eee
PYTORCH_COMMIT=017ce9b
FOVI_COMMIT=c399d3b

# Repo root, derived from this script's own location (slurm/runs/<group>/<run>.sh),
# so the run submits from YOUR clone rather than one hardcoded checkout.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
mkdir -p logs/ade20k
export TASK NGPU EXTRA_ARGS ADE20K_ROOT PRETRAIN_COMMIT PYTORCH_COMMIT FOVI_COMMIT
for v in $(compgen -v); do [[ "$v" == CFG_* ]] && export "$v"; done

sbatch \
    --gpus-per-node=A100:$NGPU \
    --ntasks-per-node=$NGPU \
    --mem=$MEM \
    --time=$TIME \
    --output="logs/ade20k/harness-finetune-%j.log" \
    --error="logs/ade20k/harness-finetune-%j.log" \
    --export=ALL \
    slurm/harness_train.sbatch
