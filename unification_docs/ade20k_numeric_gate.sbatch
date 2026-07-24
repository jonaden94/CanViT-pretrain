#!/bin/bash
# NUMERIC QUALITY GATE: does the harness ADE20K probe reach the same mIoU as the
# standalone probe it replaces?
#
# The standalone probe (canvit_pretrain/ade20k/train.py) has NO seeding — no cfg.seed,
# no torch.manual_seed — so probe init, augmentation and viewpoint sampling all differ
# run to run. Bit-parity with it is therefore impossible by construction, and a single
# harness-vs-standalone number would be unfalsifiable: any gap could be either a port
# bug or ordinary run-to-run noise.
#
# So this runs FOUR arms under one identical environment:
#   tasks 0,1,2  standalone replicates  -> their spread IS the measured noise floor
#   task  3      harness (python -m canvit_pretrain.harness.run ade20k)
# PASS = the harness curve sits inside the standalone band at every eval point.
#
# Both arms share the same eval primitives (ade20k.metrics.eval_probe_on_batch +
# mIoUAccumulator), so this gates the TRAINING path: spec/optimizer/schedule wiring,
# rollout, loss, and the eval cadence.
#
# Submit: sbatch unification_docs/ade20k_numeric_gate.sbatch
#SBATCH -p grete:shared
#SBATCH -A nib00021
#SBATCH --gpus-per-node=A100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -t 0-06:00:00
#SBATCH --qos=normal
#SBATCH -C inet
#SBATCH -J ade20k-gate
#SBATCH --array=0-3
#SBATCH -o /user/henrich1/u25995/output/ade20k-gate-%A_%a.out

cd /user/henrich1/u25995/jonathan/repos/CanViT-pretrain
# `set -u` must come AFTER .bashrc: /etc/bashrc reads an unset BASHRCSOURCED and
# would abort the job (same ordering as base_train.sbatch).
source /user/henrich1/u25995/.bashrc
set -uo pipefail
source slurm_nhr/env.sh

export WANDB_PROJECT=canvit-pretrain
ADE20K_ROOT=/user/henrich1/u25995/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016
export ADE20K_ROOT
# Overridable so a re-run of one arm can use a fresh root instead of racing with arms
# still writing under the previous root: GATE_OUT=... sbatch --array=3-3 ...
OUT="${GATE_OUT:-/mnt/vast-nhr/projects/nib00021/jonathan/_harness_smoke_ckpts/ade20k_gate}"
VENV=".venv-cu126"

# Short enough to fit the gate in one job (~20 min/arm), long enough for mIoU to
# separate from noise and to give SIX comparison points rather than one.
# max_steps also sets the LR schedule horizon, so every arm anneals identically.
STEPS=6000
VAL_EVERY=1000

ARM="${SLURM_ARRAY_TASK_ID:-3}"
# Each arm wipes ONLY ITS OWN dir (never $OUT — the arms run concurrently and would
# race). Load-bearing for the harness arm: it resumes from `latest.pt` by default, so a
# leftover checkpoint from an earlier gate makes it silently continue that run instead of
# training the schedule under test. The standalone has no resume and cannot show this.
ARM_DIR="$OUT/$([ "$ARM" = 3 ] && echo harness || echo "standalone-$ARM")"
rm -rf "$ARM_DIR"
mkdir -p "$ARM_DIR"

echo "=== ADE20K numeric gate :: arm $ARM :: job ${SLURM_JOB_ID} ==="
echo "node $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

case "$ARM" in
  0|1|2)
    echo "=== ARM $ARM: STANDALONE (python -m canvit_pretrain.ade20k), replicate $ARM ==="
    "$VENV/bin/python" -m canvit_pretrain.ade20k \
        --ade20k-root "$ADE20K_ROOT" \
        --max-steps "$STEPS" --val-every "$VAL_EVERY" \
        --probe-ckpt-dir "$ARM_DIR" \
        --tracker wandb --wandb-project "$WANDB_PROJECT"
    ;;
  3)
    # HARNESS_SEED lets a rerun sample the harness's OWN seed distribution (the harness
    # is deterministically seeded — torch.manual_seed(seed) — so without this every rerun
    # is byte-identical and tells you nothing about run-to-run spread). Unset => seed 0.
    echo "=== ARM 3: HARNESS (python -m canvit_pretrain.harness.run ade20k) seed=${HARNESS_SEED:-0} ==="
    "$VENV/bin/python" -m canvit_pretrain.harness.run ade20k \
        --cfg.ade20k-root "$ADE20K_ROOT" \
        --cfg.max-steps "$STEPS" --cfg.val-every "$VAL_EVERY" \
        --cfg.probe-ckpt-dir "$ARM_DIR" \
        ${HARNESS_SEED:+--opts.seed "$HARNESS_SEED"} \
        --cfg.tracker wandb --cfg.wandb-project "$WANDB_PROJECT"
    ;;
esac
echo "=== arm $ARM exit: $? ==="
