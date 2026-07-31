#!/bin/bash
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0-00:50:00
#SBATCH -A nib00021
#SBATCH --mem=64G
#SBATCH --array=0-5
#SBATCH -o /user/henrich1/u25995/jonathan/repos/CanViT-train/logs/in1k_gate/gate-%A_%a.log
#SBATCH -e /user/henrich1/u25995/jonathan/repos/CanViT-train/logs/in1k_gate/gate-%A_%a.log
#SBATCH --job-name=in1k-gate

# IN1k top-1 numeric gate: does the harness in1k task reproduce the standalone probe?
# Both are SEEDED identically (torch.manual_seed(seed+rank)) and share the loaders
# (make_train_loader / make_val_loader) AND the eval fn (in1k/train.evaluate). Run-to-run
# noise is large at intermediate steps, so (like the ade20k gate) compare BANDS: n=3 each.
#   arms 0,1,2: standalone (python -m canvit_train.in1k)        seeds 0,1,2
#   arms 3,4,5: harness    (python -m canvit_train.harness.run) seeds 0,1,2
# PASS = harness band overlaps standalone band at each overlapping eval step. tracker=none:
# top-1 is read from stdout ("Step N: top1=..." / "step N  eval: {'top1': ...}").
#
# Capped protocol (a full ~200k-step probe is ~60 GPU-h; this is a code-path gate, not an
# accuracy run): max_steps=1024, warmup_steps=50, val_every=256, T=10, batch=64, 25 val
# batches. warmup_steps MUST be « max_steps here (the config default 10k is for full runs).
# Commit-pin for a frozen snapshot (edit-safe while it runs): export PRETRAIN_COMMIT /
# PYTORCH_COMMIT / FOVI_COMMIT before sbatch.

cd /user/henrich1/u25995/jonathan/repos/CanViT-train
source /user/henrich1/u25995/.bashrc
set -euo pipefail

source slurm/env.sh
mkdir -p logs/in1k_gate

# Commit pinning (offline git archive -> PYTHONPATH, PYTHONSAFEPATH=1 authoritative).
_REPO_BASE=/user/henrich1/u25995/jonathan/repos
_CODE_DIR="$TMPDIR/canvit_src"
if [ -n "${PRETRAIN_COMMIT:-}" ] || [ -n "${PYTORCH_COMMIT:-}" ] || [ -n "${FOVI_COMMIT:-}" ]; then
    mkdir -p "$_CODE_DIR"
    if [ -n "${PRETRAIN_COMMIT:-}" ]; then
        mkdir -p "$_CODE_DIR/CanViT-train"
        git -C "$_REPO_BASE/CanViT-train" archive "$PRETRAIN_COMMIT" | tar -x -C "$_CODE_DIR/CanViT-train"
        export PYTHONPATH="$_CODE_DIR/CanViT-train${PYTHONPATH:+:$PYTHONPATH}"
        echo "Pinned canvit_train -> $PRETRAIN_COMMIT"
    fi
    if [ -n "${PYTORCH_COMMIT:-}" ]; then
        mkdir -p "$_CODE_DIR/CanViT-PyTorch"
        git -C "$_REPO_BASE/CanViT-PyTorch" archive "$PYTORCH_COMMIT" | tar -x -C "$_CODE_DIR/CanViT-PyTorch"
        export PYTHONPATH="$_CODE_DIR/CanViT-PyTorch${PYTHONPATH:+:$PYTHONPATH}"
        echo "Pinned canvit_pytorch -> $PYTORCH_COMMIT"
    fi
    if [ -n "${FOVI_COMMIT:-}" ]; then
        mkdir -p "$_CODE_DIR/fovi"
        git -C "$_REPO_BASE/fovi" archive "$FOVI_COMMIT" | tar -x -C "$_CODE_DIR/fovi"
        export PYTHONPATH="$_CODE_DIR/fovi${PYTHONPATH:+:$PYTHONPATH}"
        echo "Pinned fovi -> $FOVI_COMMIT"
    fi
fi

VENV=.venv-cu126
CKPT=/user/henrich1/u25995/jonathan/repos/CanViT-train/logs/jon_exp22_full_runs/exp22-uniform16/checkpoints/step-1515520-hf
OUT="$TMPDIR/in1k_gate_arm${SLURM_ARRAY_TASK_ID}"   # ckpts: node-local, never read back
mkdir -p "$OUT"

RUN=(env TMPDIR="$TMPDIR" PYTHONSAFEPATH=1 ${PYTHONPATH:+PYTHONPATH="$PYTHONPATH"} "$VENV/bin/python")

echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
case "$SLURM_ARRAY_TASK_ID" in
  0|1|2)
    SEED=$SLURM_ARRAY_TASK_ID
    echo "=== ARM $SLURM_ARRAY_TASK_ID: STANDALONE (python -m canvit_train.in1k) seed=$SEED ==="
    "${RUN[@]}" -m canvit_train.in1k \
        --model-repo "$CKPT" --mode frozen \
        --max-steps 1024 --warmup-steps 50 --val-every 256 --n-timesteps 10 --batch-size 64 \
        --limit-val-batches 25 --num-workers 8 --tracker none \
        --seed "$SEED" --clf-ckpt-dir "$OUT"
    ;;
  3|4|5)
    SEED=$((SLURM_ARRAY_TASK_ID - 3))
    echo "=== ARM $SLURM_ARRAY_TASK_ID: HARNESS (python -m canvit_train.harness.run in1k) seed=$SEED ==="
    "${RUN[@]}" -m canvit_train.harness.run in1k \
        --cfg.model-repo "$CKPT" --cfg.mode frozen \
        --cfg.max-steps 1024 --cfg.warmup-steps 50 --cfg.val-every 256 --cfg.n-timesteps 10 --cfg.batch-size 64 \
        --cfg.limit-val-batches 25 --cfg.num-workers 8 --cfg.tracker none \
        --opts.seed "$SEED" --opts.ckpt-dir "$OUT"
    ;;
esac
echo "=== ARM $SLURM_ARRAY_TASK_ID DONE ==="
