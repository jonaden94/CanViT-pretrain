#!/bin/bash
# DDP correctness check on the REAL target: 2x A100, NCCL (production transport).
#
# Structured to separate "GPU/transport" from "my harness logic", starting from the
# known-good baseline and adding one divergence at a time:
#   probe 1: plain NCCL all_reduce, NO device_id   -> exactly minimal_distributed_training.py
#   probe 2: plain NCCL all_reduce, WITH device_id -> the one thing train/dist.py adds
#   smoke  : harness_ddp_smoke.py, leg1 (srun -n2) vs leg2 (srun -n1), then compare
# The probes are non-fatal so one job yields the whole matrix even if a leg dies.
#
# grete:shared A100:2 backfills fast for a short job; QOS normal matches the exp arrays.
# Submit: sbatch unification_docs/ddp_smoke_a100.sbatch
#SBATCH -p grete:shared
#SBATCH -A nib00021
#SBATCH --gpus-per-node=A100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-00:15:00
#SBATCH --qos=normal
#SBATCH -J canvit-ddp-a100
#SBATCH -o /user/henrich1/u25995/output/ddp-a100-%J.out

set -uo pipefail   # NOT -e: probes are allowed to fail without aborting the smoke
cd /user/henrich1/u25995/jonathan/repos/CanViT-pretrain

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID % 10000)))
export HF_HOME=/user/henrich1/u25995/.cache/huggingface
export HF_HUB_OFFLINE=1
export NCCL_DEBUG=WARN   # surface transport warnings without INFO's flood
VENV=".venv-cu126"
OUT=/mnt/vast-nhr/projects/nib00021/jonathan/_harness_smoke_ckpts/ddp
rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== node $(hostname)  gpus: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"
"$VENV/bin/python" -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'nccl',torch.cuda.nccl.version(),'avail',torch.cuda.is_available())"

echo "=== probe 1: NCCL all_reduce, NO device_id (known-good baseline) ==="
srun --ntasks=2 env WORLD_SIZE=2 MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
    "$VENV/bin/python" unification_docs/ddp_transport_probe.py \
    || echo "  PROBE 1 FAILED (exit $?)"

echo "=== probe 2: NCCL all_reduce, WITH device_id (train/dist.py's divergence) ==="
srun --ntasks=2 env WORLD_SIZE=2 MASTER_ADDR="$MASTER_ADDR" MASTER_PORT=$((MASTER_PORT+1)) \
    "$VENV/bin/python" unification_docs/ddp_transport_probe.py --device-id \
    || echo "  PROBE 2 FAILED (exit $?)"

echo "=== smoke leg 1: 2-rank DDP (NCCL) ==="
srun --ntasks=2 env WORLD_SIZE=2 MASTER_ADDR="$MASTER_ADDR" MASTER_PORT=$((MASTER_PORT+2)) \
    "$VENV/bin/python" unification_docs/harness_ddp_smoke.py --tag ddp2 \
    || echo "  SMOKE LEG 1 FAILED (exit $?)"

echo "=== smoke leg 2: 1-rank reference (batch 2x) ==="
srun --ntasks=1 env WORLD_SIZE=1 \
    "$VENV/bin/python" unification_docs/harness_ddp_smoke.py --tag single \
    || echo "  SMOKE LEG 2 FAILED (exit $?)"

echo "=== comparing (cross-rank identity = hard; DDP-vs-1GPU = informational) ==="
"$VENV/bin/python" unification_docs/harness_ddp_compare.py
RC_CMP=$?
echo "=== compare exit code: $RC_CMP ==="
if [ "$RC_CMP" -ne 0 ]; then exit 1; fi
echo "=== ALL HARD CHECKS PASSED ==="
