#!/bin/bash
# Direct proof DDP really works on the real distill path: each rank reads DISJOINT
# webdataset shards (different data), gradients DIFFER before AllReduce and are IDENTICAL
# after. 2x A100, NCCL. Submit: sbatch unification_docs/ddp_data_grad_check.sbatch
#SBATCH -p grete:shared
#SBATCH -A nib00021
#SBATCH --gpus-per-node=A100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 0-00:15:00
#SBATCH --qos=normal
#SBATCH -J canvit-ddp-datagrad
#SBATCH -o /user/henrich1/u25995/output/ddp-datagrad-%J.out

set -uo pipefail
cd /user/henrich1/u25995/jonathan/repos/CanViT-train

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((22000 + (SLURM_JOB_ID % 10000)))
export HF_HOME=/user/henrich1/u25995/.cache/huggingface
export HF_HUB_OFFLINE=1
export NCCL_DEBUG=WARN
# Dataset paths are set in ddp_data_grad_check.py (it builds the Config directly).
VENV=".venv-cu126"
OUT=/mnt/vast-nhr/projects/nib00021/jonathan/_harness_smoke_ckpts/ddp_datagrad
rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== node $(hostname)  gpus: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ' ')"

echo "=== 2-rank run: each rank records shards + batch fingerprint + grad norms ==="
srun --ntasks=2 \
    env WORLD_SIZE=2 MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
    "$VENV/bin/python" unification_docs/ddp_data_grad_check.py

echo "=== verdict ==="
"$VENV/bin/python" unification_docs/ddp_data_grad_check.py --compare
RC=$?
echo "=== compare exit code: $RC ==="
exit $RC
