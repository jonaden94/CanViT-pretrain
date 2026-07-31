#!/bin/bash
# Re-verify the 8-config harness run() integration matrix on a real GPU.
# Submit: sbatch unification_docs/harness_integration_a100.sbatch
#SBATCH -p grete:shared
#SBATCH -A nib00021
#SBATCH --gpus-per-node=A100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 0-00:45:00
#SBATCH --qos=normal
#SBATCH -J canvit-harness-integ
#SBATCH -o /user/henrich1/u25995/output/harness-integ-%J.out

set -uo pipefail
cd /user/henrich1/u25995/jonathan/repos/CanViT-train

export HF_HOME=/user/henrich1/u25995/.cache/huggingface
export HF_HUB_OFFLINE=1

echo "=== node $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
.venv-cu126/bin/python unification_docs/harness_run_integration.py
RC=$?
echo "=== exit code: $RC ==="
exit $RC
