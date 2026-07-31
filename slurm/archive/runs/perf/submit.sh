#!/bin/bash
# Submit the throughput matrix pinned to exact commits.
# Launchers are read at SUBMIT time and are NOT themselves pinned (same as every other
# runs/*.sh here), so edit freely once the job is queued.
set -euo pipefail
cd /user/henrich1/u25995/jonathan/repos/CanViT-train
export PRETRAIN_COMMIT=$(git rev-parse --short HEAD)
export PYTORCH_COMMIT=$(git -C ../CanViT-PyTorch rev-parse --short HEAD)
export FOVI_COMMIT=$(git -C ../fovi rev-parse --short HEAD)
echo "pins: pretrain=$PRETRAIN_COMMIT pytorch=$PYTORCH_COMMIT fovi=$FOVI_COMMIT"
sbatch --export=ALL slurm/archive/runs/perf/throughput_matrix.sh
