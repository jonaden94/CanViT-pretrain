#!/bin/bash
#SBATCH --job-name=canvit-in1k-sweep
#SBATCH --account=rrg-skrishna_gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/eval-in1k-sweep-%A_%a.out
#SBATCH --error=logs/eval-in1k-sweep-%A_%a.err

# IN1k evaluation sweep: N rollouts of a given policy
#
# Usage:
#   sbatch --array=0-1 slurm/eval_in1k_sweep.sbatch c2f
#   sbatch --array=0-1 slurm/eval_in1k_sweep.sbatch fullrand
#   sbatch --array=0-1 slurm/eval_in1k_sweep.sbatch iid
#
# First argument is policy name: c2f, fullrand, iid

set -eux

NAME="${1:?Usage: sbatch --array=0-N eval_in1k_sweep.sbatch <c2f|fullrand|iid>}"

cd ~/scratch/CanViT-train
source slurm/env.sh
mkdir -p logs outputs

case "$NAME" in
    c2f)
        POLICY="coarse_to_fine"
        EXTRA_ARGS=""
        ;;
    fullrand)
        POLICY="random"
        EXTRA_ARGS="--random-start-full"
        ;;
    iid)
        POLICY="random"
        EXTRA_ARGS="--no-random-start-full"
        ;;
    *)
        echo "Unknown policy name: $NAME (expected c2f, fullrand, iid)"
        exit 1
        ;;
esac

ROLLOUT=$SLURM_ARRAY_TASK_ID
OUTPUT="outputs/in1k_${NAME}_r${ROLLOUT}.pt"

echo "Policy: $NAME ($POLICY), rollout: $ROLLOUT, output: $OUTPUT"
exec uv run python -m canvit_eval.in1k \
    --policy "$POLICY" \
    --output "$OUTPUT" \
    $EXTRA_ARGS
