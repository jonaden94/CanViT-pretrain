#!/bin/bash
#SBATCH --job-name=signal-test
#SBATCH --time=0:03:00
#SBATCH --signal=TERM@60
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=signal_test_%j.out

# 3-minute job, SIGTERM 60 seconds before walltime
# At 0.1s/step, we do ~10 steps/sec = ~1200 steps in 2 min before SIGTERM
# Target is 1000 steps, should complete in one job (or test signal if it doesn't)

set -euo pipefail

echo "=== Job $SLURM_JOB_ID starting at $(date) ==="
echo "Working directory: $(pwd)"

# Run the test
python3 slurm/test_signal.py

EXIT_CODE=$?
echo "Python exited with code $EXIT_CODE"

# Check if done
if [ -f "signal_test_DONE" ]; then
    echo "=== Training complete! ==="
    exit 0
fi

# Resubmit if exited cleanly
if [ $EXIT_CODE -eq 0 ]; then
    echo "=== Resubmitting... ==="
    sbatch slurm/test_signal.sbatch
else
    echo "=== Job failed, not resubmitting ==="
    exit $EXIT_CODE
fi
