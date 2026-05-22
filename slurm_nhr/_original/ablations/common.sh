# Shared ablation config for Grete. Sourced by individual ablation scripts.
#
# Ablations run shorter than the flagship: ~200k steps total.
# Adjust ARRAY and STEPS_PER_JOB together so ARRAY_SIZE × STEPS_PER_JOB ≈ 200k.
ARRAY="0-3%1"           # 4 jobs × 50000 steps = 200k steps
STEPS_PER_JOB=50000
WARMUP=20000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

submit() {
    local name="$1"; shift
    sbatch --array="$ARRAY" -J "$name" "$SCRIPT_DIR/train.sbatch" \
        --warmup-steps "$WARMUP" \
        --run-name "$name" \
        --steps-per-job "$STEPS_PER_JOB" \
        "$@"
}
