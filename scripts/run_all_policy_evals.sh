#!/usr/bin/env bash
# Evaluate CanViT-B on ADE20K with all policies, both resolutions, multiple runs.
# Produces .pt files for the policy eval pipeline (mIoU vs FLOPs figure + table).
#
# Usage:
#   ADE20K_ROOT=/datasets/ADE20k/ADEChallengeData2016 bash scripts/run_all_policy_evals.sh [output_dir]
#
# Probes are loaded from HuggingFace Hub (--probe-repo).
# Model is loaded from HuggingFace Hub (--model-repo).
# No local paths required — runs anywhere with internet + ADE20K data.

set -euo pipefail

: "${ADE20K_ROOT:?Set ADE20K_ROOT}"
export HF_HUB_DISABLE_PROGRESS_BARS=1

OUT="${1:-results_policy_eval}"
mkdir -p "$OUT"

MODEL="canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"

# (probe_repo, scene_size, canvas_grid, batch_size)
CONFIGS=(
    "canvit/probe-ade20k-40k-s512-c32-in21k:512:32:32"
    "canvit/probe-ade20k-40k-s1024-c64-in21k:1024:64:8"
)

POLICIES="coarse_to_fine fine_to_coarse full_then_random random entropy_coarse_to_fine constant_full_scene"
N_RUNS=5

run_eval() {
    local policy=$1 probe_repo=$2 scene=$3 grid=$4 bs=$5 run=$6
    local tag="${policy}_s${scene}_c${grid}_run${run}"
    local outfile="${OUT}/${tag}.pt"
    if [ -f "$outfile" ]; then
        echo "SKIP $tag"
        return
    fi

    # constant_full_scene is deterministic — skip runs > 0
    if [ "$policy" = "constant_full_scene" ] && [ "$run" != "0" ]; then
        return
    fi

    echo "$(date +%H:%M:%S) RUN $tag"
    uv run python -m canvit_eval.ade20k evaluate \
        --probe-repo "$probe_repo" \
        --model-repo "$MODEL" \
        --policy "$policy" \
        --n-timesteps 21 \
        --scene-size "$scene" \
        --canvas-grid "$grid" \
        --batch-size "$bs" \
        --output "$outfile" \
        2>&1 | tee "${OUT}/${tag}.log"
}

for run in $(seq 0 $((N_RUNS - 1))); do
    for cfg in "${CONFIGS[@]}"; do
        IFS=: read -r probe_repo scene grid bs <<< "$cfg"
        for policy in $POLICIES; do
            run_eval "$policy" "$probe_repo" "$scene" "$grid" "$bs" "$run"
        done
    done
done

echo "$(date +%H:%M:%S) ALL DONE"
echo "Results in $OUT/"
ls -lh "$OUT/"*.pt
