#!/bin/bash
# Re-evaluate all four finished exp30 ADE20K probes under COARSE-TO-FINE.
#
# exp30 trained and validated under `random` viewpoints (ADE20K's historical default,
# inherited from the specialize probe, which trained on random views). This answers the
# separate question "what do these models score under the C2F deploy convention?".
#
# n_timesteps=21 = canvit_eval's EpisodeConfig default for policy="coarse_to_fine", so
# these are the numbers CanViT-eval would report.
#
# The two FOVEATED arms pass --override-scale 2.0: C2F's quadtree scales {1.0, 0.5, 0.25}
# are ones a fixed-scale foveated backbone never trained on (`fix_size = scale * H`), so
# unpinned every glimpse is out of distribution. Pinning keeps C2F's CENTERS and fixes the
# scale at the pretraining value -- canvit_eval's `override_scale` semantics.
#
# Usage: bash scripts/eval_exp30_c2f.sh [output_dir]
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT=${1:-logs/exp30_ade20k_probe/_c2f_eval}
mkdir -p "$OUT"

export ADE20K_ROOT=${ADE20K_ROOT:-/mnt/vast-nhr/projects/nib00021/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016}
export HF_HUB_OFFLINE=1
PY=.venv-cu126/bin/python
SRC="$PWD/logs/jon_exp22_full_runs"

# run_name : pretrained backbone the probe was trained on : override-scale ("" = uniform)
RUNS=(
  "ade20k-uni16ti-803k:exp22-uniform16-teacherinit-lrdrop2-803k/checkpoints/step-16384-hf:"
  "ade20k-uni16-1516k:exp22-uniform16-lrdrop-1516k/checkpoints/step-319488-hf:"
  "ade20k-fovi-ti-1196k:exp22-fovi-teacherinit-lrdrop-1196k/checkpoints/step-155648-hf:2.0"
  "ade20k-fovi-1901k:exp22-fovi/checkpoints/step-1900544-hf:2.0"
)

for entry in "${RUNS[@]}"; do
    IFS=: read -r run repo scale <<<"$entry"
    echo "=== $run ${scale:+(foveated, scale pinned to $scale)} ==="
    extra=()
    if [ -n "$scale" ]; then extra=(--override-scale "$scale" --fixed-scale "$scale"); fi
    $PY scripts/eval_ade20k_checkpoint.py \
        --ckpt "logs/exp30_ade20k_probe/$run/checkpoints/best.pt" \
        --model-repo "$SRC/$repo" \
        --eval-policy coarse_to_fine --n-timesteps 21 \
        --eval-batch-size 16 --num-workers 8 \
        "${extra[@]}" \
        --out "$OUT/$run.json"
done

echo "=== done -> $OUT ==="
