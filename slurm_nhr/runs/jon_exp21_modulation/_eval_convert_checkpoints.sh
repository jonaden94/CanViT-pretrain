#!/bin/bash
# Convert every non-cancelled exp21 run's final checkpoint (latest.pt @ step
# 200,704) to the local HF-Hub layout (checkpoints/latest-hf/) that
# CanViTForPretrainingHFHub.from_pretrained reads for evaluation.
# Idempotent: skips runs whose latest-hf/ already exists. Pure CPU reformatting.
set -uo pipefail

ROOT=/user/henrich1/u25995/jonathan/repos
PY="$ROOT/CanViT-eval/.venv/bin/python"
CONV="$ROOT/CanViT-specialize/scripts/pretrain_ckpt_to_hf_format.py"
L="$ROOT/CanViT-pretrain/logs/jon_exp21_modulation"

RUNS=(
  exp21-uniform8 exp21-uniform16-grid4 exp21-uniform16-repro exp21-uniform16-L
  exp21-uniform-p6-grid5 exp21-uniform-p6-grid6 exp21-uniform-overlap-p16-grid7-stride8
  exp21-cond4-fovi-L exp21-cond4-fovi-film-pos-sigma4-repro exp21-cond4-fovi-film-pos-sinusoidal
  exp21-cond4-fovi-fov140-res81 exp21-cond4-fovi-fov140-res81-force
  exp21-cond4-reg-film-pos-sigma4 exp21-cond4-reg-modulate-trunk-crossattn-fourier
  exp21-cond4-reg-modulate-trunk-fourier exp21-cond4-reg-modulate-trunk-sinusoidal
  exp21-cond4-reg-prune30 exp21-cond4-reg-repro
  exp21-cond4-reg-scale-uniform-glimpse exp21-cond4-reg-scale-uniform-rollout
  exp21-cond4-reg-scale1p41 exp21-strided-gf2 exp21-strided-gf2-keepcorners
)

ok=0; fail=0; skip=0
for r in "${RUNS[@]}"; do
  d="$L/$r/checkpoints"
  if [ -d "$d/latest-hf" ]; then echo "[skip] $r (latest-hf exists)"; skip=$((skip+1)); continue; fi
  if [ ! -f "$d/latest.pt" ]; then echo "[FAIL] $r (no latest.pt)"; fail=$((fail+1)); continue; fi
  if "$PY" "$CONV" --pt-path "$d/latest.pt" --out-dir "$d/latest-hf" >/dev/null 2>"$d/.hf_convert.err"; then
    echo "[ok]   $r"; ok=$((ok+1))
  else
    echo "[FAIL] $r (see $d/.hf_convert.err)"; fail=$((fail+1))
  fi
done
echo "=== converted ok=$ok skip=$skip fail=$fail (of ${#RUNS[@]}) ==="
