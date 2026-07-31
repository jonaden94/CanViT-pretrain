#!/bin/bash
# End-to-end proof that FOVEATED pretraining is reachable + reproducible through the NEW
# unified CLI, and that the view-scale footgun is closed for harness checkpoints:
#   1. train a few steps of distill with --cfg.model.patcher-name foveated  (was IMPOSSIBLE
#      to express before: the old hand-rolled argparse had no --model.* flags at all)
#   2. convert the resulting harness checkpoint with `python -m canvit_train.checkpoint.to_hf`
#   3. assert config.json records patcher_name=foveated AND a non-None pretrain_view_scale
#      (a None here is the silent OOD footgun that broke run 15025338)
#   4. load the exported dir back as a real model
# Submit: sbatch unification_docs/fovi_cli_roundtrip_a100.sbatch
#SBATCH -p grete:shared
#SBATCH -A nib00021
#SBATCH --gpus-per-node=A100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 0-00:30:00
#SBATCH --qos=normal
#SBATCH -J canvit-fovi-roundtrip
#SBATCH -o /user/henrich1/u25995/output/fovi-roundtrip-%J.out

set -uo pipefail
cd /user/henrich1/u25995/jonathan/repos/CanViT-train

export HF_HOME=/user/henrich1/u25995/.cache/huggingface
export HF_HUB_OFFLINE=1
WDS="/mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/webdataset-imagenet-21k-with-features"
VAL="/user/henrich1/u25995/jonathan/datasets/imagenet1k-val"
VAL_INDEX="/user/henrich1/u25995/jonathan/repos/_data_cache"
LOGS="/mnt/vast-nhr/projects/nib00021/jonathan/_harness_smoke_ckpts/fovi_roundtrip"
VENV=".venv-cu126"
rm -rf "$LOGS"; mkdir -p "$LOGS"

echo "=== node $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

echo "=== [1/4] FOVEATED distill through the new CLI ==="
"$VENV/bin/python" -m canvit_train.harness.run distill --preset default \
    --cfg.model.patcher-name foveated \
    --cfg.foveated-scale.mode fixed --cfg.foveated-scale.fixed-scale 1.0 \
    --cfg.webdataset-dir "$WDS" --cfg.val-dir "$VAL" --cfg.val-index-dir "$VAL_INDEX" \
    --cfg.batch-size-per-gpu 8 --cfg.steps-per-job 4096 \
    --cfg.logs-dir "$LOGS" --cfg.run-group fovi --cfg.run-name roundtrip \
    --cfg.log-every 5 --cfg.no-compile --cfg.tracker none \
    --opts.n-steps 20 --opts.eval-every 0
RC_TRAIN=$?
echo "=== train exit: $RC_TRAIN ==="

CKPT="$LOGS/fovi/roundtrip/checkpoints/latest.pt"
echo "=== [2/4] convert $CKPT ==="
"$VENV/bin/python" -m canvit_train.checkpoint.to_hf --pt-path "$CKPT" --out-dir "$LOGS/hf"
RC_CONV=$?
echo "=== convert exit: $RC_CONV ==="

echo "=== [3/4] + [4/4] assert view-scale recorded, then load back ==="
"$VENV/bin/python" - <<'PY'
import json, os, sys
os.environ["HF_HUB_OFFLINE"] = "1"
d = "/mnt/vast-nhr/projects/nib00021/jonathan/_harness_smoke_ckpts/fovi_roundtrip/hf"
c = json.load(open(f"{d}/config.json"))
vs = c["metadata"]["pretrain_view_scale"]
print("patcher_name       :", c["model_config"].get("patcher_name"))
print("pretrain_view_scale:", vs)
ok = c["model_config"].get("patcher_name") == "foveated" and isinstance(vs, dict) \
    and vs.get("mode") == "fixed" and vs.get("fixed_scale") == 1.0
print("  PASS foveated patcher round-tripped" if c["model_config"].get("patcher_name") == "foveated"
      else "  FAIL patcher not foveated")
print("  PASS view-scale recorded (footgun closed)" if isinstance(vs, dict)
      else "  FAIL view-scale is None -> silent OOD footgun")
from canvit_pytorch import CanViTForPretrainingHFHub
m = CanViTForPretrainingHFHub.from_pretrained(d)
print("  PASS model loads back: %.1fM params" % (sum(p.numel() for p in m.parameters()) / 1e6))
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
PY
RC=$?
echo "=== verdict exit: $RC ==="
exit $RC
