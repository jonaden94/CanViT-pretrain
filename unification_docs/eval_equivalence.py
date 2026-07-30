"""Are the HARNESS eval and the VALIDATED eval the same measurement? Test it on real models.

Earlier claims about this were made from t0 alone, and worse, from numbers measured in
SEPARATE PROCESSES — which is not a comparison. This runs both implementations on TRAINED
checkpoints, in ONE process, over the full ADE20K val split, and reports every timestep:

  A) HARNESS   — `Ade20kRunTask.evaluate` (what a run logs and selects `best.pt` on)
  B) VALIDATED — the `measure_miou_order.py` loop, which reproduces the 8 published qband
                 policies to +0.0002 CE / -0.04 mIoU

Same model instance, same loader, same scorer weights. Any difference is the eval.

Usage: python eval_equivalence.py <ckpt.policy.pt> [more.policy.pt ...]
"""
import argparse
import sys

import torch
from canvit_pytorch.policy import StateEncoder, ViewpointScorer, head_logits

from canvit_pretrain.ade20k.config import Ade20kConfig
from canvit_pretrain.ade20k.data import IGNORE_LABEL, NUM_CLASSES, make_ade20k_loaders
from canvit_pretrain.ade20k.metrics import mIoUAccumulator, preds_from_logits, reward_ce
from canvit_pretrain.ade20k.rl_train import (
    advance_state,
    build_action_table,
    consumes_full_image,
    full_scene_state,
)
from canvit_pretrain.harness.loop import apply_requires_grad
from canvit_pretrain.harness.spec import TrainSpec
from canvit_pretrain.tasks.ade20k.task import Ade20kRunTask

p = argparse.ArgumentParser()
p.add_argument("ckpts", nargs="+", help="harness *.policy.pt files (scorer under 'scorer')")
args = p.parse_args()

T = 5
cfg = Ade20kConfig(resize_mode="squish", scene_size=512, canvas_grid=64, augment=False,
                   mode="frozen", probe_repo="canvit/probe-ade20k-40k-s512-c64-in21k",
                   n_timesteps=T, eval_policy="policy")
dev = torch.device("cuda")
torch.manual_seed(0)
task = Ade20kRunTask(cfg)
model, head = task.build_model(dev, prior_model_config=None)
cg = task.canvas_grid(model)
joint = task.build_policy(model, device=dev, canvas_grid=cg,
                          generator=torch.Generator(device=dev).manual_seed(0))
apply_requires_grad(model=model, head=head, joint=joint,
                    spec=TrainSpec.policy_only(freeze_model=True))
_, val_loader = make_ade20k_loaders(cfg)

# --- the validated loop, with the SAME scorer object the harness uses ------------
vp_flat, _ = build_action_table(model, _RL := __import__(
    "canvit_pretrain.ade20k.rl_train", fromlist=["PolicyTrainConfig"]).PolicyTrainConfig(run_name="x"))
vp_flat = vp_flat.to(dev)
encoder = StateEncoder(model, canvas_grid=cg, feature_groups=task.policy_feature_groups())
assert not consumes_full_image(model), "this check is written for the uniform patcher"
GLIMPSE_PX = 128


def validated_eval(scorer: ViewpointScorer) -> dict:
    ious = [mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, dev) for _ in range(T)]
    ce = torch.zeros(T, dtype=torch.float64)
    n = 0
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    was = scorer.training
    scorer.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(dev), masks.to(dev)
            with amp:
                st = full_scene_state(model, images, canvas_grid=cg, glimpse_px=GLIMPSE_PX)
                logits = head_logits(model, st.canvas, canvas_grid=cg)
            encoder.reset()
            ious[0].update(preds_from_logits(logits, masks.shape[1], masks.shape[2]), masks)
            ce[0] += reward_ce(logits, masks, score_res=None).double().sum().cpu()
            for t in range(1, T):
                f = encoder(st, logits=logits).float()
                idx = scorer(f).reshape(images.shape[0], -1).argmax(dim=1)
                with amp:
                    st = advance_state(model, images, st, vp_flat[idx], GLIMPSE_PX)
                    logits = head_logits(model, st.canvas, canvas_grid=cg)
                ious[t].update(preds_from_logits(logits, masks.shape[1], masks.shape[2]), masks)
                ce[t] += reward_ce(logits, masks, score_res=None).double().sum().cpu()
            n += images.shape[0]
    if was:
        scorer.train()
    out = {f"miou_t{t}": ious[t].compute() for t in range(T)}
    out.update({f"ce_t{t}": (ce[t] / n).item() for t in range(T)})
    out["ce_mean"] = sum(out[f"ce_t{t}"] for t in range(1, T)) / (T - 1)
    return out


bad = 0
for path in args.ckpts:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = joint.scorer.load_state_dict(ck["scorer"], strict=False)
    assert not missing, f"missing scorer keys: {missing[:5]}"
    joint.scorer.to(dev)

    h = task.evaluate(model=model, head=head, val_loader=val_loader, device=dev, step=0,
                      joint=joint)
    v = validated_eval(joint.scorer)

    print(f"\n=== {path.split('/')[-1]}")
    print(f"{'metric':>10} {'HARNESS':>10} {'VALIDATED':>10} {'delta':>9}")
    for k in [f"miou_t{t}" for t in range(T)] + ["ce_mean"]:
        d = h[k] - v[k]
        flag = "" if abs(d) < 1e-4 else "   <-- DIFFERS"
        if abs(d) >= 1e-4:
            bad += 1
        scale = 100 if k.startswith("miou") else 1
        print(f"{k:>10} {h[k] * scale:10.3f} {v[k] * scale:10.3f} {d * scale:+9.4f}{flag}")

print(f"\n{'EQUIVALENT on every metric and checkpoint' if not bad else f'{bad} METRICS DIFFER'}")
sys.exit(1 if bad else 0)
