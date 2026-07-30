"""mIoU vs glimpse step for every eval policy — the paper's Figure-4B axis.

Curves: the trained policy (one per seed, plotted as mean + min/max band), an UNTRAINED
policy (same architecture, random init — the honest "what does the scorer buy" control),
EG-C2F, C2F, and the safe-box random policy. Paper Table 4 rows are overlaid dashed where
we have a validated counterpart.

EVERYTHING IS COMPUTED IN ONE PROCESS AT ONE EVAL BATCH SIZE, on purpose: absolute ADE20K
mIoU moves by ~0.06 with eval batch size (bf16 kernels differ by batch shape and near-tied
candidate scores then flip glimpses), so curves measured in separate runs are not safely
comparable at the precision this plot shows.

Usage:
  python plot_policy_curves.py --policy-ckpts <a.policy.pt> [<b.policy.pt> ...] \
      [--out band.png] [--batch-size 32] [--limit-batches N]
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from canvit_pretrain.ade20k.config import Ade20kConfig  # noqa: E402
from canvit_pretrain.ade20k.data import make_ade20k_loaders  # noqa: E402
from canvit_pretrain.harness.eval_viewpoints import PAPER_TABLE4_C64  # noqa: E402
from canvit_pretrain.harness.loop import apply_requires_grad  # noqa: E402
from canvit_pretrain.harness.spec import TrainSpec  # noqa: E402
from canvit_pretrain.tasks.ade20k.task import Ade20kRunTask  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--policy-ckpts", nargs="*", default=[], help="harness *.policy.pt files")
p.add_argument("--out", default="band.png")
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--limit-batches", type=int, default=0, help="0 = full val split")
p.add_argument("--probe-repo", default="canvit/probe-ade20k-40k-s512-c64-in21k")
args = p.parse_args()

T, dev = 5, torch.device("cuda")
torch.manual_seed(0)


def _cfg(eval_policy):
    return Ade20kConfig(
        resize_mode="squish", scene_size=512, canvas_grid=64, augment=False, mode="frozen",
        probe_repo=args.probe_repo, n_timesteps=T, eval_policy=eval_policy,
        eval_batch_size=args.batch_size,
        limit_val_batches=args.limit_batches or None)


# One model, one loader, reused by every curve — see the module docstring.
base = Ade20kRunTask(_cfg("coarse_to_fine"))
model, head = base.build_model(dev, prior_model_config=None)
cg = base.canvas_grid(model)
apply_requires_grad(model=model, head=head, joint=None, spec=TrainSpec.probe())
model.eval()
_, val_loader = make_ade20k_loaders(_cfg("coarse_to_fine"))
print(f"eval batch={args.batch_size}  canvas_grid={cg}  "
      f"{'FULL val' if not args.limit_batches else f'{args.limit_batches} batches'}")


def curve(eval_policy, *, joint=None):
    task = Ade20kRunTask(_cfg(eval_policy))
    out = task.evaluate(model=model, head=head, val_loader=val_loader, device=dev, step=0,
                        joint=joint)
    return [out[f"miou_t{t}"] * 100 for t in range(T)]


results: dict[str, list[float]] = {}
for pol in ("entropy_coarse_to_fine", "coarse_to_fine", "random"):
    results[pol] = curve(pol)
    print(f"  {pol:24s} " + " ".join(f"{v:6.2f}" for v in results[pol]))

# --- policy curves: untrained control + one per trained seed --------------------
policy_task = Ade20kRunTask(_cfg("policy"))
joint = policy_task.build_policy(model, device=dev, canvas_grid=cg,
                                 generator=torch.Generator(device=dev).manual_seed(0))
apply_requires_grad(model=model, head=head, joint=joint,
                    spec=TrainSpec.policy_only(freeze_model=True))
model.eval()

results["policy_untrained"] = curve("policy", joint=joint)
print(f"  {'policy (untrained)':24s} " + " ".join(f"{v:6.2f}" for v in results['policy_untrained']))

seed_curves: list[list[float]] = []
for path in args.policy_ckpts:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    missing, _ = joint.scorer.load_state_dict(ck["scorer"], strict=False)
    assert not missing, f"missing scorer keys in {path}: {missing[:5]}"
    joint.scorer.to(dev)
    c = curve("policy", joint=joint)
    seed_curves.append(c)
    print(f"  {'policy ' + Path(path).stem[:16]:24s} " + " ".join(f"{v:6.2f}" for v in c))

# --- plot ----------------------------------------------------------------------
ts = np.arange(T)
fig, ax = plt.subplots(figsize=(9, 6.5))
STYLE = {
    "entropy_coarse_to_fine": ("EG-C2F", "tab:green", "-"),
    "coarse_to_fine":         ("C2F", "tab:blue", "-"),
    "random":                 ("Random (safe-box IID, NOT paper F-IID)", "black", "-"),
}

if seed_curves:
    arr = np.array(seed_curves)
    mean = arr.mean(axis=0)
    ax.fill_between(ts, arr.min(axis=0), arr.max(axis=0), color="crimson", alpha=0.18, lw=0)
    ax.plot(ts, mean, color="crimson", lw=2.8,
            label=f"Viewpoint-Q trained (n={len(seed_curves)}, band = min..max)")
ax.plot(ts, results["policy_untrained"], color="crimson", ls=":", lw=1.8,
        label="Viewpoint-Q untrained (random init)")
for key, (label, colour, ls) in STYLE.items():
    ax.plot(ts, results[key], color=colour, ls=ls, lw=1.8, label=label)
for key, ref in PAPER_TABLE4_C64.items():
    ax.plot(ts, ref, color=STYLE[key][1], ls="--", lw=1.2, alpha=0.75,
            label=f"{STYLE[key][0]} — paper Table 4")

ax.set_xlabel("glimpse step $t$   (t0 = full scene, then 4 chosen glimpses)")
ax.set_ylabel("ADE20K mIoU (%)")
ax.set_title("Viewpoint policies on the paper Fig-4B axis — c64, squish-512, full val\n"
             f"one process, eval batch {args.batch_size} (absolute mIoU shifts ~0.06 with batch size)",
             fontsize=10)
ax.set_xticks(ts)
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower right", title="policy")
fig.tight_layout()
fig.savefig(args.out, dpi=150)

out_json = Path(args.out).with_suffix(".json")
out_json.write_text(json.dumps(
    {"eval_batch_size": args.batch_size, "curves": results, "policy_seeds": seed_curves,
     "paper_table4_c64": PAPER_TABLE4_C64}, indent=2))
print(f"\nwrote {args.out} and {out_json}")
