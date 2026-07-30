"""The DEPLOY (best-checkpoint) numbers of the newest seeds vs the PUBLISHED qband band.

Scope: the 10 jobs submitted 2026-07-30 (15107841-45 = rl_train seeds 5-9,
15107846-50 = harness+fix seeds 5-9), scored on the checkpoint the reference itself
publishes.

WHICH CHECKPOINT, and why it matters. `CanViT-PyTorch-RL/docs/qband_results.md` reports the
band under TWO selections and they are not the same number:

  * "Deploy band — per-seed best-mean(t1-t4)-CE checkpoint" -> the HEADLINE band, and the
    selection behind the 8 published HF policies ("every seed's deploy ckpt (best.pt)",
    landing at steps 4000-8000 depending on seed).
  * "LAST-step (8000) band, for context" -> CE 0.7148/0.6886/0.6751/0.6665, mIoU t4 44.91.
    The doc notes the deploy rule "buys ~0.001 CE over just taking the end".

So the headline band must be compared against OUR best checkpoint, not our terminal one.
Both trainers already select it under the same rule: rl_train picks `best.pt` on
val_ce = mean(t1..t4), and the harness's `best_metric` is `neg_ce_mean` — the same quantity.

Everything is scored in ONE process at ONE eval batch size by the harness eval, which is
bit-identical to the validated eval on published HF, rl_train and harness checkpoints alike
(doc 15 §A2).
"""
import statistics as st
from glob import glob
from pathlib import Path

import torch

from canvit_pretrain.ade20k.config import Ade20kConfig
from canvit_pretrain.ade20k.data import make_ade20k_loaders
from canvit_pretrain.harness.loop import apply_requires_grad
from canvit_pretrain.harness.spec import TrainSpec
from canvit_pretrain.tasks.ade20k.task import Ade20kRunTask

ROOT = Path("/mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-pretrain")
SEEDS = [5, 6, 7, 8, 9]          # the newest batch only

# qband_results.md, "Deploy band — per-seed best-mean(t1-4)-CE checkpoint, mean ± std over 8 seeds"
PUB_CE = {1: (0.7143, 0.0012), 2: (0.6878, 0.0009), 3: (0.6741, 0.0007), 4: (0.6652, 0.0008)}
PUB_MI = {1: (42.65, 0.16), 2: (43.95, 0.16), 3: (44.62, 0.12), 4: (44.97, 0.10)}
PUB_CE_MEAN = (0.6853, 0.0007)


def rl_best(seed: int) -> Path | None:
    for d in sorted(glob(str(ROOT / f"checkpoints/canvit-ade20k-policies/exp27-policy-bneval-s{seed}_qreg_s{seed}_*"))):
        p, last = Path(d) / "best.pt", Path(d) / "last.pt"
        if p.exists() and last.exists():   # same completion guard as the harness side
            if torch.load(last, map_location="cpu", weights_only=False)["step"] >= 8000:
                return p
    return None


def harness_best(seed: int) -> Path | None:
    d = ROOT / f"logs/exp27/exp27-policy-lossfix-s{seed}/checkpoints"
    # COMPLETION GUARD: `best.policy.pt` appears from the very first eval, so a still-running
    # (or walltime-killed) job exposes an early checkpoint that looks perfectly loadable and
    # silently drags the arm's mean down and its variance up. Require the terminal checkpoint
    # as proof the run finished its 8000 steps. This bit me once: a resubmitted seed's step-0
    # best got scored as if it were a completed run.
    return (d / "best.policy.pt") if (d / "step-8000.policy.pt").exists() else None


ARMS = [("rl_train (ported)", rl_best, "net_state"),
        ("harness + fix", harness_best, "scorer")]

T, BS, dev = 5, 32, torch.device("cuda")
torch.manual_seed(0)
cfg = Ade20kConfig(resize_mode="squish", scene_size=512, canvas_grid=64, augment=False,
                   mode="frozen", probe_repo="canvit/probe-ade20k-40k-s512-c64-in21k",
                   n_timesteps=T, eval_policy="policy", eval_batch_size=BS)
base = Ade20kRunTask(cfg)
model, head_mod = base.build_model(dev, prior_model_config=None)
cg = base.canvas_grid(model)
_, val_loader = make_ade20k_loaders(cfg)
joint = base.build_policy(model, device=dev, canvas_grid=cg,
                          generator=torch.Generator(device=dev).manual_seed(0))
apply_requires_grad(model=model, head=head_mod, joint=joint,
                    spec=TrainSpec.policy_only(freeze_model=True))
model.eval()

print(f"DEPLOY (best mean-CE) checkpoints, seeds {SEEDS}, one process, eval batch {BS}, full val\n")
res: dict[str, dict[str, list]] = {}
for arm, finder, key in ARMS:
    res[arm] = {"ce": [], "mi": [], "ce_mean": [], "seeds": []}
    for s in SEEDS:
        p = finder(s)
        if p is None:
            print(f"{arm:20s} s{s}  -- no best ckpt (job pending/failed)")
            continue
        ck = torch.load(p, map_location="cpu", weights_only=False)
        missing, _ = joint.scorer.load_state_dict(ck[key], strict=False)
        assert not missing, f"missing scorer keys in {p}"
        joint.scorer.to(dev)
        out = base.evaluate(model=model, head=head_mod, val_loader=val_loader, device=dev,
                            step=0, joint=joint)
        ce = [out[f"ce_t{t}"] for t in range(1, T)]
        mi = [out[f"miou_t{t}"] * 100 for t in range(1, T)]
        res[arm]["ce"].append(ce); res[arm]["mi"].append(mi)
        res[arm]["ce_mean"].append(out["ce_mean"]); res[arm]["seeds"].append(s)
        print(f"{arm:20s} s{s}  ce " + " ".join(f"{v:.4f}" for v in ce) +
              "   miou " + " ".join(f"{v:5.2f}" for v in mi) +
              f"   mean(t1-4) CE {out['ce_mean']:.4f}")


def ms(vals):
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else float("nan"))


print(f"\n{'=' * 84}\nvs the PUBLISHED qband deploy band (8 seeds, qband_results.md)\n{'=' * 84}")
for arm in res:
    if not res[arm]["seeds"]:
        continue
    n = len(res[arm]["seeds"])
    print(f"\n{arm}  (n={n}, seeds {res[arm]['seeds']})")
    print(f"  {'':10s} {'t1':>16s} {'t2':>16s} {'t3':>16s} {'t4':>16s}")
    for label, data, pub, fmt in (("val_ce", res[arm]["ce"], PUB_CE, "{:.4f}"),
                                  ("val_miou", res[arm]["mi"], PUB_MI, "{:.2f}")):
        ours, pubs, deltas = [], [], []
        for t in range(1, T):
            m, s_ = ms([row[t - 1] for row in data])
            pm, ps = pub[t]
            ours.append(f"{fmt.format(m)}±{fmt.format(s_).lstrip('0')}")
            pubs.append(f"{fmt.format(pm)}±{fmt.format(ps).lstrip('0')}")
            deltas.append(f"{m - pm:+.4f}" if label == "val_ce" else f"{m - pm:+.2f}")
        print(f"  {label + ' ours':10s} " + " ".join(f"{v:>16s}" for v in ours))
        print(f"  {label + ' pub':10s} " + " ".join(f"{v:>16s}" for v in pubs))
        print(f"  {'  delta':10s} " + " ".join(f"{v:>16s}" for v in deltas))
    m, s_ = ms(res[arm]["ce_mean"])
    print(f"  mean(t1-4) CE: ours {m:.4f} ± {s_:.4f}   published {PUB_CE_MEAN[0]:.4f} "
          f"± {PUB_CE_MEAN[1]:.4f}   delta {m - PUB_CE_MEAN[0]:+.4f}")
    lo, hi = PUB_CE_MEAN[0] - PUB_CE_MEAN[1], PUB_CE_MEAN[0] + PUB_CE_MEAN[1]
    inside = sum(lo <= v <= hi for v in res[arm]["ce_mean"])
    print(f"  seeds inside the band's own ±1σ ({lo:.4f}..{hi:.4f}): {inside}/{n}")
