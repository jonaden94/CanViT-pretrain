"""ADE20K viewpoint-policy comparison, t = 0..4 (the paper's Figure-4B axis).

Plots the learned Viewpoint-Q policy against the open-loop baselines on one axis, with
the paper's Table 4 rows as dashed references.

Two stages, so re-styling the plot costs no GPU:

  measure  -> evaluate each baseline on full ADE20K val + read the trained-Q seeds out of
              the exp31 run logs, cache everything to JSON
  plot     -> read the JSON, draw

    python scripts/plot_policy_comparison.py --out readme_docs/ComparisonPoliciesADE20K.png
    python scripts/plot_policy_comparison.py --from-cache ...   # re-plot only

All curves share one frozen model: the published c64 pretrain + ADE20K probe that exp31
trains its scorer against (`Ade20kConfig.model_repo` / `--probe-repo` defaults below),
canvas 64, squish-512, 5 glimpses. Same model, same data, same metric code -- only the
viewpoint policy differs, which is the whole point of the figure.

**The trained-Q curve is read from the exp31 logs, not re-measured.** Each seed is taken
at its early-stop step = the eval with the lowest mean t1..t4 CE, which is the rule the
published qband is defined by (and what `Ade20kRunTask.best_metric` selects `best.pt` on
for a policy run). Re-running the eval would give the same number only if the RNG lined
up; reading the logged eval is exact.

**F-IID is NOT reproducible here** and is drawn from the paper only. Our `random` draws
its scale from the safe-box area law (p(s) ~ 1-s) rather than F-IID's fixed fovea-sized
scale, so it is a different policy plotted under its own name. `PAPER_TABLE4_C64`'s
docstring records an earlier draw landing +0.17..+0.42 pp above the F-IID row; this run
lands slightly BELOW it at t4 (42.91 vs 43.0), so treat "how far our random sits from
F-IID" as run-to-run noise on a stochastic policy, not as a stable offset.
"""

import argparse
import ast
import json
import logging
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXP31 = REPO / "logs/exp31_policy_qreg_10seed"
PROBE_REPO = "canvit/probe-ade20k-40k-s512-c64-in21k"
T = 5

# arXiv:2603.22570 Table 4, canvas 64^2, ADE20K val, squish-512, t = 0..4.
# c2f / eg-c2f are mirrored in eval_viewpoints.PAPER_TABLE4_C64; F-IID lives only here
# because no policy in this codebase reproduces it.
PAPER = {
    "coarse_to_fine":         ([39.6, 41.3, 42.5, 43.6, 44.7], [0.00, 0.10, 0.08, 0.06, 0.03]),
    "entropy_coarse_to_fine": ([39.6, 42.2, 43.3, 44.1, 44.7], None),  # deterministic, no CI
    "f_iid":                  ([39.6, 41.2, 42.0, 42.5, 43.0], [0.00, 0.10, 0.09, 0.10, 0.09]),
}

STYLE = {  # key: (label, colour, linewidth, zorder, linestyle)
    "policy_trained":         ("Viewpoint-Q (trained)", "crimson", 2.6, 5, "-"),
    "policy_untrained":       ("Viewpoint-Q (untrained)", "palevioletred", 1.6, 4, ":"),
    "entropy_coarse_to_fine": ("EG-C2F", "tab:green", 1.8, 3, "-"),
    "coarse_to_fine":         ("C2F", "tab:blue", 1.8, 3, "-"),
    "random":                 ("random (safe-box)", "dimgray", 1.8, 2, "-"),
}
PAPER_LABEL = {"coarse_to_fine": "C2F", "entropy_coarse_to_fine": "EG-C2F", "f_iid": "F-IID"}
PAPER_COLOUR = {"coarse_to_fine": "tab:blue", "entropy_coarse_to_fine": "tab:green",
                "f_iid": "black"}


# ----------------------------------------------------------------- measure
def trained_q_from_logs() -> dict:
    """Per seed: the eval with the lowest ce_mean, i.e. the early-stop checkpoint."""
    line_re = re.compile(r"step (\d+)\s+eval[^:]*:\s*(\{.*\})")
    seeds, curves, steps = [], [], []
    for d in sorted(EXP31.glob("exp31-policy-qreg-s*")):
        evals = []
        for log in sorted(d.glob("log/*.log")):
            for m in line_re.finditer(log.read_text(errors="replace")):
                evals.append((int(m.group(1)), ast.literal_eval(m.group(2))))
        if not evals:
            continue
        step, best = min(evals, key=lambda kv: kv[1]["ce_mean"])
        seeds.append(d.name)
        steps.append(step)
        curves.append([best[f"miou_t{t}"] * 100 for t in range(T)])
    if not curves:
        raise SystemExit(f"no exp31 evals found under {EXP31}")
    return {"seeds": seeds, "early_stop_steps": steps, "curves": curves}


def measure_baselines(eval_batch_size: int, num_workers: int, n_untrained_seeds: int,
                      limit_val_batches: int | None) -> dict:
    import torch

    from canvit_train.ade20k.config import Ade20kConfig
    from canvit_train.ade20k.task import Ade20kRunTask
    from canvit_train.harness.config import JointPolicyConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out: dict[str, list] = {}

    def _cfg(policy):
        return Ade20kConfig(
            probe_repo=PROBE_REPO, canvas_grid=64, n_timesteps=T, resize_mode="squish",
            eval_policy=policy, eval_batch_size=eval_batch_size, num_workers=num_workers,
            limit_val_batches=limit_val_batches, tracker="none",
        )

    # One model + one val loader, reused for every policy: identical weights and identical
    # image order, so differences between curves are the policy and nothing else.
    task0 = Ade20kRunTask(_cfg("coarse_to_fine"), rl=JointPolicyConfig())
    model, _ = task0.build_model(device)
    model.eval()
    _, val_loader = task0.build_loaders(world_size=1, rank=0)

    for policy in ("coarse_to_fine", "entropy_coarse_to_fine", "random"):
        task = Ade20kRunTask(_cfg(policy), rl=JointPolicyConfig())
        m = task.evaluate(model=model, head=model.head, val_loader=val_loader,
                          device=device, step=0)
        out[policy] = [m[f"miou_t{t}"] * 100 for t in range(T)]
        logging.warning("%-24s %s", policy, np.round(out[policy], 2))

    # Untrained Viewpoint-Q: the SAME deploy path as the trained policy (argmax over the
    # scorer), with the scorer left at its random initialisation. Several seeds, because a
    # random net's argmax is an arbitrary fixed trajectory that varies with the draw.
    untrained = []
    for seed in range(n_untrained_seeds):
        task = Ade20kRunTask(_cfg("policy"), rl=JointPolicyConfig())
        gen = torch.Generator(device=device).manual_seed(seed)
        joint = task.build_policy(model, device=device, canvas_grid=64, generator=gen)
        m = task.evaluate(model=model, head=model.head, val_loader=val_loader,
                          device=device, step=0, joint=joint)
        untrained.append([m[f"miou_t{t}"] * 100 for t in range(T)])
        logging.warning("policy(untrained) s%d      %s", seed, np.round(untrained[-1], 2))
    out["policy_untrained"] = untrained
    return out


# -------------------------------------------------------------------- plot
def _band(ax, curves, label, colour, lw, z, ls="-"):
    """Mean over seeds with a 95% CI ribbon (normal approx on the seed SEM)."""
    a = np.asarray(curves, dtype=float)
    mean = a.mean(0)
    ax.plot(range(T), mean, color=colour, lw=lw, zorder=z, ls=ls,
            label=f"{label}, n={len(a)}" if len(a) > 1 else label)
    if len(a) > 1:
        ci = 1.96 * a.std(0, ddof=1) / np.sqrt(len(a))
        ax.fill_between(range(T), mean - ci, mean + ci, color=colour, alpha=0.18,
                        lw=0, zorder=z - 1)
    return mean


def make_plot(data: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 6.4))

    for key in ("policy_trained", "policy_untrained", "entropy_coarse_to_fine",
                "coarse_to_fine", "random"):
        if key not in data:
            continue
        label, colour, lw, z, ls = STYLE[key]
        curves = data[key]
        if not isinstance(curves[0], list):     # single measured curve
            curves = [curves]
        _band(ax, curves, label, colour, lw, z, ls)

    for key, (mean, ci) in PAPER.items():
        colour, name = PAPER_COLOUR[key], PAPER_LABEL[key]
        ax.plot(range(T), mean, ls="--", lw=1.4, color=colour, alpha=0.85, zorder=1,
                label=f"{name} (paper)")
        if ci:
            lo = np.asarray(mean) - np.asarray(ci)
            hi = np.asarray(mean) + np.asarray(ci)
            ax.fill_between(range(T), lo, hi, color=colour, alpha=0.10, lw=0, zorder=0)

    ax.set_xlabel("glimpse step $t$", fontsize=12)
    ax.set_ylabel("ADE20K mIoU (%)", fontsize=12)
    ax.set_title("Viewpoint policies on ADE20K — canvas $64^2$, squish-512, full val (2000 images)\n"
                 "solid = measured on this code base   ·   dashed = paper Table 4",
                 fontsize=11.5)
    ax.set_xticks(range(T))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="lower right", ncol=2, framealpha=0.93,
              title="policy", title_fontsize=9)
    # The two things a reader would otherwise get wrong about this figure.
    ax.text(0.015, 0.975,
            "Viewpoint-Q: each seed at its early-stop step (lowest mean t1–t4 val CE);\n"
            "band = 95% CI over seeds.  F-IID is paper-only: our random draws its scale\n"
            "from the safe-box area law, so it is a DIFFERENT policy, not a reproduction.",
            transform=ax.transAxes, fontsize=7.6, va="top", ha="left", color="0.25",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8", alpha=0.9))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path,
                   default=REPO / "readme_docs/ComparisonPoliciesADE20K.png")
    p.add_argument("--cache", type=Path,
                   default=REPO / "readme_docs/_policy_comparison_data.json")
    p.add_argument("--from-cache", action="store_true",
                   help="skip all evaluation, re-plot from --cache")
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--untrained-seeds", type=int, default=3)
    p.add_argument("--limit-val-batches", type=int, default=None, help="smoke-test knob")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if args.from_cache:
        data = json.loads(args.cache.read_text())
    else:
        trained = trained_q_from_logs()
        data = {"policy_trained": trained["curves"],
                "_trained_seeds": trained["seeds"],
                "_trained_early_stop_steps": trained["early_stop_steps"]}
        data.update(measure_baselines(args.eval_batch_size, args.num_workers,
                                      args.untrained_seeds, args.limit_val_batches))
        data["_limit_val_batches"] = args.limit_val_batches
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps(data, indent=2))
        print(f"wrote {args.cache}")

    make_plot(data, args.out)


if __name__ == "__main__":
    main()
