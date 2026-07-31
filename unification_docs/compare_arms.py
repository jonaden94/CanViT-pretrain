"""Every exp27 policy arm through ONE eval, in ONE process, at ONE batch size — plus an
exact permutation test between any two arms.

The arms' scorer key sets are identical (452/452), so `rl_train`'s `net_state` loads into
the harness scorer unchanged and all three can be scored by the same code path.

  arm C  `rl_train` + BN mode (b)  — the ported reference
  arm B  harness, BN mode (b)      — BEFORE the policy-loss scale fix (0.8x gradient)
  arm D  harness + the scale fix   — pin bc0b16b

Answers two questions: is the ported trainer really better than the harness (rather than
just measured differently), and does the scale fix close the gap? Statistics are an EXACT
permutation test, because n=5 per arm makes the normal-approximation p-values meaningless —
and because at n=2 vs n=5 the attainable floor was 0.095, which is how the first read of
this comparison ended up inconclusive.
"""
import argparse
import itertools
import json
import statistics as st
from glob import glob
from pathlib import Path

import torch

from canvit_train.ade20k.config import Ade20kConfig
from canvit_train.ade20k.data import make_ade20k_loaders
from canvit_train.ade20k.task import Ade20kRunTask
from canvit_train.harness.loop import apply_requires_grad
from canvit_train.harness.spec import TrainSpec

ROOT = Path("/mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-train")


def _rl_train_ckpts(tag: str, which: str) -> dict[int, Path]:
    """rl_train run dirs carry a timestamp, so glob rather than hard-code."""
    out: dict[int, Path] = {}
    for d in sorted(glob(str(ROOT / f"checkpoints/canvit-ade20k-policies/{tag}-s*_qreg_s*"))):
        seed = int(Path(d).name.split("_qreg_s")[1].split("_")[0])
        p = Path(d) / which
        if p.exists():
            out[seed] = p
    return out


def _harness_ckpts(tag: str, which: str) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for s in range(12):   # seeds 0-9 exist; headroom so a new batch is never silently dropped
        p = ROOT / f"logs/exp27/{tag}-s{s}/checkpoints/{which}"
        if p.exists():
            out[s] = p
    return out


# TWO checkpoint selections, because the band is published under BOTH and they are NOT the same
# number (qband_results.md):
#
#   TERMINAL  step 8000. Band: mean(t1-4) CE 0.6863, mIoU t4 44.91 ("LAST-step (8000) band").
#   DEPLOY    the per-seed best-mean(t1-4)-CE checkpoint — what the doc calls the "deploy band"
#             and the selection the 8 PUBLISHED HF policies use ("every seed's deploy ckpt
#             (best.pt)", chosen at steps 4000-8000). Band: CE 0.6853 +- 0.0007, t4 44.97 +- .10.
#             The doc: "the deploy rule's early selection buys ~0.001 CE over just taking the end".
#
# Both arms already save exactly this checkpoint under the same rule — rl_train selects `best.pt`
# on val_ce = mean(t1..t4), and the harness's best_metric is `neg_ce_mean`. Comparing our
# terminal numbers against the HEADLINE (deploy) band would be a category error in the
# unfavourable direction, and comparing deploy-vs-deploy across arms is fine as long as BOTH
# sides are deploy — what must never be mixed is one arm's best against another's last.
SELECTIONS = {
    "TERMINAL (step 8000)": {
        "band": "CE 0.6863   mIoU t4 44.91",
        "arms": [("armC rl_train", _rl_train_ckpts("exp27-policy-bneval", "last.pt"), "net_state"),
                 ("armB harness", _harness_ckpts("exp27-policy-harness", "step-8000.policy.pt"), "scorer"),
                 ("armD harness+fix", _harness_ckpts("exp27-policy-lossfix", "step-8000.policy.pt"), "scorer")],
    },
    "DEPLOY (best mean-CE ckpt)": {
        "band": "CE 0.6853 +- 0.0007   mIoU t4 44.97 +- 0.10   <- the PUBLISHED band",
        "arms": [("armC rl_train", _rl_train_ckpts("exp27-policy-bneval", "best.pt"), "net_state"),
                 ("armB harness", _harness_ckpts("exp27-policy-harness", "best.policy.pt"), "scorer"),
                 ("armD harness+fix", _harness_ckpts("exp27-policy-lossfix", "best.policy.pt"), "scorer")],
    },
}

ap = argparse.ArgumentParser()
ap.add_argument("--arms", default="", help="comma-separated substrings; only matching arms are "
                                          "scored (e.g. 'armC,armD' to skip the superseded armB)")
cli = ap.parse_args()
_want = [a.strip() for a in cli.arms.split(",") if a.strip()]

T, BS, dev = 5, 32, torch.device("cuda")
torch.manual_seed(0)


def _cfg():
    return Ade20kConfig(
        resize_mode="squish", scene_size=512, canvas_grid=64, augment=False, mode="frozen",
        probe_repo="canvit/probe-ade20k-40k-s512-c64-in21k", n_timesteps=T,
        eval_policy="policy", eval_batch_size=BS)


base = Ade20kRunTask(_cfg())
model, head_mod = base.build_model(dev, prior_model_config=None)
cg = base.canvas_grid(model)
_, val_loader = make_ade20k_loaders(_cfg())

joint = base.build_policy(model, device=dev, canvas_grid=cg,
                          generator=torch.Generator(device=dev).manual_seed(0))
apply_requires_grad(model=model, head=head_mod, joint=joint,
                    spec=TrainSpec.policy_only(freeze_model=True))
model.eval()

def perm_p(a: list[float], b: list[float], *, higher_is_worse: bool) -> tuple[float, float, int]:
    """EXACT one-sided permutation test on the difference of means (b - a).

    Exact, not a t-test: at these n the normal approximation is not credible, and the
    ATTAINABLE floor matters — with n=2 vs n=5 it is 2/21 = 0.095, so no result there could
    ever have reached 0.05 however clean the separation looked.
    """
    pool = a + b
    obs = st.mean(b) - st.mean(a)
    cnt = tot = 0
    for c in itertools.combinations(range(len(pool)), len(a)):
        g1 = [pool[i] for i in c]
        g2 = [pool[i] for i in range(len(pool)) if i not in c]
        d = st.mean(g2) - st.mean(g1)
        tot += 1
        cnt += (d >= obs - 1e-12) if higher_is_worse else (d <= obs + 1e-12)
    return obs, cnt / tot, tot


def perm_var(a: list[float], b: list[float]) -> tuple[float, float]:
    """Exact permutation test on the VARIANCE ratio var(a)/var(b).

    Worth testing separately from the mean: rl_train's t4 spread came out much tighter than the
    harness's, which is what first suggested a real difference rather than noise. Part of the
    spread is now known to be GPU nondeterminism rather than seed (doc 15 §A5.8), so a variance
    difference has no obvious mechanism — hence measure, do not assume.
    """
    pool = a + b
    obs = st.variance(a) / st.variance(b)
    cnt = tot = 0
    for c in itertools.combinations(range(len(pool)), len(a)):
        g1 = [pool[i] for i in c]
        g2 = [pool[i] for i in range(len(pool)) if i not in c]
        tot += 1
        cnt += (st.variance(g1) / st.variance(g2)) <= obs + 1e-12
    return obs, cnt / tot


print(f"one process, eval batch {BS}, full val, squish-512 c64")
all_rows: dict[str, dict] = {}
for sel_name, sel in SELECTIONS.items():
    print(f"\n{'=' * 78}\n{sel_name}   band: {sel['band']}\n{'=' * 78}")
    per_arm: dict[str, dict[str, list[float]]] = {}
    for arm, table, key in sel["arms"]:
        if _want and not any(w in arm for w in _want):
            continue
        if not table:
            print(f"{arm:18s} -- no checkpoints, skipping")
            continue
        per_arm[arm] = {"ce": [], "t4": []}
        for seed, path in sorted(table.items()):
            ck = torch.load(path, map_location="cpu", weights_only=False)
            missing, _ = joint.scorer.load_state_dict(ck[key], strict=False)
            assert not missing, f"missing scorer keys in {path}: {missing[:5]}"
            joint.scorer.to(dev)
            out = base.evaluate(model=model, head=head_mod, val_loader=val_loader, device=dev,
                                step=0, joint=joint)
            mi = [out[f"miou_t{t}"] * 100 for t in range(T)]
            all_rows[f"{sel_name} | {arm} s{seed}"] = {
                "miou": mi, "ce_mean": out["ce_mean"], "ckpt": str(path)}
            per_arm[arm]["ce"].append(out["ce_mean"])
            per_arm[arm]["t4"].append(mi[4])
            print(f"{arm:18s} s{seed}  CE {out['ce_mean']:.4f}  mIoU " +
                  " ".join(f"{v:6.2f}" for v in mi))
    print()
    for arm, v in per_arm.items():
        n = len(v["ce"])
        sce = st.stdev(v["ce"]) if n > 1 else float("nan")
        st4 = st.stdev(v["t4"]) if n > 1 else float("nan")
        print(f"{arm:18s} n={n:2d}  CE {st.mean(v['ce']):.4f} +- {sce:.4f}  "
              f"t4 {st.mean(v['t4']):.3f} +- {st4:.3f}")
    print()
    names = list(per_arm)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = names[i], names[j]
            if min(len(per_arm[x]["ce"]), len(per_arm[y]["ce"])) < 2:
                continue
            dce, pce, tot = perm_p(per_arm[x]["ce"], per_arm[y]["ce"], higher_is_worse=True)
            dt4, pt4, _ = perm_p(per_arm[x]["t4"], per_arm[y]["t4"], higher_is_worse=False)
            r, pv = perm_var(per_arm[x]["t4"], per_arm[y]["t4"])
            print(f"{y} vs {x}:  dCE={dce:+.4f} p={pce:.4f}   dt4={dt4:+.3f} p={pt4:.4f}"
                  f"   ({tot} splits, floor {1 / tot:.5f})")
            print(f"{' ' * (len(y) + 8)}t4 variance ratio {r:.3f}  exact p={pv:.4f}")

Path(__file__).with_name("compare_arms_results.json").write_text(
    json.dumps({"eval_batch_size": BS, "rows": all_rows}, indent=2))
