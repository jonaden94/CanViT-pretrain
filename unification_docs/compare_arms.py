"""Arm A (`rl_train`, the ported trainer) vs arm B (the unified harness) — every checkpoint
through ONE eval, in ONE process, at ONE batch size. The two arms' scorer key sets are
identical (452/452), so rl_train's `net_state` loads into the harness scorer unchanged.

Answers exactly one question: is the ported trainer really better than the harness, or
were the two numbers the owner compared measured differently?
"""
import json
import statistics as st
from pathlib import Path

import torch

from canvit_pretrain.ade20k.config import Ade20kConfig
from canvit_pretrain.ade20k.data import make_ade20k_loaders
from canvit_pretrain.harness.loop import apply_requires_grad
from canvit_pretrain.harness.spec import TrainSpec
from canvit_pretrain.tasks.ade20k.task import Ade20kRunTask

ROOT = Path("/mnt/vast-nhr/projects/nib00021/jonathan/repos/CanViT-pretrain")
BNEVAL = {
    0: ROOT / "checkpoints/canvit-ade20k-policies/exp27-policy-bneval-s0_qreg_s0_20260729_175712/last.pt",
    1: ROOT / "checkpoints/canvit-ade20k-policies/exp27-policy-bneval-s1_qreg_s1_20260729_175713/last.pt",
}
HARNESS = {
    s: ROOT / f"logs/exp27/exp27-policy-harness-s{s}/checkpoints/step-8000.policy.pt"
    for s in range(5)
}

T, BS, dev = 5, 32, torch.device("cuda")
torch.manual_seed(0)


def _cfg():
    return Ade20kConfig(
        resize_mode="squish", scene_size=512, canvas_grid=64, augment=False, mode="frozen",
        probe_repo="canvit/probe-ade20k-40k-s512-c64-in21k", n_timesteps=T,
        eval_policy="policy", eval_batch_size=BS)


base = Ade20kRunTask(_cfg())
model, head = base.build_model(dev, prior_model_config=None)
cg = base.canvas_grid(model)
_, val_loader = make_ade20k_loaders(_cfg())

joint = base.build_policy(model, device=dev, canvas_grid=cg,
                          generator=torch.Generator(device=dev).manual_seed(0))
apply_requires_grad(model=model, head=head, joint=joint,
                    spec=TrainSpec.policy_only(freeze_model=True))
model.eval()

print(f"one process, eval batch {BS}, full val, squish-512 c64\n")
rows = {}
for arm, table, key in (("rl_train (bneval)", BNEVAL, "net_state"),
                        ("harness", HARNESS, "scorer")):
    for seed, path in table.items():
        ck = torch.load(path, map_location="cpu", weights_only=False)
        missing, _ = joint.scorer.load_state_dict(ck[key], strict=False)
        assert not missing, f"missing scorer keys in {path}: {missing[:5]}"
        joint.scorer.to(dev)
        out = base.evaluate(model=model, head=head, val_loader=val_loader, device=dev,
                            step=0, joint=joint)
        mi = [out[f"miou_t{t}"] * 100 for t in range(T)]
        rows[f"{arm} s{seed}"] = {"miou": mi, "ce_mean": out["ce_mean"],
                                  "ce_t": [out[f"ce_t{t}"] for t in range(T)]}
        print(f"{arm:20s} s{seed}  CE {out['ce_mean']:.4f}  mIoU " +
              " ".join(f"{v:6.2f}" for v in mi))

print()
for arm in ("rl_train (bneval)", "harness"):
    ks = [k for k in rows if k.startswith(arm)]
    t4 = [rows[k]["miou"][4] for k in ks]
    ce = [rows[k]["ce_mean"] for k in ks]
    sd4 = st.stdev(t4) if len(t4) > 1 else float("nan")
    sdce = st.stdev(ce) if len(ce) > 1 else float("nan")
    print(f"{arm:20s} n={len(ks)}  t4 {st.mean(t4):.3f} +- {sd4:.3f}  "
          f"CE {st.mean(ce):.4f} +- {sdce:.4f}   t4 vals {[round(v, 2) for v in t4]}")

Path(__file__).with_name("compare_arms_results.json").write_text(
    json.dumps({"eval_batch_size": BS, "rows": rows}, indent=2))
