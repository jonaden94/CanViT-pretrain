"""Verdict for the 2-rank DDP smoke — reads the JSONs harness_ddp_smoke.py wrote.

HARD criterion (decides the exit code): **cross-rank identity**. After training, rank 0
and rank 1 must hold BIT-IDENTICAL parameters. The two ranks train on DIFFERENT data, so
this can only hold if gradients are AllReduced every step and all ranks apply the same
update — i.e. it is exactly the property that makes DDP safe (no silent divergence). Also
hard: the joint (task+policy) losses must stay finite.

INFORMATIONAL (printed, never fails the job): the 6-step DDP-vs-1-GPU param/loss drift.
Exact DDP==1-GPU equivalence does NOT hold for this stack and its failure is not a sync
bug: (1) the distill rollout draws a STOCHASTIC glimpse count that DDP broadcasts from
rank 0, so a 1-GPU run takes a different trajectory by construction; (2) even with a fixed
glimpse count and dropout off, batch-B (per-rank) vs batch-2B (1-GPU) matmuls differ by
fp32 non-associativity — the step-0 loss of the identical initial model over the same
samples already differs ~0.2% — and Adam at lr 1e-2 near random init amplifies that over
6 steps. The chaos-free scale check is ddp_grad_linearity.py (grad(2B) == mean of halves).
"""

import json
import sys
from pathlib import Path

OUT = Path("/mnt/vast-nhr/projects/nib00021/jonathan/_harness_smoke_ckpts/ddp")


def _load(tag: str, rank: int) -> dict:
    return json.loads((OUT / f"{tag}-rank{rank}.json").read_text())


def _max_rel(a: dict, b: dict) -> tuple[float, str]:
    worst, name = 0.0, ""
    for k in a:
        d = abs(a[k] - b[k]) / max(abs(a[k]), abs(b[k]), 1e-8)
        if d > worst:
            worst, name = d, k
    return worst, name


def main() -> int:
    hard: dict[str, bool] = {}
    for mode in ("task_only", "joint"):
        r0, r1 = _load("ddp2", 0)[mode], _load("ddp2", 1)[mode]
        single = _load("single", 0)[mode]

        # HARD: gradients were AllReduced => the two ranks stay bit-identical.
        drift, worst = _max_rel(r0["fingerprint"], r1["fingerprint"])
        hard[f"[{mode}] ranks bit-identical after training (AllReduce works)"] = drift == 0.0
        print(f"[{mode}] cross-rank max rel param diff = {drift:.3e} (worst: {worst})  <- HARD")

        # INFORMATIONAL: the confounded DDP-vs-1-GPU comparison (see module docstring).
        vs, worst_s = _max_rel(r0["fingerprint"], single["fingerprint"])
        lm = max(abs(a - b) / max(abs(a), abs(b), 1e-8)
                 for a, b in zip(r0["losses"], single["losses"], strict=True))
        print(f"[{mode}] vs 1-GPU  max rel param diff = {vs:.3e} (worst: {worst_s})  [info: fp32/stochastic]")
        print(f"[{mode}] losses  ddp ={[round(x, 5) for x in r0['losses']]}")
        print(f"[{mode}] losses  1gpu={[round(x, 5) for x in single['losses']]}")
        print(f"[{mode}] max rel loss diff = {lm:.3e}  [info]")

        if mode == "joint":
            hard["[joint] losses finite"] = all(
                v == v and abs(v) != float("inf") for v in r0["losses"] + r1["losses"])
        print()

    print("=== SUMMARY (hard criteria) ===")
    for k, v in hard.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    allok = all(hard.values())
    print("\nALL PASS" if allok else "\nFAILURES ABOVE")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
