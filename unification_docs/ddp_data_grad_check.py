"""Direct proof that DDP REALLY works on the real distill path (not just "doesn't crash").

Records, per rank, using the REAL DistillRunTask + WebDataset loaders:
  1. the shard filenames this rank was assigned  -> must be DISJOINT across ranks;
  2. a fingerprint of its first training batch      -> must DIFFER across ranks;
  3. the trainable-gradient L2 norm BEFORE AllReduce -> must DIFFER (each rank saw its
     own data), and AFTER AllReduce                  -> must be IDENTICAL (grads synced).

(1)+(2) prove the webdataset sharding gives each rank its own data; (3) proves the
gradients are genuinely per-rank and then correctly averaged — the thing cross-rank
parameter identity implies but does not directly show.

Run: srun --ntasks=2 ... python ddp_data_grad_check.py   (each rank writes a JSON)
     python ddp_data_grad_check.py --compare              (verdict)
"""

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from canvit_train.distill.config import Config
from canvit_train.harness.cli import DistillCmd, resolve_spec
from canvit_train.harness.infra import ddp
from canvit_train.harness.loop import apply_requires_grad
from canvit_train.harness.optim import build_optimizer_and_scheduler
from canvit_train.harness.rollout import run_rollout

OUT = Path("/mnt/vast-nhr/projects/nib00021/jonathan/_harness_smoke_ckpts/ddp_datagrad")
WDS = Path("/mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/"
           "webdataset-imagenet-21k-with-features")
VAL = Path("/user/henrich1/u25995/jonathan/datasets/imagenet1k-val")
VAL_INDEX = Path("/user/henrich1/u25995/jonathan/repos/_data_cache")


def _gnorm(params) -> float:
    sq = [p.grad.detach().double().pow(2).sum() for p in params if p.grad is not None]
    return float(torch.stack(sq).sum().sqrt()) if sq else 0.0


def run() -> None:
    dinfo = ddp.setup(device="cuda", rank=0, world_size=1)
    cmd = DistillCmd(cfg=Config(webdataset_dir=WDS, val_dir=VAL, val_index_dir=VAL_INDEX,
                                batch_size_per_gpu=8, steps_per_job=4096, tracker="none"))
    task, _settings = cmd.build()
    spec = resolve_spec(task, cmd.preset, *cmd.lr_wd())

    model, head = task.build_model(dinfo.device)
    apply_requires_grad(model=model, head=head, joint=None, spec=spec)
    groups = task.trainable_param_groups(model=model, head=head, joint=None, spec=spec)
    opt, _ = build_optimizer_and_scheduler(spec, groups)

    train, _val = task.build_loaders(world_size=dinfo.world_size, rank=dinfo.rank)
    if dinfo.is_dist:  # start identical (params + normalizer buffers) — matches run.py
        ddp.broadcast_parameters(model, None)

    trainable = [p for g in opt.param_groups for p in g["params"]]
    cg = task.canvas_grid(model)
    selector = task.build_selector(device=dinfo.device, canvas_grid=cg,
                                   is_foveated=task.is_foveated(model))

    batch = train.next()
    images = task.batch_images(batch, dinfo.device)
    bound = task.bind(batch, dinfo.device, model=model, head=head)
    opt.zero_grad()
    run_rollout(model=model, images=images, task=bound, selector=selector, bptt=spec.bptt,
                branches=task.branches(), canvas_grid_size=cg, amp_ctx=nullcontext(), joint=None)

    gn_pre = _gnorm(trainable)               # per-rank grad, BEFORE averaging
    if dinfo.is_dist:
        ddp.allreduce_grads(trainable)
    gn_post = _gnorm(trainable)              # AFTER averaging -> should match across ranks

    rec = {
        "rank": dinfo.rank, "world_size": dinfo.world_size,
        "shards": sorted(Path(s).name for s in train.shard_files),
        "img_fp": float(images.detach().double().sum()),
        "gn_pre": gn_pre, "gn_post": gn_post,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"rank{dinfo.rank}.json").write_text(json.dumps(rec))
    print(f"[rank {dinfo.rank}/{dinfo.world_size}] n_shards={len(rec['shards'])} "
          f"img_fp={rec['img_fp']:.4e} gn_pre={gn_pre:.6e} gn_post={gn_post:.6e}", flush=True)
    print(f"[rank {dinfo.rank}] shards={rec['shards']}", flush=True)


def compare() -> int:
    r0 = json.loads((OUT / "rank0.json").read_text())
    r1 = json.loads((OUT / "rank1.json").read_text())
    s0, s1 = set(r0["shards"]), set(r1["shards"])

    checks = {
        "ranks read DISJOINT shards": bool(s0) and bool(s1) and s0.isdisjoint(s1),
        "ranks see DIFFERENT data (batch fingerprint)": r0["img_fp"] != r1["img_fp"],
        "gradients DIFFER before AllReduce": abs(r0["gn_pre"] - r1["gn_pre"]) > 1e-6,
        "gradients IDENTICAL after AllReduce":
            abs(r0["gn_post"] - r1["gn_post"]) <= 1e-5 * max(abs(r0["gn_post"]), 1e-12),
    }
    print(f"rank0 shards: {r0['shards']}")
    print(f"rank1 shards: {r1['shards']}")
    print(f"img_fp   rank0={r0['img_fp']:.6e}  rank1={r1['img_fp']:.6e}")
    print(f"gn_pre   rank0={r0['gn_pre']:.6e}  rank1={r1['gn_pre']:.6e}  (should DIFFER)")
    print(f"gn_post  rank0={r0['gn_post']:.6e}  rank1={r1['gn_post']:.6e}  (should MATCH)")
    print("=== SUMMARY ===")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    sys.exit(compare() if a.compare else (run() or 0))
