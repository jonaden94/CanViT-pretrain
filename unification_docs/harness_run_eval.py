"""Validate the run-level ``Task.evaluate`` methods on real data (GPU, offline).

The integration matrix ran with eval_every=0, so the eval paths (ade20k mIoU-per-t,
in1k top-1/5, distill validate()+teacher) were never exercised. This calls each task's
``evaluate()`` once on a truncated real val set and asserts a sane metric — closing the
last unvalidated seam. Distill eval loads the cached DINOv3 teacher (offline).

Run: HF_HOME=... HF_HUB_OFFLINE=1 .venv-cu126/bin/python unification_docs/harness_run_eval.py
"""

import itertools
import logging
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from canvit_train.ade20k.config import Ade20kConfig
from canvit_train.in1k.config import In1kConfig
from canvit_train.tasks.ade20k.task import Ade20kRunTask
from canvit_train.tasks.distill.task import DistillRunTask
from canvit_train.tasks.in1k.task import In1kRunTask
from canvit_train.train.config import Config

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

ADE_ROOT = Path("/user/henrich1/u25995/jonathan/datasets/"
                "zhoubolei--scene_parse_150/ADEChallengeData2016")
IN1K_VAL = Path("/user/henrich1/u25995/jonathan/datasets/imagenet1k-val")
IN21K_WDS = Path("/mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/"
                 "webdataset-imagenet-21k-with-features")
DEV = torch.device("cuda")


def _truncate(loader, k):
    return list(itertools.islice(loader, k))


def eval_ade20k():
    cfg = Ade20kConfig(ade20k_root=ADE_ROOT, scene_size=512, batch_size=8, eval_batch_size=8,
                       num_workers=4, tracker="none", n_timesteps=4)
    t = Ade20kRunTask(cfg)
    seg, _ = t.build_model(DEV)
    _, val = t.build_loaders(world_size=1, rank=0)
    metrics = t.evaluate(model=seg, head=seg.head, val_loader=_truncate(val, 3), device=DEV, step=0)
    print(f"  ade20k eval: {metrics}")
    m = metrics["miou_final"]
    assert 0.0 <= m <= 1.0 and metrics["miou_mean"] >= 0.0, metrics
    return True


def eval_in1k():
    cfg = In1kConfig(val_dir=IN1K_VAL, scene_size=512, eval_batch_size=8, num_workers=4,
                     tracker="none", n_timesteps=4, limit_val_batches=3)
    t = In1kRunTask(cfg)
    clf, _ = t.build_model(DEV)
    # val loader only (skip train wds); make_val_loader directly
    from canvit_train.in1k.data import make_val_loader
    val = make_val_loader(cfg, world_size=1, rank=0)
    metrics = t.evaluate(model=clf, head=clf.head, val_loader=val, device=DEV, step=0)
    print(f"  in1k eval: {metrics}")
    assert 0.0 <= metrics["top1"] <= 1.0 and 0.0 <= metrics["top5"] <= 1.0, metrics
    return True


def eval_distill():
    cfg = Config(webdataset_dir=IN21K_WDS, val_dir=IN1K_VAL, batch_size_per_gpu=8,
                 steps_per_job=512, num_workers=4, canvas_patch_grid_size=32, tracker="none",
                 n_val_samples=32, n_eval_viewpoints=4)
    t = DistillRunTask(cfg)
    model, _ = t.build_model(DEV)
    _, val = t.build_loaders(world_size=1, rank=0)  # inits the normalizer from the first shard
    metrics = t.evaluate(model=model, head=None, val_loader=val, device=DEV, step=0)
    print(f"  distill eval: {metrics}")
    # distill evaluate() is best-effort (returns {} on any error); a populated dict means
    # the teacher + validate() reuse actually worked end-to-end.
    assert metrics != {}, "distill evaluate() fell back to {} — validate()/teacher wiring failed"
    return True


def main():
    assert torch.cuda.is_available()
    print(f"torch={torch.__version__}")
    results = {}
    for name, fn in (("ade20k", eval_ade20k), ("in1k", eval_in1k), ("distill", eval_distill)):
        print(f"### {name} evaluate()")
        try:
            results[name] = fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[name] = False
    print("\n=== EVAL SUMMARY ===")
    for k, v in results.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("ALL PASS" if all(results.values()) else "SOME FAILED")


if __name__ == "__main__":
    main()
