"""Validate the ported distill PCA visualization on real data (GPU, offline).

Runs distill through run() with viz_every=1 and asserts that real PNG figures land
LOCALLY under {run_dir}/visualization/pca_train/ — the same content + location as the
historical train/loop.py path (plot_multistep_pca -> save_figure), never uploaded.

Run: HF_HOME=... HF_HUB_OFFLINE=1 .venv-cu126/bin/python unification_docs/harness_run_viz.py
"""

import logging
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import matplotlib
matplotlib.use("Agg")  # headless

import torch

from canvit_pretrain.harness.run import RunSettings, run
from canvit_pretrain.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TrainSpec
from canvit_pretrain.tasks.distill.task import DistillRunTask
from canvit_pretrain.train.config import Config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

IN21K_WDS = Path("/mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/"
                 "webdataset-imagenet-21k-with-features")
IN1K_VAL = Path("/user/henrich1/u25995/jonathan/datasets/imagenet1k-val")
RUN_DIR = Path("/mnt/vast-nhr/projects/nib00021/jonathan/_harness_smoke_ckpts/vizrun")


def main() -> None:
    assert torch.cuda.is_available()
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)

    bs = 8
    cfg = Config(webdataset_dir=IN21K_WDS, val_dir=IN1K_VAL, batch_size_per_gpu=bs,
                 steps_per_job=4096 // bs, num_workers=4, canvas_patch_grid_size=32,
                 tracker="none")
    task = DistillRunTask(cfg)
    spec = TrainSpec(
        train_backbone=True, train_head=False, task_grad_to_backbone=True,
        bptt=BpttSpec(mode="chunked", chunk_size=2, continue_prob=0.5),
        optim={"backbone": GroupOptim(lr=4e-4, weight_decay=1e-4,
               schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=2))},
    )
    run(task=task, spec=spec, settings=RunSettings(
        n_steps=3, device="cuda", log_every=1, run_dir=RUN_DIR, viz_every=1, resume=False))

    viz_dir = RUN_DIR / "visualization" / "pca_train"
    pngs = sorted(viz_dir.glob("step-*.png")) if viz_dir.is_dir() else []
    for p in pngs:
        print(f"  {p.name}: {p.stat().st_size / 1e3:.0f} KB")
    ok = len(pngs) >= 2 and all(p.stat().st_size > 10_000 for p in pngs)
    print(f"figures written: {len(pngs)} in {viz_dir}")
    print("PASS: distill PCA viz renders + saves locally through the harness" if ok
          else "FAIL: no/undersized figures")


if __name__ == "__main__":
    main()
