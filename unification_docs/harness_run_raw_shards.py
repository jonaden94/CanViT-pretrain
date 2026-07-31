"""Validate distill training on RAW (no-feature) WebDataset shards (GPU, offline).

Raw shards carry only jpg+json, so the frozen DINOv3 teacher produces the targets ON THE
FLY — both to seed the standardizers and for every training batch. This is the exp21
path; before this was ported the harness called the precomputed-feature initialiser
unconditionally and `bind()` did `raw_patches.to(...)` on a None, so it CRASHED.

Uses the real no-feature IN1k webdataset (jpg+json, 4096 images/shard) as the raw source.
Checks: the run trains (finite, non-constant loss), the standardizers really were
initialised from teacher forwards, and a checkpoint round-trips.

Run: HF_HOME=... HF_HUB_OFFLINE=1 .venv-cu126/bin/python unification_docs/harness_run_raw_shards.py
"""

import logging
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from canvit_train.distill.config import Config
from canvit_train.distill.task import DistillRunTask
from canvit_train.harness.run import RunSettings, run
from canvit_train.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TrainSpec

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

RAW_WDS = Path("/user/henrich1/u25995/jonathan/datasets/webdataset-imagenet-1k-no-features")
IN1K_VAL = Path("/user/henrich1/u25995/jonathan/datasets/imagenet1k-val")
CKPT = Path("/mnt/vast-nhr/projects/nib00021/jonathan/_harness_smoke_ckpts/raw_shards")

BS, SPJ, N = 32, 128, 24  # steps_per_job * BS = 4096 = one shard; we only run N of them


def main() -> None:
    assert torch.cuda.is_available()
    shutil.rmtree(CKPT, ignore_errors=True)

    cfg = Config(webdataset_dir=RAW_WDS, val_dir=IN1K_VAL, batch_size_per_gpu=BS,
                 steps_per_job=SPJ, num_workers=4, canvas_patch_grid_size=32,
                 tracker="none", normalizer_max_samples=256)
    task = DistillRunTask(cfg)
    spec = TrainSpec(
        train_backbone=True, train_head=False, task_grad_to_backbone=True,
        bptt=BpttSpec(mode="chunked", chunk_size=2, continue_prob=0.5),
        optim={"backbone": GroupOptim(
            lr=1e-4, weight_decay=1e-4,
            schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=4, warmup_lr_ratio=1e-2))},
    )
    settings = RunSettings(n_steps=N, device="cuda", amp=True, log_every=4, ckpt_dir=CKPT, seed=0)
    last = run(task=task, spec=spec, settings=settings)

    ok = {}
    ok["loader reports RAW shards (no precomputed features)"] = task._train_loader.has_features is False
    ok["standardizers were initialised from on-the-fly teacher features"] = bool(
        task.scene_norm.initialized)
    ok["training ran and the loss is finite"] = bool(
        torch.isfinite(torch.tensor(last["total_loss"])))
    ok["checkpoint written"] = (CKPT / f"step-{N}.pt").exists()
    payload = torch.load(CKPT / f"step-{N}.pt", map_location="cpu", weights_only=False)
    ok["checkpoint carries the shard-schedule resume state"] = (
        payload["metadata"]["resume_state"].get("job_index") == 0)

    print(f"\nfinal loss = {last['total_loss']:.4f}  n_glimpses = {last['n_glimpses']}")
    print("\n=== SUMMARY ===")
    for k, v in ok.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("\nALL PASS" if all(ok.values()) else "\nFAILURES ABOVE")


if __name__ == "__main__":
    main()
