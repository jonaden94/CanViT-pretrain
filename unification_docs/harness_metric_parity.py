"""Check the harness logs the SAME metric set as train/loop.py (GPU, offline, no wandb).

The success criterion for the metric port is that a harness distill run produces the
same wandb series as an existing jon_exp22 run. This checks that offline: it drives a
short REAL distill run with a recording tracker (nothing is uploaded) and diffs the
logged key set against the reference set read off train/loop.py 903-931 + 279-282.

Run: HF_HOME=... HF_HUB_OFFLINE=1 .venv-cu126/bin/python unification_docs/harness_metric_parity.py
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

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

IN21K_WDS = Path("/mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/"
                 "webdataset-imagenet-21k-with-features")
IN1K_VAL = Path("/user/henrich1/u25995/jonathan/datasets/imagenet1k-val")
CKPT = Path("/mnt/vast-nhr/projects/nib00021/jonathan/_harness_smoke_ckpts/metrics")
BS, SPJ, N = 32, 128, 12

# train/loop.py 903-931: every EMA series under train/, plus the instantaneous extras.
# Branch series come from BranchMetrics (loop.py 890-901) for each active t0 type.
_BRANCH = ["loss", "scene_patches_loss", "scene_cls_loss",
           "scene_cos_raw", "scene_cos_norm", "cls_cos_raw", "cls_cos_norm"]
REFERENCE = (
    {"train/total_loss", "train/n_glimpses", "train/lr", "train/grad_norm",
     "train/continue_prob", "train/data_pct", "train/gpu_pct"}
    | {f"train/{b}/{m}" for b in ("full", "random") for m in _BRANCH}
)


class RecordingTracker:
    """Stands in for the wandb tracker: records instead of uploading (D-G, and this
    script must stay side-effect free)."""

    def __init__(self):
        self.metric_keys: set[str] = set()
        self.params: dict = {}
        self.samples: dict = {}

    def log_metrics(self, metrics, step=None):
        self.metric_keys |= set(metrics)
        self.samples.update(metrics)

    def log_metric(self, name, value, step=None):
        self.metric_keys.add(name)
        self.samples[name] = value

    def log_parameters(self, params):
        self.params.update(params)

    def end(self):
        pass


def main() -> None:
    assert torch.cuda.is_available()
    shutil.rmtree(CKPT, ignore_errors=True)

    cfg = Config(webdataset_dir=IN21K_WDS, val_dir=IN1K_VAL, batch_size_per_gpu=BS,
                 steps_per_job=SPJ, num_workers=4, canvas_patch_grid_size=32, tracker="none")
    task = DistillRunTask(cfg)
    spec = TrainSpec(
        train_backbone=True, train_head=False, task_grad_to_backbone=True,
        bptt=BpttSpec(mode="chunked", chunk_size=2, continue_prob=0.5),
        optim={"backbone": GroupOptim(lr=1e-5, weight_decay=1e-4, schedule=ScheduleSpec(
            kind="warmup_constant", warmup_steps=4, warmup_lr_ratio=1e-2))},
    )
    rec = RecordingTracker()
    settings = RunSettings(n_steps=N, device="cuda", amp=True, log_every=2, run_dir=CKPT,
                           seed=0, tracker="wandb", log_grad_norms=True, log_timing=True,
                           eval_every=N - 2, viz_every=N - 2,
                           grad_norm_deep_prefixes=("patcher", "patcher.conditioner"))
    # run() builds its tracker via make_tracker when settings.tracker == "wandb"; swap in
    # the recorder so the full logging path runs with NO wandb process and no upload.
    import canvit_train.harness.tracker as T
    real_make = T.make_tracker
    T.make_tracker = lambda **kw: rec
    try:
        run(task=task, spec=spec, settings=settings)
    finally:
        T.make_tracker = real_make

    got = rec.metric_keys
    missing = sorted(REFERENCE - got)
    extra = sorted(k for k in got - REFERENCE if not k.startswith("grad_norm/"))

    print(f"\nlogged {len(got)} metric keys, {len(rec.params)} hyperparameters")
    print(f"  per-module grad norms: {sum(k.startswith('grad_norm/') for k in got)} series")
    print("  sample values: " + ", ".join(
        f"{k}={rec.samples[k]:.4g}" for k in sorted(got)[:4] if isinstance(rec.samples[k], float)))
    if missing:
        print(f"\n  MISSING vs train/loop.py: {missing}")
    if extra:
        print(f"  extra (superset, fine): {extra}")

    ok = {
        "every train/loop.py metric series is logged": not missing,
        "per-branch full/ + random/ series present": all(
            f"train/{b}/{m}" in got for b in ("full", "random") for m in _BRANCH),
        "per-module grad norms logged": any(k.startswith("grad_norm/") for k in got),
        "hyperparameters logged (flattened config + spec + param counts)": (
            {"train_spec", "trainable_params", "total_params"} <= set(rec.params)
            and len(rec.params) > 20),
        "all logged metrics are finite": all(
            torch.isfinite(torch.tensor(float(v))) for v in rec.samples.values()
            if isinstance(v, (int, float))),
        # validation phase: validate() must log its OWN per-timestep series through the
        # real tracker (it used to get a throwaway tracker="none", so all of this was lost)
        "validation logged its per-timestep val/ series": any(
            k.startswith("val/") or k.startswith("val_") for k in got),
        "distill eval returned a metric (not the {} fallback)": "eval/val_metric" in got,
        "training PCA figure written to the run dir": any(
            (CKPT / "visualization").rglob("*.png")) if (CKPT / "visualization").exists() else False,
    }
    print("\n=== SUMMARY ===")
    for k, v in ok.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("\nALL PASS" if all(ok.values()) else "\nFAILURES ABOVE")


if __name__ == "__main__":
    main()
