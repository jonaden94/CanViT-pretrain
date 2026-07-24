"""End-to-end integration test of the unified ``harness.run`` on real data (GPU).

Drives ``harness/run.py::run(task, spec, settings)`` — the SINGLE orchestration path —
across the full config matrix for all three peer tasks, on the real cached pretrained
CanViT + real datasets, fully offline. This is the run-level gate that the three
``*RunTask`` wrappers + ``run()`` reproduce every legacy trainer AND unlock the new
cross-product (joint task+policy on ade20k/in1k — the point of the unification).

Each config runs a few steps, then we assert: run() returned a finite loss and the
end-of-run checkpoint was written. Configs are independent (per-config try/except) so
one failure doesn't mask the rest; a summary table prints at the end.

Run (offline):
  HF_HOME=... HF_HUB_OFFLINE=1 .venv-cu126/bin/python unification_docs/harness_run_integration.py
"""

import logging
import os
import shutil
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from canvit_pretrain.harness.run import RunSettings, run
from canvit_pretrain.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TrainSpec
from canvit_pretrain.tasks.ade20k.task import Ade20kRunTask
from canvit_pretrain.tasks.distill.task import DistillRunTask
from canvit_pretrain.tasks.in1k.task import In1kRunTask
from canvit_pretrain.ade20k.config import Ade20kConfig
from canvit_pretrain.in1k.config import In1kConfig
from canvit_pretrain.train.config import Config, JointPolicyConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CKPT_BASE = Path("/mnt/vast-nhr/projects/nib00021/jonathan/_harness_smoke_ckpts/integration")
ADE_ROOT = Path("/user/henrich1/u25995/jonathan/datasets/"
                "zhoubolei--scene_parse_150/ADEChallengeData2016")
IN1K_TRAIN = Path("/user/henrich1/u25995/jonathan/datasets/"
                  "webdataset-imagenet-1k-no-features/train-shuffled")
IN21K_WDS = Path("/mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/"
                 "webdataset-imagenet-21k-with-features")
IN1K_VAL = Path("/user/henrich1/u25995/jonathan/datasets/imagenet1k-val")
N, T = 10, 5


def _sched(warmup=3):
    return ScheduleSpec(kind="warmup_constant", warmup_steps=warmup, warmup_lr_ratio=1e-2)


def _go(lr, wd):
    return GroupOptim(lr=lr, weight_decay=wd, schedule=_sched())


# ── config matrix: (label, task_factory, spec_factory) ─────────────────────
def ade_cfg():
    return Ade20kConfig(ade20k_root=ADE_ROOT, scene_size=512, batch_size=8, num_workers=4,
                        tracker="none", n_timesteps=T)


def in1k_cfg(mode="frozen"):
    return In1kConfig(train_dir=IN1K_TRAIN, val_dir=IN1K_VAL, scene_size=512, batch_size=8,
                      num_workers=4, tracker="none", n_timesteps=T, mode=mode)


def distill_cfg(use_rl=False):
    # WebDatasetTrainLoader asserts steps_per_job * batch is a multiple of samples_per_shard
    # (4096); we only consume N steps but the loader's shard schedule needs the alignment.
    bs = 8
    c = Config(webdataset_dir=IN21K_WDS, val_dir=IN1K_VAL, batch_size_per_gpu=bs,
               steps_per_job=4096 // bs, num_workers=4, canvas_patch_grid_size=32, tracker="none")
    c.rl = JointPolicyConfig(use_rl=use_rl)
    return c


def probe_spec():
    return TrainSpec.probe(bptt=BpttSpec(mode="none", horizon=T), optim={"head": _go(3e-4, 1e-3)})


def finetune_spec():
    return TrainSpec.finetune(bptt=BpttSpec(mode="chunked", chunk_size=2, horizon=4),
                              optim={"backbone": _go(1e-5, 1e-4), "head": _go(3e-4, 1e-3)})


def joint_frozen_spec():  # frozen backbone + train head + train policy (ade20k/in1k flagship)
    return TrainSpec(
        train_backbone=False, train_head=True, train_policy=True,
        task_weight=1.0, policy_weight=1.0, task_grad_to_backbone=False, policy_grad_to_backbone=False,
        bptt=BpttSpec(mode="none", horizon=T),
        optim={"head": _go(3e-4, 1e-3), "policy": _go(2e-4, 1e-2)},
    )


def distill_taskonly_spec():  # historical distill: train backbone(+in-forward heads), no policy
    return TrainSpec(
        train_backbone=True, train_head=False, task_grad_to_backbone=True,
        bptt=BpttSpec(mode="chunked", chunk_size=2, continue_prob=0.5),
        optim={"backbone": _go(4e-4, 1e-4)},
    )


def distill_joint_spec():  # train backbone(+in-forward heads) + train policy (P4b)
    return TrainSpec(
        train_backbone=True, train_head=False, train_policy=True,
        task_weight=1.0, policy_weight=1.0, task_grad_to_backbone=True, policy_grad_to_backbone=False,
        bptt=BpttSpec(mode="chunked", chunk_size=2, continue_prob=0.5),
        optim={"backbone": _go(4e-4, 1e-4), "policy": _go(2e-4, 1e-2)},
    )


CONFIGS = [
    ("ade20k-probe",     lambda: Ade20kRunTask(ade_cfg()),                             probe_spec),
    ("ade20k-finetune",  lambda: Ade20kRunTask(ade_cfg()),                             finetune_spec),
    ("ade20k-joint",     lambda: Ade20kRunTask(ade_cfg(), rl=JointPolicyConfig(use_rl=True)), joint_frozen_spec),
    ("in1k-frozen",      lambda: In1kRunTask(in1k_cfg("frozen")),                      probe_spec),
    ("in1k-finetune",    lambda: In1kRunTask(in1k_cfg("finetune")),                    finetune_spec),
    ("in1k-joint",       lambda: In1kRunTask(in1k_cfg("frozen"), rl=JointPolicyConfig(use_rl=True)), joint_frozen_spec),
    ("distill-finetune", lambda: DistillRunTask(distill_cfg(False)),                   distill_taskonly_spec),
    ("distill-joint",    lambda: DistillRunTask(distill_cfg(True)),                    distill_joint_spec),
]


def main() -> None:
    assert torch.cuda.is_available(), "need a GPU"
    print(f"torch={torch.__version__}  device={torch.cuda.get_device_name(0)}")
    # Every config here is a FRESH run (resume has its own scripts). Wipe stale
    # checkpoints so `ckpt_file.exists()` stays meaningful and run()'s resume path
    # doesn't pick up a previous invocation's N-step checkpoint.
    shutil.rmtree(CKPT_BASE, ignore_errors=True)
    results = []
    for label, task_factory, spec_factory in CONFIGS:
        ckpt = CKPT_BASE / label
        print(f"\n{'=' * 70}\n### {label}\n{'=' * 70}", flush=True)
        try:
            task = task_factory()
            spec = spec_factory()
            settings = RunSettings(n_steps=N, device="cuda", amp=True, log_every=4,
                                   ckpt_dir=ckpt, eval_every=0, grad_clip=1.0, seed=0)
            last = run(task=task, spec=spec, settings=settings)
            loss = last["total_loss"]
            ckpt_file = ckpt / f"step-{N}.pt"
            finite = torch.isfinite(torch.tensor(loss)).item()
            saved = ckpt_file.exists()
            ok = bool(finite and saved)
            extra = f" reward_frac={last['reward_frac']:+.4f}" if "reward_frac" in last else ""
            results.append((label, ok, f"loss={loss:.4f} ckpt={saved}{extra}"))
            print(f">>> {label}: {'PASS' if ok else 'FAIL'}  loss={loss:.4f} ckpt_saved={saved}{extra}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append((label, False, f"EXC {type(e).__name__}: {e}"))
            print(f">>> {label}: FAIL ({type(e).__name__}: {e})", flush=True)

    print(f"\n{'=' * 70}\nINTEGRATION SUMMARY\n{'=' * 70}")
    for label, ok, info in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:20s} {info}")
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_pass}/{len(results)} configs PASS")
    print("ALL PASS" if n_pass == len(results) else "SOME FAILED")


if __name__ == "__main__":
    main()
