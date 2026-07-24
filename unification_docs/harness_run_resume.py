"""Validate run()-level resume on real data (GPU, offline).

Run ade20k probe for 6 steps -> checkpoint; then call run() AGAIN with resume=True on
the same ckpt_dir. It must find_latest -> restore model/opt/sched -> resume_start_step
=> start at step 6 (not 0) and finish at step 9. Also exercises the new operational
path (EMA + grad-norm logging, provenance metadata, signal handler install) end-to-end.

Run: HF_HOME=... HF_HUB_OFFLINE=1 .venv-cu126/bin/python unification_docs/harness_run_resume.py
"""

import logging
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from canvit_pretrain.ade20k.config import Ade20kConfig
from canvit_pretrain.harness.run import RunSettings, run
from canvit_pretrain.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TrainSpec
from canvit_pretrain.tasks.ade20k.task import Ade20kRunTask

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

ADE_ROOT = Path("/user/henrich1/u25995/jonathan/datasets/"
                "zhoubolei--scene_parse_150/ADEChallengeData2016")
CKPT = Path("/mnt/vast-nhr/projects/nib00021/jonathan/_harness_smoke_ckpts/resume")


def _mk():
    cfg = Ade20kConfig(ade20k_root=ADE_ROOT, scene_size=512, batch_size=8, num_workers=4,
                       tracker="none", n_timesteps=4)
    task = Ade20kRunTask(cfg)
    spec = TrainSpec.probe(
        bptt=BpttSpec(mode="none", horizon=4),
        optim={"head": GroupOptim(lr=3e-4, weight_decay=1e-3,
               schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=3, warmup_lr_ratio=1e-2))},
    )
    return task, spec


def main() -> None:
    assert torch.cuda.is_available()
    import shutil
    if CKPT.exists():
        shutil.rmtree(CKPT)

    # First leg: 6 steps from scratch -> step-6.pt
    task, spec = _mk()
    last1 = run(task=task, spec=spec, settings=RunSettings(
        n_steps=6, device="cuda", log_every=2, ckpt_dir=CKPT, resume=True, seed=0))
    print(f"leg1 last step = {last1['step']}  (expect 5)")

    # Second leg: resume=True -> must pick up at step 6 and run 4 more -> last step 9
    task2, spec2 = _mk()
    last2 = run(task=task2, spec=spec2, settings=RunSettings(
        n_steps=4, device="cuda", log_every=1, ckpt_dir=CKPT, resume=True, seed=0))
    print(f"leg2 last step = {last2['step']}  (expect 9 if resumed at 6, 3 if it restarted)")

    ok = last1["step"] == 5 and last2["step"] == 9
    print("PASS: run()-level resume continues from the checkpoint" if ok
          else f"FAIL: leg1={last1['step']} leg2={last2['step']}")


if __name__ == "__main__":
    main()
