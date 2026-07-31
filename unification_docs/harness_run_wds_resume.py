"""Validate distill's WebDataset multi-job resume on the REAL shard schedule (GPU, offline).

This is the production SLURM-array pattern: each array task runs exactly
``steps_per_job`` steps over its own shard slice, checkpoints, exits; the next task
resumes and must read the NEXT slice. Getting the offset wrong silently re-processes or
skips training data, which is why the harness raises instead of guessing.

Two legs through ``run()`` on the real IN21k feature-webdataset, plus four refusal cases:

  leg 1  fresh          -> job_index 0, steps 0..SPJ, checkpoint carries resume_state
  leg 2  resume         -> job_index 1, steps SPJ..2*SPJ, and a DISJOINT shard list
  refuse mid-job save   -> a checkpoint whose scheduler disagrees with the job boundary
  refuse changed batch / steps_per_job / world_size (shard-schedule inputs)

Run: HF_HOME=... HF_HUB_OFFLINE=1 .venv-cu126/bin/python unification_docs/harness_run_wds_resume.py
"""

import logging
import os
import shutil
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from canvit_train.harness.checkpoint import find_latest, load_checkpoint
from canvit_train.harness.run import RunSettings, run
from canvit_train.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TrainSpec
from canvit_train.tasks.distill.task import DistillRunTask
from canvit_train.train.config import Config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

IN21K_WDS = Path("/mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/"
                 "webdataset-imagenet-21k-with-features")
IN1K_VAL = Path("/user/henrich1/u25995/jonathan/datasets/imagenet1k-val")
CKPT = Path("/mnt/vast-nhr/projects/nib00021/jonathan/_harness_smoke_ckpts/wds_resume")

# samples_per_shard is 4096; steps_per_job * batch must be a multiple of it (one shard
# per GPU per job here) — and each leg must run EXACTLY steps_per_job steps.
BS, SPJ = 32, 128


def _cfg(**over):
    return replace(Config(webdataset_dir=IN21K_WDS, val_dir=IN1K_VAL, batch_size_per_gpu=BS,
                          steps_per_job=SPJ, num_workers=4, canvas_patch_grid_size=32,
                          tracker="none"), **over)


def _spec():
    return TrainSpec(
        train_backbone=True, train_head=False, task_grad_to_backbone=True,
        bptt=BpttSpec(mode="chunked", chunk_size=2, continue_prob=0.5),
        optim={"backbone": GroupOptim(
            lr=1e-5, weight_decay=1e-4,
            schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=8, warmup_lr_ratio=1e-2))},
    )


def _settings(**over):
    return RunSettings(n_steps=SPJ, device="cuda", amp=True, log_every=32, ckpt_dir=CKPT,
                       resume=True, seed=0, **over)


def _saved_resume_state():
    payload = load_checkpoint(find_latest(CKPT), "cpu")
    return payload["metadata"]["resume_state"], payload


def _refuses(what: str, task, payload, scheduler_step: int) -> bool:
    """A leg-2 task must REFUSE this checkpoint rather than resume onto a wrong slice."""
    try:
        task.resume_start_step(payload, type("S", (), {"last_epoch": scheduler_step})())
        task.build_loaders(world_size=1, rank=0)
    except RuntimeError as e:
        print(f"  refused {what}: {str(e)[:110]}...")
        return True
    print(f"  !! ACCEPTED {what} — the schedule would be silently corrupted")
    return False


def main() -> None:
    assert torch.cuda.is_available()
    shutil.rmtree(CKPT, ignore_errors=True)
    ok = {}

    # --- leg 1: fresh job 0 -------------------------------------------------
    print("\n=== leg 1: fresh (job_index 0) ===")
    t1 = DistillRunTask(_cfg())
    last1 = run(task=t1, spec=_spec(), settings=_settings())
    rs1, _ = _saved_resume_state()
    shards1 = [p.name for p in t1._train_loader.shard_files]
    print(f"leg1 last step={last1['step']} (expect {SPJ - 1})  resume_state={rs1}  shards={shards1}")
    ok["leg1 ran job 0 and recorded it"] = (
        last1["step"] == SPJ - 1 and rs1["job_index"] == 0
        and rs1["steps_per_job"] == SPJ and rs1["batch_size_per_gpu"] == BS
    )

    # --- leg 2: resume -> next slice ---------------------------------------
    print("\n=== leg 2: resume (must advance to job_index 1) ===")
    t2 = DistillRunTask(_cfg())
    last2 = run(task=t2, spec=_spec(), settings=_settings())
    rs2, payload2 = _saved_resume_state()
    shards2 = [p.name for p in t2._train_loader.shard_files]
    print(f"leg2 last step={last2['step']} (expect {2 * SPJ - 1})  resume_state={rs2}  shards={shards2}")
    ok["leg2 started at (job_index+1)*steps_per_job"] = last2["step"] == 2 * SPJ - 1
    ok["leg2 advanced job_index 0 -> 1"] = rs2["job_index"] == 1
    ok["leg2 read a DISJOINT shard slice"] = bool(shards1) and not (set(shards1) & set(shards2))

    # --- refusals: each of these would silently corrupt the data schedule ---
    print("\n=== refusals ===")
    ok["refuses a mid-job (SIGUSR1) checkpoint"] = _refuses(
        "mid-job save", DistillRunTask(_cfg()), payload2, 2 * SPJ - 40)
    # A batch-size change ALONE is not constructible here (steps_per_job * batch must stay a
    # multiple of samples_per_shard=4096), so the two schedule inputs move together and each
    # guard catches the pair from a different side. Isolated per-input coverage — batch,
    # steps_per_job, world_size, samples_per_shard one at a time — is the parametrized CPU
    # test in tasks/tests/test_wds_resume.py.
    ok["refuses changed batch+steps_per_job (job-boundary guard)"] = _refuses(
        "batch 32->16 / steps_per_job 128->256, scheduler at the OLD boundary",
        DistillRunTask(_cfg(batch_size_per_gpu=16, steps_per_job=256)), payload2, 2 * SPJ)
    ok["refuses changed batch+steps_per_job (invariant guard)"] = _refuses(
        "batch 32->16 / steps_per_job 128->256, scheduler at the NEW boundary",
        DistillRunTask(_cfg(steps_per_job=256, batch_size_per_gpu=16)), payload2, 2 * 256)
    ok["refuses a checkpoint without resume_state"] = _refuses(
        "no job_index", DistillRunTask(_cfg()), {"metadata": {}}, 2 * SPJ)

    print("\n=== SUMMARY ===")
    for k, v in ok.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print("\nALL PASS" if all(ok.values()) else "\nFAILURES ABOVE")


if __name__ == "__main__":
    main()
