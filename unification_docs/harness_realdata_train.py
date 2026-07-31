"""Real distill training through the NEW harness loop on GPU (end-to-end plumbing check).

Drives ``harness/loop.py::run_training_loop`` with a run-level distill task over the real
IN21k feature-webdataset — real loader + model-owned normalizer + per-group optimizer +
local checkpoint, many steps on the A100. Correctness is already proven exact by
``harness_realdata_ab.py`` (reldiff 0); this confirms the loop machinery runs at scale
and the checkpoint reloads into a fresh model.

Run: .venv-cu126/bin/python unification_docs/harness_realdata_train.py
"""

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from canvit_pytorch import create_backbone

from canvit_train import CanViTForPretraining, CanViTForPretrainingConfig
from canvit_train.distill.data.webdataset import WebDatasetTrainLoader, init_normalizer_stats_from_tar
from canvit_train.distill.loss import DistillTask
from canvit_train.distill.task import BoundDistillTask
from canvit_train.harness.config import FoveatedScaleConfig
from canvit_train.harness.infra.checkpoint import find_latest, load_checkpoint, restore_into
from canvit_train.harness.loop import apply_requires_grad, run_training_loop
from canvit_train.harness.optim import build_optimizer_and_scheduler
from canvit_train.harness.rollout.selector import RandomSelector
from canvit_train.harness.rollout.viewpoint import ViewpointType
from canvit_train.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TaskCaps, TrainSpec

TRAIN_DIR = Path("/mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/"
                  "webdataset-imagenet-21k-with-features/train-shuffled")
CKPT_DIR = Path("/tmp/claude-966121/-mnt-vast-nhr-projects-nib00021-jonathan-repos-canvit-modify/"
                "d55dcc83-a887-4c35-ac0d-862a62d401be/scratchpad/distill_ckpt")
_G, _TD, _GLIMPSE_PX, _RES = 32, 768, 128, 512
_BATCH, _STEPS_PER_JOB, _N = 32, 256, 150  # 32*256 = 8192 = 2 shards


class DistillRunTask:
    """Run-level distill task surface the loop needs: extract images + bind per-batch
    (standardized) teacher targets via the model-owned normalizer."""

    def __init__(self, scene_norm, cls_norm):
        self.scene_norm, self.cls_norm = scene_norm, cls_norm

    def batch_images(self, batch, device):
        return batch[0].to(device, non_blocking=True)

    def bind(self, batch, device, *, model, head):
        _, raw_patches, raw_cls, _ = batch
        raw_patches = raw_patches.to(device, dtype=torch.float32)
        raw_cls = raw_cls.to(device, dtype=torch.float32)
        return BoundDistillTask(DistillTask(
            scene_target=self.scene_norm(raw_patches),
            cls_target=self.cls_norm(raw_cls.unsqueeze(1)).squeeze(1),
            enable_scene_patches_loss=True, enable_scene_cls_loss=True,
        ))


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  torch={torch.__version__}")
    assert dev.type == "cuda"

    torch.manual_seed(0)
    model = CanViTForPretraining(
        backbone=create_backbone("vitb16"), cfg=CanViTForPretrainingConfig(teacher_dim=_TD),
        glimpse_size_px=_GLIMPSE_PX, backbone_name="vitb16", canvas_patch_grid_sizes=[_G],
    ).to(dev)

    loader = WebDatasetTrainLoader(
        train_dir=TRAIN_DIR, seed=0, job_index=0, batch_size_per_gpu=_BATCH,
        steps_per_job=_STEPS_PER_JOB, image_size=_RES, world_size=1, rank=0, num_workers=6,
    )
    cls_norm, scene_norm = model.standardizers(_G)
    init_normalizer_stats_from_tar([loader.first_shard_path()], scene_norm, cls_norm, dev, 512)

    spec = TrainSpec(
        train_backbone=True, train_head=False, task_grad_to_backbone=True,
        bptt=BpttSpec(mode="chunked", chunk_size=2, continue_prob=0.5),
        optim={"backbone": GroupOptim(
            lr=4e-4, weight_decay=1e-4,
            schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=50, warmup_lr_ratio=1e-3))},
    )
    spec.validate(TaskCaps(has_head=False, supports_policy=True))  # distill: heads ride in the backbone group
    apply_requires_grad(model=model, head=None, joint=None, spec=spec)
    opt, sched = build_optimizer_and_scheduler(spec, {"backbone": list(model.parameters())})
    selector = RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.05)

    losses: list[float] = []

    def on_log(step, m):
        losses.append(m["total_loss"])
        print(f"step {step:3d}  loss={m['total_loss']:.5f}  n_glimpses={m['n_glimpses']}  "
              f"lr={sched.get_last_lr()[0]:.2e}", flush=True)

    def batches():
        while True:
            yield loader.next()

    last = run_training_loop(
        task=DistillRunTask(scene_norm, cls_norm), model=model, head=None, optimizer=opt, scheduler=sched,
        selector=selector, spec=spec, branches=[ViewpointType.FULL, ViewpointType.RANDOM],
        canvas_grid=_G, device=dev, train_batches=batches(), n_steps=_N, task_name="distill",
        model_config={"teacher_dim": _TD, "canvas_grid": _G}, amp_ctx=torch.autocast("cuda", dtype=torch.bfloat16),
        log_every=10, ckpt_dir=CKPT_DIR, on_log=on_log,
    )

    assert losses, "on_log never fired — no losses captured"
    early = sum(losses[:3]) / 3
    late = sum(losses[-3:]) / 3
    print(f"\nfinal step={last['step']}  early-avg-loss={early:.5f}  late-avg-loss={late:.5f}  "
          f"delta={late - early:+.5f}")

    # checkpoint reloads into a fresh model.
    latest = find_latest(CKPT_DIR)
    assert latest is not None and latest.name == f"step-{_N}.pt", latest
    payload = load_checkpoint(latest, "cpu")
    fresh = CanViTForPretraining(
        backbone=create_backbone("vitb16"), cfg=CanViTForPretrainingConfig(teacher_dim=_TD),
        glimpse_size_px=_GLIMPSE_PX, backbone_name="vitb16", canvas_patch_grid_sizes=[_G],
    )
    restore_into(payload, model=fresh)
    ok = all(torch.allclose(p.cpu(), q) for (p, q) in
             zip(model.state_dict().values(), fresh.state_dict().values()))
    print(f"checkpoint reload matches trained model: {ok}")
    print("PASS: harness trains distill on real data end-to-end + checkpoint round-trips"
          if (torch.isfinite(torch.tensor(late)) and ok) else "FAIL")


if __name__ == "__main__":
    main()
