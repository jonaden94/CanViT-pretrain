"""Real-data A/B: the generalized rollout engine == the historical distill training_step
on REAL IN21k feature-webdataset batches, on GPU with AMP (extends the synthetic
byte-exact parity, digest 9a0100a1a3de3acd, to production data/numerics).

Per step: from the SAME model state + SAME real batch + SAME RNG, compute total_loss via
(a) the old ``train/step.py::training_step`` and (b) the new ``harness/rollout.py::run_rollout``
(distill adapter), and report the relative difference. Then step via the new path's grads
so the model actually trains (loss should fall). No HF (random backbone, precomputed
features), no teacher forward.

Run: .venv-cu126/bin/python unification_docs/harness_realdata_ab.py
"""

import os
import random

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from canvit_pytorch import create_backbone

from canvit_pretrain import CanViTForPretraining, CanViTForPretrainingConfig
from canvit_pretrain.harness.rollout import run_rollout
from canvit_pretrain.harness.spec import BpttSpec
from canvit_pretrain.tasks.distill.task import BoundDistillTask
from canvit_pretrain.train.config import FoveatedScaleConfig
from canvit_pretrain.train.data.webdataset import WebDatasetTrainLoader, init_normalizer_stats_from_tar
from canvit_pretrain.train.selector import RandomSelector
from canvit_pretrain.train.step import training_step
from canvit_pretrain.train.task import DistillTask
from canvit_pretrain.train.viewpoint import ViewpointType
from pathlib import Path

TRAIN_DIR = Path("/mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/"
                  "webdataset-imagenet-21k-with-features/train-shuffled")
_G, _TD, _GLIMPSE_PX, _RES = 32, 768, 128, 512
_BATCH, _STEPS_PER_JOB, _K = 32, 128, 20  # 32*128 = 4096 = 1 shard; run 20 steps


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}  cuda={torch.cuda.is_available()}  torch={torch.__version__}")
    assert dev.type == "cuda", "expected a GPU node"

    torch.manual_seed(0)
    model = CanViTForPretraining(
        backbone=create_backbone("vitb16"), cfg=CanViTForPretrainingConfig(teacher_dim=_TD),
        glimpse_size_px=_GLIMPSE_PX, backbone_name="vitb16", canvas_patch_grid_sizes=[_G],
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    loader = WebDatasetTrainLoader(
        train_dir=TRAIN_DIR, seed=0, job_index=0, batch_size_per_gpu=_BATCH,
        steps_per_job=_STEPS_PER_JOB, image_size=_RES, world_size=1, rank=0, num_workers=4,
    )
    cls_norm, scene_norm = model.standardizers(_G)
    init_normalizer_stats_from_tar([loader.first_shard_path()], scene_norm, cls_norm, dev, 512)
    print("normalizer initialized from first shard")

    selector = RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.05)
    branches = [ViewpointType.FULL, ViewpointType.RANDOM]
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    max_reldiff = 0.0
    for k in range(_K):
        images, raw_patches, raw_cls, _ = loader.next()
        images = images.to(dev, non_blocking=True)
        raw_patches = raw_patches.to(dev, dtype=torch.float32)
        raw_cls = raw_cls.to(dev, dtype=torch.float32)
        norm_patches = scene_norm(raw_patches)
        norm_cls = cls_norm(raw_cls.unsqueeze(1)).squeeze(1)

        rng = (torch.get_rng_state(), torch.cuda.get_rng_state(), random.getstate())

        opt.zero_grad()
        old = training_step(
            model=model, images=images, scene_target=norm_patches, cls_target=norm_cls,
            raw_scene_target=raw_patches, raw_cls_target=raw_cls,
            scene_denorm=scene_norm.destandardize, cls_denorm=cls_norm.destandardize,
            enable_scene_patches_loss=True, enable_scene_cls_loss=True,
            glimpse_size_px=_GLIMPSE_PX, canvas_grid_size=_G,
            n_full_start_branches=1, n_random_start_branches=1, chunk_size=2, continue_prob=0.5,
            min_viewpoint_scale=0.05, foveated_scale=FoveatedScaleConfig(), amp_ctx=amp_ctx,
            collect_viz=False,
        ).total_loss.item()

        opt.zero_grad()
        torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1]); random.setstate(rng[2])

        task = BoundDistillTask(DistillTask(
            scene_target=norm_patches, cls_target=norm_cls,
            enable_scene_patches_loss=True, enable_scene_cls_loss=True,
        ))
        result = run_rollout(
            model=model, images=images, task=task, selector=selector,
            bptt=BpttSpec(mode="chunked", chunk_size=2, continue_prob=0.5), branches=branches,
            canvas_grid_size=_G, amp_ctx=amp_ctx,
        )
        new = result.total_loss.item()

        reldiff = abs(old - new) / max(abs(old), 1e-8)
        max_reldiff = max(max_reldiff, reldiff)
        print(f"step {k:2d}  old={old:.6f}  new={new:.6f}  reldiff={reldiff:.2e}  n_glimpses={result.n_glimpses}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    print(f"\nMAX reldiff over {_K} steps = {max_reldiff:.2e}")
    print("PASS: engine matches training_step on real data" if max_reldiff < 1e-3
          else "FAIL: engine diverges from training_step on real data")


if __name__ == "__main__":
    main()
