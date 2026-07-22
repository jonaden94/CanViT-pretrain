"""P1 seam tests: training_step honors an injected Selector / Task.

The refactor's default path is covered by the parity probe (byte-identical
losses); these tests cover the other half of the contract — that explicitly
passed seams are actually consulted, with the expected cardinality."""

from contextlib import nullcontext

import pytest
import torch
from canvit_pytorch import create_backbone
from torch import Tensor

from canvit_pretrain import CanViTForPretraining, CanViTForPretrainingConfig

from .config import FoveatedScaleConfig
from .selector import RandomSelector
from .step import training_step
from .task import DistillTask

_DEVICE = torch.device("cpu")
_B, _G, _D = 2, 8, 384


@pytest.fixture(scope="module")
def model() -> CanViTForPretraining:
    torch.manual_seed(0)
    backbone = create_backbone("vits16").to(_DEVICE)
    return CanViTForPretraining(
        backbone=backbone,
        cfg=CanViTForPretrainingConfig(teacher_dim=_D),
        glimpse_size_px=128,
        backbone_name="vits16",
        canvas_patch_grid_sizes=[_G],
    ).to(_DEVICE)


def test_injected_selector_and_task_are_used(model: CanViTForPretraining) -> None:
    torch.manual_seed(7)
    tensors: dict[str, Tensor] = {
        "images": torch.randn(_B, 3, 224, 224, device=_DEVICE),
        "scene_target": torch.randn(_B, _G * _G, _D, device=_DEVICE),
        "cls_target": torch.randn(_B, _D, device=_DEVICE),
        "raw_scene_target": torch.randn(_B, _G * _G, _D, device=_DEVICE),
        "raw_cls_target": torch.randn(_B, _D, device=_DEVICE),
    }
    calls = {"start_rollout": 0, "select": 0, "step_loss": 0}
    base_sel = RandomSelector(
        is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.1
    )
    base_task = DistillTask(
        scene_target=tensors["scene_target"],
        cls_target=tensors["cls_target"],
        enable_scene_patches_loss=True,
        enable_scene_cls_loss=True,
    )

    class SpySelector:
        def start_rollout(self, **kw):
            calls["start_rollout"] += 1
            return base_sel.start_rollout(**kw)

        def select(self, **kw):
            calls["select"] += 1
            return base_sel.select(**kw)

    class SpyTask:
        def step_loss(self, out):
            calls["step_loss"] += 1
            return base_task.step_loss(out)

    model.zero_grad()
    metrics = training_step(
        model=model,
        images=tensors["images"],
        scene_target=tensors["scene_target"],
        cls_target=tensors["cls_target"],
        raw_scene_target=tensors["raw_scene_target"],
        raw_cls_target=tensors["raw_cls_target"],
        scene_denorm=lambda x: x,
        cls_denorm=lambda x: x,
        enable_scene_patches_loss=True,
        enable_scene_cls_loss=True,
        glimpse_size_px=128,
        canvas_grid_size=_G,
        n_full_start_branches=1,
        n_random_start_branches=1,
        chunk_size=2,
        continue_prob=0.0,  # -> n_glimpses == chunk_size == 2, deterministic
        min_viewpoint_scale=0.1,
        foveated_scale=FoveatedScaleConfig(),
        amp_ctx=nullcontext(),
        collect_viz=False,
        selector=SpySelector(),
        task=SpyTask(),
    )
    # 2 branches (full-start + random-start), 2 glimpses each:
    # start_rollout once per branch; select + step_loss once per glimpse per branch.
    assert metrics.n_glimpses == 2
    assert calls == {"start_rollout": 2, "select": 4, "step_loss": 4}
    assert torch.isfinite(metrics.total_loss)
