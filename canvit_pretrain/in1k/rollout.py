"""Patcher-aware glimpse rollout on CanViTForImageClassification (unification P5).

The classification analogue of ade20k/rollout.py: run the recurrent rollout, then
read the per-timestep CLS token (the classifier's head input). Glimpse routing is
identical to the ADE20K probe and canvit_eval/episode.py (uniform -> pre-crop at
the training-matched pixel size; foveated/square -> full image), so the helpers are
reused directly (they only touch ``.canvit``, which the classifier also exposes).

Frozen mode runs the backbone under no_grad and trains only LN+head; finetune runs
the whole graph. The head is applied by the caller (``clf.head(clf.norm(cls))``) so
the same CLS stream feeds training and evaluation.
"""

from contextlib import nullcontext

import torch
from canvit_pytorch import CanViTForImageClassification, sample_at_viewpoint
from canvit_pytorch.policies import coarse_to_fine_viewpoints, repeated_full_scene
from canvit_pytorch.viewpoint import Viewpoint
from torch import Tensor

from ..ade20k.rollout import consumes_full_image, derive_glimpse_px, make_random_viewpoints
from ..train.config import FoveatedScaleConfig

__all__ = [
    "consumes_full_image",
    "derive_glimpse_px",
    "eval_viewpoints",
    "make_random_viewpoints",
    "rollout_cls_tokens",
]


def eval_viewpoints(
    policy: str, batch_size: int, device: torch.device, n: int, *,
    is_foveated: bool, foveated_scale: FoveatedScaleConfig,
) -> list[Viewpoint]:
    """Deploy-time viewpoint sequence. ``coarse_to_fine`` (canvit_eval default):
    quadtree full->quadrants->…; ``full``: repeated full scene; ``random``: the
    training random law (patcher-aware). Foveated/square honor ``fix_size=scale*H``,
    so C2F's varying scales are only in-distribution for a per-glimpse-scale model;
    fixed-scale foveated models should deploy with ``full`` or a scale-pinned policy."""
    if policy == "coarse_to_fine":
        return coarse_to_fine_viewpoints(batch_size, device, n)
    if policy == "full":
        return repeated_full_scene(batch_size, device, n)
    if policy == "random":
        return make_random_viewpoints(
            batch_size, device, n, min_scale=0.05, max_scale=1.0, start_with_full_scene=True,
            is_foveated=is_foveated, foveated_scale=foveated_scale,
        )
    raise ValueError(f"unknown eval policy: {policy}")


def rollout_cls_tokens(
    *,
    clf: CanViTForImageClassification,
    images: Tensor,
    viewpoints: list[Viewpoint],
    canvas_grid: int,
    glimpse_px: int | None,
    freeze_backbone: bool,
) -> list[Tensor]:
    """Run the rollout; return the CLS token [B, D] after each timestep. In frozen
    mode the backbone runs under no_grad (only the caller's head carries grad)."""
    B = images.shape[0]
    full_image = consumes_full_image(clf)
    px = None if full_image else derive_glimpse_px(clf, glimpse_px)

    state = clf.init_state(batch_size=B, canvas_grid_size=canvas_grid)
    cls_tokens: list[Tensor] = []
    ctx = torch.no_grad() if freeze_backbone else nullcontext()
    with ctx:
        for vp in viewpoints:
            model_input = images if full_image else sample_at_viewpoint(
                spatial=images, viewpoint=vp, glimpse_size_px=px
            )
            out = clf.canvit(image=model_input, state=state, viewpoint=vp)
            state = out.state
            cls_tokens.append(out.state.recurrent_cls[:, 0].float())
    return cls_tokens
