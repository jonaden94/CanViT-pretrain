"""Patcher-aware canvas-feature rollout on CanViTForSemanticSegmentation.

This is P2's centerpiece: the unified replacement for specialize's
extract_canvas_features (which was uniform-only and reached past the task
wrapper into the raw pretraining model — the root cause of the 3-month silent
breakage, unification-status §4). Routing follows canvit_eval/episode.py:

- uniform patcher: pre-crop the glimpse at the training-matched pixel size
  (derived from glimpse_grid_size × patch size/stride, token-count hard guard);
  the wrapper is loaded with glimpse_size_px=None so the patcher treats its
  input as the already-cropped glimpse.
- foveated/square patcher: pass the FULL image (pre-cropping would double-crop).

Only canvas_hidden features are produced (recon_normalized dropped, D3).
"""

import torch
from canvit_pytorch import CanViTForSemanticSegmentation, sample_at_viewpoint
from canvit_pytorch.patcher.foveated import FoveatedPatcher
from canvit_pytorch.patcher.square import SquarePatcher
from canvit_pytorch.policies import random_viewpoints
from canvit_pytorch.viewpoint import Viewpoint
from torch import Tensor


def consumes_full_image(seg: CanViTForSemanticSegmentation) -> bool:
    return isinstance(getattr(seg.canvit, "patcher", None), (FoveatedPatcher, SquarePatcher))


def derive_glimpse_px(seg: CanViTForSemanticSegmentation, glimpse_px: int | None) -> int:
    """Training-matched uniform glimpse crop size (canvit_eval's rule): the
    patch-embed conv must yield exactly glimpse_grid_size tokens per side."""
    canvit = seg.canvit
    patch_size = canvit.backbone.patch_size_px
    stride = getattr(canvit.backbone, "patch_stride_px", patch_size)
    glimpse_grid = getattr(canvit, "glimpse_grid_size", None)
    grid = glimpse_grid if glimpse_grid is not None else 8
    if glimpse_px is None:
        glimpse_px = (grid - 1) * stride + patch_size
    assert (glimpse_px - patch_size) % stride == 0 and glimpse_px >= patch_size, (
        f"glimpse_px={glimpse_px} incompatible with patch_size_px={patch_size}, "
        f"patch_stride_px={stride} (need (glimpse_px - patch) divisible by stride)"
    )
    tokens = (glimpse_px - patch_size) // stride + 1
    if glimpse_grid is not None:
        assert tokens == glimpse_grid, (
            f"glimpse_px={glimpse_px} yields {tokens} tokens/side but the model "
            f"trained with glimpse_grid_size={glimpse_grid}"
        )
    return glimpse_px


def make_random_viewpoints(
    batch_size: int, device: torch.device, n: int, *,
    min_scale: float, max_scale: float, start_with_full_scene: bool,
) -> list[Viewpoint]:
    """Specialize's training/val viewpoint distribution (core's random_viewpoints:
    the same L²-safe-box law as pretrain's Viewpoint.random — master plan §3)."""
    return random_viewpoints(
        batch_size, device, n,
        min_scale=min_scale, max_scale=max_scale,
        start_with_full_scene=start_with_full_scene,
    )


def rollout_canvas_hidden(
    *,
    seg: CanViTForSemanticSegmentation,
    images: Tensor,
    viewpoints: list[Viewpoint],
    canvas_grid: int,
    glimpse_px: int | None,
) -> list[Tensor]:
    """Run the recurrent rollout, return canvas_hidden [B, G, G, D] per timestep.

    Steps ``seg.canvit`` directly (CanViT-only execution, blessed by the wrapper
    docstring); the probe head is applied by the caller so the same features can
    feed training and evaluation.
    """
    B = images.shape[0]
    full_image = consumes_full_image(seg)
    px = None if full_image else derive_glimpse_px(seg, glimpse_px)

    hidden: list[Tensor] = []
    state = seg.init_state(batch_size=B, canvas_grid_size=canvas_grid)
    for vp in viewpoints:
        model_input = images if full_image else sample_at_viewpoint(
            spatial=images, viewpoint=vp, glimpse_size_px=px
        )
        out = seg.canvit(image=model_input, state=state, viewpoint=vp)
        state = out.state
        hidden.append(seg.canvit.get_spatial(state.canvas).view(B, canvas_grid, canvas_grid, -1))
    return hidden
