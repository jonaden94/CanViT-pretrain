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

from ..harness.config import FoveatedScaleConfig
from ..harness.selector import RandomSelector
from ..harness.viewpoint import ViewpointType


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
    is_foveated: bool = False,
    foveated_scale: FoveatedScaleConfig | None = None,
) -> list[Viewpoint]:
    """The probe rollout's viewpoint distribution, PATCHER-AWARE (master plan §3).

    Uniform patcher: specialize's law (core ``random_viewpoints`` — the same
    L²-safe-box law as pretrain's ``Viewpoint.random``).

    Foveated/square: delegated to :class:`RandomSelector`, the canonical random
    policy extracted from the pretraining loop in P1, so the probe sees exactly
    the scale/center law the backbone was trained under. This is not cosmetic:
    the foveated patcher derives its fixation window from the viewpoint scale
    (``fix_size = scale * H``), so feeding it the uniform safe-box law (scales
    ≤ 1) after it was pretrained at, say, ``fixed_scale=2.0`` puts every glimpse
    out of distribution — measured as mIoU *decreasing* monotonically with more
    glimpses (job 15025338; see p2-notes "foveated scale mismatch").
    """
    if not is_foveated:
        return random_viewpoints(
            batch_size, device, n,
            min_scale=min_scale, max_scale=max_scale,
            start_with_full_scene=start_with_full_scene,
        )
    sel = RandomSelector(
        is_foveated=True,
        foveated_scale=foveated_scale or FoveatedScaleConfig(),
        min_viewpoint_scale=min_scale,
    )
    t0 = ViewpointType.FULL if start_with_full_scene else ViewpointType.RANDOM
    ctx = sel.start_rollout(t0_type=t0, batch_size=batch_size, device=device)
    types = [t0] + [ViewpointType.RANDOM] * (n - 1)
    return [
        sel.select(
            vp_type=vt, ctx=ctx, t=t, batch_size=batch_size, device=device, state=None  # type: ignore[arg-type]
        )
        for t, vt in enumerate(types)
    ]


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
