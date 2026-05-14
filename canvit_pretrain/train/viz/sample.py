"""Sample extraction for visualization (shared by train and validation)."""

from dataclasses import dataclass

import numpy as np
import torch
from canvit_pytorch import CanViTOutput
from canvit_pytorch.viewpoint import Viewpoint, sample_at_viewpoint
from torch import Tensor

from canvit_pretrain import CanViTForPretraining

from .image import imagenet_denormalize_to_numpy


@dataclass
class FoveatedVizData:
    """Foveated-patcher viz tensors extracted for a single sample.

    All numpy, on CPU. Two coord systems are exposed so the renderer doesn't
    have to know about fovi's internals:

    - Visual-field frame ``[-1, 1]^2`` (row, col): retinal samples and patch
      centers relative to the fixation. Used by the "patches over RT
      (relative)" overlay column.
    - Image-pixel frame (x, y) in pixels: same retinal samples and patch
      centers projected into the full-image pixel frame, using the per-step
      viewpoint's fixation point. Used by all "absolute" foveated columns.
      Some pixel coords land outside the image when fixations are near edges
      — that's expected (out-of-image samples).
    """

    sample_cart_rowcol: np.ndarray         # [N_samples, 2] in [-1, 1]^2 (relative)
    sample_xy_pixel: np.ndarray            # [N_samples, 2] (x, y) image pixels
    sample_colors: np.ndarray              # [N_samples, 3] in [0, 1]
    sample_sizes: np.ndarray               # [N_samples] scatter sizes
    patch_cart_rowcol: np.ndarray          # [N_patches, 2] in [-1, 1]^2 (relative)
    patch_xy_pixel: np.ndarray             # [N_patches, 2] (x, y) image pixels
    knn_indices: np.ndarray                # [k, N_patches] int sample indices
    knn_pad_mask: np.ndarray               # [k, N_patches] bool, True = padded slot
    cart_pad_rowcol: np.ndarray | None     # [N_pad, 2] padded sample slots (for overlay context)
    out_polar_r: np.ndarray                # [N_patches] radii for ring coloring


@dataclass
class VizSampleData:
    """Viz data extracted for a single sample.

    Shape annotations:
        G = canvas grid size (e.g., 32)
        g = glimpse grid size (e.g., 8)
        D = teacher feature dim (e.g., 768)
        C = canvas hidden dim
    """

    glimpse: np.ndarray  # [H, W, 3] denormalized RGB (uniform: cropped window; foveated: full image)
    predicted_scene: np.ndarray  # [G², D] teacher-space prediction
    canvas_spatial: np.ndarray  # [G², C] raw hidden state
    local_patches: np.ndarray | None  # [g², D] local stream patches (None if not available)
    foveated: FoveatedVizData | None  # set only when model.patcher is foveated


def _as_numpy(x, dtype=None):
    """Tensor or numpy -> numpy array. fovi mixes both types on the patcher."""
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _rowcol_to_image_xy_pixel(
    rowcol_vf: np.ndarray,
    fix_center_rowcol: np.ndarray,
    image_size: int,
) -> np.ndarray:
    """Map visual-field rowcol in [-1, 1]^2 to image-pixel (x, y).

    Assumes full-image foveation: the fixation window equals the image, so a
    visual-field rowcol of (r, c) maps to image rowcol of ``fix + (r, c)``
    (in [-1, 1] image coords). Then convert to pixel space:
        col_px = (col_image + 1) / 2 * (W - 1)
        row_px = (row_image + 1) / 2 * (H - 1)
    and return as (x=col_px, y=row_px).
    """
    image_rowcol = fix_center_rowcol[None, :] + rowcol_vf  # [N, 2]
    row_px = (image_rowcol[:, 0] + 1.0) * 0.5 * (image_size - 1)
    col_px = (image_rowcol[:, 1] + 1.0) * 0.5 * (image_size - 1)
    return np.stack([col_px, row_px], axis=1)  # (x, y)


def _extract_foveated_sample0(
    *,
    model: CanViTForPretraining,
    image: Tensor,
    viewpoint: Viewpoint,
) -> FoveatedVizData | None:
    """If ``model.patcher`` is foveated, capture the sample-level retinal
    output and the patcher's KNN partitioning state for the full image at the
    given viewpoint's fixation point.

    Cheap (~3 ms): re-runs only the retina on ``image[0:1]`` to obtain RGB at
    each sample point.
    """
    patcher = getattr(model, "patcher", None)
    if patcher is None or getattr(patcher, "kpe", None) is None:
        return None
    # Duck-type: foveated patcher has `retina` and `kpe` (KNNPartitioningPatchEmbedding).
    if not hasattr(patcher, "retina") or not hasattr(patcher, "kpe"):
        return None

    img0 = image[0:1]  # [1, 3, H, W]
    H = int(img0.shape[-1])
    fix_loc = (viewpoint.centers[0:1].to(torch.float32) + 1.0) * 0.5  # [1, 2] in [0, 1]
    with torch.no_grad():
        sensor = patcher.retina(img0, fix_loc=fix_loc, fixation_size=H)  # [1, 3, N_samples]

    # Per-sample RGB in [0, 1]. RetinalTransform output mirrors the ImageNet-
    # normalized image it consumed; denormalize so colors are display-ready.
    # Sensor shape [B, 3, N] -> reshape to [B, N, 3] for denormalize helper.
    s_chw = sensor[0].detach().cpu()  # [3, N_samples]
    s_nc = s_chw.transpose(0, 1).contiguous()  # [N_samples, 3]
    # Re-use imagenet_denormalize on a (1, 3, N, 1) faux-image then squeeze.
    s_faux = s_nc.transpose(0, 1).unsqueeze(-1)  # [3, N_samples, 1]
    sample_colors = imagenet_denormalize_to_numpy(s_faux)[..., 0, :]  # actually [N_samples, 3]
    sample_colors = np.clip(np.asarray(sample_colors, dtype=np.float32), 0.0, 1.0)

    sample_cart_rowcol = _as_numpy(patcher.retina.sampler.coords.cartesian_rowcol, dtype=np.float32)
    raw_sizes = getattr(patcher.retina, "scatter_sizes", None)
    if raw_sizes is not None:
        sample_sizes = _as_numpy(raw_sizes, dtype=np.float32) * 4.0 + 1.5
    else:
        sample_sizes = np.full(sample_cart_rowcol.shape[0], 3.0, dtype=np.float32)

    patch_cart_rowcol = _as_numpy(patcher.kpe.out_coords.cartesian_rowcol, dtype=np.float32)
    knn_indices = _as_numpy(patcher.kpe.knn_indices)
    knn_pad_mask = _as_numpy(patcher.kpe.knn_indices_pad_mask).astype(bool)
    out_polar_r = _as_numpy(patcher.kpe.out_coords.polar[:, 0], dtype=np.float32)

    cart_pad_rowcol: np.ndarray | None = None
    pad_attr = getattr(patcher.kpe.in_coords, "cartesian_pad_rowcol", None)
    if pad_attr is None:
        # fallback: older fovi versions named it `cartesian_pad_coords`
        pad_attr = getattr(patcher.kpe.in_coords, "cartesian_pad_coords", None)
    if pad_attr is not None:
        cart_pad_rowcol = _as_numpy(pad_attr, dtype=np.float32)

    # Map every (row, col) in visual-field frame to image-pixel (x, y) using
    # the viewpoint's fixation. With fix_size == image_size the mapping is
    # ``scene_rowcol = fix_center + vf_rowcol``, then to pixel space.
    fix_center_rowcol = viewpoint.centers[0].detach().cpu().to(torch.float32).numpy()
    sample_xy_pixel = _rowcol_to_image_xy_pixel(sample_cart_rowcol, fix_center_rowcol, H)
    patch_xy_pixel = _rowcol_to_image_xy_pixel(patch_cart_rowcol, fix_center_rowcol, H)

    return FoveatedVizData(
        sample_cart_rowcol=sample_cart_rowcol,
        sample_xy_pixel=sample_xy_pixel,
        sample_colors=sample_colors,
        sample_sizes=sample_sizes,
        patch_cart_rowcol=patch_cart_rowcol,
        patch_xy_pixel=patch_xy_pixel,
        knn_indices=knn_indices,
        knn_pad_mask=knn_pad_mask,
        cart_pad_rowcol=cart_pad_rowcol,
        out_polar_r=out_polar_r,
    )


def extract_sample0_viz(
    out: CanViTOutput,
    image: Tensor,
    viewpoint: Viewpoint,
    predicted_scene: Tensor,
    model: CanViTForPretraining,
    glimpse_size_px: int,
) -> VizSampleData:
    """Extract viz data for sample 0, move to CPU as numpy.

    Uniform mode: computes the cropped glimpse from ``image`` at ``viewpoint``
    for the "Glimpse" column. Foveated mode: stores the full image (the
    "Glimpse" column is replaced with foveated-specific panels in ``plot.py``).
    """
    is_foveated = getattr(model.cfg, "patcher_name", "uniform") == "foveated"

    if is_foveated:
        # No crop — the foveated column shows the full image directly.
        glimpse_np = imagenet_denormalize_to_numpy(image[0].detach().cpu())
    else:
        glimpse_t = sample_at_viewpoint(
            spatial=image, viewpoint=viewpoint, glimpse_size_px=glimpse_size_px,
        )
        glimpse_np = imagenet_denormalize_to_numpy(glimpse_t[0].detach().cpu())

    scene_cpu = predicted_scene[0].detach().cpu().float()
    scene_np = scene_cpu.numpy()

    canvas_single = out.state.canvas[0:1].detach()
    spatial = model.get_spatial(canvas_single)[0]
    spatial_np = spatial.cpu().float().numpy()

    # Extract local stream patches if available
    local_np: np.ndarray | None = None
    if out.local_patches is not None:
        local_np = out.local_patches[0].detach().cpu().float().numpy()

    foveated = _extract_foveated_sample0(model=model, image=image, viewpoint=viewpoint) if is_foveated else None

    return VizSampleData(
        glimpse=glimpse_np,
        predicted_scene=scene_np,
        canvas_spatial=spatial_np,
        local_patches=local_np,
        foveated=foveated,
    )
