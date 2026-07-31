"""ADE20K segmentation visualization figure.

Port of canvit_specialize/training/ade20k/viz.py, trimmed to the one feature type the
unified repo keeps (canvas_hidden — recon_normalized was dropped, D3) and re-pointed at
this repo's helpers. The layout is unchanged from the specialize runs, one row per
sample:

    Image | GT | canvas_h t0 | corr t0 | PCA t0 | canvas_h t-1 | corr t-1 | PCA t-1

The one deliberate delta is the sink: specialize uploaded these as wandb images (~140 MB
of Media per probe run); here the caller writes them to ``{run_dir}/visualization/`` with
``train/viz/disk.py::save_figure``, the convention the rest of pretrain already uses.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from canvit_pytorch.preprocess import imagenet_denormalize
from matplotlib.figure import Figure
from torch import Tensor

from ..harness.viz.pca import fit_pca
from .data import IGNORE_LABEL, NUM_CLASSES
from .metrics import upsample_preds

# Deterministic palette, seeded exactly as specialize's so mask colours are comparable
# against the old figures. The extra row (index NUM_CLASSES) is the ignore slot.
_PALETTE = np.random.RandomState(42).randint(0, 255, (NUM_CLASSES + 1, 3), dtype=np.uint8)
_PALETTE[NUM_CLASSES] = 0


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    return _PALETTE[np.where(mask == IGNORE_LABEL, NUM_CLASSES, mask)]


def correctness_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Green = correct, red = wrong, gray = ignored."""
    out = np.zeros((*pred.shape, 3), dtype=np.uint8)
    valid = gt != IGNORE_LABEL
    out[(pred == gt) & valid] = [0, 200, 0]
    out[(pred != gt) & valid] = [200, 0, 0]
    out[~valid] = [128, 128, 128]
    return out


def _pca_to_rgb(pca, feats: np.ndarray, H: int, W: int) -> np.ndarray:
    """[H*W, D] -> [H, W, 3] via a 2/98-percentile stretch of the first 3 components.

    Deliberately NOT ``train/viz/pca.py::pca_rgb``, which maps through a sigmoid: that is
    the distill figures' convention, this is the ADE20K figures' one, and keeping it makes
    the new panels directly comparable to the specialize runs.
    """
    if pca is None:
        return np.full((H, W, 3), 0.5, dtype=np.float32)
    proj = pca.transform(feats)[:, :3]
    lo = np.percentile(proj, 2, axis=0, keepdims=True)
    hi = np.percentile(proj, 98, axis=0, keepdims=True)
    rgb = np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1)
    return rgb.reshape(H, W, 3).astype(np.float32)


def _resize_rgb(rgb: np.ndarray, H: int, W: int) -> np.ndarray:
    """Bilinearly resize an [h, w, 3] float image to [H, W, 3]. The PCA panels are drawn
    at mask resolution, not the 32x32 canvas grid — without this they read as coarse
    blocks instead of the smooth feature maps the specialize figures showed."""
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    up = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return up[0].permute(1, 2, 0).numpy()


def make_seg_viz_figure(
    *,
    hidden: Sequence[Tensor],
    preds: Sequence[Tensor],
    images: Tensor,
    masks: Tensor,
) -> Figure:
    """Build the 8-column segmentation figure, one row per sample.

    ``hidden`` / ``preds`` are per-timestep canvas features [n, G, G, D] and argmax
    predictions [n, G, G] (CPU); only the FIRST and LAST entries are drawn, so callers may
    pass the whole rollout or just those two. ``images`` [n, 3, H, W] is
    ImageNet-normalized and ``masks`` [n, H, W] are the raw GT labels.

    PCA is fit ONCE per sample on the FINAL features and reused for t0, so the two PCA
    panels share a colour space and the change across glimpses is what you actually see
    (specialize did the same — fitting per timestep would recolour both panels
    independently and make them incomparable).
    """
    n = images.shape[0]
    n_cols = 8
    fig, axes = plt.subplots(n, n_cols, figsize=(2.5 * n_cols, 2.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    steps = [(0, "t0"), (len(hidden) - 1, "t-1")]
    for i in range(n):
        # imagenet_denormalize returns CHW in [0, 1]; imshow wants HWC uint8.
        img = (imagenet_denormalize(images[i]).cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        gt = masks[i].cpu().numpy()
        panels: list[tuple[np.ndarray, str]] = [(img, "Image"), (colorize_mask(gt), "GT")]

        feat_final = hidden[-1][i].cpu().float().numpy()
        H, W, D = feat_final.shape
        pca = fit_pca(feat_final.reshape(-1, D), n_components=3)

        for t, t_name in steps:
            pred_up = upsample_preds(preds[t][i][None].cpu(), gt.shape[0], gt.shape[1])[0].numpy()
            feat = hidden[t][i].cpu().float().numpy()
            pca_rgb = _pca_to_rgb(pca, feat.reshape(-1, D), H, W)
            panels += [
                (colorize_mask(pred_up), f"canvas_h {t_name}"),
                (correctness_map(pred_up, gt), f"corr {t_name}"),
                (_resize_rgb(pca_rgb, *gt.shape), f"PCA {t_name}"),
            ]

        for col, (panel, title) in enumerate(panels):
            axes[i, col].imshow(panel)
            axes[i, col].set_title(title if i == 0 else "")
            axes[i, col].axis("off")

    plt.tight_layout()
    return fig


__all__ = ["colorize_mask", "correctness_map", "make_seg_viz_figure"]
