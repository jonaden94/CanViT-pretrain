"""Foveated PCA plot helpers.

Replacements for the uniform-patcher columns in `plot_multistep_pca` when the
model is run with the foveated patcher. The three "Glimpse"-column replacements
(samples scatter, samples reconstruction, patches-over-foveation) are adapted
from `fovi/notebooks/plot.py`; the patch-Voronoi reconstruction (col 8) is new.

Coord systems used here:
- **Pixel frame**: matplotlib's `imshow` default, y increases downward. Used
  by `plot_samples_scatter_absolute` and `plot_patch_voronoi_absolute` so
  they overlay the full image directly without extent tricks. Pixel coords
  are pre-computed in :class:`FoveatedVizData` using the per-step viewpoint
  fixation — under full-image foveation, samples near edge fixations may
  fall outside the image bounds.
- **Visual-field frame**: ``[-1, 1]^2`` (row, col), math orientation. Used
  by `plot_patches_overlay_relative` since it's self-contained (no
  underlying image — just sample dots + patch hulls).
"""

import logging
from typing import Sequence  # noqa: F401 (kept for downstream import compat)

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, cKDTree

log = logging.getLogger(__name__)

_RING_PALETTE = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)


def plot_samples_scatter_absolute(
    ax,
    img: NDArray[np.floating],
    sample_xy: NDArray[np.floating],
    sample_colors: NDArray[np.floating],
    sizes: NDArray[np.floating] | float = 4.0,
    title: str = "Samples",
) -> None:
    """Foveated retinal samples scattered on top of the full image.

    `sample_xy` is (N, 2) in pixel coords (x, y), `sample_colors` is (N, 3) in [0, 1].
    """
    ax.imshow(img, alpha=0.45)
    ax.scatter(sample_xy[:, 0], sample_xy[:, 1], c=sample_colors, s=sizes, linewidths=0)
    ax.set_title(title)
    ax.axis("off")


def _voronoi_reconstruction_image(
    *,
    H: int,
    W: int,
    seed_xy: NDArray[np.floating],
    seed_colors: NDArray[np.floating],
    hull_xy: NDArray[np.floating],
    bg_color: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> NDArray[np.floating]:
    """Build an (H, W, 3) image where each pixel inherits the color of the
    nearest seed point, but pixels outside the convex hull of `hull_xy`
    are filled with `bg_color`.
    """
    xs = np.arange(W) + 0.5
    ys = np.arange(H) + 0.5
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    pixel_xy = np.stack([xx.ravel(), yy.ravel()], axis=1)

    _, idx = cKDTree(seed_xy).query(pixel_xy)
    recon = seed_colors[idx].reshape(H, W, seed_colors.shape[1])

    try:
        hull = ConvexHull(hull_xy)
        inside = Path(hull_xy[hull.vertices]).contains_points(pixel_xy).reshape(H, W)
        recon[~inside] = np.asarray(bg_color, dtype=recon.dtype)
    except Exception as e:
        # Hull failures (e.g. collinear samples) shouldn't kill the run.
        log.warning(f"ConvexHull failed in reconstruction: {e}; skipping hull mask")

    return np.clip(recon, 0.0, 1.0)


def plot_samples_reconstruction_absolute(
    ax,
    sample_xy: NDArray[np.floating],
    sample_colors: NDArray[np.floating],
    img_h: int,
    img_w: int,
    title: str = "Sample recon",
) -> None:
    """Nearest-sample reconstruction of the full image from foveated samples.

    Pixels outside the convex hull of `sample_xy` are gray.
    """
    recon = _voronoi_reconstruction_image(
        H=img_h, W=img_w,
        seed_xy=sample_xy,
        seed_colors=sample_colors,
        hull_xy=sample_xy,
    )
    ax.imshow(recon)
    ax.set_title(title)
    ax.axis("off")


def plot_patch_voronoi_absolute(
    ax,
    patch_xy: NDArray[np.floating],
    patch_colors: NDArray[np.floating],
    sample_hull_xy: NDArray[np.floating],
    img_h: int,
    img_w: int,
    title: str = "Patch recon",
) -> None:
    """Voronoi reconstruction from patch centers, colored by per-patch PCA RGB.

    Mask uses the sample-hull (matches `plot_samples_reconstruction_absolute`).
    """
    recon = _voronoi_reconstruction_image(
        H=img_h, W=img_w,
        seed_xy=patch_xy,
        seed_colors=patch_colors,
        hull_xy=sample_hull_xy,
    )
    ax.imshow(recon)
    ax.set_title(title)
    ax.axis("off")


def _ring_index_per_patch(out_polar: NDArray[np.floating]) -> NDArray[np.integer]:
    """Map each output coord (patch) to its concentric-ring index.

    `out_polar` is `kpe.out_coords.polar[:, 0]` (radii).
    """
    radii = out_polar
    unique = np.unique(radii)
    return np.searchsorted(unique, radii)


def plot_patches_overlay_relative(
    ax,
    sample_cart_xy: NDArray[np.floating],
    sample_colors: NDArray[np.floating],
    sample_sizes: NDArray[np.floating] | float,
    knn_indices: NDArray[np.integer],
    knn_pad_mask: NDArray[np.bool_],
    out_polar_r: NDArray[np.floating],
    cart_pad_xy: NDArray[np.floating] | None = None,
    title: str = "Patches over RT",
    outline_lw: float = 1.2,
    show_padding: bool = False,
) -> None:
    """Foveation pattern (sample scatter) + per-patch convex-hull outlines.

    Adapted from fovi/notebooks/plot.py:plot_patches_overlay. Operates entirely
    in the visual-field frame [-1, 1]^2 (no underlying image).

    When ``show_padding`` is True and ``cart_pad_xy`` is provided, out-of-field
    (padding) neighbor slots are also drawn as black 'x' markers AND included in
    each patch's convex hull -- reflecting the patch as the model actually
    processes it (a fixed kernel footprint, with peripheral cells filled by
    padding). With ``show_padding`` False (default), hulls cover only real
    samples (the image-coverage view) and padding is not drawn.
    """
    ax.scatter(
        sample_cart_xy[:, 0], sample_cart_xy[:, 1],
        c=sample_colors, s=sample_sizes, linewidths=0, zorder=1,
    )

    rings = _ring_index_per_patch(out_polar_r)
    full_xy = sample_cart_xy if cart_pad_xy is None else np.concatenate(
        [sample_cart_xy, cart_pad_xy], axis=0
    )
    n_full = full_xy.shape[0]
    draw_pad = show_padding and cart_pad_xy is not None and len(cart_pad_xy) > 0

    # fovi convention: knn_indices is [k, N_patches] (axis 1 = patches), and
    # padded slots may carry index = -1 (sentinel for some variants). Filter
    # to in-range indices; include or exclude padding slots per ``show_padding``.
    hull_segments: list[NDArray[np.floating]] = []
    hull_colors: list[str] = []
    pad_pts: list[NDArray[np.floating]] = []
    n_patches = knn_indices.shape[1]
    for p in range(n_patches):
        col_idx = knn_indices[:, p]
        col_pad = knn_pad_mask[:, p]
        in_range = (col_idx >= 0) & (col_idx < n_full)
        # Hull members: include padding slots only when drawing padding.
        hull_mask = in_range if draw_pad else (in_range & ~col_pad)
        members = col_idx[hull_mask]
        if draw_pad:
            pad_slot = in_range & col_pad
            if pad_slot.any():
                pad_pts.append(full_xy[col_idx[pad_slot]])
        if members.size < 3:
            continue
        pts = full_xy[members]
        try:
            hull = ConvexHull(pts)
        except Exception:
            continue
        verts = pts[hull.vertices]
        verts = np.concatenate([verts, verts[:1]], axis=0)
        for i in range(len(verts) - 1):
            hull_segments.append(verts[i:i + 2])
        ring = int(rings[p])
        for _ in range(len(verts) - 1):
            hull_colors.append(_RING_PALETTE[ring % len(_RING_PALETTE)])

    if hull_segments:
        lc = LineCollection(hull_segments, colors=hull_colors, linewidths=outline_lw, zorder=2)
        ax.add_collection(lc)

    # Padding markers: single uniform style (black 'x') so they are unmistakably
    # distinct from the image-colored sample dots and the ring-colored hulls.
    if draw_pad and pad_pts:
        pad_all = np.concatenate(pad_pts, axis=0)
        ax.scatter(
            pad_all[:, 0], pad_all[:, 1], marker="x",
            c="black", s=14, linewidths=0.8, zorder=3, label="padding",
        )

    # Widen the view so the padding halo (radius > 1) stays visible.
    lim = 1.05
    if draw_pad:
        lim = max(lim, float(np.abs(cart_pad_xy).max()) * 1.05)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(lim, -lim)  # match image y-down so it visually aligns with neighbors
    ax.set_xticks([])
    ax.set_yticks([])
