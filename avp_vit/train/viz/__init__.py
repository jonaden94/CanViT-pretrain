"""Visualization utilities for training."""

from .comet import log_curve, log_figure
from .image import imagenet_denormalize
from .metrics import compute_spatial_stats, cosine_dissimilarity
from .pca import fit_pca, pca_rgb
from .plot import (
    RGBA,
    TimestepPredictions,
    plot_multistep_pca,
    plot_pca_grid,
    plot_trajectory,
    timestep_colors,
)
from ._core import (
    ValAccumulator,
    VizSampleData,
    validate,
    viz_and_log,
)

__all__ = [
    # comet
    "log_curve",
    "log_figure",
    # image
    "imagenet_denormalize",
    # metrics
    "compute_spatial_stats",
    "cosine_dissimilarity",
    # pca
    "fit_pca",
    "pca_rgb",
    # plot
    "RGBA",
    "TimestepPredictions",
    "plot_multistep_pca",
    "plot_pca_grid",
    "plot_trajectory",
    "timestep_colors",
    # _core (orchestration)
    "ValAccumulator",
    "VizSampleData",
    "validate",
    "viz_and_log",
]
