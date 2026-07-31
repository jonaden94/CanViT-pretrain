"""Shared pretraining LIBRARY: config, data, probe, scheduler, viewpoint, selectors,
RL objectives, JointPolicy, tracker, viz.

Despite the name this is no longer a trainer. The distill training loop (``loop.py`` +
``step.py`` + ``__main__.py``) was deleted in the 2026-07-31 consolidation; the harness
(``canvit_train.harness``) is the single entry point and imports the modules here as
its shared substrate. The package keeps its name because ~15 import sites and every
pinned historical launcher reference it.
"""

from canvit_train.train.data import (
    Batch,
    InfiniteLoader,
    Loaders,
    create_loaders,
    scene_size_px,
)
from canvit_train.train.probe import (
    IN1K_NUM_CLASSES,
    PROBE_REGISTRY,
    ProbeInfo,
    TopKPrediction,
    compute_in1k_top1,
    get_imagenet_class_names,
    get_probe_resolution,
    get_top_k_predictions,
    labels_are_in1k,
    load_probe,
)
from canvit_train.train.scheduler import warmup_constant_scheduler
from canvit_train.train.viewpoint import (
    PixelBox,
    Viewpoint,
    make_eval_viewpoints,
    viewpoint_to_pixel_box,
)
from canvit_train.train.viz import (
    TimestepPredictions,
    fit_pca,
    imagenet_denormalize_to_numpy,
    pca_rgb,
    plot_multistep_pca,
    plot_pca_grid,
    plot_trajectory,
    timestep_colors,
    validate,
)

__all__ = [
    # Data
    "Batch",
    "InfiniteLoader",
    "Loaders",
    "create_loaders",
    "scene_size_px",
    # Probe
    "IN1K_NUM_CLASSES",
    "PROBE_REGISTRY",
    "ProbeInfo",
    "TopKPrediction",
    "compute_in1k_top1",
    "get_imagenet_class_names",
    "get_probe_resolution",
    "get_top_k_predictions",
    "labels_are_in1k",
    "load_probe",
    # Scheduler
    "warmup_constant_scheduler",
    # Viewpoint
    "PixelBox",
    "Viewpoint",
    "make_eval_viewpoints",
    "viewpoint_to_pixel_box",
    # Viz
    "TimestepPredictions",
    "fit_pca",
    "imagenet_denormalize_to_numpy",
    "pca_rgb",
    "plot_multistep_pca",
    "plot_pca_grid",
    "plot_trajectory",
    "timestep_colors",
    "validate",
]
