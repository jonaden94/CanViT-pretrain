"""Distill: passive-to-active dense latent distillation from DINOv3 — the pretraining
objective, and this repo's oldest task.

Layout matches the other two tasks (``ade20k/``, ``in1k/``):

- ``task.py``   — the harness adapter (build model, loaders, eval, viz hooks)
- ``loss.py``   — ``DistillTask``, the per-step loss object the rollout engine drives
- ``config.py`` — ``Config``, distill's own knobs (teacher, canvas/glimpse sizes, schedule)
- ``model.py``  — student construction / teacher wiring
- ``probe.py``  — the frozen IN1k probes used to score distill validation
- ``data/``     — the in21k feature webdataset, shard bookkeeping, tar readers
- ``viz/``      — distill's validation figures

Anything shared with ade20k/in1k (viewpoint sampling, selectors, RL objectives,
schedulers, EMA, tracker, the ``FoveatedScaleConfig`` / ``JointPolicyConfig`` knobs)
lives in ``canvit_train.harness`` instead — see that package's docstring.

Until 2026-07-31 all of this sat in a folder called ``train/`` together with the shared
substrate, because distill *was* the whole repo and so never needed a task-name prefix.
"""

from canvit_train.distill.data import (
    Batch,
    InfiniteLoader,
    Loaders,
    create_loaders,
    scene_size_px,
)
from canvit_train.distill.probe import (
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
from canvit_train.distill.viz import (
    TimestepPredictions,
    imagenet_denormalize_to_numpy,
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
    # Viz
    "TimestepPredictions",
    "imagenet_denormalize_to_numpy",
    "plot_multistep_pca",
    "plot_pca_grid",
    "plot_trajectory",
    "timestep_colors",
    "validate",
]
