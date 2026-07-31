"""Task-agnostic visualization leaves: PCA projection, figure I/O, feature metrics.

These three modules are pure (only third-party imports) and are used by more than one
task — ``ade20k/viz.py`` needs ``fit_pca``, ``ade20k/task.py`` needs ``save_figure`` —
so they sit with the harness. Everything that is specific to how DISTILL renders its
validation figures (sample extraction, multistep PCA plots, the foveated overlay,
``validate``) lives in ``distill/viz/``.
"""

from .disk import plot_combined_curves, save_figure
from .metrics import cosine_dissimilarity
from .pca import fit_pca, pca_rgb

__all__ = [
    "cosine_dissimilarity",
    "fit_pca",
    "pca_rgb",
    "plot_combined_curves",
    "save_figure",
]
