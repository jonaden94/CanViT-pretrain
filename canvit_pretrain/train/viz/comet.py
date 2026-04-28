"""Figure / curve logging utilities. Backend-agnostic — `exp` is a Tracker."""

import gc
import io
import logging

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from PIL import Image as PILImage

from ..tracker import Tracker

log = logging.getLogger(__name__)

# Curve budget — Comet imposes a hard cap per experiment, and image-based
# wandb curves also accumulate storage. Skip silently once exhausted.
_curve_count = 0
_CURVE_BUDGET = 900


def log_curve(exp: Tracker, name: str, **kwargs) -> None:
    """Log curve with budget enforcement. Skips silently once exhausted."""
    global _curve_count
    if _curve_count >= _CURVE_BUDGET:
        if _curve_count == _CURVE_BUDGET:
            log.warning(f"Curve budget exhausted ({_CURVE_BUDGET}), skipping further curves")
            _curve_count += 1  # only warn once
        return
    try:
        exp.log_curve(name, **kwargs)
        _curve_count += 1
    except Exception as e:
        log.exception(f"Failed to log curve {name}: {e}")


def log_figure(exp: Tracker, fig: Figure, name: str, step: int) -> None:
    """Log matplotlib figure to the active tracker. Aggressively cleans up to prevent leaks."""
    try:
        with io.BytesIO() as buf:
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img = PILImage.open(buf)
            img.load()  # decode while buf is alive — both backends accept PIL.Image
        exp.log_image(img, name=name, step=step)
    except Exception as e:
        log.exception(f"Failed to log figure {name} at step {step}: {e}")
    finally:
        for ax in fig.axes:
            ax.clear()
        fig.clf()
        plt.close(fig)
        gc.collect()
