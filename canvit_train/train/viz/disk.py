"""Disk-side visualization sink: save figures under run_dir/visualization/."""

import gc
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

log = logging.getLogger(__name__)


def save_figure(fig: Figure, run_dir: Path, subdir: str, step: int) -> None:
    """Write a Figure to {run_dir}/visualization/{subdir}/step-{step}.png.

    Cleans up matplotlib state aggressively to prevent the leaks that the
    previous wandb-backed log_figure() had to defend against.
    """
    out_dir = run_dir / "visualization" / subdir
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"step-{step}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    except Exception as e:
        log.exception(f"Failed to save figure to {out_dir} at step {step}: {e}")
    finally:
        for ax in fig.axes:
            ax.clear()
        fig.clf()
        plt.close(fig)
        gc.collect()


def plot_combined_curves(
    *,
    scene_cos_raw: list[float] | None,
    scene_cos_norm: list[float] | None,
    cls_cos_raw: list[float] | None,
    cls_cos_norm: list[float] | None,
    in1k_accs: list[float] | None,
) -> Figure:
    """Build a 2x3 figure with the 5 validation curves vs eval-trajectory timestep.

    Layout:
        | scene_cos_raw  | scene_cos_norm | in1k_tts_top1 |
        | cls_cos_raw    | cls_cos_norm   |   (unused)    |

    Missing inputs (None or empty list) render as a grayed-out subplot with an
    "N/A" annotation, so the layout stays fixed across runs.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    panels = [
        (axes[0, 0], "scene_cos_raw_vs_timestep",     scene_cos_raw,  (-1.0, 1.0)),
        (axes[0, 1], "scene_cos_norm_vs_timestep",    scene_cos_norm, (-1.0, 1.0)),
        (axes[0, 2], "in1k_tts_top1_vs_timestep",     in1k_accs,      (0.0, 1.0)),
        (axes[1, 0], "cls_cos_raw_vs_timestep",       cls_cos_raw,    (-1.0, 1.0)),
        (axes[1, 1], "cls_cos_norm_vs_timestep",      cls_cos_norm,   (-1.0, 1.0)),
    ]
    for ax, title, data, ylim in panels:
        ax.set_title(title)
        ax.set_xlabel("timestep")
        if data:
            ax.plot(range(len(data)), data, marker="o")
            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, color="gray", fontsize=14)
            ax.set_facecolor((0.95, 0.95, 0.95))
            ax.set_xticks([])
            ax.set_yticks([])

    # Empty bottom-right slot
    axes[1, 2].axis("off")

    fig.tight_layout()
    return fig
