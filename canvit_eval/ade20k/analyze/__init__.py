"""Post-hoc analysis of ADE20K evaluation results.

Loads .pt files from SLURM jobs, computes bootstrap confidence intervals.
Runs on CPU - no CUDA needed.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

FeatureType = Literal["hidden", "predicted_norm", "teacher_glimpse", "teacher_full"]


@dataclass
class PolicyStats:
    """Aggregated statistics for one policy."""

    policy: str
    n_images: int
    n_timesteps: int
    # Per-timestep aggregated mIoU (from IoUAccumulator)
    mious: np.ndarray  # [T]
    # Bootstrap CI from per-image variation
    ci_low: np.ndarray  # [T]
    ci_high: np.ndarray  # [T]
    # Raw per-image data for further analysis
    per_image: np.ndarray  # [N, T]


def load_results(paths: list[Path]) -> dict:
    """Load and merge results from multiple .pt files.

    Returns dict with:
      - per_image_mious: {feat: [N_total, T]} concatenated across files
      - viewpoints: [N_total, T, 3] concatenated
      - metadata: from first file
    """
    all_per_image: dict[str, list[torch.Tensor]] = {}
    all_viewpoints: list[torch.Tensor] = []
    metadata = None

    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if metadata is None:
            metadata = data["metadata"]
        for feat, tensor in data["per_image_mious"].items():
            all_per_image.setdefault(feat, []).append(tensor)
        if data["viewpoints"] is not None:
            all_viewpoints.append(data["viewpoints"])

    return {
        "per_image_mious": {k: torch.cat(v, dim=0) for k, v in all_per_image.items()},
        "viewpoints": torch.cat(all_viewpoints, dim=0) if all_viewpoints else None,
        "metadata": metadata,
        "n_files": len(paths),
    }


def bootstrap_ci(
    per_image: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute bootstrap confidence interval for mIoU.

    Args:
        per_image: [N, T] per-image mIoU values
        n_bootstrap: number of bootstrap samples
        ci: confidence interval (0.95 = 95%)
        seed: random seed for reproducibility

    Returns:
        (ci_low, ci_high): [T] arrays
    """
    rng = np.random.default_rng(seed)
    N, T = per_image.shape
    means = np.empty((n_bootstrap, T))

    for i in range(n_bootstrap):
        indices = rng.choice(N, size=N, replace=True)
        means[i] = per_image[indices].mean(axis=0)

    alpha = (1 - ci) / 2
    ci_low = np.percentile(means, 100 * alpha, axis=0)
    ci_high = np.percentile(means, 100 * (1 - alpha), axis=0)
    return ci_low, ci_high


def compute_stats(
    paths: list[Path],
    feature: FeatureType = "predicted_norm",
    n_bootstrap: int = 1000,
) -> PolicyStats:
    """Compute aggregated statistics with bootstrap CI for a policy.

    Args:
        paths: list of .pt files from SLURM jobs (same policy)
        feature: which feature type to analyze
        n_bootstrap: number of bootstrap samples

    Returns:
        PolicyStats with mIoU curve and confidence intervals
    """
    data = load_results(paths)
    per_image = data["per_image_mious"][feature].numpy()  # [N, T]
    N, T = per_image.shape

    # Mean mIoU per timestep
    mious = per_image.mean(axis=0)

    # Bootstrap CI
    ci_low, ci_high = bootstrap_ci(per_image, n_bootstrap=n_bootstrap)

    # Extract policy name from metadata
    cfg = data["metadata"].get("config", {})
    policy = cfg.get("policy", "unknown")
    # Only append _startfull for random policy (c2f ignores this flag)
    if policy == "random" and cfg.get("start_full"):
        policy = "random_startfull"

    return PolicyStats(
        policy=policy,
        n_images=N,
        n_timesteps=T,
        mious=mious,
        ci_low=ci_low,
        ci_high=ci_high,
        per_image=per_image,
    )


def find_result_files(output_dir: Path, policy_prefix: str) -> list[Path]:
    """Find all .pt result files for a given policy prefix."""
    return sorted(output_dir.glob(f"ade20k_{policy_prefix}_*.pt"))
