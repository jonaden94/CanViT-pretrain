"""Image transformation utilities for visualization."""

from canvit_pytorch.preprocess import imagenet_denormalize
from torch import Tensor


def imagenet_denormalize_to_numpy(img: Tensor):
    """Denormalize ImageNet-normalized tensor and return [H, W, C] numpy in [0, 1]."""
    return imagenet_denormalize(img).detach().cpu().permute(1, 2, 0).numpy()


__all__ = ["imagenet_denormalize_to_numpy"]
