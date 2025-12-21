"""Viewpoint sampling for training and evaluation."""

import random

import torch

from avp_vit.glimpse import Viewpoint


def random_viewpoint(
    B: int,
    device: torch.device,
    min_scale: float = 0.0,
    max_scale: float = 1.0,
) -> Viewpoint:
    """Sample random viewpoints with uniform safe-box-area distribution.

    Geometry
    --------
    A viewpoint has center (x, y) ∈ [-1, 1]² and scale s ∈ [min_scale, max_scale].
    The constraint is: |x| + s ≤ 1 and |y| + s ≤ 1 (viewpoint must fit in scene).

    Given scale s, the valid centers form a "safe box": [-(1-s), (1-s)]²
    with area A = 4·(1-s)².

    Derivation
    ----------
    We want to sample UNIFORMLY OVER SAFE-BOX AREA. Why? Because:
    - Large s → small safe box → few center choices → should be rarer
    - Small s → large safe box → many center choices → should be more common
    Uniform over A balances "degrees of freedom" for center placement.

    Let L = 1 - s be the safe-box half-width. Then A ∝ L².
    For uniform area sampling within [min_scale, max_scale]:
    - L_max = 1 - min_scale (largest safe box, at smallest scale)
    - L_min = 1 - max_scale (smallest safe box, at largest scale)
    - Sample L² uniformly in [L_min², L_max²]

    Then sample center uniformly in the safe box: x, y ~ Uniform(-L, L).
    """
    assert 0.0 <= min_scale <= max_scale <= 1.0

    L_min = 1 - max_scale
    L_max = 1 - min_scale

    # Sample L² uniformly in [L_min², L_max²], then take sqrt
    u = torch.rand(B, device=device)
    L_sq = L_min**2 + u * (L_max**2 - L_min**2)
    L = torch.sqrt(L_sq)

    scales = 1 - L

    # Sample center uniformly within each sample's safe box [-L, L]²
    centers = (torch.rand(B, 2, device=device) * 2 - 1) * L.unsqueeze(1)

    return Viewpoint(name="random", centers=centers, scales=scales)


def make_eval_viewpoints(B: int, device: torch.device) -> list[Viewpoint]:
    """Full scene followed by 4 quadrants in shuffled order.

    Intentionally non-deterministic: quadrant order varies each call to test
    that the model generalizes across orderings, not just a fixed sequence.
    Uses Python random (not torch) since reproducibility is NOT desired here.
    """
    vps = [Viewpoint.full_scene(B, device)]
    quadrants = [(0, 0), (0, 1), (1, 0), (1, 1)]
    random.shuffle(quadrants)
    for qx, qy in quadrants:
        vps.append(Viewpoint.quadrant(B, device, qx, qy))
    return vps
