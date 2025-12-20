"""Viewpoint sampling for training and evaluation."""

import random

import torch

from avp_vit.glimpse import Viewpoint


def random_viewpoint(B: int, device: torch.device) -> Viewpoint:
    """Sample random viewpoints with uniform safe-box-area distribution.

    Geometry
    --------
    A viewpoint has center (x, y) ∈ [-1, 1]² and scale s ∈ (0, 1].
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
    For A to be uniform on [0, 1], we need L² ~ Uniform(0, 1).
    So L = sqrt(U) where U ~ Uniform(0, 1).

    Therefore:
        L = sqrt(U)
        s = 1 - L = 1 - sqrt(U)

    Then sample center uniformly in the safe box: x, y ~ Uniform(-L, L).

    Properties
    ----------
    - Scale distribution: PDF ∝ (1-s), biased toward smaller scales
    - Center distribution: clustered toward origin (large s forces small L)
    - Constraint |center| + scale ≤ 1 satisfied BY CONSTRUCTION
    """
    # Sample safe-box half-width L such that L² is uniform.
    # L² ~ Uniform(0, 1)  →  L = sqrt(U)
    u = torch.rand(B, device=device)
    L = torch.sqrt(u)

    # Scale is complement of safe-box half-width: s = 1 - L
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
