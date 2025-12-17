"""Viewpoint sampling for training and evaluation."""

import math
import random

import torch

from avp_vit.glimpse import Viewpoint


def random_viewpoint(
    B: int, device: torch.device, min_scale: float, max_scale: float
) -> Viewpoint:
    """Random viewpoint with log-uniform scale, center constrained to stay in bounds."""
    log_min, log_max = math.log(min_scale), math.log(max_scale)
    scales = torch.exp(torch.rand(B, device=device) * (log_max - log_min) + log_min)
    max_offset = (1 - scales).unsqueeze(1)
    centers = (torch.rand(B, 2, device=device) * 2 - 1) * max_offset
    return Viewpoint(name="random", centers=centers, scales=scales)


def make_eval_viewpoints(B: int, device: torch.device) -> list[Viewpoint]:
    """Full scene followed by 4 quadrants in shuffled order."""
    vps = [Viewpoint.full_scene(B, device)]
    quadrants = [(0, 0), (0, 1), (1, 0), (1, 1)]
    random.shuffle(quadrants)
    for qx, qy in quadrants:
        vps.append(Viewpoint.quadrant(B, device, qx, qy))
    return vps
