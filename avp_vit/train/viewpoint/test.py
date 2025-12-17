"""Tests for viewpoint sampling."""

import torch

from avp_vit.train.viewpoint import make_eval_viewpoints, random_viewpoint


def test_random_viewpoint_shapes():
    B = 8
    vp = random_viewpoint(B, torch.device("cpu"), min_scale=0.3, max_scale=1.0)
    assert vp.centers.shape == (B, 2)
    assert vp.scales.shape == (B,)


def test_random_viewpoint_bounds():
    B = 100
    vp = random_viewpoint(B, torch.device("cpu"), min_scale=0.3, max_scale=0.7)
    assert (vp.scales >= 0.3).all()
    assert (vp.scales <= 0.7).all()


def test_eval_viewpoints():
    vps = make_eval_viewpoints(4, torch.device("cpu"))
    assert len(vps) == 5  # full + 4 quadrants
    assert vps[0].name == "full"
    assert {vp.name for vp in vps[1:]} == {"TL", "TR", "BL", "BR"}
