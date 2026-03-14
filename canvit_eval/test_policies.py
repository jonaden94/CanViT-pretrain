"""Smoketests for evaluation policies."""

import torch
import pytest
from canvit_eval.policies import (
    StaticPolicy, EntropyGuidedC2F, make_eval_policy,
    _level_viewpoints, _tile_mean_uncertainty,
)


def test_level_viewpoints_counts():
    assert len(_level_viewpoints(0)) == 1
    assert len(_level_viewpoints(1)) == 4
    assert len(_level_viewpoints(2)) == 16


def test_level_viewpoints_scales():
    for cy, cx, s in _level_viewpoints(0):
        assert s == 1.0
    for cy, cx, s in _level_viewpoints(1):
        assert s == 0.5
    for cy, cx, s in _level_viewpoints(2):
        assert s == 0.25


def test_level_viewpoints_cover_scene():
    """All crops at a given level should tile [-1, 1]² without gaps."""
    for level in range(3):
        crops = _level_viewpoints(level)
        n = 2**level
        s = 1.0 / n
        centers_y = sorted(set(cy for cy, _, _ in crops))
        centers_x = sorted(set(cx for _, cx, _ in crops))
        assert len(centers_y) == n
        assert len(centers_x) == n
        # First crop center should be at -1 + s, last at 1 - s
        assert abs(centers_y[0] - (-1.0 + s)) < 1e-6
        assert abs(centers_y[-1] - (1.0 - s)) < 1e-6


def test_static_policy_returns_pregenerated():
    from canvit import Viewpoint
    B, device = 2, torch.device("cpu")
    vps = [
        Viewpoint(centers=torch.zeros(B, 2), scales=torch.ones(B)),
        Viewpoint(centers=torch.ones(B, 2) * 0.5, scales=torch.ones(B) * 0.5),
    ]
    pol = StaticPolicy("test", vps)
    assert pol.name == "test"
    v0 = pol.step(0, None)
    assert torch.equal(v0.centers, vps[0].centers)
    v1 = pol.step(1, None)
    assert torch.equal(v1.scales, vps[1].scales)


def test_make_eval_policy_c2f():
    pol = make_eval_policy("c2f", batch_size=4, device=torch.device("cpu"), n_viewpoints=5)
    assert pol.name == "coarse_to_fine"
    vp = pol.step(0, None)
    assert vp.centers.shape == (4, 2)
    assert vp.scales.shape == (4,)
    # t=0 of C2F should be full scene: scale=1, center=(0,0)
    assert (vp.scales == 1.0).all()


def test_make_eval_policy_aliases():
    pol_c2f = make_eval_policy("c2f", 2, torch.device("cpu"), 5)
    pol_iid = make_eval_policy("iid", 2, torch.device("cpu"), 5)
    pol_fullrand = make_eval_policy("fullrand", 2, torch.device("cpu"), 5)
    assert pol_c2f.name == "coarse_to_fine"
    assert pol_iid.name == "random"
    assert pol_fullrand.name == "full_then_random"


def test_make_eval_policy_unknown_raises():
    with pytest.raises(ValueError, match="Unknown policy"):
        make_eval_policy("nonexistent", 2, torch.device("cpu"), 5)


def test_tile_mean_uncertainty_shape():
    B, G = 3, 8
    uncertainty = torch.randn(B, G, G)
    crops = _level_viewpoints(1)  # 4 crops
    scores = _tile_mean_uncertainty(uncertainty, crops, G)
    assert scores.shape == (B, 4)


def test_tile_mean_uncertainty_uniform():
    """Uniform uncertainty should give equal scores across tiles."""
    B, G = 2, 32
    uncertainty = torch.ones(B, G, G)
    crops = _level_viewpoints(1)
    scores = _tile_mean_uncertainty(uncertainty, crops, G)
    assert torch.allclose(scores, scores[:, :1].expand_as(scores), atol=0.01)


def test_entropy_c2f_needs_21_viewpoints():
    with pytest.raises(AssertionError, match="n_viewpoints=21"):
        make_eval_policy("entropy_c2f", 2, torch.device("cpu"), 10,
                         probe=torch.nn.Identity(), get_spatial_fn=lambda x: x)
