"""Tests for position-aware running normalization."""

import torch

from avp_vit.glimpse import Viewpoint
from avp_vit.train.norm import PositionAwareNorm


def test_output_shape():
    norm = PositionAwareNorm(n_tokens=16, embed_dim=64, grid_size=4)
    x = torch.randn(8, 16, 64)
    y = norm(x)
    assert y.shape == x.shape


def test_first_batch_initializes_stats():
    norm = PositionAwareNorm(n_tokens=16, embed_dim=64, grid_size=4)
    norm.train()
    assert not norm.initialized

    x = torch.randn(8, 16, 64)
    norm(x)

    assert norm.initialized
    assert norm.mean.shape == (16, 64)
    assert norm.var.shape == (16, 64)


def test_running_stats_update():
    norm = PositionAwareNorm(n_tokens=16, embed_dim=64, grid_size=4, momentum=0.5)
    norm.train()

    x1 = torch.randn(8, 16, 64)
    norm(x1)
    mean_after_first = norm.mean.clone()

    x2 = torch.randn(8, 16, 64) + 10  # Shift by 10
    norm(x2)

    # Mean should have moved toward new batch
    assert not torch.allclose(norm.mean, mean_after_first)


def test_eval_mode_no_update():
    norm = PositionAwareNorm(n_tokens=16, embed_dim=64, grid_size=4)
    norm.train()

    x = torch.randn(8, 16, 64)
    norm(x)
    mean_after_train = norm.mean.clone()

    norm.eval()
    x2 = torch.randn(8, 16, 64) + 100
    norm(x2)

    # Mean should NOT have changed
    assert torch.allclose(norm.mean, mean_after_train)


def test_normalize_at_viewpoint_shape():
    norm = PositionAwareNorm(n_tokens=16, embed_dim=64, grid_size=4)
    norm.train()
    norm(torch.randn(8, 16, 64))  # Initialize stats

    glimpse_grid_size = 3
    x = torch.randn(glimpse_grid_size ** 2, 64)
    vp = Viewpoint.full_scene(1, x.device)

    y = norm.normalize_at_viewpoint(x, vp, glimpse_grid_size)
    assert y.shape == x.shape


def test_normalize_at_viewpoint_quadrant():
    norm = PositionAwareNorm(n_tokens=64, embed_dim=32, grid_size=8)
    norm.train()
    norm(torch.randn(4, 64, 32))

    glimpse_grid_size = 4
    x = torch.randn(glimpse_grid_size ** 2, 32)
    vp = Viewpoint.quadrant(1, x.device, 0, 0)  # Top-left quadrant

    y = norm.normalize_at_viewpoint(x, vp, glimpse_grid_size)
    assert y.shape == x.shape


def test_state_dict_contains_buffers():
    norm = PositionAwareNorm(n_tokens=16, embed_dim=64, grid_size=4)
    sd = norm.state_dict()
    assert "mean" in sd
    assert "var" in sd
