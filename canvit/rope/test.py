"""Tests for RoPE utilities."""

import torch


def test_make_rope_periods():
    from canvit.rope import make_rope_periods
    periods = make_rope_periods(head_dim=64, dtype=torch.float32)
    assert periods.shape == (16,)  # head_dim // 4
    assert periods[0] == 1.0  # base^0 = 1


def test_grid_offsets():
    from canvit.rope import grid_offsets
    offsets = grid_offsets(2, 3, torch.device("cpu"), torch.float32)
    assert offsets.shape == (6, 2)
    assert offsets.min() >= -1 and offsets.max() <= 1


def test_make_grid_positions():
    from canvit.rope import make_grid_positions
    pos = make_grid_positions(4, 4, torch.device("cpu"), torch.float32)
    assert pos.shape == (16, 2)


def test_glimpse_positions():
    from canvit.rope import glimpse_positions
    centers = torch.zeros(2, 2)  # B=2, centered
    scales = torch.ones(2)  # full scale
    pos = glimpse_positions(centers, scales, 4, 4, torch.float32)
    assert pos.shape == (2, 16, 2)


def test_compute_rope():
    from canvit.rope import compute_rope, make_rope_periods
    positions = torch.randn(2, 16, 2)
    periods = make_rope_periods(64, torch.float32)
    sin, cos = compute_rope(positions, periods)
    assert sin.shape == cos.shape == (2, 1, 16, 64)


def test_rope_rotate_half():
    from canvit.rope import rope_rotate_half
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rotated = rope_rotate_half(x)
    assert torch.allclose(rotated, torch.tensor([-3.0, -4.0, 1.0, 2.0]))


def test_rope_apply():
    from canvit.rope import rope_apply
    x = torch.ones(1, 1, 4, 8)
    sin = torch.zeros(1, 1, 4, 8)
    cos = torch.ones(1, 1, 4, 8)
    out = rope_apply(x, sin, cos)
    assert torch.allclose(out, x)  # cos=1, sin=0 => identity


def test_rope_apply_with_prefix():
    from canvit.rope import rope_apply_with_prefix
    x = torch.ones(2, 4, 10, 8)  # 10 tokens
    sin = torch.zeros(2, 1, 8, 8)  # 8 spatial tokens
    cos = torch.ones(2, 1, 8, 8)
    out = rope_apply_with_prefix(x, (sin, cos))
    assert out.shape == x.shape
    assert torch.allclose(out[:, :, :2], x[:, :, :2])  # prefix unchanged
