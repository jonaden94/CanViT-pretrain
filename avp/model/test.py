import torch
from torch import nn

from . import AVPConfig, AVPViT


class MockRopeEmbed(nn.Module):
    dtype: torch.dtype
    periods: torch.Tensor

    def __init__(self, head_dim: int):
        super().__init__()
        self.dtype = torch.float32
        self.register_buffer("periods", torch.ones(head_dim // 4))


class MockBlock(nn.Module):
    def forward(self, x, rope):
        return x  # identity


def test_avp_forward_shapes():
    embed_dim, num_heads, n_blocks = 64, 4, 2
    cfg = AVPConfig(scene_grid_size=4, glimpse_grid_size=3)

    blocks = nn.ModuleList([MockBlock() for _ in range(n_blocks)])
    rope_embed = MockRopeEmbed(embed_dim // num_heads)

    avp = AVPViT(
        blocks=blocks,
        rope_embed=rope_embed,
        embed_dim=embed_dim,
        num_heads=num_heads,
        n_prefix_tokens=1,
        cfg=cfg,
    )

    B, n_prefix, n_patches = 2, 1, 9
    local = torch.randn(B, n_prefix + n_patches, embed_dim)
    centers = torch.rand(B, 2)
    scales = torch.rand(B)

    out_local, out_scene = avp(local, centers, scales)

    assert out_local.shape == local.shape
    assert out_scene.shape == (B, 16, embed_dim)  # 4x4 scene grid


def test_gate_init():
    cfg = AVPConfig(scene_grid_size=4, gate_init=0.5)
    blocks = nn.ModuleList([MockBlock(), MockBlock()])
    rope_embed = MockRopeEmbed(16)

    avp = AVPViT(blocks, rope_embed, 64, 4, 1, cfg)

    for g in avp.read_gate:
        assert (g == 0.5).all()
    for g in avp.write_gate:
        assert (g == 0.5).all()
