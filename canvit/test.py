"""Tests for CanViT."""

import torch
from torch import Tensor, nn

from canvit import CanViT, CanViTConfig
from canvit.backbone import ViTBackbone
from canvit.rope import compute_rope, make_rope_periods


class MockBackbone(ViTBackbone, nn.Module):
    """Minimal backbone for testing."""

    def __init__(self, dim: int = 64, heads: int = 8, blocks: int = 6, patch_px: int = 16):
        nn.Module.__init__(self)
        self._dim = dim
        self._heads = heads
        self._blocks = blocks
        self._patch_px = patch_px
        self._periods = make_rope_periods(dim // heads, torch.float32)
        self.block_modules = nn.ModuleList([nn.Identity() for _ in range(blocks)])

    @property
    def embed_dim(self) -> int:
        return self._dim

    @property
    def num_heads(self) -> int:
        return self._heads

    @property
    def n_prefix_tokens(self) -> int:
        return 1

    @property
    def n_blocks(self) -> int:
        return self._blocks

    @property
    def patch_size_px(self) -> int:
        return self._patch_px

    @property
    def rope_periods(self) -> Tensor:
        return self._periods

    @property
    def rope_dtype(self) -> torch.dtype:
        return torch.float32

    def forward_block(self, idx: int, x: Tensor, rope: tuple[Tensor, Tensor] | None) -> Tensor:
        return self.block_modules[idx](x)

    def prepare_tokens(self, images: Tensor) -> tuple[Tensor, int, int]:
        B, _, H, W = images.shape
        gh, gw = H // self._patch_px, W // self._patch_px
        tokens = torch.randn(B, 1 + gh * gw, self._dim, device=images.device)
        return tokens, gh, gw


def test_canvit_init_canvas():
    backbone = MockBackbone(dim=64)
    cfg = CanViTConfig(n_registers=8)
    model = CanViT(backbone, cfg)

    canvas = model.init_canvas(batch_size=2, canvas_grid_size=4)
    # 1 cls + 8 registers + 16 spatial = 25
    assert canvas.shape == (2, 1 + 8 + 16, 64)


def test_canvit_forward():
    backbone = MockBackbone(dim=64, heads=8, blocks=6, patch_px=16)
    cfg = CanViTConfig(n_registers=8, adapter_stride=2, layer_scale_init=1e-3)
    model = CanViT(backbone, cfg)

    B, D = 2, 64
    canvas_grid = 4
    n_spatial = canvas_grid ** 2
    n_local = 17  # 1 prefix + 16 patches

    local = torch.randn(B, n_local, D)
    canvas = model.init_canvas(B, canvas_grid)

    local_pos = torch.randn(B, n_local - 1, 2)
    canvas_pos = torch.randn(B, n_spatial, 2)
    local_rope = compute_rope(local_pos, backbone.rope_periods)
    canvas_rope = compute_rope(canvas_pos, backbone.rope_periods)

    local_out, canvas_out = model(local, canvas, local_rope, canvas_rope)
    assert local_out.shape == local.shape
    assert canvas_out.shape == canvas.shape


def test_canvit_n_adapters():
    backbone = MockBackbone(blocks=12)
    cfg = CanViTConfig(adapter_stride=2)
    model = CanViT(backbone, cfg)
    assert model.n_adapters == 5  # (12-1) // 2 = 5

    cfg3 = CanViTConfig(adapter_stride=3)
    model3 = CanViT(backbone, cfg3)
    assert model3.n_adapters == 3  # (12-1) // 3 = 3


def test_canvit_gradients_flow():
    backbone = MockBackbone(dim=64, heads=8, blocks=4, patch_px=16)
    cfg = CanViTConfig(n_registers=4, adapter_stride=2)
    model = CanViT(backbone, cfg)

    B, D = 2, 64
    canvas_grid = 4
    n_local = 17

    local = torch.randn(B, n_local, D, requires_grad=True)
    canvas = model.init_canvas(B, canvas_grid).detach().requires_grad_(True)

    local_rope = compute_rope(torch.randn(B, n_local - 1, 2), backbone.rope_periods)
    canvas_rope = compute_rope(torch.randn(B, canvas_grid ** 2, 2), backbone.rope_periods)

    local_out, canvas_out = model(local, canvas, local_rope, canvas_rope)
    loss = local_out.sum() + canvas_out.sum()
    loss.backward()

    assert local.grad is not None
    assert canvas.grad is not None
    assert local.grad.abs().sum() > 0
    assert canvas.grad.abs().sum() > 0
