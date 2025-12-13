"""Unified AVP tests against all backbones."""

import pytest
import torch
from dinov3.models.vision_transformer import vit_small as dinov3_vit_small
from ijepa.models.vision_transformer import vit_small as ijepa_vit_small

from avp_vit import AVPConfig, AVPViT
from avp_vit.backbone import ViTBackbone
from avp_vit.backbone.dinov3 import DINOv3Backbone
from avp_vit.backbone.ijepa import IJEPABackbone
from avp_vit.rope import compute_rope, glimpse_positions


@pytest.fixture(params=["dinov3", "ijepa"])
def backbone(request: pytest.FixtureRequest) -> ViTBackbone:
    torch.manual_seed(42)
    if request.param == "dinov3":
        native = dinov3_vit_small(img_size=112, patch_size=16)
        native.init_weights()
        return DINOv3Backbone(native)
    else:
        native = ijepa_vit_small(img_size=[112], patch_size=16)
        return IJEPABackbone(native)


def test_avp_identity_init(backbone: ViTBackbone) -> None:
    """With γ=0, AVP should be identity: local = backbone(local), scene = scene."""
    cfg = AVPConfig(scene_grid_size=8, glimpse_grid_size=7, gate_init=0.0)
    avp = AVPViT(backbone, cfg)

    B, H, W = 2, 7, 7
    local = torch.randn(B, backbone.n_prefix_tokens + H * W, backbone.embed_dim)
    centers = torch.zeros(B, 2)
    scales = torch.ones(B)

    positions = glimpse_positions(centers, scales, H, W, dtype=backbone.rope_dtype)
    rope = compute_rope(positions, backbone.rope_periods)

    expected = local.clone()
    for i in range(backbone.n_blocks):
        expected = backbone.forward_block(i, expected, rope)

    actual_local, actual_scene = avp(local.clone(), centers, scales)

    assert torch.allclose(actual_local, expected, atol=1e-5)
    assert torch.allclose(actual_scene, avp.scene_tokens.expand(B, -1, -1), atol=1e-5)


def test_avp_forward_shapes(backbone: ViTBackbone) -> None:
    """AVP forward produces correct output shapes."""
    cfg = AVPConfig(scene_grid_size=8, glimpse_grid_size=7)
    avp = AVPViT(backbone, cfg)

    B, H, W = 2, 7, 7
    local = torch.randn(B, backbone.n_prefix_tokens + H * W, backbone.embed_dim)
    centers = torch.rand(B, 2) * 2 - 1
    scales = torch.rand(B) * 0.5 + 0.5

    out_local, out_scene = avp(local, centers, scales)

    assert out_local.shape == local.shape
    assert out_scene.shape == (B, 64, backbone.embed_dim)


def test_different_glimpses_differ(backbone: ViTBackbone) -> None:
    """Different glimpse positions should produce different outputs."""
    cfg = AVPConfig(scene_grid_size=8, glimpse_grid_size=7, gate_init=1.0)
    avp = AVPViT(backbone, cfg)

    B, H, W = 2, 7, 7
    local = (
        torch.randn(1, backbone.n_prefix_tokens + H * W, backbone.embed_dim)
        .expand(B, -1, -1)
        .clone()
    )
    centers = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
    scales = torch.tensor([0.3, 0.7])

    out_local, out_scene = avp(local, centers, scales)

    assert not torch.allclose(out_local[0], out_local[1], atol=1e-3)
    assert not torch.allclose(out_scene[0], out_scene[1], atol=1e-3)
