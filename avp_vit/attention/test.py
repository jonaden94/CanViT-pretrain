import torch
from torch import nn

from avp_vit.attention import AttentionConfig, RoPEReadCrossAttention, RoPEWriteCrossAttention


def _default_cfg() -> AttentionConfig:
    return AttentionConfig()


def test_read_shapes():
    B, N_q, N_kv, D, heads = 2, 10, 20, 64, 4
    head_dim = D // heads
    attn = RoPEReadCrossAttention(D, heads, _default_cfg())
    q = torch.randn(B, N_q, D)
    kv = torch.randn(B, N_kv, D)
    q_rope = (torch.randn(B, 1, N_q, head_dim), torch.randn(B, 1, N_q, head_dim))
    kv_rope = (torch.randn(B, 1, N_kv, head_dim), torch.randn(B, 1, N_kv, head_dim))
    out = attn(q, kv, q_rope, kv_rope)
    assert out.shape == (B, N_q, D)


def test_write_shapes():
    B, N_q, N_kv, D, heads = 2, 20, 10, 64, 4
    head_dim = D // heads
    attn = RoPEWriteCrossAttention(D, heads, _default_cfg())
    q = torch.randn(B, N_q, D)
    kv = torch.randn(B, N_kv, D)
    q_rope = (torch.randn(B, 1, N_q, head_dim), torch.randn(B, 1, N_q, head_dim))
    kv_rope = (torch.randn(B, 1, N_kv, head_dim), torch.randn(B, 1, N_kv, head_dim))
    out = attn(q, kv, q_rope, kv_rope)
    assert out.shape == (B, N_q, D)


def test_write_v_identity_init():
    """V projection starts as identity when identity_init_v=True."""
    D = 64
    cfg = AttentionConfig(identity_init_v=True)
    attn = RoPEWriteCrossAttention(D, num_heads=4, cfg=cfg)
    v = attn.v_transform
    assert isinstance(v, nn.Linear)
    assert torch.allclose(v.weight, torch.eye(D))
    assert torch.allclose(v.bias, torch.zeros(D))


def test_write_v_default_init():
    """V projection uses default init when identity_init_v=False."""
    D = 64
    cfg = AttentionConfig(identity_init_v=False)
    attn = RoPEWriteCrossAttention(D, num_heads=4, cfg=cfg)
    v = attn.v_transform
    assert isinstance(v, nn.Linear)
    assert not torch.allclose(v.weight, torch.eye(D))


def test_flops_read():
    """Read attention: Q and O projections on queries."""
    D = 64
    attn = RoPEReadCrossAttention(D, num_heads=4, cfg=_default_cfg())
    f = attn.flops(n_q=10, n_kv=20)
    # attention + Q proj + O proj
    assert f == 4 * 10 * 20 * D + 2 * 10 * D * D + 2 * 10 * D * D


def test_flops_write():
    """Write attention: K and V projections on keys/values."""
    D = 64
    attn = RoPEWriteCrossAttention(D, num_heads=4, cfg=_default_cfg())
    f = attn.flops(n_q=20, n_kv=10)
    # attention + K proj + V proj
    assert f == 4 * 20 * 10 * D + 2 * 10 * D * D + 2 * 10 * D * D


def test_flops_projection_placement_matters():
    """Which tokens get projected affects FLOPs significantly."""
    D = 64
    cfg = _default_cfg()
    read = RoPEReadCrossAttention(D, 4, cfg)
    write = RoPEWriteCrossAttention(D, 4, cfg)
    # Asymmetric: 10 queries, 100 keys/values
    read_f = read.flops(n_q=10, n_kv=100)
    write_f = write.flops(n_q=10, n_kv=100)
    # Read projects small tensor (queries), write projects large tensor (keys/values)
    assert write_f > read_f * 4


def test_pre_affine_enabled():
    """Pre-transform EWAs are ElementwiseAffine when use_pre_affine=True."""
    from ytch.nn.elementwise_affine import ElementwiseAffine

    D = 64
    cfg = AttentionConfig(use_pre_affine=True)
    attn = RoPEReadCrossAttention(D, 4, cfg)
    assert isinstance(attn.affine_q, ElementwiseAffine)
    assert isinstance(attn.affine_k, ElementwiseAffine)
    assert isinstance(attn.affine_v, ElementwiseAffine)


def test_pre_affine_disabled():
    """Pre-transform EWAs are Identity when use_pre_affine=False."""
    D = 64
    cfg = AttentionConfig(use_pre_affine=False)
    attn = RoPEReadCrossAttention(D, 4, cfg)
    assert isinstance(attn.affine_q, nn.Identity)
    assert isinstance(attn.affine_k, nn.Identity)
    assert isinstance(attn.affine_v, nn.Identity)


def test_post_rope_affine_enabled():
    """Post-RoPE EWAs are ElementwiseAffine when use_post_rope_affine=True."""
    from ytch.nn.elementwise_affine import ElementwiseAffine

    D, heads = 64, 4
    cfg = AttentionConfig(use_post_rope_affine=True)
    attn = RoPEReadCrossAttention(D, heads, cfg)
    assert isinstance(attn.post_rope_q, ElementwiseAffine)
    assert isinstance(attn.post_rope_k, ElementwiseAffine)


def test_post_rope_affine_disabled():
    """Post-RoPE EWAs are Identity when use_post_rope_affine=False."""
    D = 64
    cfg = AttentionConfig(use_post_rope_affine=False)
    attn = RoPEReadCrossAttention(D, 4, cfg)
    assert isinstance(attn.post_rope_q, nn.Identity)
    assert isinstance(attn.post_rope_k, nn.Identity)
