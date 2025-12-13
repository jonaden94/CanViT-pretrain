import torch

from . import CrossAttention


def test_shapes():
    B, N_q, N_kv, D, heads = 2, 10, 20, 64, 4
    attn = CrossAttention(D, heads)
    q = torch.randn(B, N_q, D)
    kv = torch.randn(B, N_kv, D)
    out = attn(q, kv)
    assert out.shape == (B, N_q, D)


def test_with_rope():
    B, N_q, N_kv, D, heads = 2, 10, 20, 64, 4
    head_dim = D // heads
    attn = CrossAttention(D, heads)
    q = torch.randn(B, N_q, D)
    kv = torch.randn(B, N_kv, D)
    q_rope = (torch.randn(B, 1, N_q, head_dim), torch.randn(B, 1, N_q, head_dim))
    kv_rope = (torch.randn(B, 1, N_kv, head_dim), torch.randn(B, 1, N_kv, head_dim))
    out = attn(q, kv, q_rope=q_rope, kv_rope=kv_rope)
    assert out.shape == (B, N_q, D)
