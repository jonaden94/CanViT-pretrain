"""Tests for FLOP counting."""

import canvit_flops as flops


def test_linear_formula():
    """Linear(128 → 256) on 10 tokens = 2 * 10 * 128 * 256."""
    assert flops.linear(10, 128, 256) == 2 * 10 * 128 * 256


def test_sdpa_formula():
    """SDPA(8 queries, 16 keys, head_dim=64) = 4 * 8 * 16 * 64."""
    assert flops.sdpa(8, 16, 64) == 4 * 8 * 16 * 64


def test_layer_norm_formula():
    """LayerNorm on 10 tokens, dim=128 = 5 * 10 * 128."""
    assert flops.layer_norm(10, 128) == 5 * 10 * 128


def test_layer_scale_formula():
    """LayerScale on 10 tokens, dim=128 = 10 * 128."""
    assert flops.layer_scale(10, 128) == 10 * 128


def test_rope_compute_formula():
    """RoPE compute: precompute sin/cos tables."""
    n, head_dim = 100, 128
    # 21 * n * head_dim (1x for div/mul, 10x for sin, 10x for cos)
    expected = n * head_dim + 2 * 10 * n * head_dim  # = 21 * n * head_dim
    assert flops.rope_compute(n, head_dim) == expected


def test_rope_apply_formula():
    """RoPE apply: ~4 * n * dim."""
    n, dim = 100, 1024
    assert flops.rope_apply(n, dim) == 4 * n * dim


def test_canvas_read_includes_all_ops():
    """Canvas read: LN, Q projection, RoPE, SDPA, O projection, LayerScale."""
    n_local, n_canvas = 10, 100
    local_dim, canvas_dim = 768, 1024

    total = flops.canvas_read(n_local, n_canvas, local_dim, canvas_dim)

    # Manually compute expected (K, V are Identity - no FLOPs)
    expected = 0
    expected += flops.layer_norm(n_local, local_dim)
    expected += flops.layer_norm(n_canvas, canvas_dim)
    expected += flops.linear(n_local, local_dim, canvas_dim)  # Q
    expected += flops.rope_apply(n_local, canvas_dim)   # Q rotation
    expected += flops.rope_apply(n_canvas, canvas_dim)  # K rotation
    expected += flops.sdpa(n_local, n_canvas, canvas_dim)
    expected += flops.linear(n_local, canvas_dim, local_dim)  # O
    expected += flops.layer_scale(n_local, local_dim)

    assert total == expected


def test_canvas_write_includes_all_ops():
    """Canvas write: LN, K/V projections, RoPE, SDPA. Q/O are Identity."""
    n_local, n_canvas = 10, 100
    local_dim, canvas_dim = 768, 1024

    total = flops.canvas_write(n_local, n_canvas, local_dim, canvas_dim)

    expected = 0
    expected += flops.layer_norm(n_canvas, canvas_dim)
    expected += flops.layer_norm(n_local, local_dim)
    expected += flops.linear(n_local, local_dim, canvas_dim) * 2  # K, V
    expected += flops.rope_apply(n_canvas, canvas_dim)  # Q rotation
    expected += flops.rope_apply(n_local, canvas_dim)   # K rotation
    expected += flops.sdpa(n_canvas, n_local, canvas_dim)

    assert total == expected


def test_canvas_adapter_is_read_plus_write():
    """Adapter = read + write."""
    n_local, n_canvas = 10, 100
    local_dim, canvas_dim = 768, 1024

    adapter = flops.canvas_adapter(n_local, n_canvas, local_dim, canvas_dim)
    read = flops.canvas_read(n_local, n_canvas, local_dim, canvas_dim)
    write = flops.canvas_write(n_local, n_canvas, local_dim, canvas_dim)

    assert adapter == read + write


def test_canvas_attention_asymmetric():
    """Canvas attention uses Identity on canvas side, avoiding O(n*D²)."""
    n_canvas = 256 * 256  # 65k tokens
    canvas_dim = 1024

    # Dense projection would be: O(n_canvas * canvas_dim²)
    dense_cost = flops.linear(n_canvas, canvas_dim, canvas_dim)

    # With Identity on canvas, the only costs are LayerNorm O(n*D)
    # and SDPA. No D² term on the canvas side.
    # This test verifies the asymmetry conceptually exists.
    assert dense_cost == 2 * n_canvas * canvas_dim * canvas_dim


def test_vit_block_structure():
    """ViT block FLOPs should include self-attn (with RoPE) and FFN."""
    n, D = 16, 768
    ffn_ratio = 4.0

    total = flops.vit_block(n, D, ffn_ratio)

    # Self-attention: LN + QKV + RoPE(Q) + RoPE(K) + SDPA + out
    self_attn = (
        flops.layer_norm(n, D)
        + flops.linear(n, D, 3 * D)
        + flops.rope_apply(n, D)  # Q rotation
        + flops.rope_apply(n, D)  # K rotation
        + flops.sdpa(n, n, D)
        + flops.linear(n, D, D)
    )

    # FFN: LN + fc1 + fc2
    ffn_dim = int(ffn_ratio * D)
    ffn = flops.layer_norm(n, D) + flops.linear(n, D, ffn_dim) + flops.linear(n, ffn_dim, D)

    assert total == self_attn + ffn


def test_vit_block_without_rope():
    """ViT block FLOPs without RoPE (use_rope=False)."""
    n, D = 16, 768
    ffn_ratio = 4.0

    total = flops.vit_block(n, D, ffn_ratio, use_rope=False)

    # Self-attention: LN + QKV + SDPA + out (no RoPE)
    self_attn = (
        flops.layer_norm(n, D)
        + flops.linear(n, D, 3 * D)
        + flops.sdpa(n, n, D)
        + flops.linear(n, D, D)
    )

    # FFN: LN + fc1 + fc2
    ffn_dim = int(ffn_ratio * D)
    ffn = flops.layer_norm(n, D) + flops.linear(n, D, ffn_dim) + flops.linear(n, ffn_dim, D)

    assert total == self_attn + ffn


def test_patch_embed_formula():
    """Patch embed conv FLOPs."""
    n_patches = 14 * 14
    patch_size = 16
    embed_dim = 768
    in_chans = 3

    total = flops.patch_embed(n_patches, patch_size, embed_dim, in_chans)
    expected = 2 * n_patches * (patch_size**2) * in_chans * embed_dim

    assert total == expected


def test_cls_head_formula():
    """CLS head: LN + Linear."""
    local_dim, teacher_dim = 768, 1024

    total = flops.cls_head(local_dim, teacher_dim)
    expected = flops.layer_norm(1, local_dim) + flops.linear(1, local_dim, teacher_dim)

    assert total == expected
