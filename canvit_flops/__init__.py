"""FLOP counting for CanViT architectures.

Accurate formulas including all operations:
- Linear: 2 * n * in * out (matmul)
- SDPA: 4 * n_q * n_kv * dim (Q@K^T + softmax@V)
- LayerNorm: 5 * n * dim (mean, var, normalize, affine)
- LayerScale: n * dim (per-channel scale)
- RoPE compute: 21 * n * head_dim (divisions, muls, sin, cos)
- RoPE apply: 4 * n * dim (two muls, one add, negations)
"""

# Sin/cos approximated as ~10 FLOPs each (polynomial approximation)
_TRIG_FLOPS = 10


def rope_compute(n: int, head_dim: int) -> int:
    """Precompute RoPE sin/cos tables for n positions.

    Operations:
    - Division by periods: n * head_dim/2
    - Multiply by 2π: n * head_dim/2
    - Sin: n * head_dim (×10 FLOPs for polynomial)
    - Cos: n * head_dim (×10 FLOPs for polynomial)

    Total: ~21 * n * head_dim
    """
    return n * head_dim + 2 * _TRIG_FLOPS * n * head_dim


def rope_apply(n: int, dim: int) -> int:
    """Apply RoPE rotation to n tokens.

    Operations: out = x * cos + rotate_half(x) * sin
    - x * cos: n * dim muls
    - rotate_half negation: n * dim/2
    - rotate_half(x) * sin: n * dim muls
    - addition: n * dim

    Total: ~3.5 * n * dim ≈ 4 * n * dim (conservative)
    """
    return 4 * n * dim


def linear(n: int, in_dim: int, out_dim: int) -> int:
    """Linear projection: 2 FLOPs per multiply-add."""
    return 2 * n * in_dim * out_dim


def sdpa(n_q: int, n_kv: int, head_dim: int) -> int:
    """Scaled dot-product attention: Q@K^T + softmax@V."""
    return 4 * n_q * n_kv * head_dim


def layer_norm(n: int, dim: int) -> int:
    """LayerNorm: mean, variance, normalize, affine transform."""
    return 5 * n * dim


def layer_scale(n: int, dim: int) -> int:
    """LayerScale: per-channel multiplication."""
    return n * dim


def canvas_read(
    n_local: int,
    n_canvas: int,
    local_dim: int,
    canvas_dim: int,
    use_layer_scale: bool = True,
) -> int:
    """CanvasReadAttention FLOPs: local queries canvas.

    Components:
    - pre_q_ln: LayerNorm on local
    - pre_kv_ln: LayerNorm on canvas
    - Q: Linear(local_dim → canvas_dim) on n_local
    - K, V: Identity on canvas (no FLOPs)
    - RoPE apply on Q (n_local tokens)
    - RoPE apply on K (n_canvas tokens)
    - SDPA: n_local × n_canvas
    - O: Linear(canvas_dim → local_dim) on n_local
    - LayerScale on output (if use_layer_scale)

    Note: RoPE compute (sin/cos precompute) is amortized across all
    attentions in a forward pass - count separately via rope_compute().
    """
    total = 0
    total += layer_norm(n_local, local_dim)
    total += layer_norm(n_canvas, canvas_dim)
    total += linear(n_local, local_dim, canvas_dim)
    total += rope_apply(n_local, canvas_dim)   # Q rotation
    total += rope_apply(n_canvas, canvas_dim)  # K rotation
    total += sdpa(n_local, n_canvas, canvas_dim)
    total += linear(n_local, canvas_dim, local_dim)
    if use_layer_scale:
        total += layer_scale(n_local, local_dim)
    return total


def canvas_write(
    n_local: int,
    n_canvas: int,
    local_dim: int,
    canvas_dim: int,
) -> int:
    """CanvasWriteAttention FLOPs: canvas queries local.

    Components:
    - pre_q_ln: LayerNorm on canvas
    - pre_kv_ln: LayerNorm on local
    - Q, O: Identity on canvas (no FLOPs)
    - K: Linear(local_dim → canvas_dim) on n_local
    - V: Linear(local_dim → canvas_dim) on n_local
    - RoPE apply on Q (n_canvas tokens)
    - RoPE apply on K (n_local tokens)
    - SDPA: n_canvas × n_local

    Note: RoPE compute (sin/cos precompute) is amortized across all
    attentions in a forward pass - count separately via rope_compute().
    """
    total = 0
    total += layer_norm(n_canvas, canvas_dim)
    total += layer_norm(n_local, local_dim)
    total += linear(n_local, local_dim, canvas_dim)  # K
    total += linear(n_local, local_dim, canvas_dim)  # V
    total += rope_apply(n_canvas, canvas_dim)  # Q rotation
    total += rope_apply(n_local, canvas_dim)   # K rotation
    total += sdpa(n_canvas, n_local, canvas_dim)
    return total


def canvas_adapter(
    n_local: int,
    n_canvas: int,
    local_dim: int,
    canvas_dim: int,
    use_layer_scale: bool = True,
) -> int:
    """One read + one write = one canvas adapter."""
    return canvas_read(
        n_local, n_canvas, local_dim, canvas_dim, use_layer_scale
    ) + canvas_write(n_local, n_canvas, local_dim, canvas_dim)


def vit_block(n: int, embed_dim: int, ffn_ratio: float, *, use_rope: bool = True) -> int:
    """Standard ViT transformer block FLOPs.

    Self-attention:
    - LayerNorm
    - QKV projection: Linear(D → 3D)
    - RoPE apply on Q, K (if use_rope)
    - SDPA: n × n
    - Out projection: Linear(D → D)

    FFN:
    - LayerNorm
    - fc1: Linear(D → ffn_dim)
    - fc2: Linear(ffn_dim → D)

    Note: RoPE compute (sin/cos precompute) is amortized across all blocks
    in a forward pass - count separately via rope_compute().
    """
    D = embed_dim
    ffn_dim = int(ffn_ratio * D)

    total = 0
    # Self-attention
    total += layer_norm(n, D)
    total += linear(n, D, 3 * D)  # QKV
    if use_rope:
        total += rope_apply(n, D)  # Q rotation
        total += rope_apply(n, D)  # K rotation
    total += sdpa(n, n, D)
    total += linear(n, D, D)  # out

    # FFN
    total += layer_norm(n, D)
    total += linear(n, D, ffn_dim)
    total += linear(n, ffn_dim, D)

    return total


def patch_embed(n_patches: int, patch_size: int, embed_dim: int, in_chans: int = 3) -> int:
    """Patch embedding convolution FLOPs."""
    return 2 * n_patches * (patch_size**2) * in_chans * embed_dim


def vit_backbone(
    n_patches: int, patch_size: int, embed_dim: int, n_blocks: int, ffn_ratio: float, *, use_rope: bool = True
) -> int:
    """Full ViT backbone: patch embed + blocks.

    Note: RoPE compute cost is amortized - add rope_compute(n_tokens, head_dim)
    once per forward pass if use_rope=True.
    """
    n_tokens = n_patches + 1  # +CLS
    return patch_embed(n_patches, patch_size, embed_dim) + n_blocks * vit_block(n_tokens, embed_dim, ffn_ratio, use_rope=use_rope)


def scene_head(n_spatial: int, canvas_dim: int, teacher_dim: int) -> int:
    """Scene head: LN + Linear per spatial token."""
    return layer_norm(n_spatial, canvas_dim) + linear(n_spatial, canvas_dim, teacher_dim)


def cls_head(local_dim: int, teacher_dim: int) -> int:
    """CLS head: LN(cls) + Linear(cls → teacher_dim)."""
    return layer_norm(1, local_dim) + linear(1, local_dim, teacher_dim)


__all__ = [
    "rope_compute",
    "rope_apply",
    "linear",
    "sdpa",
    "layer_norm",
    "layer_scale",
    "canvas_read",
    "canvas_write",
    "canvas_adapter",
    "vit_block",
    "patch_embed",
    "vit_backbone",
    "scene_head",
    "cls_head",
]
