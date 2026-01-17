"""Benchmark attention fusion - find what actually compiles well."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.profiler import profile, ProfilerActivity

# Actual dimensions from CanViT config
B = 64
N_LOCAL = 71      # local tokens
N_CANVAS = 1040   # canvas tokens
CANVAS_DIM = 1024
LOCAL_DIM = 768
NUM_HEADS = 8
HEAD_DIM = CANVAS_DIM // NUM_HEADS  # 128
N_PREFIX = 16     # canvas registers

N_WARMUP = 20
N_BENCH = 50


def to_multihead(x: Tensor, num_heads: int) -> Tensor:
    B, N, D = x.shape
    return x.view(B, N, num_heads, D // num_heads).transpose(1, 2)


def from_multihead(x: Tensor) -> Tensor:
    B, H, N, head_dim = x.shape
    return x.transpose(1, 2).reshape(B, N, H * head_dim)


class SplitEWA(nn.Module):
    """Minimal SplitElementwiseAffine for testing."""
    def __init__(self, dim: int, n_prefix: int):
        super().__init__()
        self.n_prefix = n_prefix
        self.prefix_scale = nn.Parameter(torch.ones(n_prefix, dim))
        self.prefix_bias = nn.Parameter(torch.zeros(n_prefix, dim))
        self.rest_scale = nn.Parameter(torch.ones(dim))
        self.rest_bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        prefix = x[:, :self.n_prefix] * self.prefix_scale + self.prefix_bias
        rest = x[:, self.n_prefix:] * self.rest_scale + self.rest_bias
        return torch.cat([prefix, rest], dim=1)


class CurrentAttention(nn.Module):
    """Current CanvasWriteAttention structure (canvas queries local)."""
    def __init__(self):
        super().__init__()
        self.pre_q_ln = nn.LayerNorm(CANVAS_DIM)
        self.pre_kv_ln = nn.LayerNorm(LOCAL_DIM)
        self.q_transform = SplitEWA(CANVAS_DIM, N_PREFIX)
        self.k_transform = nn.Linear(LOCAL_DIM, CANVAS_DIM)
        self.v_transform = nn.Linear(LOCAL_DIM, CANVAS_DIM)
        self.out_transform = SplitEWA(CANVAS_DIM, N_PREFIX)

    def forward(self, query: Tensor, kv: Tensor) -> Tensor:
        q = to_multihead(self.q_transform(self.pre_q_ln(query)), NUM_HEADS)
        kv_normed = self.pre_kv_ln(kv)
        k = to_multihead(self.k_transform(kv_normed), NUM_HEADS)
        v = to_multihead(self.v_transform(kv_normed), NUM_HEADS)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_transform(from_multihead(out))


class FusedQKVAttention(nn.Module):
    """Try fusing K/V projection into one."""
    def __init__(self):
        super().__init__()
        self.pre_q_ln = nn.LayerNorm(CANVAS_DIM)
        self.pre_kv_ln = nn.LayerNorm(LOCAL_DIM)
        self.q_transform = SplitEWA(CANVAS_DIM, N_PREFIX)
        self.kv_transform = nn.Linear(LOCAL_DIM, CANVAS_DIM * 2)  # Fused K+V
        self.out_transform = SplitEWA(CANVAS_DIM, N_PREFIX)

    def forward(self, query: Tensor, kv: Tensor) -> Tensor:
        q = to_multihead(self.q_transform(self.pre_q_ln(query)), NUM_HEADS)
        kv_normed = self.pre_kv_ln(kv)
        kv_proj = self.kv_transform(kv_normed)
        k, v = kv_proj.chunk(2, dim=-1)
        k = to_multihead(k, NUM_HEADS)
        v = to_multihead(v, NUM_HEADS)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_transform(from_multihead(out))


class ContiguousReshapeAttention(nn.Module):
    """Use contiguous() before reshape to avoid hidden copies."""
    def __init__(self):
        super().__init__()
        self.pre_q_ln = nn.LayerNorm(CANVAS_DIM)
        self.pre_kv_ln = nn.LayerNorm(LOCAL_DIM)
        self.q_transform = SplitEWA(CANVAS_DIM, N_PREFIX)
        self.k_transform = nn.Linear(LOCAL_DIM, CANVAS_DIM)
        self.v_transform = nn.Linear(LOCAL_DIM, CANVAS_DIM)
        self.out_transform = SplitEWA(CANVAS_DIM, N_PREFIX)

    def forward(self, query: Tensor, kv: Tensor) -> Tensor:
        q = to_multihead(self.q_transform(self.pre_q_ln(query)), NUM_HEADS)
        kv_normed = self.pre_kv_ln(kv)
        k = to_multihead(self.k_transform(kv_normed), NUM_HEADS)
        v = to_multihead(self.v_transform(kv_normed), NUM_HEADS)
        out = F.scaled_dot_product_attention(q, k, v)
        # Explicit contiguous before reshape
        out = out.transpose(1, 2).contiguous().view(out.shape[0], -1, CANVAS_DIM)
        return self.out_transform(out)


class NoEWAAttention(nn.Module):
    """Baseline without EWA - just identity transforms on Q/out."""
    def __init__(self):
        super().__init__()
        self.pre_q_ln = nn.LayerNorm(CANVAS_DIM)
        self.pre_kv_ln = nn.LayerNorm(LOCAL_DIM)
        self.k_transform = nn.Linear(LOCAL_DIM, CANVAS_DIM)
        self.v_transform = nn.Linear(LOCAL_DIM, CANVAS_DIM)

    def forward(self, query: Tensor, kv: Tensor) -> Tensor:
        q = to_multihead(self.pre_q_ln(query), NUM_HEADS)
        kv_normed = self.pre_kv_ln(kv)
        k = to_multihead(self.k_transform(kv_normed), NUM_HEADS)
        v = to_multihead(self.v_transform(kv_normed), NUM_HEADS)
        out = F.scaled_dot_product_attention(q, k, v)
        return from_multihead(out)


def bench(module, name: str, query: Tensor, kv: Tensor, compile_mode: str | None):
    if compile_mode:
        module = torch.compile(module, mode=compile_mode)

    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = module(query, kv)
    torch.cuda.synchronize()

    # Bench
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(N_BENCH):
                _ = module(query, kv)
            torch.cuda.synchronize()

    total = sum(e.self_device_time_total for e in prof.key_averages() if e.self_device_time_total > 0)
    per_iter = total / 1000 / N_BENCH
    mode_str = compile_mode or "eager"
    print(f"{name:30s} [{mode_str:12s}]: {per_iter:.3f} ms/iter")

    # Show top kernels for first run
    if compile_mode is None:
        print("  Top kernels:")
        for e in sorted(prof.key_averages(), key=lambda x: -x.self_device_time_total)[:5]:
            if e.self_device_time_total > 0:
                print(f"    {e.key[:50]:50s} {e.self_device_time_total/1000/N_BENCH:.3f} ms")
    return per_iter


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Query (canvas): [{B}, {N_CANVAS}, {CANVAS_DIM}]")
    print(f"KV (local):     [{B}, {N_LOCAL}, {LOCAL_DIM}]")
    print()

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)

    query = torch.randn(B, N_CANVAS, CANVAS_DIM, device=device, dtype=torch.bfloat16)
    kv = torch.randn(B, N_LOCAL, LOCAL_DIM, device=device, dtype=torch.bfloat16)

    variants = [
        ("Current (EWA)", CurrentAttention),
        ("Fused KV proj", FusedQKVAttention),
        ("Contiguous reshape", ContiguousReshapeAttention),
        ("No EWA baseline", NoEWAAttention),
    ]

    results = {}
    for name, cls in variants:
        module = cls().to(device, dtype=torch.bfloat16).eval()
        print(f"\n{'='*60}")
        print(f"{name}")
        print('='*60)
        results[f"{name}_eager"] = bench(module, name, query, kv, None)

        # Fresh module for compile
        module = cls().to(device, dtype=torch.bfloat16).eval()
        results[f"{name}_default"] = bench(module, name, query, kv, "default")

        module = cls().to(device, dtype=torch.bfloat16).eval()
        try:
            results[f"{name}_reduce"] = bench(module, name, query, kv, "reduce-overhead")
        except Exception as e:
            print(f"  reduce-overhead failed: {e}")

        module = cls().to(device, dtype=torch.bfloat16).eval()
        try:
            results[f"{name}_autotune"] = bench(module, name, query, kv, "max-autotune")
        except Exception as e:
            print(f"  max-autotune failed: {e}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {k:45s}: {v:.3f} ms")


if __name__ == "__main__":
    main()
