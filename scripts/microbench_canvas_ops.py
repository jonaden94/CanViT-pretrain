"""Microbenchmark individual canvas operations to understand overhead.

Isolates:
1. LayerNorm on canvas (1040 tokens × 1024 dim)
2. Cross-attention (71 local vs 1040 canvas)
3. EWA (SplitElementwiseAffine with torch.cat)

All in bf16 with proper warmup.
"""

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity

# Match training config exactly
BATCH = 64
CANVAS_TOKENS = 1040  # 16 registers + 1024 spatial
CANVAS_DIM = 1024
LOCAL_TOKENS = 71  # 1 VPE + 1 rec_cls + 1 eph_cls + 4 regs + 64 patches
LOCAL_DIM = 768
NUM_HEADS = 16
N_WARMUP = 50
N_BENCH = 100


def bench_layernorm():
    """Benchmark LayerNorm on canvas-sized tensor."""
    print("=" * 70)
    print("LAYERNORM ON CANVAS")
    print(f"  Shape: [{BATCH}, {CANVAS_TOKENS}, {CANVAS_DIM}]")
    print(f"  Elements: {BATCH * CANVAS_TOKENS * CANVAS_DIM:,}")
    print("=" * 70)

    device = torch.device("cuda")
    ln = nn.LayerNorm(CANVAS_DIM).to(device)
    x = torch.randn(BATCH, CANVAS_TOKENS, CANVAS_DIM, device=device, dtype=torch.bfloat16)

    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = ln(x)
    torch.cuda.synchronize()

    # Benchmark with profiler
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(N_BENCH):
                _ = ln(x)
            torch.cuda.synchronize()

    total_cuda = sum(e.self_device_time_total for e in prof.key_averages() if e.self_device_time_total > 0)
    print(f"Total CUDA time: {total_cuda/1000:.2f} ms ({N_BENCH} iters)")
    print(f"Per-iter: {total_cuda/1000/N_BENCH:.3f} ms")
    print()
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=10))
    return total_cuda / N_BENCH


def bench_layernorm_compiled():
    """Benchmark compiled LayerNorm."""
    print("=" * 70)
    print("LAYERNORM ON CANVAS (COMPILED)")
    print("=" * 70)

    device = torch.device("cuda")
    ln = nn.LayerNorm(CANVAS_DIM).to(device)
    ln = torch.compile(ln)
    x = torch.randn(BATCH, CANVAS_TOKENS, CANVAS_DIM, device=device, dtype=torch.bfloat16)

    # Warmup (more for compilation)
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = ln(x)
    torch.cuda.synchronize()

    # Benchmark
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(N_BENCH):
                _ = ln(x)
            torch.cuda.synchronize()

    total_cuda = sum(e.self_device_time_total for e in prof.key_averages() if e.self_device_time_total > 0)
    print(f"Total CUDA time: {total_cuda/1000:.2f} ms ({N_BENCH} iters)")
    print(f"Per-iter: {total_cuda/1000/N_BENCH:.3f} ms")
    print()
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=10))
    return total_cuda / N_BENCH


def bench_sdpa_cross_attention():
    """Benchmark raw SDPA for cross-attention pattern (local queries canvas)."""
    print("=" * 70)
    print("SDPA CROSS-ATTENTION (local queries canvas)")
    print(f"  Q: [{BATCH}, {NUM_HEADS}, {LOCAL_TOKENS}, {CANVAS_DIM // NUM_HEADS}]")
    print(f"  K/V: [{BATCH}, {NUM_HEADS}, {CANVAS_TOKENS}, {CANVAS_DIM // NUM_HEADS}]")
    print("=" * 70)

    device = torch.device("cuda")
    head_dim = CANVAS_DIM // NUM_HEADS

    q = torch.randn(BATCH, NUM_HEADS, LOCAL_TOKENS, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(BATCH, NUM_HEADS, CANVAS_TOKENS, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(BATCH, NUM_HEADS, CANVAS_TOKENS, head_dim, device=device, dtype=torch.bfloat16)

    # Force flash attention
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)

    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()

    # Benchmark
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(N_BENCH):
                _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()

    total_cuda = sum(e.self_device_time_total for e in prof.key_averages() if e.self_device_time_total > 0)
    print(f"Total CUDA time: {total_cuda/1000:.2f} ms ({N_BENCH} iters)")
    print(f"Per-iter: {total_cuda/1000/N_BENCH:.3f} ms")
    print()
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=10))
    return total_cuda / N_BENCH


def bench_ewa_cat():
    """Benchmark the EWA pattern: split, scale/bias, cat."""
    print("=" * 70)
    print("EWA (SplitElementwiseAffine pattern)")
    print(f"  Input: [{BATCH}, {CANVAS_TOKENS}, {CANVAS_DIM}]")
    print(f"  Prefix (registers): 16 tokens")
    print(f"  Rest (spatial): 1024 tokens")
    print("=" * 70)

    device = torch.device("cuda")
    n_prefix = 16

    x = torch.randn(BATCH, CANVAS_TOKENS, CANVAS_DIM, device=device, dtype=torch.bfloat16)
    prefix_scale = torch.randn(n_prefix, CANVAS_DIM, device=device, dtype=torch.bfloat16)
    prefix_bias = torch.randn(n_prefix, CANVAS_DIM, device=device, dtype=torch.bfloat16)
    rest_scale = torch.randn(CANVAS_DIM, device=device, dtype=torch.bfloat16)
    rest_bias = torch.randn(CANVAS_DIM, device=device, dtype=torch.bfloat16)

    def ewa_forward(x):
        prefix = x[:, :n_prefix] * prefix_scale + prefix_bias
        rest = x[:, n_prefix:] * rest_scale + rest_bias
        return torch.cat([prefix, rest], dim=1)

    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = ewa_forward(x)
    torch.cuda.synchronize()

    # Benchmark
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(N_BENCH):
                _ = ewa_forward(x)
            torch.cuda.synchronize()

    total_cuda = sum(e.self_device_time_total for e in prof.key_averages() if e.self_device_time_total > 0)
    print(f"Total CUDA time: {total_cuda/1000:.2f} ms ({N_BENCH} iters)")
    print(f"Per-iter: {total_cuda/1000/N_BENCH:.3f} ms")
    print()
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=10))
    return total_cuda / N_BENCH


def bench_ewa_cat_compiled():
    """Benchmark compiled EWA."""
    print("=" * 70)
    print("EWA (COMPILED)")
    print("=" * 70)

    device = torch.device("cuda")
    n_prefix = 16

    x = torch.randn(BATCH, CANVAS_TOKENS, CANVAS_DIM, device=device, dtype=torch.bfloat16)
    prefix_scale = torch.randn(n_prefix, CANVAS_DIM, device=device, dtype=torch.bfloat16)
    prefix_bias = torch.randn(n_prefix, CANVAS_DIM, device=device, dtype=torch.bfloat16)
    rest_scale = torch.randn(CANVAS_DIM, device=device, dtype=torch.bfloat16)
    rest_bias = torch.randn(CANVAS_DIM, device=device, dtype=torch.bfloat16)

    def ewa_forward(x):
        prefix = x[:, :n_prefix] * prefix_scale + prefix_bias
        rest = x[:, n_prefix:] * rest_scale + rest_bias
        return torch.cat([prefix, rest], dim=1)

    ewa_compiled = torch.compile(ewa_forward)

    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = ewa_compiled(x)
    torch.cuda.synchronize()

    # Benchmark
    with torch.no_grad():
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            for _ in range(N_BENCH):
                _ = ewa_compiled(x)
            torch.cuda.synchronize()

    total_cuda = sum(e.self_device_time_total for e in prof.key_averages() if e.self_device_time_total > 0)
    print(f"Total CUDA time: {total_cuda/1000:.2f} ms ({N_BENCH} iters)")
    print(f"Per-iter: {total_cuda/1000/N_BENCH:.3f} ms")
    print()
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=10))
    return total_cuda / N_BENCH


def main():
    assert torch.cuda.is_available(), "CUDA required"

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Warmup iters: {N_WARMUP}, Bench iters: {N_BENCH}")
    print()

    results = {}

    results["ln_eager"] = bench_layernorm()
    print()
    results["ln_compiled"] = bench_layernorm_compiled()
    print()
    results["sdpa"] = bench_sdpa_cross_attention()
    print()
    results["ewa_eager"] = bench_ewa_cat()
    print()
    results["ewa_compiled"] = bench_ewa_cat_compiled()

    print()
    print("=" * 70)
    print("SUMMARY (μs per call)")
    print("=" * 70)
    for name, us in results.items():
        print(f"  {name:20s}: {us:.1f} μs")


if __name__ == "__main__":
    main()
