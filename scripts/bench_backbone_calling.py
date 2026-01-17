"""Compare backbone calling patterns: forward_features vs forward_block loop.

Same backbone, same input, different calling patterns.
"""

import torch
from torch.profiler import profile, ProfilerActivity

from canvit.hub import create_backbone
from canvit.rope import compute_rope

BATCH = 64
GLIMPSE_PX = 128
N_WARMUP = 20
N_BENCH = 50


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda")

    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)

    # Same backbone for both tests
    backbone = create_backbone("dinov3_vitb16", pretrained=False).to(device).eval()

    # Compile blocks
    for block in backbone.vit.blocks:
        block.compile()

    img = torch.randn(BATCH, 3, GLIMPSE_PX, GLIMPSE_PX, device=device)

    print(f"Backbone: {backbone.n_blocks} blocks, dim={backbone.embed_dim}")
    print(f"Input: [{BATCH}, 3, {GLIMPSE_PX}, {GLIMPSE_PX}]")
    print()

    # =========================================================================
    # Method 1: forward_norm_features (all blocks in one call)
    # =========================================================================
    print("=" * 60)
    print("METHOD 1: forward_norm_features (single call)")
    print("=" * 60)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(N_WARMUP):
            _ = backbone.forward_norm_features(img)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CUDA]) as prof1:
            for _ in range(N_BENCH):
                _ = backbone.forward_norm_features(img)
            torch.cuda.synchronize()

    t1 = sum(e.self_device_time_total for e in prof1.key_averages() if e.self_device_time_total > 0)
    print(f"Total: {t1/1000:.1f} ms ({N_BENCH} iters)")
    print(f"Per-iter: {t1/1000/N_BENCH:.2f} ms")
    print()

    # =========================================================================
    # Method 2: forward_block loop (12 individual calls)
    # =========================================================================
    print("=" * 60)
    print("METHOD 2: forward_block loop (12 calls per iter)")
    print("=" * 60)

    # Prepare tokens once
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x, H, W = backbone.prepare_tokens(img)
        # Compute RoPE like CanViT does
        from canvit.coords import grid_coords
        positions = grid_coords(H=H, W=W, device=device).flatten(0, 1).unsqueeze(0).expand(BATCH, -1, -1)
        rope = compute_rope(positions=positions, periods=backbone.rope_periods)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(N_WARMUP):
            tokens = x.clone()
            for idx in range(backbone.n_blocks):
                tokens = backbone.forward_block(idx, tokens, rope)
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CUDA]) as prof2:
            for _ in range(N_BENCH):
                tokens = x.clone()
                for idx in range(backbone.n_blocks):
                    tokens = backbone.forward_block(idx, tokens, rope)
            torch.cuda.synchronize()

    t2 = sum(e.self_device_time_total for e in prof2.key_averages() if e.self_device_time_total > 0)
    print(f"Total: {t2/1000:.1f} ms ({N_BENCH} iters)")
    print(f"Per-iter: {t2/1000/N_BENCH:.2f} ms")
    print()

    # =========================================================================
    # Compare
    # =========================================================================
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"forward_norm_features: {t1/1000/N_BENCH:.2f} ms/iter")
    print(f"forward_block loop:    {t2/1000/N_BENCH:.2f} ms/iter")
    print(f"Ratio: {t2/t1:.2f}x")


if __name__ == "__main__":
    main()
