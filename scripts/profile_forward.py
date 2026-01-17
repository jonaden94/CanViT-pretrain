"""Comprehensive CanViT overhead investigation.

Tests multiple hypotheses in ONE run to minimize H100 time:
1. Profile teacher vs canvit (current component-level compile)
2. Test FULL forward compilation (compile entire forward, not just parts)
3. Test NO compilation baseline
4. Check for graph breaks
5. Output kernel summaries

Usage:
    uv run python scripts/profile_forward.py --device cuda
"""

import logging
import time
from dataclasses import dataclass

import torch
import torch._dynamo as dynamo
import tyro

from avp_vit import ActiveCanViT, ActiveCanViTConfig
from canvit.hub import create_backbone
from canvit.viewpoint import Viewpoint

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    device: str = "cuda"
    batch: int = 64
    glimpse_grid: int = 8
    canvas_grid: int = 32
    patch_size: int = 16
    warmup: int = 5
    iters: int = 30


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, warmup: int, iters: int, device: torch.device) -> float:
    """Benchmark without per-iteration syncs. Returns ms/iter."""
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / iters * 1000


def main(cfg: Config) -> None:
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        log.info(f"GPU: {torch.cuda.get_device_name()}")

    glimpse_px = cfg.glimpse_grid * cfg.patch_size

    log.info("=" * 70)
    log.info("COMPREHENSIVE CANVIT OVERHEAD INVESTIGATION")
    log.info("=" * 70)
    log.info(f"Batch: {cfg.batch}, Glimpse: {cfg.glimpse_grid}x{cfg.glimpse_grid}, Canvas: {cfg.canvas_grid}x{cfg.canvas_grid}")
    log.info("")

    logging.getLogger("dinov3").setLevel(logging.WARNING)

    # Create inputs once
    glimpse = torch.randn(cfg.batch, 3, glimpse_px, glimpse_px, device=device)
    vp = Viewpoint(
        centers=torch.zeros(cfg.batch, 2, device=device),
        scales=torch.ones(cfg.batch, device=device),
    )

    results = {}

    # =========================================================================
    # TEST 1: Teacher baseline
    # =========================================================================
    log.info("=" * 70)
    log.info("TEST 1: Teacher (DINOv3 backbone only)")
    log.info("=" * 70)

    teacher = create_backbone("dinov3_vitb16", pretrained=False).to(device).eval()

    # 1a. Teacher eager
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        results["teacher_eager"] = bench(
            lambda: teacher.forward_norm_features(glimpse),
            cfg.warmup, cfg.iters, device
        )
    log.info(f"  Eager:    {results['teacher_eager']:.2f} ms")

    # 1b. Teacher compiled
    teacher.compile()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        results["teacher_compiled"] = bench(
            lambda: teacher.forward_norm_features(glimpse),
            cfg.warmup, cfg.iters, device
        )
    log.info(f"  Compiled: {results['teacher_compiled']:.2f} ms")
    log.info("")

    # =========================================================================
    # TEST 2: CanViT - NO compilation (pure eager baseline)
    # =========================================================================
    log.info("=" * 70)
    log.info("TEST 2: CanViT EAGER (no compilation at all)")
    log.info("=" * 70)

    backbone_eager = create_backbone("dinov3_vitb16", pretrained=False)
    model_cfg = ActiveCanViTConfig(teacher_dim=backbone_eager.embed_dim)
    model_eager = ActiveCanViT(backbone=backbone_eager, cfg=model_cfg, policy=None)
    model_eager.to(device).eval()
    state_eager = model_eager.init_state(batch_size=cfg.batch, canvas_grid_size=cfg.canvas_grid)

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        results["canvit_eager"] = bench(
            lambda: model_eager.forward(glimpse=glimpse, state=state_eager, viewpoint=vp),
            cfg.warmup, cfg.iters, device
        )
    log.info(f"  Time: {results['canvit_eager']:.2f} ms")
    log.info(f"  vs Teacher eager: x{results['canvit_eager']/results['teacher_eager']:.2f}")
    log.info("")

    del model_eager, backbone_eager, state_eager
    torch.cuda.empty_cache()

    # =========================================================================
    # TEST 3: CanViT - Component compilation (current approach)
    # =========================================================================
    log.info("=" * 70)
    log.info("TEST 3: CanViT COMPONENT COMPILE (current: backbone + attn + rope)")
    log.info("=" * 70)

    backbone_comp = create_backbone("dinov3_vitb16", pretrained=False)
    model_comp = ActiveCanViT(backbone=backbone_comp, cfg=model_cfg, policy=None)
    model_comp.to(device).eval()
    model_comp.compile()  # This compiles backbone blocks + attention modules + _compute_ropes
    state_comp = model_comp.init_state(batch_size=cfg.batch, canvas_grid_size=cfg.canvas_grid)

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        results["canvit_component"] = bench(
            lambda: model_comp.forward(glimpse=glimpse, state=state_comp, viewpoint=vp),
            cfg.warmup, cfg.iters, device
        )
    log.info(f"  Time: {results['canvit_component']:.2f} ms")
    log.info(f"  vs Teacher compiled: x{results['canvit_component']/results['teacher_compiled']:.2f}")
    log.info("")

    # =========================================================================
    # TEST 4: CanViT - FULL forward compilation
    # =========================================================================
    log.info("=" * 70)
    log.info("TEST 4: CanViT FULL FORWARD COMPILE (wrap entire forward)")
    log.info("=" * 70)

    backbone_full = create_backbone("dinov3_vitb16", pretrained=False)
    model_full = ActiveCanViT(backbone=backbone_full, cfg=model_cfg, policy=None)
    model_full.to(device).eval()
    # Don't call model.compile() - instead compile the entire forward
    state_full = model_full.init_state(batch_size=cfg.batch, canvas_grid_size=cfg.canvas_grid)

    # Wrap forward in torch.compile
    @torch.compile
    def compiled_forward(model, glimpse, state, vp):
        return model.forward(glimpse=glimpse, state=state, viewpoint=vp)

    log.info("  Warming up full compile (may take a while)...")
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        for _ in range(cfg.warmup):
            _ = compiled_forward(model_full, glimpse, state_full, vp)
    sync(device)
    log.info("  Warmup done.")

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        results["canvit_full"] = bench(
            lambda: compiled_forward(model_full, glimpse, state_full, vp),
            cfg.warmup, cfg.iters, device
        )
    log.info(f"  Time: {results['canvit_full']:.2f} ms")
    log.info(f"  vs Teacher compiled: x{results['canvit_full']/results['teacher_compiled']:.2f}")
    log.info("")

    # =========================================================================
    # TEST 5: Check graph breaks in component compile
    # =========================================================================
    log.info("=" * 70)
    log.info("TEST 5: Graph break analysis")
    log.info("=" * 70)

    dynamo.reset()
    graph_count = [0]

    def count_backend(gm, example_inputs):
        graph_count[0] += 1
        return gm

    backbone_gb = create_backbone("dinov3_vitb16", pretrained=False)
    model_gb = ActiveCanViT(backbone=backbone_gb, cfg=model_cfg, policy=None)
    model_gb.to(device).eval()
    state_gb = model_gb.init_state(batch_size=cfg.batch, canvas_grid_size=cfg.canvas_grid)

    # Compile entire forward with counting backend
    compiled_count = torch.compile(model_gb.forward, backend=count_backend)

    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        _ = compiled_count(glimpse=glimpse, state=state_gb, viewpoint=vp)

    log.info(f"  Graph fragments when compiling model.forward: {graph_count[0]}")
    if graph_count[0] > 1:
        log.info(f"  ⚠️  {graph_count[0]} graph breaks detected!")
        log.info("     Each break = Python overhead + potential sync")
    else:
        log.info("  ✓ Single graph (good)")
    log.info("")

    # =========================================================================
    # TEST 6: Profile top kernels (component compile)
    # =========================================================================
    log.info("=" * 70)
    log.info("TEST 6: Kernel profiling (component compile)")
    log.info("=" * 70)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for _ in range(3):
                _ = model_comp.forward(glimpse=glimpse, state=state_comp, viewpoint=vp)

    log.info("  Top 15 CUDA kernels by time:")
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
    for line in table.split("\n"):
        log.info(f"  {line}")
    log.info("")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info("")
    log.info("Timing (ms/forward):")
    log.info(f"  Teacher eager:           {results['teacher_eager']:7.2f}")
    log.info(f"  Teacher compiled:        {results['teacher_compiled']:7.2f}")
    log.info(f"  CanViT eager:            {results['canvit_eager']:7.2f}  (x{results['canvit_eager']/results['teacher_eager']:.2f} vs teacher eager)")
    log.info(f"  CanViT component:        {results['canvit_component']:7.2f}  (x{results['canvit_component']/results['teacher_compiled']:.2f} vs teacher compiled)")
    log.info(f"  CanViT full compile:     {results['canvit_full']:7.2f}  (x{results['canvit_full']/results['teacher_compiled']:.2f} vs teacher compiled)")
    log.info("")
    log.info("Expected overhead: x1.26-1.40 (from FLOP analysis)")
    log.info("")

    if results['canvit_full'] < results['canvit_component'] * 0.8:
        log.info("DIAGNOSIS: Full forward compile is FASTER than component compile")
        log.info("  → Graph breaks between compiled components are the bottleneck")
        log.info("  → FIX: Compile entire model.forward() instead of individual parts")
    elif results['canvit_eager'] < results['canvit_component'] * 1.1:
        log.info("DIAGNOSIS: Compilation provides little benefit")
        log.info("  → Model is likely memory-bound, not compute-bound")
        log.info("  → Or compilation isn't actually happening")
    elif results['canvit_full'] / results['teacher_compiled'] > 2.0:
        log.info("DIAGNOSIS: Even full compile has high overhead")
        log.info("  → Something fundamental is slow (memory layout? tensor creation?)")
        log.info("  → Check kernel profile above for clues")
    else:
        log.info("DIAGNOSIS: Full compile brings overhead close to expected")
        log.info("  → Solution: compile entire forward instead of components")


if __name__ == "__main__":
    main(tyro.cli(Config))
