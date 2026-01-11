"""Interactive benchmarking preload for ipython.

Usage:
    srun --account=rrg-skrishna_gpu --gres=gpu:h100:1 --mem=32G --cpus-per-task=8 --time=1:00:00 --pty bash
    cd ~/scratch/avp-vit && source slurm/env.sh
    uv run ipython -i scripts/bench_interactive.py

Then:
    bench_teacher()
    bench_teacher(batch=32)
    bench_student(glimpse=8, canvas=32)
    bench_student(glimpse=16, canvas=64, batch=32)
"""

import os
import time
import torch
from tqdm import trange

DEVICE = torch.device("cuda")
N = 100
WARMUP = 5

# === Load models ===
print("Loading teacher...")
from canvit.hub import create_backbone
from canvit.viewpoint import Viewpoint
from avp_vit import ActiveCanViT, ActiveCanViTConfig

CKPT = os.path.expanduser(os.environ.get(
    'AVP_TEACHER_CKPT',
    '~/projects/def-skrishna/checkpoints/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
))
teacher = create_backbone('dinov3_vitb16', weights=CKPT).to(DEVICE).eval()
print(f"  teacher: {teacher.embed_dim}d, {teacher.n_blocks} blocks")

print("Loading student...")
backbone = create_backbone('dinov3_vitb16', pretrained=False).to(DEVICE)
student = ActiveCanViT(backbone=backbone, cfg=ActiveCanViTConfig(teacher_dim=768)).to(DEVICE).eval()
PATCH = backbone.patch_size_px
print(f"  student: patch={PATCH}px")


def _bench(fn, batch):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in trange(N, unit_scale=batch, unit="img"):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"  → {N/elapsed:.1f} steps/s, {N*batch/elapsed:.0f} img/s")


def bench_teacher(batch=64, size=512):
    x = torch.randn(batch, 3, size, size, device=DEVICE)
    with torch.no_grad():
        _bench(lambda: teacher.forward_norm_features(x), batch)


def bench_student(glimpse=8, canvas=32, batch=64, size=512):
    x = torch.randn(batch, 3, size, size, device=DEVICE)
    vp = Viewpoint.full_scene(batch_size=batch, device=DEVICE)
    glimpse_px = glimpse * PATCH
    with torch.no_grad():
        def step():
            state = student.init_state(batch_size=batch, canvas_grid_size=canvas)
            student.forward_step(image=x, state=state, viewpoint=vp, glimpse_size_px=glimpse_px)
        _bench(step, batch)


print("\nReady:")
print("  bench_teacher(batch=64, size=512)")
print("  bench_student(glimpse=8, canvas=32, batch=64, size=512)")
