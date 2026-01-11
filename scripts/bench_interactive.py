"""Interactive benchmarking preload for ipython.

Usage:
    cd ~/scratch/avp-vit && source slurm/env.sh
    uv run ipython -i scripts/bench_interactive.py

Then:
    bench_teacher()
    bench_teacher(batch=32, amp=True, compile=True)
    bench_student(glimpse=8, canvas=32)
    bench_student(glimpse=16, canvas=64, amp=True)
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
teacher_compiled = None
print(f"  teacher: {teacher.embed_dim}d, {teacher.n_blocks} blocks")

print("Loading student...")
backbone = create_backbone('dinov3_vitb16', pretrained=False).to(DEVICE)
student = ActiveCanViT(backbone=backbone, cfg=ActiveCanViTConfig(teacher_dim=768)).to(DEVICE).eval()
student_compiled = None
PATCH = backbone.patch_size_px
print(f"  student: patch={PATCH}px")


def _get_teacher(compile: bool):
    global teacher_compiled
    if not compile:
        return teacher
    if teacher_compiled is None:
        print("  compiling teacher...")
        teacher.compile()
        teacher_compiled = teacher
    return teacher_compiled


def _get_student(compile: bool):
    global student_compiled
    if not compile:
        return student
    if student_compiled is None:
        print("  compiling student...")
        student.compile()
        student_compiled = student
    return student_compiled


def _bench(fn, batch, amp: bool):
    ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if amp else torch.no_grad()

    with torch.no_grad():
        with ctx if amp else torch.enable_grad():
            for _ in range(WARMUP):
                fn()
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        with ctx if amp else torch.enable_grad():
            for _ in trange(N, unit_scale=batch, unit="img"):
                fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"  → {N/elapsed:.1f} steps/s, {N*batch/elapsed:.0f} img/s")


def bench_teacher(batch=64, size=512, amp=False, compile=False):
    print(f"bench_teacher(batch={batch}, size={size}, amp={amp}, compile={compile})")
    model = _get_teacher(compile)
    x = torch.randn(batch, 3, size, size, device=DEVICE)
    _bench(lambda: model.forward_norm_features(x), batch, amp)


def bench_student(glimpse=8, canvas=32, batch=64, size=512, amp=False, compile=False):
    print(f"bench_student(glimpse={glimpse}, canvas={canvas}, batch={batch}, amp={amp}, compile={compile})")
    model = _get_student(compile)
    x = torch.randn(batch, 3, size, size, device=DEVICE)
    vp = Viewpoint.full_scene(batch_size=batch, device=DEVICE)
    glimpse_px = glimpse * PATCH
    def step():
        state = model.init_state(batch_size=batch, canvas_grid_size=canvas)
        model.forward_step(image=x, state=state, viewpoint=vp, glimpse_size_px=glimpse_px)
    _bench(step, batch, amp)


print(f"\nReady. CUDA: {torch.cuda.get_device_name()}")
print("  bench_teacher(batch=64, size=512, amp=False, compile=False)")
print("  bench_student(glimpse=8, canvas=32, batch=64, amp=False, compile=False)")
