"""Does compiling the DINOv3 teacher change the distillation targets?

`train/loop.py` compiles BOTH under `cfg.compile` ("Compiling teacher and model"); the
harness compiles only the student ("Compiling model (torch.compile)"). Verified from the
exp23 job logs, so every harness-vs-oldloop comparison so far has had a compiled teacher
on one side and an eager one on the other.

That matters because the teacher produces:
  * VALIDATION targets  (tasks/distill/task.py::evaluate)  -> val/scene_cos_norm_t*, the
    exact series used to judge exp23/exp26
  * RAW-shard TRAINING targets (tasks/distill/task.py::_teacher_targets), the exp21
    on-the-fly path

If eager and compiled teachers agree to ~1e-7 this is a non-issue. If they differ at
1e-3, it is a candidate explanation for the residual uniform gap (harness sat ~0.0021
below the old loop, negative at 15/18 eval points) and must be fixed before any further
A/B is trusted.

Run on a GPU node:
  .venv-cu126/bin/python unification_docs/teacher_compile_delta.py
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

from canvit_train.train.config import Config
from canvit_train.train.model import load_teacher

_B, _RES = 4, 512


def _feats(teacher, images):
    # DINOv3Teacher has no forward(); the distill path calls forward_norm_features,
    # which is exactly what produces the val/raw-shard targets.
    with torch.no_grad():
        out = teacher.forward_norm_features(images)
    return out.patches.float(), out.cls.float()


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} torch={torch.__version__}")
    if dev.type != "cuda":
        print("SKIP: needs a GPU (compile on CPU is not the production path)")
        return 0

    torch.set_float32_matmul_precision("high")  # what train/__main__ and the harness set
    cfg = Config()
    torch.manual_seed(0)
    images = torch.randn(_B, 3, _RES, _RES, device=dev)

    eager = load_teacher(cfg)
    p_eager, c_eager = _feats(eager, images)

    compiled = load_teacher(cfg)
    from canvit_train.train.model import compile_teacher
    compile_teacher(compiled)
    p_comp, c_comp = _feats(compiled, images)

    def rep(name, a, b):
        if a is None or b is None:
            print(f"  {name}: n/a")
            return 0.0
        absd = (a - b).abs().max().item()
        rel = absd / max(a.abs().max().item(), 1e-8)
        # what actually matters downstream: the cosine the distill metric reports
        cos = torch.nn.functional.cosine_similarity(
            a.flatten(1).float(), b.flatten(1).float(), dim=-1).mean().item()
        print(f"  {name}: max|d|={absd:.3e}  rel={rel:.3e}  mean_cos={cos:.10f}  "
              f"1-cos={1 - cos:.3e}")
        return 1 - cos

    print("\nteacher features, eager vs compiled (same weights, same input):")
    d_patch = rep("patches", p_eager, p_comp)
    d_cls = rep("cls    ", c_eager, c_comp)

    worst = max(d_patch, d_cls)
    print()
    if worst < 1e-6:
        print(f"NEGLIGIBLE (1-cos = {worst:.2e}): compiling the teacher does not move the "
              "targets; the harness/oldloop teacher-compile asymmetry cannot explain the gap.")
    else:
        print(f"MATERIAL (1-cos = {worst:.2e}): the compiled and eager teachers produce "
              "DIFFERENT targets. The harness must compile the teacher too, or every\n"
              "harness-vs-oldloop val comparison carries this offset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
