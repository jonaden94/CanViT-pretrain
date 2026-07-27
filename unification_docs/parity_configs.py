"""Cross-CONFIG parity: the harness rollout == the old ``training_step``, per patcher.

Why this exists: every pre-existing parity check ran ONE configuration — the uniform
patcher on a non-modulated backbone (``parity_probe.py`` builds ``create_backbone("vits16")``
with the default patcher; ``harness_realdata_ab.py`` hardcodes ``is_foveated=False``). The
foveated path therefore had zero same-seed coverage, which is why the exp23 foveated
regression could only be found by a 12-hour production run. This closes that hole for
every patcher we actually use.

Per config, per step: from the SAME model state, SAME batch and SAME RNG, compute the
loss via (a) the old ``train/step.py::training_step`` and (b) the new
``harness/rollout.py::run_rollout``, and require them to agree. Also asserts both paths
derive ``is_foveated`` identically — the "square counts as foveated" question that was a
real bug once.

CPU + synthetic tensors: no GPU, no webdataset, no HF. Runs anywhere, so it can gate a
refactor instead of waiting for a queue.

Run:  .venv-cu126/bin/python unification_docs/parity_configs.py [--steps N] [--only NAME]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from canvit_pytorch import create_backbone
from canvit_pytorch.model.base.config import ViTModulationConfig
from canvit_pytorch.patcher import FoveatedPatcherConfig, SquarePatcherConfig
from canvit_pytorch.patcher.conditioning import PatchConditioningConfig

from canvit_pretrain import CanViTForPretraining, CanViTForPretrainingConfig
from canvit_pretrain.harness.rollout import run_rollout
from canvit_pretrain.harness.spec import BpttSpec
from canvit_pretrain.tasks.distill.task import BoundDistillTask
from canvit_pretrain.train.config import FoveatedScaleConfig
from canvit_pretrain.train.selector import RandomSelector
from canvit_pretrain.train.step import training_step
from canvit_pretrain.train.task import DistillTask
from canvit_pretrain.train.viewpoint import ViewpointType

_B, _G, _D = 2, 8, 384
_RES, _GLIMPSE = 224, 128
_TOL = 1e-5  # fp32 CPU, same ops in the same order -> agreement is near-exact


def _cfg(**kw) -> CanViTForPretrainingConfig:
    return CanViTForPretrainingConfig(teacher_dim=_D, **kw)


# Small patcher geometry so the foveated/square paths stay CPU-cheap while exercising
# the real sampling code (resolution/cart_patch_size drive the retinal grid).
_FOV = FoveatedPatcherConfig(resolution=16, cart_patch_size=4)
_SQR = SquarePatcherConfig(resolution=16, cart_patch_size=4)


def configs() -> dict[str, tuple[str, CanViTForPretrainingConfig]]:
    """name -> (backbone_name, model config). ``*_modulate`` backbones carry the
    adaLN-style TokenModulation params, a separate code path from the plain ViT."""
    mod = ViTModulationConfig(enabled=True)
    film = PatchConditioningConfig(mode="film")
    return {
        "uniform": ("vits16", _cfg(patcher_name="uniform")),
        "uniform+modulated": ("vits16_modulate", _cfg(patcher_name="uniform", vit_modulation=mod)),
        "foveated": ("vits16", _cfg(patcher_name="foveated", foveated_patcher=_FOV)),
        "foveated+film": ("vits16", _cfg(patcher_name="foveated",
                                         foveated_patcher=replace(_FOV, conditioning=film))),
        "foveated+modulated": ("vits16_modulate", _cfg(patcher_name="foveated",
                                                      foveated_patcher=_FOV, vit_modulation=mod)),
        "square": ("vits16", _cfg(patcher_name="square", square_patcher=_SQR)),
        "square+modulated": ("vits16_modulate", _cfg(patcher_name="square",
                                                    square_patcher=_SQR, vit_modulation=mod)),
    }


def _build(backbone_name: str, cfg: CanViTForPretrainingConfig) -> CanViTForPretraining:
    torch.manual_seed(1234)
    return CanViTForPretraining(
        backbone=create_backbone(backbone_name), cfg=cfg, glimpse_size_px=_GLIMPSE,
        backbone_name=backbone_name, canvas_patch_grid_sizes=[_G],
    )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "images": torch.randn(_B, 3, _RES, _RES),
        "scene_target": torch.randn(_B, _G * _G, _D),
        "cls_target": torch.randn(_B, _D),
        "raw_scene_target": torch.randn(_B, _G * _G, _D),
        "raw_cls_target": torch.randn(_B, _D),
    }


def run_config(name: str, backbone_name: str, cfg: CanViTForPretrainingConfig,
               n_steps: int) -> dict:
    model = _build(backbone_name, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    fov_scale = FoveatedScaleConfig()

    # Both stacks must classify the patcher the same way. `square` counting as uniform
    # was a real bug; assert the two derivations agree rather than trusting either.
    from canvit_pretrain.tasks.distill.task import DistillRunTask
    from canvit_pretrain.train.config import Config as PretrainConfig
    old_is_fov = getattr(model.cfg, "patcher_name", "uniform") in ("foveated", "square")
    new_is_fov = DistillRunTask(PretrainConfig()).is_foveated(model)
    assert old_is_fov == new_is_fov, (
        f"{name}: is_foveated disagrees — old={old_is_fov} new={new_is_fov}")

    selector = RandomSelector(is_foveated=new_is_fov, foveated_scale=fov_scale,
                              min_viewpoint_scale=0.1)
    branches = [ViewpointType.FULL, ViewpointType.RANDOM]

    random.seed(4321)
    torch.manual_seed(5678)
    worst, losses = 0.0, []
    for _ in range(n_steps):
        t = _batch()
        rng = (torch.get_rng_state(), random.getstate())

        opt.zero_grad()
        old = training_step(
            model=model, images=t["images"], scene_target=t["scene_target"],
            cls_target=t["cls_target"], raw_scene_target=t["raw_scene_target"],
            raw_cls_target=t["raw_cls_target"],
            scene_denorm=lambda x: x, cls_denorm=lambda x: x,
            enable_scene_patches_loss=True, enable_scene_cls_loss=True,
            glimpse_size_px=_GLIMPSE, canvas_grid_size=_G,
            n_full_start_branches=1, n_random_start_branches=1,
            chunk_size=2, continue_prob=0.5, min_viewpoint_scale=0.1,
            foveated_scale=fov_scale, amp_ctx=nullcontext(), collect_viz=False,
        ).total_loss.item()

        # Rewind every RNG the rollout consumes so the new path draws the SAME
        # trajectory length and the SAME viewpoints as the old one.
        opt.zero_grad()
        torch.set_rng_state(rng[0])
        random.setstate(rng[1])
        new = run_rollout(
            model=model, images=t["images"],
            task=BoundDistillTask(DistillTask(
                scene_target=t["scene_target"], cls_target=t["cls_target"],
                enable_scene_patches_loss=True, enable_scene_cls_loss=True)),
            selector=selector,
            bptt=BpttSpec(mode="chunked", chunk_size=2, continue_prob=0.5),
            branches=branches, canvas_grid_size=_G, amp_ctx=nullcontext(),
        ).total_loss.item()

        reldiff = abs(old - new) / max(abs(old), 1e-8)
        worst = max(worst, reldiff)
        losses.append(new.hex())

        # Both paths run chunked BPTT, which backwards at each chunk boundary — the
        # grads are already accumulated, so step directly (no explicit .backward()).
        opt.step()

    return {"is_foveated": new_is_fov, "max_reldiff": worst, "losses_hex": losses,
            "digest": hashlib.sha256("".join(losses).encode()).hexdigest()[:16]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--only", default=None, help="run a single config by name")
    args = ap.parse_args()

    torch.use_deterministic_algorithms(True)
    all_cfgs = configs()
    if args.only:
        all_cfgs = {args.only: all_cfgs[args.only]}

    results, failures = {}, []
    for name, (bb, cfg) in all_cfgs.items():
        try:
            r = run_config(name, bb, cfg, args.steps)
        except Exception as e:
            print(f"{name:22s} ERROR  {type(e).__name__}: {e}")
            failures.append(name)
            results[name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        ok = r["max_reldiff"] < _TOL
        failures += [] if ok else [name]
        print(f"{name:22s} fov={str(r['is_foveated']):5s} "
              f"max_reldiff={r['max_reldiff']:.2e}  digest={r['digest']}  "
              f"{'PASS' if ok else 'FAIL'}")
        results[name] = r

    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True, check=True).stdout.strip())
    out_dir = Path(__file__).parent / "parity"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"configs_{rev}{'-dirty' if dirty else ''}.json"
    out.write_text(json.dumps({"git_rev": rev, "steps": args.steps,
                               "torch": torch.__version__, "results": results}, indent=2) + "\n")
    print(f"\nwrote {out}")
    print("ALL PASS" if not failures else f"FAILURES: {', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
