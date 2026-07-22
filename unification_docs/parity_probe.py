"""Deterministic CPU parity probe for the P1 harness refactor (master plan §7/§8).

Runs training_step N times with fully pinned RNG on a tiny CPU model + synthetic
batches and records every loss to full float precision. Run it BEFORE and AFTER
any refactor of the training loop; the refactor is parity-clean iff the JSON
records are byte-identical (same seeds -> same trajectory lengths, same
viewpoints, same losses).

Usage (from the CanViT-pretrain repo root):

    .venv-cu126/bin/python unification_docs/parity_probe.py

Writes unification_docs/parity/record_<git-rev>[ -dirty ].json and prints a
digest. Compare records with a plain diff.
"""

import hashlib
import json
import random
import subprocess
from contextlib import nullcontext
from pathlib import Path

import torch
from canvit_pytorch import create_backbone

from canvit_pretrain import CanViTForPretraining, CanViTForPretrainingConfig
from canvit_pretrain.train.config import FoveatedScaleConfig
from canvit_pretrain.train.step import training_step

_B, _G, _D = 2, 8, 384
_N_STEPS = 25
_DEVICE = torch.device("cpu")


def _build_model() -> CanViTForPretraining:
    torch.manual_seed(1234)
    backbone = create_backbone("vits16").to(_DEVICE)
    cfg = CanViTForPretrainingConfig(teacher_dim=_D)
    return CanViTForPretraining(
        backbone=backbone,
        cfg=cfg,
        glimpse_size_px=128,
        backbone_name="vits16",
        canvas_patch_grid_sizes=[_G],
    ).to(_DEVICE)


def main() -> None:
    torch.use_deterministic_algorithms(True)
    model = _build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    random.seed(4321)  # drives trajectory length (continue_prob draws)
    torch.manual_seed(5678)  # drives batches + viewpoint sampling
    losses: list[str] = []
    for _ in range(_N_STEPS):
        tensors = {
            "images": torch.randn(_B, 3, 224, 224, device=_DEVICE),
            "scene_target": torch.randn(_B, _G * _G, _D, device=_DEVICE),
            "cls_target": torch.randn(_B, _D, device=_DEVICE),
            "raw_scene_target": torch.randn(_B, _G * _G, _D, device=_DEVICE),
            "raw_cls_target": torch.randn(_B, _D, device=_DEVICE),
        }
        opt.zero_grad()
        metrics = training_step(
            model=model,
            images=tensors["images"],
            scene_target=tensors["scene_target"],
            cls_target=tensors["cls_target"],
            raw_scene_target=tensors["raw_scene_target"],
            raw_cls_target=tensors["raw_cls_target"],
            scene_denorm=lambda x: x,
            cls_denorm=lambda x: x,
            enable_scene_patches_loss=True,
            enable_scene_cls_loss=True,
            glimpse_size_px=128,
            canvas_grid_size=_G,
            n_full_start_branches=1,
            n_random_start_branches=1,
            chunk_size=2,
            continue_prob=0.5,
            min_viewpoint_scale=0.1,
            foveated_scale=FoveatedScaleConfig(),
            amp_ctx=nullcontext(),
            collect_viz=False,
        )
        opt.step()
        losses.append(metrics.total_loss.item().hex())

    rev = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
    )
    record = {
        "git_rev": rev + ("-dirty" if dirty else ""),
        "n_steps": _N_STEPS,
        "torch": torch.__version__,
        "losses_hex": losses,  # full precision — byte-diffable
    }
    out_dir = Path(__file__).parent / "parity"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"record_{record['git_rev']}.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    digest = hashlib.sha256("".join(losses).encode()).hexdigest()[:16]
    print(f"wrote {out}")
    print(f"loss-stream sha256[:16] = {digest}")


if __name__ == "__main__":
    main()
