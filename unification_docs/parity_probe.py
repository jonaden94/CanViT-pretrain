"""Deterministic CPU parity probe for the distill rollout (master plan §7/§8).

Runs the distill rollout N times with fully pinned RNG on a tiny CPU model + synthetic
batches and records every loss to full float precision. Run it BEFORE and AFTER any
refactor of the training path; the refactor is parity-clean iff the JSON records are
byte-identical (same seeds -> same trajectory lengths, same viewpoints, same losses).

Originally drove ``train/step.py::training_step``, which the harness consolidation
deleted. It now drives ``harness/rollout.py::run_rollout`` through the same distill
adapter as ``harness/tests/test_rollout_parity.py`` — the two are the same computation
(the test asserts this stream still hashes to ``9a0100a1a3de3acd``, the digest recorded
back when the old loop was the one producing it).

What that costs, stated plainly: the digest can still be REGENERATED and diffed across
commits, but it can no longer be re-derived from the original implementation, because
that implementation is gone. The recorded constant in the test is now the sole reference —
which is exactly the trade the design doc sanctioned, conditional on the harness having
reproduced it byte-for-byte first (it did).

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
from canvit_pretrain.harness.rollout import GlimpseOut, run_rollout
from canvit_pretrain.harness.spec import BpttSpec
from canvit_pretrain.train.config import FoveatedScaleConfig
from canvit_pretrain.train.selector import RandomSelector
from canvit_pretrain.train.task import DistillTask
from canvit_pretrain.train.viewpoint import ViewpointType


class _DistillAdapter:
    """Wraps a per-step ``DistillTask`` as a ``RolloutTask`` — verbatim from
    ``harness/tests/test_rollout_parity.py`` so the two cannot drift."""

    def __init__(self, task: DistillTask):
        self.task = task

    def forward_glimpse(self, *, model, images, state, viewpoint, backbone_no_grad):
        ctx = torch.no_grad() if backbone_no_grad else nullcontext()
        with ctx:
            out = model(image=images, state=state, viewpoint=viewpoint)
        return GlimpseOut(readout=out, state=out.state, vpe=out.vpe)

    def step_loss(self, readout):
        return self.task.step_loss(readout)

    def per_image_loss(self, readout):
        return self.task.per_image_loss(readout)

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

    selector = RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(),
                              min_viewpoint_scale=0.1)
    bptt = BpttSpec(mode="chunked", chunk_size=2, continue_prob=0.5)
    branches = [ViewpointType.FULL, ViewpointType.RANDOM]  # n_full=1, n_random=1 (full first)

    random.seed(4321)  # drives trajectory length (continue_prob draws)
    torch.manual_seed(5678)  # drives batches + viewpoint sampling
    losses: list[str] = []
    for _ in range(_N_STEPS):
        images = torch.randn(_B, 3, 224, 224, device=_DEVICE)
        scene_target = torch.randn(_B, _G * _G, _D, device=_DEVICE)
        cls_target = torch.randn(_B, _D, device=_DEVICE)
        # raw_* targets feed only distill's cos-sim metrics, never the loss stream — but
        # the draws must still happen, or every later batch shifts and the digest changes.
        _ = torch.randn(_B, _G * _G, _D, device=_DEVICE)
        _ = torch.randn(_B, _D, device=_DEVICE)

        task = _DistillAdapter(DistillTask(
            scene_target=scene_target, cls_target=cls_target,
            enable_scene_patches_loss=True, enable_scene_cls_loss=True,
        ))
        opt.zero_grad()
        result = run_rollout(
            model=model, images=images, task=task, selector=selector,
            bptt=bptt, branches=branches, canvas_grid_size=_G, amp_ctx=nullcontext(),
        )
        opt.step()
        losses.append(result.total_loss.item().hex())

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
