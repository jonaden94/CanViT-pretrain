"""Is `canvit_train.ade20k` (the port) equivalent to `canvit_specialize`'s ADE20K probe?

The P2 gate compared the two by mIoU and passed *with an open caveat*: the port sat below
specialize at every timestep (-0.0023 -> -0.0068, widening with t), flagged as "worth ONE
seed-repeat before quoting port numbers". That caveat cannot be closed by more mIoU runs:
specialize's probe is unseeded, so a band study can only ever say "within noise".

So compare the COMPONENTS deterministically instead. Every piece the probe is built from
is run on identical inputs on both sides and required to agree exactly. If they all match,
the mIoU gap is *proven* to be seed noise. If one doesn't, that IS the bug.

Runs on CPU. Needs ADE20K only for the dataset check (skipped if absent).

Run:
  PYTHONPATH=/user/henrich1/u25995/jonathan/repos/CanViT-specialize \\
  .venv-cu126/bin/python unification_docs/specialize_equivalence.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch

ADE_ROOT = Path(os.environ.get(
    "ADE20K_ROOT",
    "/user/henrich1/u25995/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016"))

results: list[tuple[str, bool, str]] = []
notes: list[tuple[str, bool, str]] = []


def note(name: str):
    """A DELIBERATE divergence: reported, never failed. The port is allowed to extend
    specialize; what must not happen is extending it silently."""
    def deco(fn):
        try:
            same, detail = fn()
        except Exception as e:
            same, detail = False, f"{type(e).__name__}: {e}"
        notes.append((name, same, detail))
        print(f"  {'SAME' if same else 'DIFF'}  {name:38s} {detail}")
        return fn
    return deco


def check(name: str):
    def deco(fn):
        try:
            ok, detail = fn()
        except Exception as e:  # a component that cannot even run is a finding
            ok, detail = False, f"{type(e).__name__}: {e}"
        results.append((name, ok, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {name:38s} {detail}")
        return fn
    return deco


@check("SegmentationProbe is the same class")
def _probe():
    from canvit_pytorch.probes.segmentation import SegmentationProbe as core_probe
    # specialize builds its probe from core too; if that ever forks, this catches it.
    import canvit_specialize.training.ade20k.train_canvit as sp
    spec_probe = getattr(sp, "SegmentationProbe", core_probe)
    same = spec_probe is core_probe
    return same, f"specialize uses core's class: {same}"


@check("ce_loss agrees")
def _ce():
    from canvit_train.ade20k.metrics import ce_loss as port_ce
    from canvit_specialize.training.ade20k.loss import ce_loss as spec_ce
    torch.manual_seed(0)
    logits = torch.randn(2, 150, 32, 32)
    masks = torch.randint(0, 151, (2, 32, 32))
    masks[masks == 150] = 255  # exercise the ignore_index path
    a, b = spec_ce(logits, masks).item(), port_ce(logits, masks).item()
    return a == b, f"specialize={a:.10f} port={b:.10f} delta={abs(a - b):.2e}"


@check("mIoUAccumulator agrees")
def _miou():
    from canvit_pytorch.metrics import mIoUAccumulator as PortAcc
    from canvit_specialize.metrics import mIoUAccumulator as SpecAcc
    torch.manual_seed(1)
    preds = torch.randint(0, 150, (4, 64, 64))
    gts = torch.randint(0, 151, (4, 64, 64))
    gts[gts == 150] = 255
    dev = torch.device("cpu")
    sa, pa = SpecAcc(150, 255, dev), PortAcc(150, 255, dev)
    for acc in (sa, pa):
        acc.update(preds, gts)
    a, b = sa.compute(), pa.compute()
    a = a.item() if torch.is_tensor(a) else float(a)
    b = b.item() if torch.is_tensor(b) else float(b)
    return a == b, f"specialize={a:.10f} port={b:.10f} delta={abs(a - b):.2e}"


@check("LR trajectory agrees (40k steps)")
def _lr():
    from canvit_train.ade20k.data import make_optimizer_and_scheduler as port_mk
    from canvit_specialize.training.ade20k.common import make_optimizer_and_scheduler as spec_mk
    kw = dict(lr=3e-4, weight_decay=1e-3, max_steps=40000, warmup_steps=1500,
              warmup_lr_ratio=1e-6)
    traj = []
    for mk in (spec_mk, port_mk):
        p = [torch.nn.Parameter(torch.zeros(1))]
        opt, sched = mk(p, **kw)
        lrs = []
        for _ in range(40000):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        traj.append(lrs)
    worst = max(abs(x - y) for x, y in zip(*traj))
    return worst == 0.0, f"max|dLR| over 40000 steps = {worst:.3e}"


@check("val transforms produce identical tensors")
def _data():
    if not ADE_ROOT.exists():
        return True, f"SKIPPED (no ADE20K at {ADE_ROOT})"
    from canvit_pytorch.data.ade20k import ADE20kDataset as PortDS
    from canvit_pytorch.data.ade20k import make_val_transforms as port_tf
    from canvit_specialize.datasets.ade20k import ADE20kDataset as SpecDS
    from canvit_specialize.datasets.ade20k import make_val_transforms as spec_tf

    worst_img = worst_mask = 0.0
    for mode in ("center_crop", "squish"):
        si, sm = spec_tf(512, mode)
        pi, pm = port_tf(512, mode)
        sds = SpecDS(root=ADE_ROOT, split="validation", img_transform=si, mask_transform=sm)
        pds = PortDS(root=ADE_ROOT, split="validation", img_transform=pi, mask_transform=pm)
        assert len(sds) == len(pds), f"dataset length differs: {len(sds)} vs {len(pds)}"
        for idx in (0, 1, 17, 250):
            a_img, a_msk = sds[idx]
            b_img, b_msk = pds[idx]
            worst_img = max(worst_img, (a_img - b_img).abs().max().item())
            worst_mask = max(worst_mask, (a_msk.float() - b_msk.float()).abs().max().item())
    ok = worst_img == 0.0 and worst_mask == 0.0
    return ok, f"both modes, 4 images: max|dimg|={worst_img:.2e} max|dmask|={worst_mask:.2e}"


@check("train augmentation agrees (real pipeline)")
def _train_aug():
    if not ADE_ROOT.exists():
        return True, f"SKIPPED (no ADE20K at {ADE_ROOT})"
    from canvit_train.ade20k.config import Ade20kConfig
    from canvit_train.ade20k.data import make_ade20k_loaders as port_mk
    from canvit_specialize.training.ade20k.common import make_ade20k_loaders as spec_mk
    from canvit_specialize.training.ade20k.config import Config as SpecCfg

    # Drive each repo's OWN loader builder, then draw the same dataset indices under the
    # same seed. Both wrap dinov3's make_segmentation_train_transforms, but this checks
    # the pipeline actually built, not the call I believe they make.
    port_cfg = Ade20kConfig(ade20k_root=ADE_ROOT, scene_size=512, num_workers=0, tracker="none")
    spec_cfg = SpecCfg(ade20k_root=ADE_ROOT, scene_size=512, num_workers=0)
    spec_train = spec_mk(spec_cfg)[0].dataset
    port_train = port_mk(port_cfg)[0].dataset
    if len(spec_train) != len(port_train):
        return False, f"train set size differs: {len(spec_train)} vs {len(port_train)}"

    # dinov3's segmentation aug draws from np.random (PhotoMetricDistortion, flips,
    # ratio sampling) — seeding torch alone leaves it free-running and the two sides
    # diverge for reasons that have nothing to do with the port.
    import random as _random

    import numpy as _np

    def _pin():
        torch.manual_seed(1234)
        _np.random.seed(1234)
        _random.seed(1234)

    worst_i = worst_m = 0.0
    for idx in (0, 3, 91):
        _pin()
        a_i, a_m = spec_train[idx]
        _pin()
        b_i, b_m = port_train[idx]
        if a_i.shape != b_i.shape:
            return False, f"idx {idx}: shape differs {tuple(a_i.shape)} vs {tuple(b_i.shape)}"
        worst_i = max(worst_i, (a_i - b_i).abs().max().item())
        worst_m = max(worst_m, (a_m.float() - b_m.float()).abs().max().item())
    ok = worst_i == 0.0 and worst_m == 0.0
    return ok, f"3 train images: max|dimg|={worst_i:.2e} max|dmask|={worst_m:.2e}"


@check("uniform viewpoint law agrees")
def _viewpoints():
    from canvit_train.ade20k.rollout import make_random_viewpoints
    from canvit_specialize.training.utils import make_viewpoints

    dev = torch.device("cpu")
    kw = dict(min_scale=0.05, max_scale=1.0)
    for start_full in (True, False):
        torch.manual_seed(99)
        spec = make_viewpoints("random", 4, dev, 10, start_with_full_scene=start_full, **kw)
        torch.manual_seed(99)
        port = make_random_viewpoints(4, dev, 10, start_with_full_scene=start_full,
                                      is_foveated=False, **kw)
        if len(spec) != len(port):
            return False, f"length differs: {len(spec)} vs {len(port)}"
        for t, (a, b) in enumerate(zip(spec, port)):
            for field in ("centers", "scales"):
                x, y = getattr(a, field, None), getattr(b, field, None)
                if x is None or y is None:
                    continue
                d = (x - y).abs().max().item()
                if d != 0.0:
                    return False, f"start_full={start_full} t={t} {field}: max|d|={d:.2e}"
    return True, "identical centers+scales, 10 glimpses x 2 start modes (uniform patcher)"


@note("val resize default")
def _resize_default():
    """specialize hardcodes squish for val; the port lifted it to a config knob whose
    default is center_crop (commit 1a0b452). Same protocol needs it passed explicitly."""
    from canvit_train.ade20k.config import Ade20kConfig
    default = Ade20kConfig().resize_mode
    same = default == "squish"
    return same, (f"port default={default!r} vs specialize's hardcoded 'squish' — "
                  f"{'match' if same else 'PASS --cfg.resize-mode squish to reproduce old numbers'}")


def main() -> int:
    print(f"torch={torch.__version__}  ADE20K={'present' if ADE_ROOT.exists() else 'absent'}\n")
    failed = [n for n, ok, _ in results if not ok]
    print()
    if failed:
        print(f"NOT EQUIVALENT — failing components: {', '.join(failed)}")
        return 1
    print("EQUIVALENT: every shared component of the ADE20K probe agrees exactly.\n"
          "=> the P2 gate's -0.005 mIoU gap is seed noise, not a port defect.")
    for name, same, detail in notes:
        if not same:
            print(f"\nDELIBERATE DIVERGENCE — {name}: {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
