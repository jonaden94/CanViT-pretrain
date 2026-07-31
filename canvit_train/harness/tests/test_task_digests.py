"""Pinning digests for the ade20k / in1k rollouts — the refactor guard.

WHAT THESE ARE. ``test_rollout_parity.py``'s distill digest (``9a0100a1a3de3acd``)
originally certified agreement with the now-deleted standalone trainer. ade20k and in1k
have no old side left to agree with, so the digests here are **pinning** digests: they
assert only that today's numbers equal the numbers recorded when this file was written.
That is exactly the guarantee a pure file-move refactor needs, and nothing more — a
pre-existing bug is pinned in along with everything else. They are NOT a correctness
claim.

Recorded 2026-07-31 on the tree at ``8554c1f``, immediately before the package
restructure (``train/`` split into ``harness/`` + ``distill/``). If the restructure is a
pure move, every digest below is unchanged.

WHAT EACH DIGEST COVERS. Both halves of the training path:
  * the per-step ``total_loss`` stream — forward numerics, glimpse routing, BPTT
    chunking, loss reduction;
  * a fingerprint of every parameter after N optimizer steps — so a change to gradient
    flow (what is frozen, where ``detach`` lands, how chunks normalise) moves it too,
    even if step 0's loss is untouched.

FIDELITY. The spec of each config (bptt mode, branch composition, what is frozen) is the
production one from ``unification_docs/capability_matrix.md``. ``horizon`` and step count
are cut down to keep this a CPU unit test; the digest's job is to detect change, not to
reproduce a training curve.

Run: ``.venv-cu126/bin/python -m pytest canvit_train/harness/tests/test_task_digests.py``
"""

import hashlib
from contextlib import nullcontext

import torch
from canvit_pytorch import CanViTForImageClassification, CanViTForSemanticSegmentation

from canvit_train.ade20k.data import IGNORE_LABEL, NUM_CLASSES
from canvit_train.harness.rollout import run_rollout
from canvit_train.harness.spec import BpttSpec
from canvit_train.tasks.ade20k.task import BoundAde20kTask
from canvit_train.tasks.in1k.task import BoundIn1kTask
from canvit_train.train.config import FoveatedScaleConfig
from canvit_train.train.selector import RandomSelector
from canvit_train.train.viewpoint import ViewpointType

_B, _G, _IMG, _N_CLS = 2, 8, 224, 10
_N_STEPS = 6
_HORIZON = 3
_DEVICE = torch.device("cpu")

# Recorded on 8554c1f. A pure file move must not change these.
_EXPECTED = {
    "ade20k_probe": "b9fd07bdac4f68bd",
    "ade20k_finetune": "28fb8bff5010a010",
    "in1k_probe": "00ed4e8f2279b20f",
    "in1k_finetune": "6f4accd7c2ad3dba",
}


def _selector() -> RandomSelector:
    return RandomSelector(
        is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.1
    )


def _param_fingerprint(model: torch.nn.Module) -> str:
    """sha256 over every parameter's raw bytes, in sorted-name order. Sensitive to any
    numeric drift, including one that only shows up after several optimizer steps."""
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        h.update(name.encode())
        h.update(p.detach().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def _run_digest(*, model, make_task, bptt: BpttSpec, trainable, seed: int) -> str:
    """N seeded steps of ``run_rollout`` + ``opt.step()``; hash the loss stream and the
    resulting weights together."""
    torch.use_deterministic_algorithms(True)
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    selector = _selector()

    torch.manual_seed(seed)  # batches + viewpoint sampling + dropout
    losses: list[str] = []
    for _ in range(_N_STEPS):
        images = torch.randn(_B, 3, _IMG, _IMG, device=_DEVICE)
        task = make_task()
        opt.zero_grad()
        result = run_rollout(
            model=model, images=images, task=task, selector=selector, bptt=bptt,
            branches=[ViewpointType.FULL],  # both tasks: train_start_full=True
            canvas_grid_size=_G, amp_ctx=nullcontext(),
        )
        opt.step()
        assert torch.isfinite(result.total_loss), "digest is meaningless on a NaN stream"
        losses.append(result.total_loss.item().hex())

    return hashlib.sha256(
        ("".join(losses) + _param_fingerprint(model)).encode()
    ).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# ADE20K — probe (head only, bptt=none) and finetune (backbone+head, bptt=full).
# --------------------------------------------------------------------------- #
def _seg() -> CanViTForSemanticSegmentation:
    torch.manual_seed(0)
    return CanViTForSemanticSegmentation(
        backbone_name="vits16", model_config={}, num_classes=NUM_CLASSES
    ).to(_DEVICE)


def _seg_masks() -> torch.Tensor:
    m = torch.randint(0, NUM_CLASSES, (_B, _IMG, _IMG), device=_DEVICE)
    m[:, :8] = IGNORE_LABEL
    return m


def ade20k_probe_digest() -> str:
    seg = _seg()
    seg.canvit.requires_grad_(False)
    seg.canvit.eval()  # frozen backbone runs in eval mode (BN), as the probe recipe does
    return _run_digest(
        model=seg,
        make_task=lambda: BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G),
        bptt=BpttSpec(mode="none", horizon=_HORIZON),
        trainable=list(seg.head.parameters()),
        seed=101,
    )


def ade20k_finetune_digest() -> str:
    seg = _seg()
    return _run_digest(
        model=seg,
        make_task=lambda: BoundAde20kTask(seg=seg, masks=_seg_masks(), canvas_grid=_G),
        bptt=BpttSpec(mode="full", horizon=_HORIZON),
        trainable=list(seg.parameters()),
        seed=102,
    )


# --------------------------------------------------------------------------- #
# IN1k — probe (head only, bptt=none) and finetune (backbone+head, bptt=full).
# --------------------------------------------------------------------------- #
def _clf() -> CanViTForImageClassification:
    torch.manual_seed(0)
    return CanViTForImageClassification(
        backbone_name="vits16", model_config={}, n_classes=_N_CLS, glimpse_grid_size=_G,
    ).to(_DEVICE)


def _targets() -> torch.Tensor:
    return torch.randint(0, _N_CLS, (_B,), device=_DEVICE)


def in1k_probe_digest() -> str:
    clf = _clf()
    clf.canvit.requires_grad_(False)
    clf.canvit.eval()
    return _run_digest(
        model=clf,
        make_task=lambda: BoundIn1kTask(clf=clf, targets=_targets(), canvas_grid=_G),
        bptt=BpttSpec(mode="none", horizon=_HORIZON),
        trainable=list(clf.head.parameters()),
        seed=201,
    )


def in1k_finetune_digest() -> str:
    clf = _clf()
    return _run_digest(
        model=clf,
        make_task=lambda: BoundIn1kTask(clf=clf, targets=_targets(), canvas_grid=_G),
        bptt=BpttSpec(mode="full", horizon=_HORIZON),
        trainable=list(clf.parameters()),
        seed=202,
    )


DIGESTS = {
    "ade20k_probe": ade20k_probe_digest,
    "ade20k_finetune": ade20k_finetune_digest,
    "in1k_probe": in1k_probe_digest,
    "in1k_finetune": in1k_finetune_digest,
}


def test_ade20k_probe_digest():
    assert ade20k_probe_digest() == _EXPECTED["ade20k_probe"]


def test_ade20k_finetune_digest():
    assert ade20k_finetune_digest() == _EXPECTED["ade20k_finetune"]


def test_in1k_probe_digest():
    assert in1k_probe_digest() == _EXPECTED["in1k_probe"]


def test_in1k_finetune_digest():
    assert in1k_finetune_digest() == _EXPECTED["in1k_finetune"]


if __name__ == "__main__":  # record / re-check the digests
    for name, fn in DIGESTS.items():
        print(f"{name:20s} {fn()}")
