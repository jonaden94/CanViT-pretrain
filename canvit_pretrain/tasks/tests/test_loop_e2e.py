"""End-to-end CPU smoke of the harness vertical (design §Loop): the neutral
``run_training_loop`` drives an ADE20K probe on synthetic data through the engine +
per-group optimizer, trains only the head, and the local checkpoint round-trips.

Proves loop → task.bind → run_rollout → optimizer/scheduler → checkpoint wiring on
CPU. Real-data ``build_loaders``/``evaluate`` + the ``python -m canvit_pretrain.train``
dispatch are validated against the datasets separately.
"""

import torch
from canvit_pytorch import CanViTForSemanticSegmentation

from canvit_pretrain.ade20k.data import IGNORE_LABEL, NUM_CLASSES
from canvit_pretrain.harness.checkpoint import find_latest, load_checkpoint, restore_into
from canvit_pretrain.harness.loop import apply_requires_grad, run_training_loop
from canvit_pretrain.harness.optim import build_optimizer_and_scheduler
from canvit_pretrain.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TrainSpec
from canvit_pretrain.tasks.ade20k.task import BoundAde20kTask
from canvit_pretrain.train.config import FoveatedScaleConfig
from canvit_pretrain.train.selector import RandomSelector
from canvit_pretrain.train.viewpoint import ViewpointType

_B, _G, _IMG = 2, 8, 224


class _StubAdeTask:
    """Minimal Task surface the loop needs (the run-level Task wrapper adds
    build_model/build_loaders/evaluate; here we supply a synthetic batch stream)."""

    def batch_images(self, batch, device):
        return batch[0].to(device)

    def bind(self, batch, device, *, model, head):
        return BoundAde20kTask(seg=model, masks=batch[1].to(device), canvas_grid=_G)


def _batches():
    torch.manual_seed(2)
    while True:
        m = torch.randint(0, NUM_CLASSES, (_B, _IMG, _IMG))
        m[:, :8] = IGNORE_LABEL
        yield (torch.randn(_B, 3, _IMG, _IMG), m)


def _seg():
    torch.manual_seed(0)
    return CanViTForSemanticSegmentation(backbone_name="vits16", model_config={}, num_classes=NUM_CLASSES)


def test_loop_trains_probe_and_checkpoint_roundtrips(tmp_path):
    seg = _seg()
    spec = TrainSpec.probe(
        bptt=BpttSpec(mode="none", horizon=2),
        optim={"head": GroupOptim(lr=1e-2, schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=1))},
    )
    apply_requires_grad(model=seg, head=seg.head, joint=None, spec=spec)
    opt, sched = build_optimizer_and_scheduler(spec, {"head": list(seg.head.parameters())})
    selector = RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.05)

    before = next(seg.head.parameters()).detach().clone()
    last = run_training_loop(
        task=_StubAdeTask(), model=seg, head=seg.head, optimizer=opt, scheduler=sched,
        selector=selector, spec=spec, branches=[ViewpointType.FULL], canvas_grid=_G,
        device=torch.device("cpu"), train_batches=_batches(), n_steps=4,
        task_name="ade20k", model_config={"num_classes": NUM_CLASSES}, ckpt_dir=tmp_path,
    )
    assert torch.isfinite(torch.tensor(last["total_loss"])) and last["step"] == 3
    assert not torch.allclose(before, next(seg.head.parameters()))       # head trained
    assert all(p.grad is None for p in seg.canvit.parameters())          # backbone frozen throughout

    # end-of-run checkpoint round-trips into a fresh model.
    latest = find_latest(tmp_path)
    assert latest is not None and latest.name == "step-4.pt"
    payload = load_checkpoint(latest, "cpu")
    assert payload["task"] == "ade20k" and payload["step"] == 4
    assert payload["train_spec"]["train_head"] is True and payload["train_spec"]["train_backbone"] is False
    assert payload["model_config"] == {"num_classes": NUM_CLASSES}

    fresh = _seg()
    restore_into(payload, model=fresh)
    for (n, p), (_, q) in zip(seg.head.named_parameters(), fresh.head.named_parameters()):
        assert torch.allclose(p, q), f"restored head mismatch at {n}"
