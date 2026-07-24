"""CPU tests for the harness's operational features (parity with train/loop.py):
per-module grad norms, EMA-smoothed loss, SIGUSR1-style checkpoint-on-signal, and the
resume start-step hook. The full run()-level resume (find_latest -> restore -> continue)
is validated on real data by the GPU resume smoke.
"""

import canvit_pretrain.harness.loop as L
import torch
from canvit_pytorch import CanViTForSemanticSegmentation
from torch import nn

from canvit_pretrain.ade20k.config import Ade20kConfig
from canvit_pretrain.ade20k.data import IGNORE_LABEL, NUM_CLASSES
from canvit_pretrain.harness.loop import (
    apply_requires_grad,
    grad_norms_by_module,
    request_checkpoint,
    run_training_loop,
)
from canvit_pretrain.harness.optim import build_optimizer_and_scheduler
from canvit_pretrain.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TrainSpec
from canvit_pretrain.in1k.config import In1kConfig
from canvit_pretrain.tasks.ade20k.task import Ade20kRunTask, BoundAde20kTask
from canvit_pretrain.tasks.distill.task import DistillRunTask
from canvit_pretrain.tasks.in1k.task import In1kRunTask
from canvit_pretrain.train.config import Config, FoveatedScaleConfig
from canvit_pretrain.train.selector import RandomSelector
from canvit_pretrain.train.viewpoint import ViewpointType

_B, _G, _IMG = 2, 8, 224


# --- grad_norms_by_module (pure) ------------------------------------------
def test_grad_norms_grouping_and_deep_prefixes():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
            self.head = nn.Linear(4, 2)

    net = Net()
    net.head(net.backbone(torch.randn(3, 4))).pow(2).sum().backward()
    g = grad_norms_by_module(net, depth=1)
    assert {"backbone", "head"} <= set(g) and all(v > 0 for v in g.values())
    # deep_prefixes zooms into backbone one level deeper without changing 'head'.
    g2 = grad_norms_by_module(net, depth=1, deep_prefixes=("backbone",))
    assert "backbone.0" in g2 and "backbone.1" in g2 and "head" in g2
    assert "backbone" not in g2


# --- tiny ADE20K vertical for the loop-level ops --------------------------
class _StubTask:
    def batch_images(self, batch, device):
        return batch[0].to(device)

    def bind(self, batch, device, *, model, head):
        return BoundAde20kTask(seg=model, masks=batch[1].to(device), canvas_grid=_G)


def _batches():
    torch.manual_seed(2)
    while True:
        m = torch.randint(0, NUM_CLASSES, (_B, _IMG, _IMG))
        yield (torch.randn(_B, 3, _IMG, _IMG), m)


def _setup():
    torch.manual_seed(0)
    seg = CanViTForSemanticSegmentation(backbone_name="vits16", model_config={}, num_classes=NUM_CLASSES)
    spec = TrainSpec.probe(
        bptt=BpttSpec(mode="none", horizon=2),
        optim={"head": GroupOptim(lr=1e-2, schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=1))},
    )
    apply_requires_grad(model=seg, head=seg.head, joint=None, spec=spec)
    opt, sched = build_optimizer_and_scheduler(spec, {"head": list(seg.head.parameters())})
    sel = RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.05)
    return seg, spec, opt, sched, sel


def test_ema_and_grad_norms_surface_in_metrics():
    seg, spec, opt, sched, sel = _setup()
    seen: list[dict] = []
    run_training_loop(
        task=_StubTask(), model=seg, head=seg.head, optimizer=opt, scheduler=sched, selector=sel,
        spec=spec, branches=[ViewpointType.RANDOM], canvas_grid=_G, device=torch.device("cpu"),
        train_batches=_batches(), n_steps=3, log_every=1, ema_alpha=0.5, log_grad_norms=True,
        on_log=lambda step, m: seen.append(m),
    )
    assert seen and "total_loss_ema" in seen[-1]
    assert any(k.startswith("grad_norm/") for k in seen[-1])  # head at least


def test_signal_checkpoint_saves_midrun(tmp_path):
    seg, spec, opt, sched, sel = _setup()
    L._checkpoint_requested = False  # isolate from other tests (module global)
    try:
        run_training_loop(
            task=_StubTask(), model=seg, head=seg.head, optimizer=opt, scheduler=sched, selector=sel,
            spec=spec, branches=[ViewpointType.RANDOM], canvas_grid=_G, device=torch.device("cpu"),
            train_batches=_batches(), n_steps=3, log_every=1, ckpt_dir=tmp_path, ckpt_every=0,
            # request a checkpoint mid-run (as SIGUSR1 would) at step 0
            on_log=lambda step, m: request_checkpoint() if step == 0 else None,
        )
        # ckpt_every=0 => normally only the end (step-3) is written; the signal save
        # produced step-0 mid-run.
        assert (tmp_path / "step-0.pt").exists(), "signal-triggered mid-run checkpoint missing"
        assert (tmp_path / "step-3.pt").exists(), "end-of-run checkpoint missing"
    finally:
        L._checkpoint_requested = False


def test_resume_start_step_hook_returns_scheduler_epoch():
    class _Sched:
        last_epoch = 7

    tasks = [
        Ade20kRunTask(Ade20kConfig(tracker="none")),
        In1kRunTask(In1kConfig(tracker="none")),
        DistillRunTask(Config(webdataset_dir="/nonexistent")),
    ]
    for t in tasks:
        assert t.resume_start_step({}, _Sched()) == 7, t.name
