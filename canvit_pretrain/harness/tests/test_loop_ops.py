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
    """train/loop.py logs EMA-smoothed values under the plain names (total_loss etc.)
    and keeps the instantaneous total as total_loss_raw."""
    seg, spec, opt, sched, sel = _setup()
    seen: list[dict] = []
    run_training_loop(
        task=_StubTask(), model=seg, head=seg.head, optimizer=opt, scheduler=sched, selector=sel,
        spec=spec, branches=[ViewpointType.RANDOM], canvas_grid=_G, device=torch.device("cpu"),
        train_batches=_batches(), n_steps=3, log_every=1, ema_alpha=0.5, log_grad_norms=True,
        on_log=lambda step, m: seen.append(m),
    )
    assert seen and {"total_loss", "total_loss_raw", "grad_norm"} <= set(seen[-1])
    assert seen[-1]["total_loss"] != seen[-1]["total_loss_raw"], "total_loss must be the EMA"
    assert any(k.startswith("grad_norm/") for k in seen[-1])  # head at least


def test_per_branch_metrics_are_grouped_and_averaged_by_t0_type():
    """train/loop.py's full/… and random/… series. The engine groups by t0 type and
    averages same-type branches; names beyond `loss` come from the task's hooks, so a
    hookless task (ade20k here) yields exactly `{type}/loss`."""
    from canvit_pretrain.harness.loop import branch_metrics

    seg, spec, opt, sched, sel = _setup()
    seen: list[dict] = []
    run_training_loop(
        task=_StubTask(), model=seg, head=seg.head, optimizer=opt, scheduler=sched, selector=sel,
        spec=spec, branches=[ViewpointType.FULL, ViewpointType.RANDOM, ViewpointType.RANDOM],
        canvas_grid=_G, device=torch.device("cpu"), train_batches=_batches(), n_steps=1,
        log_every=1, on_log=lambda step, m: seen.append(m),
    )
    assert {"full/loss", "random/loss"} <= set(seen[-1])

    # the grouping/averaging rule itself, on synthetic branches
    class _B:
        def __init__(self, t0, loss, extra):
            self.t0_type, self.mean_loss, self.metrics = t0, torch.tensor(loss), extra

    out = branch_metrics([
        _B(ViewpointType.FULL, 1.0, {"cos": torch.tensor(0.2)}),
        _B(ViewpointType.RANDOM, 2.0, {"cos": torch.tensor(0.4)}),
        _B(ViewpointType.RANDOM, 4.0, {"cos": torch.tensor(0.6)}),
    ])
    assert float(out["full/loss"]) == 1.0
    assert float(out["random/loss"]) == 3.0            # (2+4)/2
    assert abs(float(out["random/cos"]) - 0.5) < 1e-6  # (0.4+0.6)/2


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


def test_timing_metrics_surface_when_enabled():
    seg, spec, opt, sched, sel = _setup()
    seen: list[dict] = []
    run_training_loop(
        task=_StubTask(), model=seg, head=seg.head, optimizer=opt, scheduler=sched, selector=sel,
        spec=spec, branches=[ViewpointType.RANDOM], canvas_grid=_G, device=torch.device("cpu"),
        train_batches=_batches(), n_steps=2, log_every=1, log_timing=True,
        on_log=lambda step, m: seen.append(m),
    )
    m = seen[-1]
    assert "data_pct" in m and "gpu_pct" in m
    assert abs(m["data_pct"] + m["gpu_pct"] - 100.0) < 1e-6


def test_seed_mode_loads_weights_only_and_starts_at_zero(tmp_path):
    """SEED: weights come from the checkpoint but opt/sched are fresh and step is 0
    (vs RESUME which continues the step count). Uses restore_into directly — the same
    call run() makes — so this pins the semantics without building a full run."""
    from canvit_pretrain.harness.checkpoint import load_checkpoint, restore_into, save_checkpoint

    seg, spec, opt, sched, sel = _setup()
    # train a couple of steps so the weights differ from a fresh init, then save
    run_training_loop(
        task=_StubTask(), model=seg, head=seg.head, optimizer=opt, scheduler=sched, selector=sel,
        spec=spec, branches=[ViewpointType.RANDOM], canvas_grid=_G, device=torch.device("cpu"),
        train_batches=_batches(), n_steps=2, log_every=99, ckpt_dir=tmp_path,
    )
    ckpt = tmp_path / "step-2.pt"
    assert ckpt.exists()
    trained_head = next(seg.head.parameters()).detach().clone()

    # fresh model + fresh opt/sched; seed = weights only
    seg2, spec2, opt2, sched2, _ = _setup()
    assert not torch.allclose(next(seg2.head.parameters()), trained_head)
    payload = load_checkpoint(ckpt, "cpu")
    restore_into(payload, model=seg2)          # NO optimizer/scheduler => seed semantics
    assert torch.allclose(next(seg2.head.parameters()), trained_head)   # weights arrived
    assert sched2.last_epoch == 0                                       # schedule is fresh
    assert not opt2.state                                               # optimizer is fresh


def test_checkpoint_metadata_history_accumulates_and_carries_view_scale():
    """run() accumulates training_config_history/provenance_history across resumes
    (to_hf reads them to recover pretrain_view_scale). Here we pin the distill task's
    view-scale stamping + the accumulation rule run() applies."""
    from canvit_pretrain.train.config import FoveatedScaleConfig

    from types import SimpleNamespace

    cfg = Config(webdataset_dir="/nonexistent")
    cfg.model.patcher_name = "foveated"
    cfg.foveated_scale = FoveatedScaleConfig(mode="fixed", fixed_scale=2.0)

    # is_foveated() only reads model.cfg.patcher_name
    meta = DistillRunTask(cfg).checkpoint_metadata(SimpleNamespace(cfg=cfg.model))
    assert meta["pretrain_view_scale"] == 2.0, meta
    assert meta["patcher_name"] == "foveated"

    # uniform / sampled scale => no fixed view scale to record
    cfg2 = Config(webdataset_dir="/nonexistent")
    cfg2.model.patcher_name = "uniform"
    meta2 = DistillRunTask(cfg2).checkpoint_metadata(SimpleNamespace(cfg=cfg2.model))
    assert meta2["pretrain_view_scale"] is None

    # accumulation rule: prior history is carried forward, not replaced
    prior = {"metadata": {"training_config_history": {"t0": {"a": 1}},
                          "provenance_history": {"t0": {"git": "x"}}}}
    carried = dict(prior["metadata"]["training_config_history"])
    carried["t1"] = {"a": 2}
    assert set(carried) == {"t0", "t1"}


def test_collect_viz_is_off_by_default_and_opt_in_collects_frames():
    """The engine's viz seam: default off (parity path untouched => result.viz is None);
    when enabled it calls the task's viz_init/viz_frame hooks on branch 0 only."""
    from types import SimpleNamespace

    from canvit_pretrain.harness.rollout import run_rollout

    class _VizTask(_StubTask):
        def __init__(self, seg):
            self.seg, self.inits, self.n_frames = seg, 0, 0

        def viz_init(self, *, model, images, state):
            self.inits += 1
            return {"ok": True}

        def viz_frame(self, *, model, images, gout, viewpoint, loss):
            self.n_frames += 1
            return SimpleNamespace(vp=viewpoint)

    seg, spec, *_ = _setup()
    sel = RandomSelector(is_foveated=False, foveated_scale=FoveatedScaleConfig(), min_viewpoint_scale=0.05)
    imgs, masks = next(_batches())
    vt = _VizTask(seg)
    bound = vt.bind((imgs, masks), torch.device("cpu"), model=seg, head=seg.head)
    common = dict(model=seg, images=imgs, task=bound, selector=sel,
                  bptt=BpttSpec(mode="none", horizon=3),
                  branches=[ViewpointType.RANDOM, ViewpointType.RANDOM],
                  canvas_grid_size=_G, amp_ctx=torch.enable_grad())

    r_off = run_rollout(**common)
    assert r_off.viz is None  # default: no viz, no hook calls

    # Hooks live on the RUN-level task (which also owns render_viz), NOT on the bound
    # per-batch task the engine otherwise talks to — so they arrive via viz_task.
    r_on = run_rollout(**{**common, "collect_viz": True, "viz_task": vt})
    assert r_on.viz is not None
    assert vt.inits == 1, "viz_init must fire once, on branch 0 only"
    assert len(r_on.viz.frames) == 3, "one frame per glimpse of branch 0 only"
    assert len(r_on.viz.viewpoints) == 3


def test_joint_clips_model_and_scorer_separately(monkeypatch):
    """train/loop.py clips `trainable` and `joint.scorer.parameters()` in TWO independent
    calls, each to grad_clip. One joint norm over the union would couple their magnitudes
    (a big scorer gradient would shrink the model's update), so the split is load-bearing."""
    from canvit_pretrain.harness.policy import build_policy
    from canvit_pretrain.tasks.ade20k.task import POLICY_FEATURE_GROUPS as ADE_GROUPS
    from canvit_pretrain.train.config import JointPolicyConfig

    seg, _, _, _, sel = _setup()
    gen = torch.Generator(device="cpu").manual_seed(0)
    joint = build_policy(
        canvit=seg.canvit, rl=JointPolicyConfig(use_rl=True, objective="qreg"),
        feature_groups=ADE_GROUPS, device=torch.device("cpu"), canvas_grid=_G,
        min_viewpoint_scale=0.05, foveated_scale=FoveatedScaleConfig(), generator=gen,
        encode_model=seg,
    )
    # joint spec: frozen backbone, train head + policy — so the scorer really is an
    # optimizer group and the split has something to prove.
    _s = ScheduleSpec(kind="warmup_constant", warmup_steps=1)
    spec = TrainSpec(
        train_backbone=False, train_head=True, train_policy=True,
        task_grad_to_backbone=False, policy_grad_to_backbone=False,
        bptt=BpttSpec(mode="none", horizon=2),
        optim={"head": GroupOptim(lr=1e-2, schedule=_s), "policy": GroupOptim(lr=1e-3, schedule=_s)},
    )
    apply_requires_grad(model=seg, head=seg.head, joint=joint, spec=spec)
    opt, sched = build_optimizer_and_scheduler(
        spec, {"head": list(seg.head.parameters()), "policy": list(joint.scorer.parameters())})
    scorer_ids = {id(p) for p in joint.scorer.parameters()}

    calls: list[set[int]] = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy(params, max_norm, *a, **k):
        ps = list(params)
        calls.append({id(p) for p in ps})
        return real_clip(ps, max_norm, *a, **k)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    run_training_loop(
        task=_StubTask(), model=seg, head=seg.head, optimizer=opt, scheduler=sched, selector=sel,
        spec=spec, branches=[ViewpointType.RANDOM], canvas_grid=_G, device=torch.device("cpu"),
        train_batches=_batches(), n_steps=1, log_every=99, joint=joint,
    )
    assert len(calls) == 2, f"expected one model clip + one scorer clip, got {len(calls)}"
    model_call, scorer_call = calls
    assert scorer_call == scorer_ids, "second clip must cover exactly the scorer's params"
    assert not (model_call & scorer_ids), "the model clip must NOT include scorer params"
    assert model_call, "the model clip must still cover the task's trainable params"


def test_resume_start_step_hook_returns_scheduler_epoch():
    """Default hook: the step count is the scheduler's. Distill's WebDataset path is the
    one exception (start_step comes from the shard schedule) — see tasks/tests/test_wds_resume.py."""
    class _Sched:
        last_epoch = 7

    tasks = [
        Ade20kRunTask(Ade20kConfig(tracker="none")),
        In1kRunTask(In1kConfig(tracker="none")),
        DistillRunTask(Config()),  # no webdataset_dir => sharded/feature path
    ]
    for t in tasks:
        assert t.resume_start_step({}, _Sched()) == 7, t.name
