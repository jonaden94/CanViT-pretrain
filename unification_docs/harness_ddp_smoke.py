"""2-rank DDP correctness check for the unified harness (design §9).

Two independent properties, both on real GPUs with NCCL:

  1. **Ranks stay identical.** After N steps every parameter must be bit-identical across
     ranks. If gradients were not averaged, each rank applies a different update and the
     models silently diverge — the failure mode §9 exists to prevent.
  2. **DDP == single-GPU.** Data-parallel training over 2 ranks with batch B each must
     equal one process with batch 2B over the SAME samples: mean-of-means over equal-size
     slices is the mean over the union, so the gradients (and hence the weights) agree.

To make (2) exact the dataset is fixed and sliced deterministically — rank r takes the
r-th contiguous half of the same tensor the 1-rank leg consumes whole. Any real loader
would give the two legs different data and make the comparison meaningless.

Both a task-only and a joint (task+policy) spec are covered, since the scorer is synced
through its own hook (``joint.allreduce_grads``) rather than the harness's.

Launched by ``unification_docs/ddp_smoke.sbatch``: leg 1 is ``srun -n2``, leg 2 is
``srun -n1``, in one job so both legs run on the same node and GPU model.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from canvit_pytorch import CanViTForSemanticSegmentation

from canvit_pretrain.harness import ddp
from canvit_pretrain.harness.loop import apply_requires_grad, run_training_loop
from canvit_pretrain.harness.optim import build_optimizer_and_scheduler
from canvit_pretrain.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TrainSpec
from canvit_pretrain.tasks.ade20k.task import POLICY_FEATURE_GROUPS, BoundAde20kTask
from canvit_pretrain.train.config import FoveatedScaleConfig, JointPolicyConfig
from canvit_pretrain.train.viewpoint import Viewpoint, ViewpointType

OUT = Path("/mnt/vast-nhr/projects/nib00021/jonathan/_harness_smoke_ckpts/ddp")
G, IMG, CLASSES, PER_RANK, STEPS = 8, 224, 20, 2, 6


class _FixedData:
    """The same samples every run; rank r gets the r-th contiguous slice."""

    def __init__(self, world_size: int, rank: int, device):
        torch.manual_seed(1234)  # identical on every rank => one shared dataset
        n = PER_RANK * 2
        self.images = torch.randn(n, 3, IMG, IMG)
        self.masks = torch.randint(0, CLASSES, (n, IMG, IMG))
        if world_size > 1:  # 2 ranks x PER_RANK
            lo, hi = rank * PER_RANK, (rank + 1) * PER_RANK
            self.images, self.masks = self.images[lo:hi], self.masks[lo:hi]
        self.device = device

    def __iter__(self):
        while True:
            yield (self.images, self.masks)


class _FixedSelector:
    """Viewpoints that depend only on ``t`` — never on the batch or any RNG.

    This is load-bearing for check (2). ``RandomSelector`` draws a viewpoint PER IMAGE, so
    the 2x(batch B) legs and the 1x(batch 2B) leg would consume the RNG differently and
    look at different crops; the runs would then diverge for a reason that has nothing to
    do with gradient averaging. Fixing the viewpoints makes the two legs mathematically
    identical up to the gradient reduction, which is exactly what is under test.
    """

    def start_rollout(self, *, t0_type, batch_size, device):
        return None

    def select(self, *, vp_type, ctx, t, batch_size, device, state):
        frac = (t % 3) / 3.0
        centers = torch.full((batch_size, 2), frac - 0.5, device=device)
        scales = torch.full((batch_size,), 0.5 + 0.25 * frac, device=device)
        return Viewpoint(centers=centers, scales=scales, name=f"fixed{t}")


class _Task:
    def __init__(self, model):
        self.model = model

    def batch_images(self, batch, device):
        return batch[0].to(device)

    def bind(self, batch, device, *, model, head):
        return BoundAde20kTask(seg=model, masks=batch[1].to(device), canvas_grid=G)


def _spec(joint: bool) -> TrainSpec:
    sched = ScheduleSpec(kind="warmup_constant", warmup_steps=0)
    optim = {"head": GroupOptim(lr=1e-2, weight_decay=0.0, schedule=sched)}
    if joint:
        optim["policy"] = GroupOptim(lr=1e-3, weight_decay=0.0, schedule=sched)
    return TrainSpec(
        train_backbone=False, train_head=True, train_policy=joint,
        task_grad_to_backbone=False, policy_grad_to_backbone=False,
        bptt=BpttSpec(mode="none", horizon=3), optim=optim,
    )


def _fingerprint(model, joint) -> dict:
    """Per-parameter checksums — enough to catch any cross-rank divergence."""
    mods = {"model": model} | ({"scorer": joint.scorer} if joint is not None else {})
    return {f"{tag}.{n}": float(p.detach().double().sum())
            for tag, m in mods.items() for n, p in m.named_parameters()}


def run_leg(joint_mode: bool, dinfo) -> dict:
    torch.manual_seed(0)  # identical init on every rank (broadcast also enforces it)
    # dropout=0: the ONLY stochastic layer in this model is the seg head's Dropout2d
    # (the CanViT backbone has none). With it active, the 2x(batch B) and 1x(batch 2B)
    # legs draw different-shaped dropout masks and consume RNG differently, so the forward
    # is no longer a deterministic per-sample function and grad([0:2B]) != mean of the two
    # halves — breaking the DDP==1-GPU equivalence for a reason unrelated to gradient sync.
    # Turning it off makes the equivalence exact, so this check actually validates the
    # AllReduce SCALE (a sum-instead-of-average bug would leave params 2x off here while
    # the cross-rank-identity check still passed).
    seg = CanViTForSemanticSegmentation(
        backbone_name="vits16", model_config={}, num_classes=CLASSES, dropout=0.0).to(dinfo.device)

    joint = None
    if joint_mode:
        from canvit_pretrain.harness.policy import build_policy
        gen = torch.Generator(device=dinfo.device).manual_seed(0)
        joint = build_policy(
            canvit=seg.canvit, rl=JointPolicyConfig(use_rl=True, objective="qreg"),
            feature_groups=POLICY_FEATURE_GROUPS, device=dinfo.device, canvas_grid=G,
            min_viewpoint_scale=0.05, foveated_scale=FoveatedScaleConfig(),
            generator=gen, encode_model=seg)

    spec = _spec(joint_mode)
    apply_requires_grad(model=seg, head=seg.head, joint=joint, spec=spec)
    groups = {"head": list(seg.head.parameters())}
    if joint_mode:
        groups["policy"] = list(joint.scorer.parameters())
    opt, sched = build_optimizer_and_scheduler(spec, groups)

    if dinfo.is_dist:  # the run() step: make every rank start from the same weights
        ddp.broadcast_parameters(seg, joint.scorer if joint is not None else None)

    losses: list[float] = []
    run_training_loop(
        task=_Task(seg), model=seg, head=seg.head, optimizer=opt, scheduler=sched,
        selector=_FixedSelector(),
        spec=spec, branches=[ViewpointType.FULL], canvas_grid=G, device=dinfo.device,
        train_batches=iter(_FixedData(dinfo.world_size, dinfo.rank, dinfo.device)),
        n_steps=STEPS, log_every=1, joint=joint,
        is_dist=dinfo.is_dist, rank=dinfo.rank, ema_alpha=0.0,
        on_log=lambda s, m: losses.append(m["total_loss"]),
    )
    return {"losses": losses, "fingerprint": _fingerprint(seg, joint)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="output tag, e.g. ddp2 / single")
    a = ap.parse_args()

    dinfo = ddp.setup(device="cuda", rank=0, world_size=1)
    OUT.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(dinfo.device) if dinfo.device.type == "cuda" else "cpu"
    print(f"[{a.tag}] rank {dinfo.rank}/{dinfo.world_size} device={dinfo.device} gpu={gpu}",
          flush=True)

    out = {mode: run_leg(mode == "joint", dinfo) for mode in ("task_only", "joint")}
    (OUT / f"{a.tag}-rank{dinfo.rank}.json").write_text(json.dumps(out))
    print(f"[{a.tag}] rank {dinfo.rank} wrote results", flush=True)


if __name__ == "__main__":
    main()
