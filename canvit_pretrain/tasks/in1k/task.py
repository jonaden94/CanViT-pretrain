"""IN1k classification task — engine-facing core (design §3.1, §0 table).

Readout = per-glimpse recurrent CLS token [B,D]; the ``LN + Linear`` head is applied
OUTSIDE the backbone forward (frozen backbone → head still trains). Glimpse routing
matches ADE20K (uniform pre-crop / foveated full-image), so the ade20k helpers are
reused. The run-level ``Task`` wrapper (build_model via ``from_pretrained_with_new_head``,
webdataset loaders with ``with_epoch``, top-1/5 eval) lands with the neutral loop.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from canvit_pytorch import CanViTForImageClassification, sample_at_viewpoint
from canvit_pytorch import RecurrentState
from torch import Tensor

from canvit_pytorch.policy.features import INTRINSIC_GROUPS

from canvit_pretrain.ade20k.rollout import consumes_full_image, derive_glimpse_px
from canvit_pretrain.harness.rollout import GlimpseOut, TaskLoss
from canvit_pretrain.train.viewpoint import ViewpointType

# in1k's classifier reads the CLS token, not a spatial probe, so the probe-entropy
# groups don't apply — the scorer uses the probe-free INTRINSIC groups (like distill).
POLICY_FEATURE_GROUPS: tuple[str, ...] = INTRINSIC_GROUPS


class BoundIn1kTask:
    """Per-batch IN1k :class:`RolloutTask`. Binds this batch's class labels; holds the
    classifier wrapper (``.canvit`` + ``.norm`` + ``.head``) and the glimpse routing."""

    def __init__(
        self, *, clf: CanViTForImageClassification, targets: Tensor, canvas_grid: int,
        glimpse_px: int | None = None, label_smoothing: float = 0.0,
    ):
        self.clf = clf
        self.targets = targets  # [B] class idx
        self.canvas_grid = canvas_grid
        self.label_smoothing = label_smoothing
        self.full_image = consumes_full_image(clf)
        self.glimpse_px = None if self.full_image else derive_glimpse_px(clf, glimpse_px)

    def forward_glimpse(
        self, *, model: Any, images: Tensor, state: RecurrentState,
        viewpoint: Any, backbone_no_grad: bool,
    ) -> GlimpseOut:
        clf = getattr(model, "module", model)  # unwrap DDP; in1k steps .canvit directly
        model_input = images if self.full_image else sample_at_viewpoint(
            spatial=images, viewpoint=viewpoint, glimpse_size_px=self.glimpse_px,
        )
        ctx = torch.no_grad() if backbone_no_grad else nullcontext()
        with ctx:
            out = clf.canvit(image=model_input, state=state, viewpoint=viewpoint)
        cls = out.state.recurrent_cls[:, 0].float()  # [B, D]
        return GlimpseOut(readout=cls, state=out.state, vpe=out.vpe)

    def _logits(self, readout: Tensor) -> Tensor:
        return self.clf.head(self.clf.norm(readout))  # [B, C]

    def step_loss(self, readout: Any) -> TaskLoss:
        logits = self._logits(readout)
        return TaskLoss(combined=F.cross_entropy(logits, self.targets, label_smoothing=self.label_smoothing))

    def per_image_loss(self, readout: Any) -> Tensor:
        return F.cross_entropy(self._logits(readout), self.targets, reduction="none")  # [B]


class In1kRunTask:
    """Run-level IN1k :class:`~canvit_pretrain.harness.run.RunTask`. The trainable
    "head" is LN(``clf.norm``) + Linear(``clf.head``): ``from_pretrained_with_new_head``
    leaves ``clf.norm`` at requires_grad=True and the harness freezes only the trunk, so
    norm stays trainable in both frozen and finetune; the optimizer's "head" group is
    norm+head. Config composed (design D-B): task holds its ``In1kConfig``; joint policy
    config passed in (``rl``)."""

    name = "in1k"

    def __init__(self, cfg, *, rl=None):
        self.cfg = cfg
        self.rl = rl

    def caps(self):
        from canvit_pretrain.harness.spec import TaskCaps
        return TaskCaps(has_head=True, supports_policy=True)

    def default_spec(self):
        """cfg.mode drives the default: 'frozen' => probe (backbone frozen, bptt none);
        'finetune' => train backbone + head end to end (full-graph)."""
        from canvit_pretrain.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TrainSpec
        T = self.cfg.n_timesteps
        head_go = GroupOptim(lr=self.cfg.peak_lr, weight_decay=self.cfg.weight_decay,
                             schedule=ScheduleSpec(kind="warmup_constant", warmup_steps=0))
        if self.cfg.mode == "finetune":
            return TrainSpec.finetune(bptt=BpttSpec(mode="full", horizon=T),
                                      optim={"backbone": head_go, "head": head_go})
        return TrainSpec.probe(bptt=BpttSpec(mode="none", horizon=T), optim={"head": head_go})

    def build_model(self, device, prior_model_config=None):
        # prior_model_config is unused: the backbone arch comes from the HF repo the head
        # was built on, so a resume rebuilds the same model from cfg.model_repo already.
        from canvit_pretrain.in1k.config import NUM_CLASSES
        clf = CanViTForImageClassification.from_pretrained_with_new_head(
            pretrained_repo=self.cfg.model_repo, n_classes=NUM_CLASSES,
        ).to(device)
        return clf, clf.head

    def canvas_grid(self, model):
        if self.cfg.canvas_grid is not None:
            return self.cfg.canvas_grid
        return self.cfg.scene_size // model.canvit.backbone.patch_size_px

    def is_foveated(self, model):
        return consumes_full_image(model)

    def branches(self):
        return [ViewpointType.FULL if self.cfg.train_start_full else ViewpointType.RANDOM]

    def build_loaders(self, *, world_size, rank):
        from canvit_pretrain.in1k.data import make_train_loader, make_val_loader
        loader, _ = make_train_loader(self.cfg, world_size=world_size, rank=rank)
        val = make_val_loader(self.cfg, world_size=world_size, rank=rank) if self.cfg.val_dir.is_dir() else None
        return loader, val

    def build_selector(self, *, device, canvas_grid, is_foveated):
        from canvit_pretrain.train.selector import RandomSelector
        return RandomSelector(is_foveated=is_foveated, foveated_scale=self.cfg.foveated_scale,
                              min_viewpoint_scale=self.cfg.min_vp_scale)

    def build_policy(self, model, *, device, canvas_grid, generator):
        from canvit_pretrain.harness.policy import build_policy
        from canvit_pretrain.train.config import JointPolicyConfig
        rl = self.rl or JointPolicyConfig(use_rl=True, feature_groups=POLICY_FEATURE_GROUPS)
        return build_policy(
            canvit=model.canvit, rl=rl, feature_groups=POLICY_FEATURE_GROUPS, device=device,
            canvas_grid=canvas_grid, min_viewpoint_scale=self.cfg.min_vp_scale,
            foveated_scale=self.cfg.foveated_scale, generator=generator, encode_model=model,
        )

    def policy_feature_groups(self):
        return POLICY_FEATURE_GROUPS

    def trainable_param_groups(self, *, model, head, joint, spec):
        groups: dict[str, list] = {}
        if spec.train_backbone:
            groups["backbone"] = list(model.canvit.parameters())
        if spec.train_head:  # in1k head = LN(norm) + Linear(head)
            groups["head"] = list(model.norm.parameters()) + list(model.head.parameters())
        if spec.train_policy:
            assert joint is not None
            groups["policy"] = list(joint.scorer.parameters())
        return groups

    def resume_start_step(self, payload, scheduler):
        return scheduler.last_epoch  # with_epoch wds: steps == scheduler.step() calls

    def resume_state(self):
        return {}  # with_epoch reshuffles every epoch: no cross-job shard cursor

    def batch_images(self, batch, device):
        return batch[0].to(device, non_blocking=True)

    def bind(self, batch, device, *, model, head):
        _, labels = batch
        return BoundIn1kTask(
            clf=model, targets=torch.as_tensor(labels, dtype=torch.long, device=device),
            canvas_grid=self.canvas_grid(model), glimpse_px=self.cfg.glimpse_px,
            label_smoothing=self.cfg.label_smoothing,
        )

    @torch.no_grad()
    def evaluate(self, *, model, head, val_loader, device, step):
        """Top-1/5 over the eval policy (reuses in1k/train.py::evaluate)."""
        if val_loader is None:
            return {}
        from canvit_pretrain.in1k.train import evaluate as _eval
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        accs = _eval(model, self.cfg, val_loader, device=device, canvas_grid=self.canvas_grid(model),
                     amp_ctx=amp, is_foveated=consumes_full_image(model))
        return {"top1": accs[1], "top5": accs[5]}

    def model_config(self, model):
        from canvit_pretrain.in1k.config import NUM_CLASSES
        return {"task": "in1k", "n_classes": NUM_CLASSES, "canvas_grid": self.canvas_grid(model),
                "model_repo": self.cfg.model_repo, "mode": self.cfg.mode}

    def checkpoint_metadata(self, model):
        return {"task": "in1k", "mode": self.cfg.mode, "scene_size": self.cfg.scene_size,
                "n_timesteps": self.cfg.n_timesteps}


__all__ = ["POLICY_FEATURE_GROUPS", "BoundIn1kTask", "In1kRunTask"]
