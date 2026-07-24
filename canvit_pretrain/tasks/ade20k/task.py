"""ADE20K semantic-segmentation task — engine-facing core (design §3.1, §0 table).

Readout = per-glimpse ``canvas_hidden`` [B,G,G,D]; the segmentation probe head is
applied OUTSIDE the backbone forward (so a frozen backbone still trains the head —
the probe cell). Glimpse routing follows canvit_eval/the existing ade20k rollout:
uniform → pre-crop at the training-matched pixel size; foveated/square → full image.

Reuses the ported, tested helpers (``consumes_full_image`` / ``derive_glimpse_px`` /
``ce_loss``) rather than duplicating them. The run-level ``Task`` wrapper
(build_model via ``from_pretrained_with_new_probe`` / ``from_pretrained_with_probe``,
loaders, mIoU eval) lands with the neutral loop.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from canvit_pytorch import CanViTForSemanticSegmentation, sample_at_viewpoint
from canvit_pytorch import RecurrentState
from torch import Tensor

from canvit_pretrain.ade20k.data import IGNORE_LABEL
from canvit_pretrain.ade20k.metrics import ce_loss
from canvit_pytorch.policy.features import FEATURE_GROUPS

from canvit_pretrain.ade20k.rollout import consumes_full_image, derive_glimpse_px
from canvit_pretrain.harness.rollout import GlimpseOut, TaskLoss
from canvit_pretrain.train.viewpoint import ViewpointType

log = logging.getLogger(__name__)

# ade20k has a spatial segmentation probe, so its scorer reads the full feature set
# INCLUDING probe entropy (ent / ent_delta) — the RL repo's canonical seg-policy features.
POLICY_FEATURE_GROUPS: tuple[str, ...] = FEATURE_GROUPS


class BoundAde20kTask:
    """Per-batch ADE20K :class:`RolloutTask`. Binds this batch's masks; holds the
    seg wrapper (``.canvit`` + ``.head``) and the glimpse routing derived once."""

    def __init__(
        self, *, seg: CanViTForSemanticSegmentation, masks: Tensor, canvas_grid: int,
        glimpse_px: int | None = None,
    ):
        self.seg = seg
        self.masks = masks  # [B, H, W] long
        self.canvas_grid = canvas_grid
        self.full_image = consumes_full_image(seg)
        self.glimpse_px = None if self.full_image else derive_glimpse_px(seg, glimpse_px)

    def forward_glimpse(
        self, *, model: Any, images: Tensor, state: RecurrentState,
        viewpoint: Any, backbone_no_grad: bool,
    ) -> GlimpseOut:
        seg = getattr(model, "module", model)  # unwrap DDP; ade20k steps .canvit directly
        B = images.shape[0]
        model_input = images if self.full_image else sample_at_viewpoint(
            spatial=images, viewpoint=viewpoint, glimpse_size_px=self.glimpse_px,
        )
        ctx = torch.no_grad() if backbone_no_grad else nullcontext()
        with ctx:
            out = seg.canvit(image=model_input, state=state, viewpoint=viewpoint)
            hidden = seg.canvit.get_spatial(out.state.canvas).view(B, self.canvas_grid, self.canvas_grid, -1)
        return GlimpseOut(readout=hidden, state=out.state, vpe=out.vpe)

    def _logits(self, readout: Tensor) -> Tensor:
        return self.seg.head(readout.float())  # [B, C, G, G]

    def step_loss(self, readout: Any) -> TaskLoss:
        return TaskLoss(combined=ce_loss(self._logits(readout), self.masks))

    def per_image_loss(self, readout: Any) -> Tensor:
        logits = self._logits(readout)
        masks = self.masks
        if masks.shape[1:] != logits.shape[2:]:
            masks = F.interpolate(masks.unsqueeze(1).float(), logits.shape[2:], mode="nearest").squeeze(1).long()
        per_px = F.cross_entropy(logits, masks, ignore_index=IGNORE_LABEL, reduction="none")  # [B, G, G]
        valid = (masks != IGNORE_LABEL).float()
        return (per_px * valid).flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1.0)


class Ade20kRunTask:
    """Run-level ADE20K :class:`~canvit_pretrain.harness.run.RunTask` — the full seam
    ``harness.run`` drives: model construction (pretrained backbone + fresh probe),
    ADE20K loaders, mIoU eval, joint-policy assembly, and per-batch ``bind`` into the
    :class:`BoundAde20kTask` engine core. Config is composed (design D-B): the task holds
    its ``Ade20kConfig``; the policy config for joint runs is passed in (``rl``) so the
    live ade20k config stays untouched.
    """

    name = "ade20k"
    best_metric = "miou_final"
    """Eval key the harness maximizes for `best.pt` — the last-timestep mIoU, which is
    what `ade20k/train.py` selects its best probe checkpoint on (`probe.best_last_miou`)."""

    def __init__(self, cfg, *, rl=None):
        self.cfg = cfg
        self.rl = rl  # JointPolicyConfig | None; only needed for train_policy runs

    # --- capabilities & defaults ------------------------------------------
    def caps(self):
        from canvit_pretrain.harness.spec import TaskCaps
        return TaskCaps(has_head=True, supports_policy=True)

    def default_spec(self):
        """Frozen-backbone probe (the historical ade20k regime), fixed horizon = n_timesteps.
        The LR schedule is ``warmup_onecycle``, which reproduces the standalone probe's
        AdamW + ``WarmupOneCycleLR`` step for step (``ade20k/data.make_optimizer_and_scheduler``
        with the same max_steps / warmup_steps / warmup_lr_ratio)."""
        from canvit_pretrain.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TrainSpec
        return TrainSpec.probe(
            bptt=BpttSpec(mode="none", horizon=self.cfg.n_timesteps),
            optim={"head": GroupOptim(
                lr=self.cfg.peak_lr, weight_decay=self.cfg.weight_decay,
                schedule=ScheduleSpec(kind="warmup_onecycle", warmup_steps=self.cfg.warmup_steps,
                                      total_steps=self.cfg.max_steps,
                                      warmup_lr_ratio=self.cfg.warmup_lr_ratio))},
        )

    # --- construction ------------------------------------------------------
    def build_model(self, device, prior_model_config=None):
        # prior_model_config is unused: the backbone arch comes from the HF repo the probe
        # was built on, so a resume rebuilds the same model from cfg.model_repo already.
        seg = CanViTForSemanticSegmentation.from_pretrained_with_new_probe(
            pretrained_repo=self.cfg.model_repo, num_classes=self._num_classes(),
            dropout=self.cfg.dropout, use_ln=True,
        ).to(device)
        # The OOD footgun that silently ruined run 15025338 (ade20k/train.py:97): a
        # foveated backbone derives its fixation window as `fix_size = scale * H`, so a
        # probe rollout at a scale the backbone never saw makes EVERY glimpse
        # out-of-distribution. It does not crash — mIoU just falls as glimpses
        # accumulate. Warn loudly, exactly as the standalone does.
        if consumes_full_image(seg):
            fs = self.cfg.foveated_scale
            detail = (f"fixed_scale={fs.fixed_scale}" if fs.mode == "fixed"
                      else f"{fs.distribution} in [{fs.min_scale}, {fs.max_scale}]")
            log.warning(
                "  foveated view scale: mode=%s, %s — this MUST match the backbone's "
                "pretraining scale or every glimpse is out of distribution "
                "(symptom: mIoU falls as glimpses accumulate).", fs.mode, detail)
        return seg, seg.head

    def _num_classes(self):
        from canvit_pretrain.ade20k.data import NUM_CLASSES
        return NUM_CLASSES

    def canvas_grid(self, model):
        if self.cfg.canvas_grid is not None:
            return self.cfg.canvas_grid
        return self.cfg.scene_size // model.canvit.backbone.patch_size_px

    def is_foveated(self, model):
        return consumes_full_image(model)

    def branches(self):
        return [ViewpointType.FULL if self.cfg.train_start_full else ViewpointType.RANDOM]

    def build_loaders(self, *, world_size, rank):
        from canvit_pretrain.ade20k.data import make_ade20k_loaders
        return make_ade20k_loaders(self.cfg)

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
        if spec.train_head:
            groups["head"] = list(model.head.parameters())
        if spec.train_policy:
            assert joint is not None
            groups["policy"] = list(joint.scorer.parameters())
        return groups

    def resume_start_step(self, payload, scheduler):
        return scheduler.last_epoch  # map dataset: steps == scheduler.step() calls

    def resume_state(self):
        return {}  # re-iterable map dataset: nothing to carry across jobs

    # --- per-batch (engine-facing) ----------------------------------------
    def batch_images(self, batch, device):
        return batch[0].to(device, non_blocking=True)

    def bind(self, batch, device, *, model, head):
        _, masks = batch
        return BoundAde20kTask(seg=model, masks=masks.to(device),
                               canvas_grid=self.canvas_grid(model), glimpse_px=self.cfg.glimpse_px)

    # --- eval & checkpoint -------------------------------------------------
    @torch.no_grad()
    def evaluate(self, *, model, head, val_loader, device, step, tracker=None, run_dir=None):
        # tracker/run_dir unused: this task returns its scalars for the caller to log
        # and renders no validation figures (owner: distill viz only).
        """mIoU per timestep over the val set (the historical ade20k eval), reusing the
        tested rollout + probe-eval helpers. Returns t0 / final / mean mIoU."""
        from canvit_pretrain.ade20k.data import IGNORE_LABEL, NUM_CLASSES
        from canvit_pretrain.ade20k.metrics import eval_probe_on_batch, mIoUAccumulator
        from canvit_pretrain.ade20k.rollout import make_random_viewpoints, rollout_canvas_hidden
        T = self.cfg.n_timesteps
        cg = self.canvas_grid(model)
        is_fov = consumes_full_image(model)
        was_training = model.head.training
        model.head.eval()
        ious = [mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) for _ in range(T)]
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        for vi, vm in val_loader:
            vi, vm = vi.to(device), vm.to(device)
            vps = make_random_viewpoints(vi.shape[0], device, T, min_scale=self.cfg.min_vp_scale,
                                         max_scale=self.cfg.max_vp_scale, start_with_full_scene=True,
                                         is_foveated=is_fov, foveated_scale=self.cfg.foveated_scale)
            with amp:
                hidden = rollout_canvas_hidden(seg=model, images=vi, viewpoints=vps,
                                               canvas_grid=cg, glimpse_px=self.cfg.glimpse_px)
            for t in range(T):
                eval_probe_on_batch(model.head, hidden[t], vm, ious[t])
        mious = [m.compute() for m in ious]
        if was_training:
            model.head.train()
        # EVERY timestep, not just the endpoints: mIoU-vs-glimpse-count is the whole
        # point of a canvas probe (and how the foveated OOD symptom shows up — mIoU
        # FALLING as glimpses accumulate). ade20k/train.py:179 logs val_miou_t{t} for
        # all t; the caller namespaces these as eval/miou_t{t}.
        out = {f"miou_t{t}": v for t, v in enumerate(mious)}
        out["miou_final"] = mious[-1]   # the best-checkpoint key (see best_metric)
        out["miou_mean"] = sum(mious) / T
        return out

    def model_config(self, model):
        return {"task": "ade20k", "num_classes": self._num_classes(),
                "canvas_grid": self.canvas_grid(model), "model_repo": self.cfg.model_repo}

    def checkpoint_metadata(self, model):
        return {"task": "ade20k", "scene_size": self.cfg.scene_size,
                "n_timesteps": self.cfg.n_timesteps,
                "pretrain_view_scale": getattr(self.cfg.foveated_scale, "fixed_scale", None)}


__all__ = ["POLICY_FEATURE_GROUPS", "BoundAde20kTask", "Ade20kRunTask"]
