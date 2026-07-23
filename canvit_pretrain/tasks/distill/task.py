"""Distill task — the DINOv3 feature-regression peer (design §3.1, §0 table).

Engine-facing core only (the per-batch :class:`RolloutTask` the rollout engine
consumes). Distill's readout is special: its recon/CLS heads live INSIDE the
pretraining forward, so ``forward_glimpse`` IS the model forward and ``step_loss``/
``per_image_loss`` delegate straight to the historical ``DistillTask`` (train/task.py).
This is the exact adapter proven byte-for-byte against the parity digest
``9a0100a1a3de3acd`` in ``harness/tests/test_rollout_parity.py``.

The run-level ``Task`` wrapper (build_model/build_loaders/evaluate) is added with the
neutral loop (design §11); its data pipeline is distill's existing webdataset +
normalizer machinery, unchanged.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from canvit_pytorch import RecurrentState
from torch import Tensor

from canvit_pytorch.policy.features import INTRINSIC_GROUPS

from canvit_pretrain.harness.rollout import GlimpseOut
from canvit_pretrain.train.task import DistillTask
from canvit_pretrain.train.viewpoint import ViewpointType

# distill scorer uses the probe-free INTRINSIC feature groups (no task head to read).
POLICY_FEATURE_GROUPS: tuple[str, ...] = INTRINSIC_GROUPS


class BoundDistillTask:
    """Per-batch distill :class:`RolloutTask`. ``distill`` binds this batch's
    (standardized) teacher targets + the active loss terms."""

    def __init__(self, distill: DistillTask):
        self.distill = distill

    def forward_glimpse(
        self, *, model: Any, images: Tensor, state: RecurrentState,
        viewpoint: Any, backbone_no_grad: bool,
    ) -> GlimpseOut:
        # The pretraining wrapper handles glimpse cropping internally (glimpse_size_px
        # baked in) and computes scene/cls preds inside its forward; call it directly
        # (through the DDP wrapper when present, so head grads are AllReduced).
        ctx = torch.no_grad() if backbone_no_grad else nullcontext()
        with ctx:
            out = model(image=images, state=state, viewpoint=viewpoint)
        return GlimpseOut(readout=out, state=out.state, vpe=out.vpe)

    def step_loss(self, readout: Any) -> Any:
        return self.distill.step_loss(readout)

    def per_image_loss(self, readout: Any) -> Tensor:
        return self.distill.per_image_loss(readout)


class DistillRunTask:
    """Run-level distill :class:`~canvit_pretrain.harness.run.RunTask`. Distill's heads
    ride INSIDE the pretraining forward, so ``head=None`` (``has_head=False``) and the
    backbone group carries them. Reuses distill's existing machinery unchanged: model
    (``create_model``), webdataset loaders + model-owned normalizers (``create_loaders``
    / ``init_normalizer_stats_from_tar``), and the ``validate()`` eval. The normalizer is
    model-state, so it is initialized once (in ``build_loaders``, which by then has both
    the model — stashed in ``build_model`` — and the first shard)."""

    name = "distill"

    def __init__(self, cfg):
        self.cfg = cfg
        self._model = None
        self._device = None
        self._glimpse_size_px = None
        self.scene_norm = None
        self.cls_norm = None
        self._teacher = None

    def caps(self):
        from canvit_pretrain.harness.spec import TaskCaps
        return TaskCaps(has_head=False, supports_policy=True)  # heads live in the forward

    def default_spec(self):
        """The historical distill regime: train backbone (+ in-forward heads), stochastic
        chunked TBPTT. Byte-exact to train/step.py under the parity digest."""
        from canvit_pretrain.harness.spec import BpttSpec, GroupOptim, ScheduleSpec, TrainSpec
        sched = ScheduleSpec(
            kind="warmup_cosine" if self.cfg.cosine_total_steps else "warmup_constant",
            warmup_steps=self.cfg.warmup_steps, total_steps=self.cfg.cosine_total_steps,
            start_lr=self.cfg.start_lr,
        )
        return TrainSpec(
            train_backbone=True, train_head=False, task_grad_to_backbone=True,
            bptt=BpttSpec(mode="chunked", chunk_size=self.cfg.chunk_size,
                          continue_prob=self.cfg.continue_prob),
            optim={"backbone": GroupOptim(lr=self.cfg.peak_lr, weight_decay=self.cfg.weight_decay,
                                          schedule=sched)},
        )

    def build_model(self, device):
        from canvit_pretrain.train.model import create_model, load_student_backbone
        teacher = None
        if self.cfg.init_backbone_from_teacher:
            teacher = self._load_teacher(device)
        backbone = load_student_backbone(self.cfg, teacher=teacher)
        bundle = create_model(backbone, self.cfg.model.teacher_dim, self.cfg)
        self._model, self._device = bundle.model, device
        self._glimpse_size_px = bundle.glimpse_size_px
        self.cls_norm, self.scene_norm = bundle.model.standardizers(self.cfg.canvas_patch_grid_size)
        return bundle.model, None

    def _load_teacher(self, device):
        if self._teacher is None:
            from canvit_pretrain.train.model import load_teacher
            self._teacher = load_teacher(self.cfg)
        return self._teacher

    def canvas_grid(self, model):
        return self.cfg.canvas_patch_grid_size

    def is_foveated(self, model):
        return getattr(model.cfg, "patcher_name", "uniform") in ("foveated", "square")

    def branches(self):
        return ([ViewpointType.FULL] * self.cfg.n_full_start_branches
                + [ViewpointType.RANDOM] * self.cfg.n_random_start_branches)

    def build_loaders(self, *, world_size, rank):
        from canvit_pretrain.train.data import create_loaders
        from canvit_pretrain.train.data.webdataset import (
            WebDatasetTrainLoader, init_normalizer_stats_from_tar,
        )
        loaders = create_loaders(self.cfg, self.cfg.start_step if hasattr(self.cfg, "start_step") else 0,
                                 job_index=0, world_size=world_size, rank=rank)
        train, val = loaders.train, loaders.val
        assert isinstance(train, WebDatasetTrainLoader), (
            "DistillRunTask currently supports the webdataset path only (set cfg.webdataset_dir)"
        )
        if not self.scene_norm.initialized:
            init_normalizer_stats_from_tar(
                train.first_shard_path(), self.scene_norm, self.cls_norm, self._device,
                self.cfg.normalizer_max_samples or 512,
            )
        return train, val

    def build_selector(self, *, device, canvas_grid, is_foveated):
        from canvit_pretrain.train.selector import RandomSelector
        return RandomSelector(is_foveated=is_foveated, foveated_scale=self.cfg.foveated_scale,
                              min_viewpoint_scale=self.cfg.min_viewpoint_scale)

    def build_policy(self, model, *, device, canvas_grid, generator):
        from canvit_pretrain.harness.policy import build_policy
        # distill: canvit == the pretraining model (its .cfg has canvas_dim/patcher_name);
        # encode_model defaults to SimpleNamespace(canvit=model) (probe-free INTRINSIC groups).
        return build_policy(
            canvit=model, rl=self.cfg.rl, feature_groups=POLICY_FEATURE_GROUPS, device=device,
            canvas_grid=canvas_grid, min_viewpoint_scale=self.cfg.min_viewpoint_scale,
            foveated_scale=self.cfg.foveated_scale, generator=generator, encode_model=None,
        )

    def policy_feature_groups(self):
        return POLICY_FEATURE_GROUPS

    def trainable_param_groups(self, *, model, head, joint, spec):
        groups: dict[str, list] = {}
        if spec.train_backbone:  # distill's in-forward heads ride the backbone group
            groups["backbone"] = list(model.parameters())
        if spec.train_policy:
            assert joint is not None
            groups["policy"] = list(joint.scorer.parameters())
        return groups

    def batch_images(self, batch, device):
        return batch[0].to(device, non_blocking=True)

    def bind(self, batch, device, *, model, head):
        _, raw_patches, raw_cls, _ = batch
        raw_patches = raw_patches.to(device, dtype=torch.float32)
        raw_cls = raw_cls.to(device, dtype=torch.float32)
        return BoundDistillTask(DistillTask(
            scene_target=self.scene_norm(raw_patches),
            cls_target=self.cls_norm(raw_cls.unsqueeze(1)).squeeze(1),
            enable_scene_patches_loss=self.cfg.enable_scene_patches_loss,
            enable_scene_cls_loss=self.cfg.enable_scene_cls_loss,
        ))

    @torch.no_grad()
    def evaluate(self, *, model, head, val_loader, device, step):
        """Reuse the existing distill ``validate()`` (cos-sim / recon per timestep). Needs
        the teacher (cached, offline) to compute val targets on the fly. Best-effort: a
        metric readout, not parity-gated — returns {} if it can't run."""
        import tempfile
        from pathlib import Path

        from canvit_pytorch.backbone.vit import NormFeatures

        from canvit_pretrain.train.data import scene_size_px
        from canvit_pretrain.train.tracker import make_tracker
        from canvit_pretrain.train.viz import validate
        try:
            teacher = self._load_teacher(device)
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

            def compute_raw_targets(images, sz):
                with amp:
                    if images.shape[-1] != sz:
                        images = torch.nn.functional.interpolate(images, size=(sz, sz),
                                                                 mode="bilinear", align_corners=False)
                    feats = teacher.forward_norm_features(images)
                    return NormFeatures(patches=feats.patches.float(), cls=feats.cls.float())

            exp = make_tracker(tracker="none", is_main=True, is_seeding=False, run_name="distill-eval",
                               wandb_project=None, wandb_entity=None, wandb_dir=None,
                               prev_comet_id=None, prev_wandb_id=None)
            metric = validate(
                exp=exp, step=step, model=model, compute_raw_targets=compute_raw_targets,
                scene_normalizer=self.scene_norm, cls_normalizer=self.cls_norm,
                val_batches=val_loader.batches(), device=device,
                canvas_grid_size=self.cfg.canvas_patch_grid_size,
                scene_size_px=scene_size_px(self.cfg.canvas_patch_grid_size,
                                            model.backbone.patch_size_px),
                glimpse_size_px=self._glimpse_size_px,
                run_dir=Path(tempfile.mkdtemp(prefix="distill_eval_")),
                n_eval_viewpoints=self.cfg.n_eval_viewpoints,
                min_viewpoint_scale=self.cfg.min_viewpoint_scale, prefix="val",
            )
            return {"val_metric": float(metric)}
        except Exception as e:  # eval is a readout; never let it kill training
            import logging
            logging.getLogger(__name__).warning("distill evaluate() skipped: %s", e)
            return {}

    def model_config(self, model):
        return {"task": "distill", "teacher_dim": self.cfg.model.teacher_dim,
                "canvas_grid": self.cfg.canvas_patch_grid_size, "backbone_name": self.cfg.backbone_name}

    def checkpoint_metadata(self, model):
        return {"task": "distill", "scene_resolution": self.cfg.scene_resolution,
                "dataset": self.cfg.dataset}


__all__ = ["POLICY_FEATURE_GROUPS", "BoundDistillTask", "DistillRunTask"]
