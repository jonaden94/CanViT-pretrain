"""Training and validation orchestration for visualization.

This module ties together plotting, metrics, and Comet logging for
training and validation visualization.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")

import comet_ml
import numpy as np
import torch
import torch.nn.functional as F
from canvit.backbone.dinov3 import DINOv3Backbone, NormFeatures
from canvit.model.active.base import GlimpseOutput
from canvit.viewpoint import Viewpoint as CanvitViewpoint
from torch import Tensor

from dinov3_probes import DINOv3LinearClassificationHead

from avp_vit import ActiveCanViT
from ..norm import PositionAwareNorm
from ..probe import (
    compute_in1k_top1,
    get_imagenet_class_names,
    get_probe_resolution,
    get_top_k_predictions,
    labels_are_in1k,
)
from ..viewpoint import Viewpoint, make_eval_viewpoints
from .comet import log_curve, log_figure
from .image import imagenet_denormalize
from .metrics import compute_spatial_stats
from .plot import TimestepPredictions, plot_multistep_pca

log = logging.getLogger(__name__)


@dataclass
class VizSampleData:
    """Viz data extracted for a single sample during streaming validation.

    Shape annotations use:
        G = canvas grid size (e.g., 32)
        g = glimpse grid size (e.g., 3)
        D = teacher feature dim (e.g., 768)
        C = canvas hidden dim
    """

    glimpse: np.ndarray  # [g, g, 3] denormalized RGB
    predicted_scene: np.ndarray  # [G², D] teacher-space prediction
    canvas_spatial: np.ndarray  # [G², C] raw hidden state


@dataclass
class ValAccumulator:
    """Accumulator for streaming validation metrics.

    MEMORY OPTIMIZATION:
    - Validation metrics (scene_cos_sim, cls_cos_sim, IN1k acc) require the FULL BATCH
      to compute batch-averaged values. These are computed in step_fn, producing scalars,
      then the full-batch tensors are discarded (go out of scope).
    - PCA visualization only needs ONE sample. We extract sample 0's data in step_fn,
      move to CPU as numpy, and store in viz_samples. This is O(T) for one sample,
      not O(B×T) for all samples.

    Result: memory footprint is O(1) for metrics + O(T × single_sample) for viz,
    instead of O(B × T × tensor_size) if we kept all intermediate batch tensors.
    """

    scene_cos_sims: list[float] = field(default_factory=list)
    cls_cos_sims: list[float] = field(default_factory=list)
    in1k_accs: list[float] = field(default_factory=list)
    pca_predictions: list[TimestepPredictions] = field(default_factory=list)
    viz_samples: list[VizSampleData] = field(default_factory=list)
    initial_scene: np.ndarray | None = None
    initial_canvas_spatial: np.ndarray | None = None


def _extract_sample0_viz(
    out: GlimpseOutput,
    predicted_scene: Tensor,
    model: "ActiveCanViT",
) -> VizSampleData:
    """Extract viz data for sample 0, move to CPU as numpy."""
    glimpse_cpu = out.glimpse[0].cpu()
    glimpse_np = imagenet_denormalize(glimpse_cpu).numpy()

    scene_cpu = predicted_scene[0].cpu().float()
    scene_np = scene_cpu.numpy()

    canvas_single = out.canvas[0:1]
    spatial = model.get_spatial(canvas_single)[0]
    spatial_np = spatial.cpu().float().numpy()

    return VizSampleData(
        glimpse=glimpse_np,
        predicted_scene=scene_np,
        canvas_spatial=spatial_np,
    )


@dataclass
class _TrainVizAccumulator:
    """Accumulator for training viz (uses forward_reduce, sample 0 only)."""

    scene_cos_sims: list[float] = field(default_factory=list)
    cls_cos_sims: list[float] = field(default_factory=list)
    viz_samples: list[VizSampleData] = field(default_factory=list)
    final_predicted_scene: Tensor | None = None


def viz_and_log(
    *,
    exp: comet_ml.Experiment,
    step: int,
    prefix: str,
    model: ActiveCanViT,
    teacher: DINOv3Backbone,
    normalizer: PositionAwareNorm,
    images: Tensor,
    viewpoints: list[Viewpoint],
    target: Tensor,
    canvas: Tensor,
    glimpse_size_px: int,
    cls_target: Tensor | None = None,
    log_spatial_stats: bool = True,
    log_curves: bool = True,
) -> None:
    """Run forward pass and log PCA visualization (training viz)."""
    assert isinstance(model.backbone, DINOv3Backbone)
    n_spatial = canvas.shape[1] - model.n_canvas_registers
    canvas_grid_size = int(n_spatial**0.5)
    assert canvas_grid_size**2 == n_spatial
    glimpse_grid_size = glimpse_size_px // model.backbone.patch_size_px
    has_cls = cls_target is not None

    with torch.inference_mode():
        initial_scene_np = model.predict_teacher_scene(canvas)[0].cpu().float().numpy()
        initial_canvas_spatial_np = model.get_spatial(canvas[0:1])[0].cpu().float().numpy()

        def init_fn(_canvas: Tensor, _cls: Tensor) -> _TrainVizAccumulator:
            return _TrainVizAccumulator()

        def step_fn(
            acc: _TrainVizAccumulator, out: GlimpseOutput, _vp: CanvitViewpoint
        ) -> _TrainVizAccumulator:
            predicted_scene = model.predict_teacher_scene(out.canvas)

            acc.scene_cos_sims.append(
                F.cosine_similarity(predicted_scene, target, dim=-1).mean().item()
            )
            if has_cls:
                assert cls_target is not None
                predicted_cls = model.predict_teacher_cls(out.cls, out.canvas)
                acc.cls_cos_sims.append(
                    F.cosine_similarity(predicted_cls, cls_target, dim=-1).mean().item()
                )

            acc.viz_samples.append(_extract_sample0_viz(out, predicted_scene, model))
            acc.final_predicted_scene = predicted_scene
            return acc

        acc, _, _ = model.forward_reduce(
            image=images,
            viewpoints=viewpoints,  # pyright: ignore[reportArgumentType]
            glimpse_size_px=glimpse_size_px,
            canvas_grid_size=canvas_grid_size,
            init_fn=init_fn,
            step_fn=step_fn,
            canvas=canvas,
        )

        if log_curves:
            log_curve(
                exp,
                f"{prefix}/scene_cos_sim_vs_timestep",
                x=list(range(len(acc.scene_cos_sims))),
                y=acc.scene_cos_sims,
                step=step,
            )
            if acc.cls_cos_sims:
                log_curve(
                    exp,
                    f"{prefix}/cls_cos_sim_vs_timestep",
                    x=list(range(len(acc.cls_cos_sims))),
                    y=acc.cls_cos_sims,
                    step=step,
                )

        if log_spatial_stats and acc.final_predicted_scene is not None:
            target_stats = compute_spatial_stats(target)
            pred_stats = compute_spatial_stats(acc.final_predicted_scene)
            exp.log_metrics(
                {
                    f"{prefix}/target_spatial_mean": target_stats["mean"],
                    f"{prefix}/target_spatial_std": target_stats["std"],
                    f"{prefix}/pred_spatial_mean": pred_stats["mean"],
                    f"{prefix}/pred_spatial_std": pred_stats["std"],
                },
                step=step,
            )

        H, W = images.shape[-2], images.shape[-1]
        boxes = [vp.to_pixel_box(0, H, W) for vp in viewpoints]
        names = [vp.name for vp in viewpoints]

        fig_pca = plot_multistep_pca(
            full_img=imagenet_denormalize(images[0].cpu()).numpy(),
            teacher=target[0].cpu().float().numpy(),
            scenes=[vs.predicted_scene for vs in acc.viz_samples],
            glimpses=[vs.glimpse for vs in acc.viz_samples],
            boxes=boxes,
            names=names,
            scene_grid_size=canvas_grid_size,
            glimpse_grid_size=glimpse_grid_size,
            initial_scene=initial_scene_np,
            hidden_spatials=[vs.canvas_spatial for vs in acc.viz_samples],
            initial_hidden_spatial=initial_canvas_spatial_np,
        )
        log_figure(exp, fig_pca, f"{prefix}/pca", step)


def _log_pca_from_accumulator(
    *,
    exp: comet_ml.Experiment,
    step: int,
    prefix: str,
    acc: ValAccumulator,
    full_img: np.ndarray,
    teacher_np: np.ndarray,
    boxes: list,
    names: list[str],
    canvas_grid_size: int,
    glimpse_grid_size: int,
    log_spatial_stats: bool,
    log_curves: bool,
) -> None:
    """Log PCA visualization from pre-computed accumulator data."""
    assert acc.initial_scene is not None
    scenes = [vs.predicted_scene for vs in acc.viz_samples]
    glimpses = [vs.glimpse for vs in acc.viz_samples]
    canvas_spatials = [vs.canvas_spatial for vs in acc.viz_samples]

    fig_pca = plot_multistep_pca(
        full_img=full_img,
        teacher=teacher_np,
        scenes=scenes,
        glimpses=glimpses,
        boxes=boxes,
        names=names,
        scene_grid_size=canvas_grid_size,
        glimpse_grid_size=glimpse_grid_size,
        initial_scene=acc.initial_scene,
        hidden_spatials=canvas_spatials if canvas_spatials[0] is not None else None,
        initial_hidden_spatial=acc.initial_canvas_spatial,
        timestep_predictions=acc.pca_predictions if acc.pca_predictions else None,
    )
    log_figure(exp, fig_pca, f"{prefix}/pca", step)

    if log_spatial_stats and acc.viz_samples:
        target_stats = {"mean": float(np.mean(teacher_np)), "std": float(np.std(teacher_np))}
        pred_stats = {"mean": float(np.mean(scenes[-1])), "std": float(np.std(scenes[-1]))}
        exp.log_metrics(
            {
                f"{prefix}/target_spatial_mean": target_stats["mean"],
                f"{prefix}/target_spatial_std": target_stats["std"],
                f"{prefix}/pred_spatial_mean": pred_stats["mean"],
                f"{prefix}/pred_spatial_std": pred_stats["std"],
            },
            step=step,
        )

    if log_curves:
        log_curve(
            exp,
            f"{prefix}/scene_cos_sim_vs_timestep",
            x=list(range(len(acc.scene_cos_sims))),
            y=acc.scene_cos_sims,
            step=step,
        )
        if acc.cls_cos_sims:
            log_curve(
                exp,
                f"{prefix}/cls_cos_sim_vs_timestep",
                x=list(range(len(acc.cls_cos_sims))),
                y=acc.cls_cos_sims,
                step=step,
            )


def validate(
    *,
    exp: comet_ml.Experiment,
    step: int,
    model: ActiveCanViT,
    compute_raw_targets: Callable[[Tensor, int], "NormFeatures"],
    scene_normalizer: PositionAwareNorm,
    cls_normalizer: PositionAwareNorm,
    images: Tensor,
    canvas_grid_size: int,
    scene_size_px: int,
    glimpse_size_px: int,
    prefix: str = "val",
    probe: DINOv3LinearClassificationHead | None = None,
    labels: Tensor | None = None,
    log_curves: bool = False,
    log_pca: bool = False,
    teacher: DINOv3Backbone | None = None,
    log_spatial_stats: bool = False,
    backbone: str | None = None,
) -> float:
    """Run validation with streaming metrics (no O(B×T) memory)."""
    assert not log_pca or teacher is not None

    if probe is not None and backbone is not None:
        probe_res = get_probe_resolution(backbone)
        if scene_size_px != probe_res:
            log.warning(
                f"Resolution mismatch: model predicts teacher@{scene_size_px}, "
                f"but probe trained on teacher@{probe_res}. IN1k metrics may be unreliable."
            )

    B = images.shape[0]
    viewpoints = make_eval_viewpoints(B, images.device)
    has_cls = model.cls_proj is not None
    has_probe = probe is not None and labels is not None and labels_are_in1k(labels)

    scene_was_training = scene_normalizer.training
    cls_was_training = cls_normalizer.training
    scene_normalizer.eval()
    cls_normalizer.eval()

    try:
        with torch.inference_mode():
            raw_feats = compute_raw_targets(images, scene_size_px)
            target = scene_normalizer(raw_feats.patches)
            cls_target = (
                cls_normalizer(raw_feats.cls.unsqueeze(1)).squeeze(1) if has_cls else None
            )

            target_sample0 = target[0].cpu().float().numpy() if log_pca else None

            gt_idx = int(labels[0].item()) if has_probe and labels is not None else 0
            gt_name = get_imagenet_class_names()[gt_idx] if has_probe else ""

            if has_probe and teacher is not None:
                assert backbone is not None and probe is not None
                probe_res = get_probe_resolution(backbone)
                images_at_probe_res = F.interpolate(
                    images, size=(probe_res, probe_res), mode="bilinear", align_corners=False
                )
                teacher_cls = teacher.forward_norm_features(images_at_probe_res).cls
                teacher_logits = probe(teacher_cls)
                assert labels is not None
                teacher_acc = compute_in1k_top1(teacher_logits, labels)
                exp.log_metric(f"{prefix}/in1k_teacher_top1", teacher_acc, step=step)

            def init_fn(canvas: Tensor, cls: Tensor) -> ValAccumulator:
                acc = ValAccumulator()
                if log_pca:
                    acc.initial_scene = model.predict_teacher_scene(canvas)[0].cpu().float().numpy()
                    acc.initial_canvas_spatial = (
                        model.get_spatial(canvas[0:1])[0].cpu().float().numpy()
                    )
                return acc

            def step_fn(
                acc: ValAccumulator, out: GlimpseOutput, _vp: CanvitViewpoint
            ) -> ValAccumulator:
                predicted_scene = model.predict_teacher_scene(out.canvas)
                predicted_cls = model.predict_teacher_cls(out.cls, out.canvas) if has_cls else None

                scene_cos = F.cosine_similarity(predicted_scene, target, dim=-1).mean().item()
                acc.scene_cos_sims.append(scene_cos)

                if has_cls and cls_target is not None and predicted_cls is not None:
                    cls_cos = F.cosine_similarity(predicted_cls, cls_target, dim=-1).mean().item()
                    acc.cls_cos_sims.append(cls_cos)

                    if has_probe:
                        assert probe is not None and labels is not None
                        cls_raw = cls_normalizer.denormalize(predicted_cls)
                        logits = probe(cls_raw)
                        acc.in1k_accs.append(compute_in1k_top1(logits, labels))

                        if log_pca:
                            top_k = get_top_k_predictions(logits[0:1], k=5)[0]
                            acc.pca_predictions.append(
                                TimestepPredictions(
                                    predictions=top_k, gt_idx=gt_idx, gt_name=gt_name
                                )
                            )

                if log_pca:
                    acc.viz_samples.append(_extract_sample0_viz(out, predicted_scene, model))

                return acc

            acc, _final_canvas, _final_cls = model.forward_reduce(
                image=images,
                viewpoints=viewpoints,  # pyright: ignore[reportArgumentType]
                glimpse_size_px=glimpse_size_px,
                canvas_grid_size=canvas_grid_size,
                init_fn=init_fn,
                step_fn=step_fn,
            )

            final_cos_sim = acc.scene_cos_sims[-1]
            exp.log_metric(f"{prefix}/scene_cos_sim", final_cos_sim, step=step)

            for t, sc in enumerate(acc.scene_cos_sims):
                exp.log_metric(f"{prefix}/scene_cos_sim_t{t}", sc, step=step)

            if has_cls:
                exp.log_metric(f"{prefix}/cls_cos_sim", acc.cls_cos_sims[-1], step=step)
                for t, cc in enumerate(acc.cls_cos_sims):
                    exp.log_metric(f"{prefix}/cls_cos_sim_t{t}", cc, step=step)

            if has_probe:
                for t, ia in enumerate(acc.in1k_accs):
                    exp.log_metric(f"{prefix}/in1k_tts_top1_t{t}", ia, step=step)
                if log_curves:
                    log_curve(
                        exp,
                        f"{prefix}/in1k_tts_top1_vs_timestep",
                        x=list(range(len(acc.in1k_accs))),
                        y=acc.in1k_accs,
                        step=step,
                    )

            if log_pca:
                assert target_sample0 is not None
                H, W = images.shape[-2], images.shape[-1]
                boxes = [vp.to_pixel_box(0, H, W) for vp in viewpoints]
                names = [vp.name for vp in viewpoints]
                full_img = imagenet_denormalize(images[0].cpu()).numpy()
                glimpse_grid_size = glimpse_size_px // model.backbone.patch_size_px

                _log_pca_from_accumulator(
                    exp=exp,
                    step=step,
                    prefix=prefix,
                    acc=acc,
                    full_img=full_img,
                    teacher_np=target_sample0,
                    boxes=boxes,
                    names=names,
                    canvas_grid_size=canvas_grid_size,
                    glimpse_grid_size=glimpse_grid_size,
                    log_spatial_stats=log_spatial_stats,
                    log_curves=log_curves,
                )

            return final_cos_sim
    finally:
        if scene_was_training:
            scene_normalizer.train()
        if cls_was_training:
            cls_normalizer.train()
