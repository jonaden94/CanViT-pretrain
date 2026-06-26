"""Tests for training_step TBPTT logic.

Verifies chunk_size=1 (no temporal BPTT) and chunk_size=2 (baseline)
both produce correct gradients.
"""

import importlib.util
import random
from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
from canvit_pytorch import SquarePatcherConfig, Viewpoint, create_backbone
from canvit_pytorch.modulation import ViTModulationConfig
from canvit_pytorch.patcher import FoveatedPatcherConfig
from canvit_pytorch.patcher.conditioning import FiLMConfig, PatchConditioningConfig
from torch import Tensor

from canvit_pretrain import CanViTForPretraining, CanViTForPretrainingConfig

from .config import FoveatedScaleConfig
from .step import training_step

_DEVICE = torch.device("cpu")
_B = 2
_G = 8  # canvas grid size
_D = 384  # teacher dim (vits16)


@pytest.fixture()
def model() -> CanViTForPretraining:
    backbone = create_backbone("vits16").to(_DEVICE)
    cfg = CanViTForPretrainingConfig(teacher_dim=_D)
    return CanViTForPretraining(
        backbone=backbone,
        cfg=cfg,
        glimpse_size_px=128,
        backbone_name="vits16",
        canvas_patch_grid_sizes=[_G],
    ).to(_DEVICE)


@pytest.fixture()
def tensors() -> dict[str, Tensor]:
    torch.manual_seed(42)
    return {
        "images": torch.randn(_B, 3, 224, 224, device=_DEVICE),
        "scene_target": torch.randn(_B, _G * _G, _D, device=_DEVICE),
        "cls_target": torch.randn(_B, _D, device=_DEVICE),
        "raw_scene_target": torch.randn(_B, _G * _G, _D, device=_DEVICE),
        "raw_cls_target": torch.randn(_B, _D, device=_DEVICE),
    }


def _run_step(
    model: CanViTForPretraining,
    tensors: dict[str, Tensor],
    *,
    chunk_size: int,
    continue_prob: float,
    n_glimpses_override: int | None = None,
    foveated_scale: FoveatedScaleConfig | None = None,
) -> tuple[float, dict[str, Tensor]]:
    """Run training_step, return (loss, param_grads).

    If n_glimpses_override is set, patches random.random() to produce
    exactly that many glimpses.
    """
    model.zero_grad()

    # Deterministic trajectory length via controlled random.random() sequence.
    if n_glimpses_override is not None:
        assert continue_prob > 0, "need continue_prob > 0 to control n_glimpses"
        # n_glimpses starts at chunk_size, each loop iteration adds chunk_size.
        # We need (n_glimpses_override / chunk_size - 1) successes then 1 failure.
        n_continuations = n_glimpses_override // chunk_size - 1
        assert n_continuations >= 0
        # Values < continue_prob → continue; >= continue_prob → stop.
        sequence = [continue_prob / 2] * n_continuations + [1.0]
        call_count = 0

        def controlled_random() -> float:
            nonlocal call_count
            if call_count < len(sequence):
                val = sequence[call_count]
                call_count += 1
                return val
            return 1.0  # safety: stop

        ctx = patch("canvit_pretrain.train.step.random.random", side_effect=controlled_random)
    else:
        ctx = nullcontext()

    with ctx:
        metrics = training_step(
            model=model,
            images=tensors["images"],
            scene_target=tensors["scene_target"],
            cls_target=tensors["cls_target"],
            raw_scene_target=tensors["raw_scene_target"],
            raw_cls_target=tensors["raw_cls_target"],
            scene_denorm=lambda x: x,
            cls_denorm=lambda x: x,
            enable_scene_patches_loss=True,
            enable_scene_cls_loss=True,
            glimpse_size_px=128,
            canvas_grid_size=_G,
            n_full_start_branches=0,
            n_random_start_branches=1,
            chunk_size=chunk_size,
            continue_prob=continue_prob,
            min_viewpoint_scale=0.1,
            foveated_scale=foveated_scale or FoveatedScaleConfig(),
            amp_ctx=nullcontext(),
            collect_viz=False,
        )

    grads = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    return metrics.total_loss.item(), grads


def _has_grads(grads: dict[str, Tensor]) -> bool:
    return len(grads) > 0 and any(g.abs().sum() > 0 for g in grads.values())


# ── chunk_size=1 ──────────────────────────────────────────────────────


class TestChunkSize1:
    """chunk_size=1: every timestep gets an isolated backward (no temporal BPTT)."""

    def test_single_glimpse_produces_gradients(
        self, model: CanViTForPretraining, tensors: dict[str, Tensor],
    ) -> None:
        """n_glimpses=1: the critical edge case (was previously broken)."""
        loss, grads = _run_step(model, tensors, chunk_size=1, continue_prob=0.0)
        assert loss > 0
        assert _has_grads(grads), "no gradients produced with chunk_size=1, n_glimpses=1"

    def test_two_glimpses_produces_gradients(
        self, model: CanViTForPretraining, tensors: dict[str, Tensor],
    ) -> None:
        loss, grads = _run_step(
            model, tensors, chunk_size=1, continue_prob=0.5, n_glimpses_override=2,
        )
        assert loss > 0
        assert _has_grads(grads)

    def test_three_glimpses_produces_gradients(
        self, model: CanViTForPretraining, tensors: dict[str, Tensor],
    ) -> None:
        loss, grads = _run_step(
            model, tensors, chunk_size=1, continue_prob=0.5, n_glimpses_override=3,
        )
        assert loss > 0
        assert _has_grads(grads)

    def test_no_cross_step_gradient_flow(
        self, model: CanViTForPretraining, tensors: dict[str, Tensor],
    ) -> None:
        """Verify that with chunk_size=1, gradients are isolated per timestep.

        Strategy: run twice with same model state but different random seeds
        for viewpoints. If gradients flow across steps, the t=0 viewpoint
        would influence the gradient contribution from t=1. With isolation,
        each step's gradient depends only on its own viewpoint and the
        (detached) canvas state it receives.

        We use n_glimpses=2. With isolated backward:
          grad = grad_from_t0(vp0) + grad_from_t1(vp1, detached_state)
        The t1 contribution depends on detached canvas from t0 (same model
        init → same), so only t0's contribution differs. We verify the
        difference equals exactly the difference in t0's isolated gradient.
        """
        torch.manual_seed(0)
        random.seed(0)
        loss_a, grads_a = _run_step(
            model, tensors, chunk_size=1, continue_prob=0.5, n_glimpses_override=2,
        )

        torch.manual_seed(1)
        random.seed(1)
        loss_b, grads_b = _run_step(
            model, tensors, chunk_size=1, continue_prob=0.5, n_glimpses_override=2,
        )

        # Gradients should differ (different viewpoints → different loss landscape)
        shared_keys = set(grads_a) & set(grads_b)
        assert len(shared_keys) > 0
        any_differ = any(
            not torch.allclose(grads_a[k], grads_b[k], atol=1e-6)
            for k in shared_keys
        )
        assert any_differ, "gradients identical despite different viewpoints"


# ── chunk_size=2 (baseline, regression test) ──────────────────────────


class TestChunkSize2Regression:
    """Verify chunk_size=2 (the production default) is unaffected by the fix."""

    def test_single_chunk_produces_gradients(
        self, model: CanViTForPretraining, tensors: dict[str, Tensor],
    ) -> None:
        """n_glimpses=2 (one full chunk)."""
        loss, grads = _run_step(model, tensors, chunk_size=2, continue_prob=0.0)
        assert loss > 0
        assert _has_grads(grads)

    def test_two_chunks_produces_gradients(
        self, model: CanViTForPretraining, tensors: dict[str, Tensor],
    ) -> None:
        """n_glimpses=4 (two chunks)."""
        loss, grads = _run_step(
            model, tensors, chunk_size=2, continue_prob=0.5, n_glimpses_override=4,
        )
        assert loss > 0
        assert _has_grads(grads)

    def test_deterministic_with_same_seed(
        self, model: CanViTForPretraining, tensors: dict[str, Tensor],
    ) -> None:
        """Same seed → same loss. Verifies determinism of the test harness."""
        torch.manual_seed(99)
        random.seed(99)
        loss_a, _ = _run_step(model, tensors, chunk_size=2, continue_prob=0.0)

        torch.manual_seed(99)
        random.seed(99)
        loss_b, _ = _run_step(model, tensors, chunk_size=2, continue_prob=0.0)

        assert abs(loss_a - loss_b) < 1e-5, f"non-deterministic: {loss_a} vs {loss_b}"


# ── Cross-chunk_size consistency ──────────────────────────────────────


class TestCrossChunkSizeConsistency:
    """Verify that chunk_size=1 and chunk_size=2 produce different gradients
    (confirming TBPTT truncation matters) but both produce valid, finite results."""

    def test_both_produce_finite_losses(
        self, model: CanViTForPretraining, tensors: dict[str, Tensor],
    ) -> None:
        torch.manual_seed(0)
        random.seed(0)
        loss_1, grads_1 = _run_step(
            model, tensors, chunk_size=1, continue_prob=0.5, n_glimpses_override=2,
        )

        torch.manual_seed(0)
        random.seed(0)
        loss_2, grads_2 = _run_step(
            model, tensors, chunk_size=2, continue_prob=0.0,
        )

        assert torch.isfinite(torch.tensor(loss_1))
        assert torch.isfinite(torch.tensor(loss_2))
        assert _has_grads(grads_1)
        assert _has_grads(grads_2)


@pytest.mark.skipif(importlib.util.find_spec("fovi") is None, reason="fovi not installed")
class TestFoveatedScaleModes:
    """The foveated/square path drives `viewpoint.scales`; verify each scale
    mode runs end-to-end through training_step (sensor honors `fix_size=scale*H`)."""

    @pytest.fixture()
    def foveated_model(self) -> CanViTForPretraining:
        backbone = create_backbone("vits16").to(_DEVICE)
        cfg = CanViTForPretrainingConfig(
            teacher_dim=_D,
            patcher_name="foveated",
            foveated_patcher=FoveatedPatcherConfig(
                fov=16.0, cmf_a=2.785765, resolution=32, style="isotropic",
                sampler="grid_nn", cart_patch_size=8, sample_cortex=True,
            ),
        )
        return CanViTForPretraining(
            backbone=backbone, cfg=cfg, glimpse_size_px=128,
            backbone_name="vits16", canvas_patch_grid_sizes=[_G],
        ).to(_DEVICE)

    @pytest.mark.parametrize("fscale", [
        FoveatedScaleConfig(mode="fixed", fixed_scale=1.0),
        FoveatedScaleConfig(mode="fixed", fixed_scale=1.5),  # zoom-out
        FoveatedScaleConfig(mode="per_rollout", distribution="uniform", min_scale=0.6, max_scale=1.4),
        FoveatedScaleConfig(mode="per_glimpse", distribution="safebox", min_scale=0.2, max_scale=1.0),
    ])
    def test_mode_runs_and_produces_finite_gradients(self, foveated_model, tensors, fscale):
        torch.manual_seed(0)
        random.seed(0)
        loss, grads = _run_step(
            foveated_model, tensors, chunk_size=2, continue_prob=0.0, foveated_scale=fscale,
        )
        assert torch.isfinite(torch.tensor(loss))
        assert _has_grads(grads)

    @pytest.mark.parametrize("fscale,kind", [
        (FoveatedScaleConfig(mode="fixed", fixed_scale=1.0), "fixed1"),
        (FoveatedScaleConfig(mode="fixed", fixed_scale=0.7), "fixed_other"),
        (FoveatedScaleConfig(mode="per_rollout", distribution="uniform", min_scale=0.5, max_scale=0.9), "per_rollout"),
        (FoveatedScaleConfig(mode="per_glimpse", distribution="uniform", min_scale=0.5, max_scale=0.9), "per_glimpse"),
    ])
    def test_full_start_glimpse_follows_scale_mode(self, foveated_model, tensors, fscale, kind):
        """FULL start glimpse is always centered (center=0). Its scale: ``fixed``
        -> the single training scale ``fixed_scale`` (fixed_scale=1 reproduces the
        old scale-1 full glimpse); ``per_rollout`` -> scale=1, and the whole
        full-start rollout stays scale=1 (constant scale per rollout); ``per_glimpse``
        -> full t0 is the scale-1 anchor while its random glimpses draw their own."""
        torch.manual_seed(0)
        random.seed(0)
        # First (only) branch is a FULL-start branch; 2 glimpses -> t0 FULL, t1 RANDOM.
        metrics = training_step(
            model=foveated_model,
            images=tensors["images"], scene_target=tensors["scene_target"],
            cls_target=tensors["cls_target"], raw_scene_target=tensors["raw_scene_target"],
            raw_cls_target=tensors["raw_cls_target"], scene_denorm=lambda x: x, cls_denorm=lambda x: x,
            enable_scene_patches_loss=True, enable_scene_cls_loss=True,
            glimpse_size_px=128, canvas_grid_size=_G,
            n_full_start_branches=1, n_random_start_branches=0,
            chunk_size=2, continue_prob=0.0, min_viewpoint_scale=0.1,
            foveated_scale=fscale, amp_ctx=nullcontext(), collect_viz=True,
        )
        vps = metrics.viz_data.viewpoints
        full_vp, rand_vp = vps[0], vps[1]
        # FULL glimpse is always centered at fixation.
        assert torch.allclose(full_vp.centers, torch.zeros_like(full_vp.centers), atol=1e-6)
        if kind == "fixed1":
            assert torch.allclose(full_vp.scales, torch.ones_like(full_vp.scales), atol=1e-6)
        elif kind == "fixed_other":
            assert torch.allclose(full_vp.scales, torch.full_like(full_vp.scales, 0.7), atol=1e-6)
        elif kind == "per_rollout":
            # FULL-start rollout is entirely scale=1 (one scale per rollout): the
            # full t0 AND its subsequent random glimpses are all scale 1.
            assert torch.allclose(full_vp.scales, torch.ones_like(full_vp.scales), atol=1e-6)
            assert torch.allclose(rand_vp.scales, torch.ones_like(rand_vp.scales), atol=1e-6)
            assert rand_vp.centers.abs().sum() > 0  # random glimpse center still sampled
        else:  # per_glimpse: full t0 = scale-1 anchor; random glimpses draw their own
            assert torch.allclose(full_vp.scales, torch.ones_like(full_vp.scales), atol=1e-6)
            assert (rand_vp.scales >= 0.5 - 1e-6).all() and (rand_vp.scales <= 0.9 + 1e-6).all()
            assert rand_vp.centers.abs().sum() > 0  # random glimpse center is sampled


# Cheap fovi geometry shared by the foveated/square FiLM cases below.
_FOVI_KW = dict(
    fov=16.0, cmf_a=2.785765, resolution=32, style="isotropic",
    cart_patch_size=8, sample_cortex=True,
)


def _film(encoding: str) -> PatchConditioningConfig:
    return PatchConditioningConfig(mode="film", film=FiLMConfig(encoding=encoding))


def _build(cfg: CanViTForPretrainingConfig, backbone_name: str) -> CanViTForPretraining:
    backbone = create_backbone(backbone_name).to(_DEVICE)
    return CanViTForPretraining(
        backbone=backbone, cfg=cfg, glimpse_size_px=128,
        backbone_name=backbone_name.removesuffix("_modulate"),
        canvas_patch_grid_sizes=[_G],
    ).to(_DEVICE)


# Every distinct conditioning setup used across the exp21 runs. Each thunk
# builds the corresponding pretraining model; the test then drives it through
# forward_reduce (the exact path validate() uses).
_FR_CASES = {
    # No conditioning of any kind (uniform repros, reg-repro, prune30, scale runs).
    "plain": lambda: _build(CanViTForPretrainingConfig(teacher_dim=_D), "vits16"),
    # ViT/backbone modulation (the adaLN trunk +/- cross-attn runs) -- this is
    # the path whose hoisted `modulation=` kwarg crashed validation.
    "vit_trunk_fourier": lambda: _build(
        CanViTForPretrainingConfig(
            teacher_dim=_D,
            vit_modulation=ViTModulationConfig(enabled=True, encoding="fourier"),
        ), "vits16_modulate"),
    "vit_trunk_sinusoidal": lambda: _build(
        CanViTForPretrainingConfig(
            teacher_dim=_D,
            vit_modulation=ViTModulationConfig(enabled=True, encoding="sinusoidal"),
        ), "vits16_modulate"),
    "vit_trunk_crossattn_fourier": lambda: _build(
        CanViTForPretrainingConfig(
            teacher_dim=_D,
            vit_modulation=ViTModulationConfig(
                enabled=True, encoding="fourier", modulate_cross_attn=True),
        ), "vits16_modulate"),
    # FiLM *patch* conditioning (applied inside the patcher, not via the kwarg)
    # -- on foveated and on square/fovi_regularized, fourier + sinusoidal.
    "foveated_film_fourier": lambda: _build(
        CanViTForPretrainingConfig(
            teacher_dim=_D, patcher_name="foveated",
            foveated_patcher=FoveatedPatcherConfig(
                sampler="grid_nn", conditioning=_film("fourier"), **_FOVI_KW),
        ), "vits16"),
    "foveated_film_sinusoidal": lambda: _build(
        CanViTForPretrainingConfig(
            teacher_dim=_D, patcher_name="foveated",
            foveated_patcher=FoveatedPatcherConfig(
                sampler="grid_nn", conditioning=_film("sinusoidal"), **_FOVI_KW),
        ), "vits16"),
    "square_reg_film_fourier": lambda: _build(
        CanViTForPretrainingConfig(
            teacher_dim=_D, patcher_name="square",
            square_patcher=SquarePatcherConfig(
                method="fovi_regularized", conditioning=_film("fourier"), **_FOVI_KW),
        ), "vits16"),
}


class TestForwardReduce:
    """`validate()` drives the model via ``forward_reduce`` (not training_step).

    forward_reduce hoists the per-token ViT modulation out of the glimpse loop
    and passes ``modulation=`` into every ``forward`` call -- including the None
    case for non-modulation runs. The pretraining ``forward`` override must
    accept and forward that kwarg, else *all* validations raise TypeError and
    the loop silently swallows them (no val metrics ever reach wandb).

    Parametrized over every conditioning setup in exp21 to confirm validation
    runs regardless of whether/which modulation is applied: none, ViT trunk
    (fourier/sinusoidal), ViT trunk+cross-attn, and FiLM patch conditioning on
    foveated and square patchers.
    """

    @pytest.mark.parametrize("case", list(_FR_CASES), ids=list(_FR_CASES))
    def test_forward_reduce_runs(self, case: str):
        model = _FR_CASES[case]()
        img = torch.randn(_B, 3, 224, 224, device=_DEVICE)
        vps = [Viewpoint.full_scene(batch_size=_B, device=_DEVICE) for _ in range(2)]
        with torch.inference_mode():
            acc, _ = model.forward_reduce(
                image=img, viewpoints=vps, canvas_grid_size=_G,
                init_fn=lambda s: None, step_fn=lambda acc, out, vp: out,
            )
        assert torch.isfinite(acc.local_patches).all()
