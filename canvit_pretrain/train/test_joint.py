"""P4b tests: joint task+policy training through the P1/P4a seams (CPU, tiny model).

Covers the contract training_step honors when a JointPolicy is passed: the scorer
AND the distill task both receive gradient, the per-glimpse policy loss is folded
into the chunk backward, feats_detached gates the policy->backbone gradient, and
the keep_random_branch / PG variants run. Parity when joint is None is guarded by
the byte-for-byte parity probe, not here."""

from contextlib import nullcontext

import torch
from canvit_pytorch import RecurrentState, create_backbone

from canvit_pretrain import CanViTForPretraining, CanViTForPretrainingConfig

from .config import FoveatedScaleConfig, JointPolicyConfig
from .joint import build_joint_policy
from .step import training_step
from .viewpoint import ViewpointType

_DEVICE = torch.device("cpu")
_B, _G, _D = 2, 8, 384


def _model() -> CanViTForPretraining:
    torch.manual_seed(0)
    backbone = create_backbone("vits16").to(_DEVICE)
    return CanViTForPretraining(
        backbone=backbone,
        cfg=CanViTForPretrainingConfig(teacher_dim=_D),
        glimpse_size_px=128,
        backbone_name="vits16",
        canvas_patch_grid_sizes=[_G],
    ).to(_DEVICE)


def _tensors() -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    return {
        "images": torch.randn(_B, 3, 224, 224, device=_DEVICE),
        "scene_target": torch.randn(_B, _G * _G, _D, device=_DEVICE),
        "cls_target": torch.randn(_B, _D, device=_DEVICE),
        "raw_scene_target": torch.randn(_B, _G * _G, _D, device=_DEVICE),
        "raw_cls_target": torch.randn(_B, _D, device=_DEVICE),
    }


def _small_rl(**overrides) -> JointPolicyConfig:
    base = dict(use_rl=True, centers_per_axis=4, width=16, block_layers=1, scales=(0.5,))
    base.update(overrides)
    return JointPolicyConfig(**base)


def _build(model, rl):
    gen = torch.Generator(device=_DEVICE)
    gen.manual_seed(0)
    return build_joint_policy(
        core_model=model, rl=rl, device=_DEVICE, canvas_grid=_G,
        min_viewpoint_scale=0.1, foveated_scale=FoveatedScaleConfig(), generator=gen,
    )


def _step(model, jp, *, continue_prob=0.0, chunk_size=2, n_full=1, n_random=1):
    t = _tensors()
    model.zero_grad()
    jp.scorer.zero_grad()
    return training_step(
        model=model, images=t["images"], scene_target=t["scene_target"], cls_target=t["cls_target"],
        raw_scene_target=t["raw_scene_target"], raw_cls_target=t["raw_cls_target"],
        scene_denorm=lambda x: x, cls_denorm=lambda x: x,
        enable_scene_patches_loss=True, enable_scene_cls_loss=True,
        glimpse_size_px=128, canvas_grid_size=_G,
        n_full_start_branches=n_full, n_random_start_branches=n_random,
        chunk_size=chunk_size, continue_prob=continue_prob, min_viewpoint_scale=0.1,
        foveated_scale=FoveatedScaleConfig(), amp_ctx=nullcontext(), joint=jp,
    )


def _grad_flowed(params) -> bool:
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)


def test_qreg_trains_scorer_and_task() -> None:
    model = _model()
    jp = _build(model, _small_rl(objective="qreg", prime_on_policy=0.5))
    metrics = _step(model, jp)
    assert torch.isfinite(metrics.total_loss)
    assert _grad_flowed(jp.scorer.parameters()), "scorer got no gradient from the policy loss"
    assert _grad_flowed(model.parameters()), "distill task still trains the model"
    assert jp.last_step and torch.isfinite(jp.last_step["policy_loss"])
    assert torch.isfinite(jp.last_step["reward_frac"])


def test_pg_trains_scorer() -> None:
    model = _model()
    jp = _build(model, _small_rl(objective="pg"))
    metrics = _step(model, jp)
    assert torch.isfinite(metrics.total_loss)
    assert _grad_flowed(jp.scorer.parameters())
    assert jp.log_alpha is not None  # entropy floor active (entropy_target=1.0)


def test_keep_random_branch_runs_both() -> None:
    model = _model()
    jp = _build(model, _small_rl(keep_random_branch=True))
    metrics = _step(model, jp, n_full=1, n_random=1)
    # the FULL-start branch is the policy branch (populates last_step); both branches
    # feed the distill task, so both the scorer and the model get gradient.
    assert jp.last_step and torch.isfinite(jp.last_step["policy_loss"])
    assert _grad_flowed(jp.scorer.parameters())
    assert _grad_flowed(model.parameters())
    assert torch.isfinite(metrics.total_loss)


def test_feats_detached_gates_backbone_gradient() -> None:
    """At the selector level: feats_detached=True cuts the policy gradient to the
    canvas state (backbone), while the scorer still learns; =False lets it through."""
    model = _model()
    state = model.init_state(batch_size=_B, canvas_grid_size=_G)

    for detached, expect_state_grad in [(True, False), (False, True)]:
        jp = _build(model, _small_rl(feats_detached=detached))
        st = RecurrentState(
            canvas=state.canvas.detach().clone().requires_grad_(True),
            recurrent_cls=state.recurrent_cls.detach().clone().requires_grad_(True),
        )
        ctx = jp.policy_selector.start_rollout(t0_type=ViewpointType.RANDOM, batch_size=_B, device=_DEVICE)
        jp.policy_selector.select(
            vp_type=ViewpointType.RANDOM, ctx=ctx, t=1, batch_size=_B, device=_DEVICE, state=st
        )
        jp.policy_selector.last_aux["scores"].sum().backward()
        got = st.canvas.grad is not None and st.canvas.grad.abs().sum() > 0
        assert got == expect_state_grad, f"detached={detached}: state-grad {got}, expected {expect_state_grad}"
        assert _grad_flowed(jp.scorer.parameters()), "scorer must learn in both modes"


def test_state_dict_roundtrip() -> None:
    model = _model()
    jp = _build(model, _small_rl())
    _step(model, jp)  # populates running-norm(s)
    jp.log_alpha = None  # QReg has no dual variable
    sd = jp.state_dict()

    jp2 = _build(_model(), _small_rl())
    p1 = next(iter(jp.scorer.parameters())).detach().clone()
    with torch.no_grad():  # perturb so the load has something to restore
        next(iter(jp2.scorer.parameters())).add_(1.0)
    assert not torch.allclose(p1, next(iter(jp2.scorer.parameters())))
    jp2.load_state_dict(sd)
    assert torch.allclose(p1, next(iter(jp2.scorer.parameters())))
    assert set(jp2.running) == set(jp.running)
    for d in jp.running:
        assert torch.allclose(jp2.running[d].mean, jp.running[d].mean)
        assert jp2.running[d].count == jp.running[d].count


def test_prime_curriculum() -> None:
    model = _model()
    jp = _build(model, _small_rl(objective="qreg", prime_on_policy=0.5, policy_warmup_steps=100))
    jp.set_prime_for_step(0)
    assert jp.policy_selector.prime_on_policy == 0.0
    jp.set_prime_for_step(50)
    assert abs(jp.policy_selector.prime_on_policy - 0.25) < 1e-6
    jp.set_prime_for_step(100)
    assert abs(jp.policy_selector.prime_on_policy - 0.5) < 1e-6
    jp.set_prime_for_step(200)  # holds at target
    assert abs(jp.policy_selector.prime_on_policy - 0.5) < 1e-6

    pg = _build(_model(), _small_rl(objective="pg"))
    before = pg.policy_selector.prime_on_policy
    pg.set_prime_for_step(50)  # no-op for PG (always samples on-policy)
    assert pg.policy_selector.prime_on_policy == before


def test_action_space_matches_patcher() -> None:
    model = _model()
    uni = _build(model, _small_rl())
    assert uni.scorer.action_space == "safebox"

    model.cfg.patcher_name = "foveated"  # drives only the action-space branch
    fov = _build(model, _small_rl())
    assert fov.scorer.action_space == "fixation"
    assert torch.allclose(fov.policy_selector.vp_flat[:, 2], torch.ones(fov.policy_selector.vp_flat.shape[0]))
