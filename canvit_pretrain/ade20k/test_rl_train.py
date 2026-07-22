"""CPU smoke tests for the in-graph policy trainer (P3).

One rollout_and_loss step per objective (QReg ε-greedy, PG sampled) on a tiny
frozen wrapper: finite loss, scorer gets gradients, frozen seg gets none. Also
the foveated path end-to-end with the fixation action space."""

from contextlib import nullcontext

import torch
from canvit_pytorch import CanViTForSemanticSegmentation
from canvit_pytorch.patcher import FoveatedPatcherConfig
from canvit_pytorch.policy import StateEncoder, ViewpointScorer

from .data import NUM_CLASSES
from .rl_train import PolicyTrainConfig, build_action_table, rollout_and_loss
from canvit_pretrain.train.rl import RunningNorm

_B, _G, _IMG = 2, 8, 64


def _setup(objective: str, model_config: dict):
    torch.manual_seed(0)
    cfg = PolicyTrainConfig(
        canvas_grid=_G, scene_size=_IMG, glimpse_px=None, score_res=32,
        scales=(0.5, 0.25), centers_per_axis=4, width=16, block_layers=1,
        train_horizon=2, batch_size=_B, objective=objective,  # type: ignore[arg-type]
        device="cpu", amp=False,
    )
    obj = cfg.build_objective()
    seg = CanViTForSemanticSegmentation(
        backbone_name="vits16", model_config=model_config, num_classes=NUM_CLASSES
    )
    seg.eval().requires_grad_(False)
    vp_flat, n_scale = build_action_table(seg, cfg)
    fixation = vp_flat.shape[0] == cfg.centers_per_axis**2 and n_scale == 1
    net = ViewpointScorer(
        canvas_dim=seg.canvas_dim, width=cfg.width, n_scale=n_scale,
        scales=(1.0,) if fixation else cfg.scales,
        centers_per_axis=cfg.centers_per_axis, block_layers=cfg.block_layers,
        dueling=(objective == "qreg"),
        action_space="fixation" if fixation else "safebox",
    )
    net.train()
    encoder = StateEncoder(seg, canvas_grid=_G)
    return cfg, obj, seg, net, encoder, vp_flat


def _one_step(objective: str, model_config: dict) -> None:
    cfg, obj, seg, net, encoder, vp_flat = _setup(objective, model_config)
    torch.manual_seed(1)
    images = torch.randn(_B, 3, _IMG, _IMG)
    masks = torch.randint(0, NUM_CLASSES, (_B, _IMG, _IMG))
    gen = torch.Generator()
    gen.manual_seed(0)
    running = [RunningNorm(momentum=0.997, device=torch.device("cpu")) for _ in range(cfg.train_horizon)]
    log_alpha = torch.tensor(0.0) if objective == "pg" else None

    loss, metrics = rollout_and_loss(
        seg=seg, net=net, critic=None, encoder=encoder, images=images, masks=masks,
        vp_flat=vp_flat, cfg=cfg, obj=obj, running=running, log_alpha=log_alpha,
        gen=gen, amp_ctx=nullcontext(),
    )
    assert torch.isfinite(loss)
    loss.backward()
    net_grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert net_grads and any(g.abs().sum() > 0 for g in net_grads)
    assert all(p.grad is None for p in seg.parameters())  # frozen stack untouched
    assert torch.isfinite(metrics["reward_frac_mean"])


def test_qreg_step_uniform() -> None:
    _one_step("qreg", {})


def test_pg_step_uniform() -> None:
    _one_step("pg", {})


def test_qreg_step_foveated_fixation() -> None:
    cfg, obj, seg, net, encoder, vp_flat = _setup(
        "qreg", {"patcher_name": "foveated", "foveated_patcher": FoveatedPatcherConfig()}
    )
    assert vp_flat.shape == (cfg.centers_per_axis**2, 3)  # fixation grid, no scale axis
    _one_step("qreg", {"patcher_name": "foveated", "foveated_patcher": FoveatedPatcherConfig()})
