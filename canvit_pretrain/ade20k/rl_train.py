"""Frozen-model viewing-policy training on ADE20K (unification P3).

Recreates the CanViT-PyTorch-RL flagship as a mode of the unified repo: a frozen
CanViT + seg probe (from_pretrained_with_probe), a ViewpointScorer trained by
QReg (ε-greedy DAgger, MSE on the standardized fractional-CE reward) or PG
(on-policy score-function + entropy floor). Deploy is argmax under both.

Deltas vs the RL repo (recorded in unification_docs/p3-notes.md):
- IN-GRAPH rollout (master plan §4.3): the scorer forward that selects the action
  IS the training forward — one forward per state instead of collect-then-reforward.
  BatchNorm mode (a) [user 2026-07-22]: that forward runs in train mode, so the
  strict eval-mode DAgger selection is knowingly approximated.
- Patcher-aware glimpse routing (foveated/square models consume the full image);
  the RL repo was uniform-only. The fixation action space pairs with foveated.
- Entry: python -m canvit_pretrain.ade20k.rl_train (tyro); wandb tracker.

Recipe defaults are the RL repo's canonical values (provenance in its
training/config.py, mirrored here).
"""

import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from canvit_pytorch import CanViTForSemanticSegmentation, Viewpoint, sample_at_viewpoint
from canvit_pytorch.checkpoints import resolve_canvit_repo
from canvit_pytorch.policy import (
    FEATURE_GROUPS,
    StateEncoder,
    ViewpointScorer,
    candidate_viewpoints,
    fixation_candidates,
    head_logits,
    per_image_ce,
)
from torch import Tensor
from tqdm import tqdm

from canvit_pretrain.train.rl import PG, Objective, QReg, RunningNorm, entropy_floor_step, pg_loss, qreg_loss
from canvit_pretrain.train.tracker import make_tracker

from .config import _default_ade20k_root, _default_wandb_dir, _default_wandb_entity, _default_wandb_project
from .config import ResizeMode
from .data import IGNORE_LABEL, ADE20kDataset, make_val_transforms
from .rollout import consumes_full_image, derive_glimpse_px

log = logging.getLogger(__name__)


@dataclass
class PolicyTrainConfig:
    """Frozen-model policy training (the RL repo's canonical recipe by default)."""

    model_repo: str = resolve_canvit_repo("canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02")
    probe_repo: str = ""  # "" -> probe-ade20k-40k-s512-c{canvas_grid}-in21k (the RL repo's rule)
    ade20k_root: Path = field(default_factory=_default_ade20k_root)
    scene_size: int = 512
    canvas_grid: int = 64
    glimpse_px: int | None = 128
    resize_mode: ResizeMode = "center_crop"
    """Val/eval image resize (train and eval use the same protocol here). Default
    ``center_crop`` is aspect-preserving (matches pretraining; REQUIRED for
    foveated/square models). Pass ``--resize-mode squish`` to reproduce the
    CanViT-PyTorch-RL qband band / EG-C2F numbers, which were measured on
    aspect-distorting squish resize."""

    # Action space / net (uniform models: safebox grid; foveated/square: fixation grid)
    scales: tuple[float, ...] = (0.5, 0.25)
    centers_per_axis: int = 16
    width: int = 128
    block_layers: int = 3
    feature_groups: tuple[str, ...] = FEATURE_GROUPS

    # Objective (flat knobs; `objective` selects the sum-type member)
    objective: Literal["qreg", "pg"] = "qreg"
    prime_on_policy: float = 0.5
    dueling: bool = True
    entropy_bonus: float = 0.01
    entropy_target: float | None = 1.0
    alpha_lr: float = 0.05
    qprop: bool = False

    # Rollout / budget / optimization (RL repo canonical values)
    train_horizon: int = 4
    score_res: int | None = 128
    budget_forwards: int = 640_000
    batch_size: int = 16
    lr: float = 2e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    warmup_frac: float = 0.125
    target_momentum: float = 0.997

    # Harness
    eval_every: int = 1000
    eval_batch_size: int = 32
    num_workers: int = 4
    seed: int = 0
    device: str = "cuda"
    amp: bool = True
    run_name: str = "policy"
    ckpt_dir: Path = field(default_factory=lambda: Path(os.environ.get("CHECKPOINTS_DIR", "checkpoints")) / "canvit-ade20k-policies")
    tracker: Literal["comet", "wandb", "none"] = "wandb"
    wandb_project: str | None = field(default_factory=_default_wandb_project)
    wandb_entity: str | None = field(default_factory=_default_wandb_entity)
    wandb_dir: Path | None = field(default_factory=_default_wandb_dir)

    @property
    def resolved_probe_repo(self) -> str:
        return self.probe_repo or resolve_canvit_repo(f"probe-ade20k-40k-s512-c{self.canvas_grid}-in21k")

    @property
    def max_steps(self) -> int:
        return self.budget_forwards // (self.batch_size * (1 + self.train_horizon))

    def build_objective(self) -> Objective:
        if self.objective == "qreg":
            return QReg(prime_on_policy=self.prime_on_policy, dueling=self.dueling)
        return PG(entropy_bonus=self.entropy_bonus, entropy_target=self.entropy_target,
                  alpha_lr=self.alpha_lr, qprop=self.qprop)


def ce_from_logits(logits: Tensor, masks: Tensor, *, score_res: int | None = None) -> Tensor:
    """Per-image CE from probe logits; score_res subsamples the mask + upsample target
    (~2x cheaper, candidate ranking preserved to Spearman 0.999 at 128). None = full res."""
    full = masks.shape[-1]
    res = score_res or full
    assert full % res == 0, f"score_res {res} must divide mask res {full}"
    m = masks if res == full else masks[:, :: full // res, :: full // res].contiguous()
    up = F.interpolate(logits, size=(res, res), mode="bilinear", align_corners=False)
    return per_image_ce(up, m, ignore_label=IGNORE_LABEL).float()


def _glimpse_input(seg, images: Tensor, vp: Viewpoint, glimpse_px: int | None) -> Tensor:
    if consumes_full_image(seg):
        return images
    return sample_at_viewpoint(spatial=images, viewpoint=vp, glimpse_size_px=derive_glimpse_px(seg, glimpse_px))


def advance_state(seg, images: Tensor, st, acts: Tensor, glimpse_px: int | None):
    """Next recurrent state after the glimpse at viewpoints acts [B,3] (cy,cx,scale)."""
    vp = Viewpoint(centers=acts[:, :2].contiguous(), scales=acts[:, 2].contiguous())
    return seg.canvit(image=_glimpse_input(seg, images, vp, glimpse_px), state=st, viewpoint=vp).state


def full_scene_state(seg, images: Tensor, *, canvas_grid: int, glimpse_px: int | None):
    """t0 recurrent state from the full-scene glimpse."""
    vp = Viewpoint.full_scene(batch_size=images.shape[0], device=images.device)
    return seg.canvit(
        image=_glimpse_input(seg, images, vp, glimpse_px),
        state=seg.canvit.init_state(batch_size=images.shape[0], canvas_grid_size=canvas_grid),
        viewpoint=vp,
    ).state


def build_action_table(seg, cfg: PolicyTrainConfig) -> tuple[Tensor, int]:
    """(vp_flat [A, 3], n_scale) for the model's patcher family."""
    if consumes_full_image(seg):
        cand = fixation_candidates(cfg.centers_per_axis)
    else:
        cand = candidate_viewpoints(cfg.scales, cfg.centers_per_axis)
    return cand.reshape(-1, 3), cand.shape[0]


def rollout_and_loss(
    *,
    seg: CanViTForSemanticSegmentation,
    net: ViewpointScorer,
    critic: ViewpointScorer | None,
    encoder: StateEncoder,
    images: Tensor,
    masks: Tensor,
    vp_flat: Tensor,
    cfg: PolicyTrainConfig,
    obj: Objective,
    running: list[RunningNorm],
    log_alpha: Tensor | None,
    gen: torch.Generator,
    amp_ctx,
) -> tuple[Tensor, dict[str, Tensor]]:
    """One in-graph training rollout (master plan §4.3): the scorer forward that picks
    each glimpse is the same forward the loss reads (BN mode (a): train mode). All seg
    forwards run under no_grad (frozen backbone); only the scorer carries graph."""
    B, device = images.shape[0], images.device
    A = vp_flat.shape[0]

    with torch.no_grad(), amp_ctx:
        st = full_scene_state(seg, images, canvas_grid=cfg.canvas_grid, glimpse_px=cfg.glimpse_px)
        logits = head_logits(seg, st.canvas, canvas_grid=cfg.canvas_grid)
        cur_ce = ce_from_logits(logits, masks, score_res=cfg.score_res)
    encoder.reset()

    pred_rows: list[Tensor] = []  # per-depth [B, A] scorer outputs (grad)
    crit_rows: list[Tensor] = []
    idxs: list[Tensor] = []
    fracs: list[Tensor] = []
    for _ in range(cfg.train_horizon):
        with torch.no_grad():
            f = encoder(st, logits=logits).float()
        flat = net(f).reshape(B, -1)  # train-mode forward: selects AND trains (mode (a))
        if isinstance(obj, PG):
            probs = F.softmax(flat.detach().float(), dim=1)
            idx = torch.multinomial(probs, 1, generator=gen).squeeze(1)
            if critic is not None:
                crit_rows.append(critic(f).reshape(B, -1))
        else:
            greedy = flat.detach().argmax(dim=1)
            rand_idx = torch.randint(A, (B,), device=device, generator=gen)
            on_pol = torch.rand(B, device=device, generator=gen) < obj.prime_on_policy
            idx = torch.where(on_pol, greedy, rand_idx)
        pred_rows.append(flat)
        idxs.append(idx)

        with torch.no_grad(), amp_ctx:
            st = advance_state(seg, images, st, vp_flat[idx], cfg.glimpse_px)
            logits = head_logits(seg, st.canvas, canvas_grid=cfg.canvas_grid)
            next_ce = ce_from_logits(logits, masks, score_res=cfg.score_res)
        fracs.append((cur_ce - next_ce) / cur_ce.clamp_min(1e-4))
        cur_ce = next_ce

    # per-depth global z (the regression TARGET under qreg; the REINFORCE ADVANTAGE under pg)
    sub = isinstance(obj, PG) and obj.z_subtract_only
    target = torch.cat(
        [running[d].normalize(fracs[d], subtract_only=sub) for d in range(cfg.train_horizon)]
    ).detach()
    pred_all = torch.cat(pred_rows)  # [horizon*B, A]
    flat_idx = torch.cat(idxs)

    if isinstance(obj, PG):
        alpha = obj.entropy_bonus if log_alpha is None else log_alpha.exp()
        crit_all = torch.cat(crit_rows) if crit_rows else None
        loss, entropy, metrics = pg_loss(pred_all, flat_idx, target, alpha=alpha, crit_all=crit_all)
        if log_alpha is not None:
            assert obj.entropy_target is not None
            entropy_floor_step(
                log_alpha=log_alpha, entropy=entropy.detach(),
                target=obj.entropy_target, alpha_lr=obj.alpha_lr, alpha_min=obj.entropy_bonus,
            )
            metrics["alpha"] = log_alpha.exp()
    else:
        loss, metrics = qreg_loss(pred_all, flat_idx, target)
    metrics["reward_frac_mean"] = torch.stack(fracs).mean()
    return loss, metrics


@torch.no_grad()
def evaluate(
    *, seg, net, encoder, loader, vp_flat: Tensor, cfg: PolicyTrainConfig, device, amp_ctx
) -> float:
    """Argmax deploy over the val split; returns mean per-image CE over t1..horizon
    at FULL resolution (the trainer's model-selection objective)."""
    net.eval()
    total, count = 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        with amp_ctx:
            st = full_scene_state(seg, images, canvas_grid=cfg.canvas_grid, glimpse_px=cfg.glimpse_px)
            logits = head_logits(seg, st.canvas, canvas_grid=cfg.canvas_grid)
        encoder.reset()
        ces = []
        for _ in range(cfg.train_horizon):
            f = encoder(st, logits=logits).float()
            idx = net(f).reshape(images.shape[0], -1).argmax(dim=1)
            with amp_ctx:
                st = advance_state(seg, images, st, vp_flat[idx], cfg.glimpse_px)
                logits = head_logits(seg, st.canvas, canvas_grid=cfg.canvas_grid)
            ces.append(ce_from_logits(logits, masks, score_res=None))
        total += torch.stack(ces).mean(dim=0).sum().item()
        count += images.shape[0]
    net.train()
    return total / count


def train(cfg: PolicyTrainConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    obj = cfg.build_objective()

    log.info("=" * 60)
    log.info(f"Viewing-policy training ({cfg.objective}) — frozen {cfg.model_repo}")
    log.info(f"probe={cfg.resolved_probe_repo}  steps={cfg.max_steps}  horizon={cfg.train_horizon}")

    seg = CanViTForSemanticSegmentation.from_pretrained_with_probe(
        pretrained_repo=cfg.model_repo, probe_repo=cfg.resolved_probe_repo
    ).to(device)
    seg.eval().requires_grad_(False)

    vp_flat, n_scale = build_action_table(seg, cfg)
    vp_flat = vp_flat.to(device)
    fixation = consumes_full_image(seg)
    net = ViewpointScorer(
        canvas_dim=seg.canvas_dim, width=cfg.width, n_scale=n_scale,
        scales=(1.0,) if fixation else cfg.scales,
        centers_per_axis=cfg.centers_per_axis, block_layers=cfg.block_layers,
        groups=cfg.feature_groups,
        dueling=isinstance(obj, QReg) and obj.dueling,
        action_space="fixation" if fixation else "safebox",
    ).to(device)
    net.train()
    critic: ViewpointScorer | None = None
    if isinstance(obj, PG) and obj.qprop:
        critic = ViewpointScorer(
            canvas_dim=seg.canvas_dim, width=cfg.width, n_scale=n_scale,
            scales=(1.0,) if fixation else cfg.scales,
            centers_per_axis=cfg.centers_per_axis, block_layers=cfg.block_layers,
            groups=cfg.feature_groups, dueling=False,
            action_space="fixation" if fixation else "safebox",
        ).to(device)
        critic.train()

    encoder = StateEncoder(seg, canvas_grid=cfg.canvas_grid, feature_groups=cfg.feature_groups)

    params = list(net.parameters()) + (list(critic.parameters()) if critic else [])
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay,
                            betas=(cfg.adam_beta1, cfg.adam_beta2))
    warmup_steps = max(1, int(cfg.warmup_frac * cfg.max_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup_steps))  # ramp then HOLD

    running = [RunningNorm(momentum=cfg.target_momentum, device=device) for _ in range(cfg.train_horizon)]
    log_alpha: Tensor | None = None
    if isinstance(obj, PG) and obj.entropy_target is not None:
        log_alpha = torch.tensor(math.log(obj.entropy_bonus), device=device)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)

    # Data: the RL repo's protocol — NO augmentation, both splits use the val
    # transform. resize_mode defaults to center_crop (aspect-preserving); pass
    # --resize-mode squish to reproduce the documented qband/EG-C2F numbers.
    img_tf, mask_tf = make_val_transforms(cfg.scene_size, cfg.resize_mode)
    train_ds = ADE20kDataset(root=cfg.ade20k_root, split="training", img_transform=img_tf, mask_transform=mask_tf)
    val_ds = ADE20kDataset(root=cfg.ade20k_root, split="validation", img_transform=img_tf, mask_transform=mask_tf)
    train_loader = torch.utils.data.DataLoader(
        train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(val_ds, cfg.eval_batch_size, num_workers=cfg.num_workers, pin_memory=True)

    amp_dtype = torch.bfloat16 if cfg.amp else torch.float32
    amp_ctx = torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=cfg.amp)

    exp = make_tracker(
        tracker=cfg.tracker, is_main=True, is_seeding=False,
        run_name=f"{cfg.run_name}_{cfg.objective}_s{cfg.seed}",
        wandb_project=cfg.wandb_project, wandb_entity=cfg.wandb_entity, wandb_dir=cfg.wandb_dir,
        prev_comet_id=None, prev_wandb_id=None,
    )
    exp.log_parameters({k: str(v) for k, v in asdict(cfg).items()})

    run_dir = cfg.ckpt_dir / f"{cfg.run_name}_{cfg.objective}_s{cfg.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Checkpoints: {run_dir}")

    best_ce = float("inf")
    step = 0
    train_iter = iter(train_loader)
    pbar = tqdm(total=cfg.max_steps, desc=f"policy-{cfg.objective}")
    while step < cfg.max_steps:
        try:
            images, masks = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, masks = next(train_iter)
        images, masks = images.to(device), masks.to(device)

        opt.zero_grad()
        loss, metrics = rollout_and_loss(
            seg=seg, net=net, critic=critic, encoder=encoder, images=images, masks=masks,
            vp_flat=vp_flat, cfg=cfg, obj=obj, running=running, log_alpha=log_alpha,
            gen=gen, amp_ctx=amp_ctx,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        sched.step()

        step += 1
        pbar.update(1)
        if step % 20 == 0:
            exp.log_metrics(
                {"loss": loss.item(), "grad_norm": grad_norm.item(), "lr": sched.get_last_lr()[0]}
                | {k: v.item() for k, v in metrics.items()},
                step=step,
            )

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            val_ce = evaluate(
                seg=seg, net=net, encoder=encoder, loader=val_loader, vp_flat=vp_flat,
                cfg=cfg, device=device, amp_ctx=amp_ctx,
            )
            exp.log_metric("val_ce_mean_t1_tH", val_ce, step=step)
            log.info(f"Step {step}: val CE (mean t1..t{cfg.train_horizon}) = {val_ce:.4f}")
            torch.save({"net_state": net.state_dict(), "step": step, "val_ce": val_ce,
                        "config": {k: str(v) for k, v in asdict(cfg).items()}}, run_dir / "last.pt")
            if val_ce < best_ce:
                best_ce = val_ce
                torch.save({"net_state": net.state_dict(), "step": step, "val_ce": val_ce,
                            "config": {k: str(v) for k, v in asdict(cfg).items()}}, run_dir / "best.pt")
                net.save_pretrained(run_dir / "best-hf")

    pbar.close()
    log.info(f"Done. best val CE = {best_ce:.4f}  ({run_dir})")
    exp.end()


if __name__ == "__main__":
    import tyro

    train(tyro.cli(PolicyTrainConfig))
