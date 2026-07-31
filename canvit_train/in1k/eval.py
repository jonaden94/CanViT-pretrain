"""IN1k top-1/5 validation over a glimpse rollout.

Moved verbatim out of the (now deleted) standalone ``in1k/train.py`` during the harness
consolidation — the harness task already delegated to it so that the two entry points
could not diverge on the eval protocol. With one entry point left it belongs here.
"""

import logging

import torch
import torch.distributed as tdist

from ..harness import dist as ddp
from .config import In1kConfig
from .metrics import TopKAccuracy
from .rollout import consumes_full_image, rollout_cls_tokens

log = logging.getLogger(__name__)


def _policy_rollout_cls(*, clf, images, joint, n, canvas_grid, glimpse_px, amp_ctx):
    """Closed-loop deploy rollout: the scorer picks each glimpse by argmax from the live
    canvas. Returns the per-timestep CLS token, exactly like the open-loop
    ``rollout_cls_tokens`` it mirrors, so the accuracy code below is shared."""
    from canvit_pytorch import sample_at_viewpoint

    from ..harness.eval_viewpoints import deploy_rollout_viewpoints
    from ..harness.viewpoint import ViewpointType
    from .rollout import derive_glimpse_px

    B = images.shape[0]
    full_image = consumes_full_image(clf)
    px = None if full_image else derive_glimpse_px(clf, glimpse_px)
    cls_tokens = []

    def advance(vp, state, t):
        if state is None:  # t0: the rollout owns its own state init
            state = clf.init_state(batch_size=B, canvas_grid_size=canvas_grid)
        model_input = images if full_image else sample_at_viewpoint(
            spatial=images, viewpoint=vp, glimpse_size_px=px)
        with amp_ctx:
            out = clf.canvit(image=model_input, state=state, viewpoint=vp)
        cls_tokens.append(out.state.recurrent_cls[:, 0].float())
        return out.state

    deploy_rollout_viewpoints(joint=joint, advance=advance, t0_type=ViewpointType.FULL,
                              batch_size=B, device=images.device, n=n)
    return cls_tokens


@torch.no_grad()
def evaluate(clf, cfg: In1kConfig, val_loader, *, device, canvas_grid, amp_ctx, is_foveated,
             eval_policy: str | None = None, joint=None) -> dict[int, float]:
    """Deploy (argmax over the eval policy's viewpoints); global top-1/5 at the final
    timestep, aggregated across ranks.

    ``eval_policy`` defaults to resolving ``cfg.eval_policy`` against this task's
    historical trajectory; the harness passes the already-resolved value. ``"policy"``
    deploys ``joint``'s scorer by argmax, which needs the live canvas and so cannot
    precompute its viewpoints."""
    from ..harness.eval_viewpoints import open_loop_viewpoints, resolve
    policy = resolve(eval_policy or cfg.eval_policy, task="in1k", is_foveated=is_foveated)
    clf.eval()
    acc = TopKAccuracy(ks=(1, 5))
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        if policy == "policy":
            cls_tokens = _policy_rollout_cls(
                clf=clf, images=images, joint=joint, n=cfg.n_timesteps,
                canvas_grid=canvas_grid, glimpse_px=cfg.glimpse_px, amp_ctx=amp_ctx)
        else:
            vps = open_loop_viewpoints(
                policy, batch_size=images.shape[0], device=device, n=cfg.n_timesteps,
                is_foveated=is_foveated, foveated_scale=cfg.foveated_scale,
                min_scale=cfg.min_vp_scale, max_scale=cfg.max_vp_scale,
                foveated_eval_scale=getattr(cfg.foveated_scale, "fixed_scale", 1.0),
            )
            with amp_ctx:
                cls_tokens = rollout_cls_tokens(
                    clf=clf, images=images, viewpoints=vps, canvas_grid=canvas_grid,
                    glimpse_px=cfg.glimpse_px, freeze_backbone=True,
                )
        logits = clf.head(clf.norm(cls_tokens[-1]))
        acc.update(logits, labels)
        if cfg.limit_val_batches is not None and acc.total >= cfg.limit_val_batches * cfg.eval_batch_size:
            break
    # aggregate correct/total across ranks
    stats = torch.tensor([acc.correct[1], acc.correct[5], acc.total], dtype=torch.float64, device=device)
    if ddp.is_dist():
        tdist.all_reduce(stats, op=tdist.ReduceOp.SUM)
    top1, top5, total = stats.tolist()
    return {1: top1 / max(total, 1), 5: top5 / max(total, 1)}
