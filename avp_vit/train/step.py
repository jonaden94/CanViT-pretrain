"""Training step with truncated BPTT and balanced branches."""

import random
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Callable, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from avp_vit import ActiveCanViT, GlimpseOutput, RecurrentState
from canvit import Viewpoint

from .viewpoint import Viewpoint as NamedViewpoint, ViewpointType


class NormalizedTargets(NamedTuple):
    """Normalized teacher features (patches + CLS)."""
    patches: Tensor
    cls: Tensor


class LossOutput(NamedTuple):
    """Output from compute_loss - individual losses + combined mean."""

    scene_loss: Tensor
    scene_cls_loss: Tensor
    glimpse_patches_loss: Tensor
    glimpse_cls_loss: Tensor
    combined: Tensor  # mean of active losses (for gradient scaling)
    scene_pred: Tensor  # for cosine similarity metrics
    cls_pred: Tensor


class BranchMetrics(NamedTuple):
    """Metrics for a (t0_type, t1_type) combination."""

    loss: Tensor
    scene_loss: Tensor
    scene_cls_loss: Tensor
    glimpse_patches_loss: Tensor
    glimpse_cls_loss: Tensor
    scene_cos: Tensor
    cls_cos: Tensor


class StepMetrics(NamedTuple):
    """Output from training_step."""

    total_loss: Tensor
    branches: dict[tuple[ViewpointType, ViewpointType], BranchMetrics]
    n_glimpses: int  # trajectory length this step


@dataclass
class ChunkState:
    """State for TBPTT chunk processing."""

    state: RecurrentState  # recurrent model state (canvas, cls, local_registers)
    vpe: Tensor | None  # VPE token (ephemeral, per-glimpse)
    # Combined loss (mean of active components) - for backprop and metrics
    chunk_combined_loss: Tensor  # with grad
    total_combined_loss: Tensor  # detached
    # Individual losses (for per-component metrics)
    total_scene_loss: Tensor
    total_scene_cls_loss: Tensor
    total_glimpse_patches_loss: Tensor
    total_glimpse_cls_loss: Tensor
    n_steps: int
    scene_pred: Tensor
    cls_pred: Tensor


def training_step(
    *,
    model: ActiveCanViT,
    images: Tensor,
    scene_target: Tensor,
    cls_target: Tensor,
    compute_glimpse_targets: Callable[[Tensor], NormalizedTargets] | None,
    glimpse_size_px: int,
    canvas_grid_size: int,
    n_branches: int,
    chunk_size: int,
    continue_prob: float,
    min_viewpoint_scale: float,
    amp_ctx: AbstractContextManager,
    use_checkpointing: bool,
) -> StepMetrics:
    """Training with truncated BPTT.

    Chunks of `chunk_size` steps: backward after each chunk, gradient flows within.
    Trajectory length is stochastic: chunk_size * (1 + geometric(continue_prob)).
    Memory is O(chunk_size) due to chunking.
    """
    assert n_branches >= 2 and n_branches % 2 == 0
    assert chunk_size >= 2
    assert 0.0 <= continue_prob <= 1.0
    device = images.device
    B = images.shape[0]

    state_init = model.init_state(batch_size=B, canvas_grid_size=canvas_grid_size)

    # Sample trajectory length in chunks (shared across branches)
    n_glimpses = chunk_size
    while random.random() < continue_prob:
        n_glimpses += chunk_size

    vp_types = _assign_viewpoint_types(
        n_glimpses=n_glimpses,
        n_branches=n_branches,
        has_policy=model.policy is not None,
    )

    full_indices = [i for i in range(n_branches) if vp_types[0][i] == ViewpointType.FULL]
    random_indices = [i for i in range(n_branches) if vp_types[0][i] == ViewpointType.RANDOM]

    # Metrics accumulators (per-branch)
    traj_losses = torch.zeros(n_branches, device=device)
    scene_losses = torch.zeros(n_branches, device=device)
    scene_cls_losses = torch.zeros(n_branches, device=device)
    glimpse_patches_losses = torch.zeros(n_branches, device=device)
    glimpse_cls_losses = torch.zeros(n_branches, device=device)
    scene_cos = torch.zeros(n_branches, device=device)
    cls_cos = torch.zeros(n_branches, device=device)

    def make_vp(vp_type: ViewpointType, vpe: Tensor | None) -> Viewpoint:
        if vp_type == ViewpointType.RANDOM:
            v = NamedViewpoint.random(batch_size=B, device=device, min_scale=min_viewpoint_scale)
            return Viewpoint(centers=v.centers, scales=v.scales)
        if vp_type == ViewpointType.FULL:
            v = NamedViewpoint.full_scene(batch_size=B, device=device)
            return Viewpoint(centers=v.centers, scales=v.scales)
        assert model.policy is not None and vpe is not None
        p = model.policy(vpe)
        return Viewpoint(centers=p.position, scales=p.scale)

    def forward_glimpse(*, state: RecurrentState, vp: Viewpoint, use_ckpt: bool) -> GlimpseOutput:
        if use_ckpt:
            out = checkpoint(
                lambda cv, cl, lr, ctr, sc: model.forward_step(
                    image=images,
                    state=RecurrentState(canvas=cv, cls=cl, local_registers=lr),
                    viewpoint=Viewpoint(centers=ctr, scales=sc),
                    glimpse_size_px=glimpse_size_px,
                ),
                state.canvas, state.cls, state.local_registers, vp.centers, vp.scales,
                use_reentrant=False,
            )
            assert isinstance(out, GlimpseOutput)
            return out
        return model.forward_step(
            image=images, state=state,
            viewpoint=vp, glimpse_size_px=glimpse_size_px,
        )

    def compute_loss(out: GlimpseOutput) -> LossOutput:
        # Scene losses (targets = full image features)
        scene_pred = model.predict_teacher_scene(out.state.canvas)
        cls_pred = model.predict_scene_teacher_cls(out.state.cls, out.state.canvas)
        scene_loss = F.mse_loss(scene_pred, scene_target)
        scene_cls_loss = F.mse_loss(cls_pred, cls_target)

        # Glimpse losses (targets = glimpse features from teacher)
        if compute_glimpse_targets is not None:
            glimpse_targets = compute_glimpse_targets(out.glimpse)
            glimpse_patches_pred = model.predict_glimpse_teacher_patches(out.local_patches)
            glimpse_cls_pred = model.predict_glimpse_teacher_cls(out.local_cls)
            glimpse_patches_loss = F.mse_loss(glimpse_patches_pred, glimpse_targets.patches)
            glimpse_cls_loss = F.mse_loss(glimpse_cls_pred, glimpse_targets.cls)
        else:
            glimpse_patches_loss = torch.zeros((), device=device)
            glimpse_cls_loss = torch.zeros((), device=device)

        # Combined = mean of active losses (automatic normalization)
        active = [scene_loss, scene_cls_loss]
        if compute_glimpse_targets is not None:
            active.extend([glimpse_patches_loss, glimpse_cls_loss])
        combined = torch.stack(active).mean()

        return LossOutput(
            scene_loss=scene_loss,
            scene_cls_loss=scene_cls_loss,
            glimpse_patches_loss=glimpse_patches_loss,
            glimpse_cls_loss=glimpse_cls_loss,
            combined=combined,
            scene_pred=scene_pred,
            cls_pred=cls_pred,
        )

    def run_tbptt(
        *,
        branch_idx: int,
        out_t0: GlimpseOutput,
        loss_t0: LossOutput,
        retain_first_chunk: bool,
    ) -> None:
        """TBPTT with 1-step lookahead: backward every 2 steps, gradient flows through both."""
        L = loss_t0

        chunk = ChunkState(
            state=out_t0.state,
            vpe=out_t0.vpe,
            chunk_combined_loss=L.combined.float(),
            total_combined_loss=L.combined.detach().float(),
            total_scene_loss=L.scene_loss.detach().float(),
            total_scene_cls_loss=L.scene_cls_loss.detach().float(),
            total_glimpse_patches_loss=L.glimpse_patches_loss.detach().float(),
            total_glimpse_cls_loss=L.glimpse_cls_loss.detach().float(),
            n_steps=1,
            scene_pred=L.scene_pred,
            cls_pred=L.cls_pred,
        )

        for t in range(1, n_glimpses):
            vp = make_vp(vp_types[t][branch_idx], chunk.vpe)

            with amp_ctx:
                use_ckpt = use_checkpointing and (t % 2 == 1)
                out = forward_glimpse(state=chunk.state, vp=vp, use_ckpt=use_ckpt)
                L = compute_loss(out)

            # Accumulate combined (with grad for backprop)
            chunk.chunk_combined_loss = chunk.chunk_combined_loss + L.combined.float()
            chunk.total_combined_loss = chunk.total_combined_loss + L.combined.detach().float()
            # Accumulate individual losses (detached, for per-component metrics)
            chunk.total_scene_loss = chunk.total_scene_loss + L.scene_loss.detach().float()
            chunk.total_scene_cls_loss = chunk.total_scene_cls_loss + L.scene_cls_loss.detach().float()
            chunk.total_glimpse_patches_loss = chunk.total_glimpse_patches_loss + L.glimpse_patches_loss.detach().float()
            chunk.total_glimpse_cls_loss = chunk.total_glimpse_cls_loss + L.glimpse_cls_loss.detach().float()
            chunk.scene_pred, chunk.cls_pred = L.scene_pred, L.cls_pred
            chunk.n_steps += 1

            is_chunk_end = ((t + 1) % chunk_size == 0)
            is_last = (t == n_glimpses - 1)

            if is_chunk_end:
                # Backward chunk: combined already averaged over active losses
                loss_for_backward = chunk.chunk_combined_loss / n_glimpses / n_branches
                retain = retain_first_chunk if t == 1 else False
                loss_for_backward.backward(retain_graph=retain)

                # Detach for next chunk
                if not is_last:
                    chunk.state = RecurrentState(
                        canvas=out.state.canvas.detach(),
                        cls=out.state.cls.detach(),
                        local_registers=out.state.local_registers.detach(),
                    )
                    chunk.vpe = out.vpe.detach() if out.vpe is not None else None
                    chunk.chunk_combined_loss = torch.zeros((), device=device)
                else:
                    chunk.state = out.state
                    chunk.vpe = out.vpe
            else:
                chunk.state = out.state
                chunk.vpe = out.vpe

        # Record metrics (no trailing step - n_glimpses is always a multiple of chunk_size)
        traj_losses[branch_idx] = chunk.total_combined_loss / chunk.n_steps
        scene_losses[branch_idx] = chunk.total_scene_loss / chunk.n_steps
        scene_cls_losses[branch_idx] = chunk.total_scene_cls_loss / chunk.n_steps
        glimpse_patches_losses[branch_idx] = chunk.total_glimpse_patches_loss / chunk.n_steps
        glimpse_cls_losses[branch_idx] = chunk.total_glimpse_cls_loss / chunk.n_steps
        scene_cos[branch_idx] = F.cosine_similarity(chunk.scene_pred, scene_target, dim=-1).mean()
        cls_cos[branch_idx] = F.cosine_similarity(chunk.cls_pred, cls_target, dim=-1).mean()

    # === FULL branches: share t=0 ===
    if full_indices:
        with amp_ctx:
            vp_full = make_vp(ViewpointType.FULL, None)
            out_full = forward_glimpse(state=state_init, vp=vp_full, use_ckpt=False)
            loss_full = compute_loss(out_full)

        for idx, i in enumerate(full_indices):
            run_tbptt(
                branch_idx=i,
                out_t0=out_full,
                loss_t0=loss_full,
                retain_first_chunk=(idx < len(full_indices) - 1),
            )

    # === RANDOM branches: unique t=0 each ===
    for i in random_indices:
        with amp_ctx:
            vp_rand = make_vp(ViewpointType.RANDOM, None)
            out = forward_glimpse(state=state_init, vp=vp_rand, use_ckpt=False)
            loss = compute_loss(out)

        run_tbptt(branch_idx=i, out_t0=out, loss_t0=loss, retain_first_chunk=False)

    # Aggregate metrics
    branches = _aggregate_branch_metrics(
        vp_types=vp_types,
        n_branches=n_branches,
        has_policy=model.policy is not None,
        traj_losses=traj_losses,
        scene_losses=scene_losses,
        scene_cls_losses=scene_cls_losses,
        glimpse_patches_losses=glimpse_patches_losses,
        glimpse_cls_losses=glimpse_cls_losses,
        scene_cos=scene_cos,
        cls_cos=cls_cos,
    )

    return StepMetrics(total_loss=traj_losses.mean(), branches=branches, n_glimpses=n_glimpses)


def _assign_viewpoint_types(
    *, n_glimpses: int, n_branches: int, has_policy: bool
) -> list[list[ViewpointType]]:
    """Assign viewpoint types: random permutation of half/half at each timestep."""
    vp_types: list[list[ViewpointType]] = []
    for t in range(n_glimpses):
        if t == 0:
            base = [ViewpointType.RANDOM] * (n_branches // 2) + [ViewpointType.FULL] * (n_branches // 2)
        elif has_policy:
            base = [ViewpointType.RANDOM] * (n_branches // 2) + [ViewpointType.POLICY] * (n_branches // 2)
        else:
            base = [ViewpointType.RANDOM] * n_branches
        random.shuffle(base)
        vp_types.append(base)
    return vp_types


def _aggregate_branch_metrics(
    *,
    vp_types: list[list[ViewpointType]],
    n_branches: int,
    has_policy: bool,
    traj_losses: Tensor,
    scene_losses: Tensor,
    scene_cls_losses: Tensor,
    glimpse_patches_losses: Tensor,
    glimpse_cls_losses: Tensor,
    scene_cos: Tensor,
    cls_cos: Tensor,
) -> dict[tuple[ViewpointType, ViewpointType], BranchMetrics]:
    """Aggregate metrics by (t0_type, t1_type)."""
    branches: dict[tuple[ViewpointType, ViewpointType], BranchMetrics] = {}
    t1_options = [ViewpointType.RANDOM, ViewpointType.POLICY] if has_policy else [ViewpointType.RANDOM]

    for t0 in [ViewpointType.RANDOM, ViewpointType.FULL]:
        for t1 in t1_options:
            indices = [i for i in range(n_branches) if vp_types[0][i] == t0 and vp_types[1][i] == t1]
            if indices:
                branches[(t0, t1)] = BranchMetrics(
                    loss=traj_losses[indices].mean(),
                    scene_loss=scene_losses[indices].mean(),
                    scene_cls_loss=scene_cls_losses[indices].mean(),
                    glimpse_patches_loss=glimpse_patches_losses[indices].mean(),
                    glimpse_cls_loss=glimpse_cls_losses[indices].mean(),
                    scene_cos=scene_cos[indices].mean(),
                    cls_cos=cls_cos[indices].mean(),
                )

    return branches
