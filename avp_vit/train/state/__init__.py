"""Training state management with Bernoulli survival."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TrainState:
    """Persistent state for Bernoulli survival training.

    Bernoulli survival enables learning to generalize across arbitrary inference
    lengths despite BPTT being restricted to short horizons. Each batch item has
    probability `survival_prob` of being "carried over" between optimizer steps:
    - Surviving items: same image/target, hidden state continues from previous step
    - Non-surviving items: fresh image/target, hidden resets to learned spatial_init

    This creates a geometric distribution of effective sequence lengths.

    IMPORTANT - Naming convention (consistent with AVPViT):
    - hidden: Internal state for CONTINUATION between timesteps
    - local_prev: Local state for CONTINUATION when use_local_temporal enabled
    - scene: Projected output for loss/viz (not stored here, computed on demand)
    """

    images: Tensor  # [B, C, H, W]
    targets: Tensor  # [B, G*G, D]
    hidden: Tensor | None  # [B, G*G, D] or None (first step) - for CONTINUATION
    local_prev: Tensor | None  # [B, N, D] or None - for CONTINUATION when use_local_temporal

    @staticmethod
    def init(images: Tensor, targets: Tensor) -> "TrainState":
        """Initialize state with fresh batch, no hidden history."""
        return TrainState(images=images, targets=targets, hidden=None, local_prev=None)

    def step(
        self,
        fresh_images: Tensor,
        fresh_targets: Tensor,
        next_hidden: Tensor,
        next_local_prev: Tensor | None,
        survival_prob: float,
        hidden_init: Tensor,
        local_init: Tensor | None,
    ) -> "TrainState":
        """Update state with Bernoulli survival.

        Args:
            fresh_images: New images from dataloader [B, C, H, W]
            fresh_targets: Teacher patches for fresh images [B, G*G, D]
            next_hidden: Hidden state from forward step [B, n_tokens, D] (for CONTINUATION)
            next_local_prev: Local state from forward step [B, N, D] (when use_local_temporal)
            survival_prob: Probability of keeping current item
            hidden_init: Initialized hidden state [B, n_tokens, D] (from model._init_hidden)
            local_init: Initialized local state [B, N, D] or None (when use_local_temporal)

        Returns:
            Updated TrainState for next iteration.
        """
        B = self.images.shape[0]
        device = self.images.device

        survive = torch.rand(B, device=device) < survival_prob
        s_img = survive.view(B, 1, 1, 1)
        s_feat = survive.view(B, 1, 1)

        images = torch.where(s_img, self.images, fresh_images)
        targets = torch.where(s_feat, self.targets, fresh_targets)

        # Survivors: detach to cut BPTT across optimizer steps
        # Non-survivors: reset to hidden_init (gradients flow through hidden_init)
        hidden = torch.where(s_feat, next_hidden.detach(), hidden_init)

        # Same pattern for local_prev when use_local_temporal enabled
        if next_local_prev is not None:
            assert local_init is not None
            local_prev = torch.where(s_feat, next_local_prev.detach(), local_init)
        else:
            local_prev = None

        return TrainState(images=images, targets=targets, hidden=hidden, local_prev=local_prev)
