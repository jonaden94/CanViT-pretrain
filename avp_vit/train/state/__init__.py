"""Batch state for fresh-ratio survival across optimizer steps."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class SurvivalBatch:
    """Batch state for fresh-ratio survival.

    Replaces a fraction of batch each step; surviving samples keep hidden for continuation.
    """

    images: Tensor  # [B, C, H, W]
    targets: Tensor  # [B, G*G, D]
    hidden: Tensor  # [B, n_tokens, D]

    @staticmethod
    def init(images: Tensor, targets: Tensor, hidden: Tensor) -> "SurvivalBatch":
        return SurvivalBatch(images=images, targets=targets, hidden=hidden)

    def step(
        self,
        *,
        fresh_images: Tensor,
        fresh_targets: Tensor,
        next_hidden: Tensor,
        hidden_init: Tensor,
    ) -> "SurvivalBatch":
        """Permute batch, replace first K with fresh samples."""
        B = self.images.shape[0]
        K = fresh_images.shape[0]
        perm = torch.randperm(B, device=self.images.device)

        images = self.images[perm]
        targets = self.targets[perm]
        hidden = next_hidden[perm].detach()

        images[:K] = fresh_images
        targets[:K] = fresh_targets
        hidden[:K] = hidden_init

        return SurvivalBatch(images=images, targets=targets, hidden=hidden)
