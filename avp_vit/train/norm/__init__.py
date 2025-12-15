"""Position-aware running normalization for spatial token sequences."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from avp_vit.glimpse import Viewpoint
from avp_vit.rope import grid_offsets


class PositionAwareNorm(nn.Module):
    """Running normalization with per-position stats for [B, N, D] sequences.

    Stats shape: [N, D] - one mean/var per token position per dimension.
    Handles both scene-level normalization and glimpse normalization via interpolation.

    Stateful: first batch initializes stats directly (no warmup with wrong stats).
    All ops on GPU buffers without sync.
    """

    mean: Tensor
    var: Tensor

    def __init__(
        self,
        n_tokens: int,
        embed_dim: int,
        grid_size: int,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("mean", torch.zeros(n_tokens, embed_dim))
        self.register_buffer("var", torch.ones(n_tokens, embed_dim))
        self.initialized = False

    def forward(self, x: Tensor) -> Tensor:
        """Normalize x: [B, N, D] -> [B, N, D]. Updates stats only in train mode."""
        if self.training:
            with torch.no_grad():
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=True)

                if not self.initialized:
                    self.mean.copy_(batch_mean)
                    self.var.copy_(batch_var)
                    self.initialized = True
                else:
                    self.mean.lerp_(batch_mean, self.momentum)
                    self.var.lerp_(batch_var, self.momentum)

        return (x - self.mean) / (self.var + self.eps).sqrt()

    def normalize_at_viewpoint(
        self, x: Tensor, viewpoint: Viewpoint, glimpse_grid_size: int
    ) -> Tensor:
        """Normalize glimpse features using interpolated scene stats.

        Bilinearly interpolates scene-level mean/std to glimpse positions.

        Args:
            x: [G_glimpse^2, D] glimpse features (single sample, no batch dim)
            viewpoint: Viewpoint defining glimpse position/scale
            glimpse_grid_size: Size of glimpse grid (G_glimpse)

        Returns:
            [G_glimpse^2, D] normalized features
        """
        S, G = self.grid_size, glimpse_grid_size
        D = self.mean.shape[1]

        # Reshape stats to [1, D, S, S] for grid_sample
        mean_grid = self.mean.view(S, S, D).permute(2, 0, 1).unsqueeze(0)
        std_grid = (self.var + self.eps).sqrt().view(S, S, D).permute(2, 0, 1).unsqueeze(0)

        # Build glimpse sampling grid (same coords as extract_glimpse)
        offsets = grid_offsets(G, G, self.mean.device, dtype=torch.float32)
        offsets = offsets.view(G, G, 2).unsqueeze(0)  # [1, G, G, 2]

        # Single sample: take first element of batch viewpoint
        center = viewpoint.centers[0:1].view(1, 1, 1, 2)
        scale = viewpoint.scales[0:1].view(1, 1, 1, 1)
        grid = center + scale * offsets
        grid = grid.flip(-1)  # (y, x) -> (x, y) for grid_sample

        # Interpolate stats
        mean_local = F.grid_sample(mean_grid, grid, mode="bilinear", align_corners=False)
        std_local = F.grid_sample(std_grid, grid, mode="bilinear", align_corners=False)

        # Reshape back to [G^2, D]
        mean_local = mean_local.squeeze(0).permute(1, 2, 0).reshape(G * G, D)
        std_local = std_local.squeeze(0).permute(1, 2, 0).reshape(G * G, D)

        return (x - mean_local) / std_local
