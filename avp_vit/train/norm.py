"""Fixed normalization for spatial token sequences.

Computes stats once from a data sample, then freezes them.
No online updates, no train/eval mode differences.
"""

import torch
from torch import Tensor, nn


class PositionAwareNorm(nn.Module):
    """Fixed normalization with per-position stats for [B, N, D] sequences.

    Stats shape: [N, D] - one mean/var per token position per dimension.
    Stats are computed once via set_stats() and then frozen.
    No train/eval mode difference - always uses fixed stats.
    """

    mean: Tensor
    var: Tensor
    _initialized: Tensor

    def __init__(
        self,
        n_tokens: int,
        embed_dim: int,
        grid_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.eps = eps
        self.register_buffer("mean", torch.zeros(n_tokens, embed_dim))
        self.register_buffer("var", torch.ones(n_tokens, embed_dim))
        self.register_buffer("_initialized", torch.tensor(False))

    @property
    def initialized(self) -> bool:
        val = self._initialized.item()
        assert isinstance(val, bool)
        return val

    def set_stats(self, data: Tensor) -> None:
        """Compute and set fixed stats from data tensor.

        Args:
            data: [N_samples, N_tokens, D] tensor to compute stats from
        """
        assert data.dim() == 3, f"Expected [N, tokens, D], got {data.shape}"
        assert data.shape[1] == self.n_tokens, f"Token mismatch: {data.shape[1]} vs {self.n_tokens}"
        assert data.shape[2] == self.embed_dim, f"Dim mismatch: {data.shape[2]} vs {self.embed_dim}"

        with torch.no_grad():
            self.mean.copy_(data.mean(dim=0))
            self.var.copy_(data.var(dim=0, unbiased=False))
            self._initialized.fill_(True)

    def forward(self, x: Tensor) -> Tensor:
        """Normalize x: [B, N, D] -> [B, N, D]. Always uses fixed stats."""
        assert self.initialized, "Stats not initialized - call set_stats() first"
        return (x - self.mean) / (self.var + self.eps).sqrt()

    def denormalize(self, x: Tensor) -> Tensor:
        """Invert normalization: x * sqrt(var + eps) + mean."""
        return x * (self.var + self.eps).sqrt() + self.mean
