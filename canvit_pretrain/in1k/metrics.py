"""IN1k classification metrics: cross-entropy loss + top-k accuracy."""

import torch.nn.functional as F
from torch import Tensor


def ce_loss(logits: Tensor, targets: Tensor, *, label_smoothing: float = 0.0) -> Tensor:
    """Mean cross-entropy over the batch. logits [B, C], targets [B] (class idx)."""
    return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)


def topk_correct(logits: Tensor, targets: Tensor, ks: tuple[int, ...] = (1, 5)) -> dict[int, int]:
    """Count of correct predictions at each k (SUMMED over the batch, for streaming
    accumulation across batches). logits [B, C], targets [B]."""
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1)  # [B, maxk]
    hit = pred.eq(targets[:, None])  # [B, maxk]
    return {k: int(hit[:, :k].any(dim=1).sum().item()) for k in ks}


class TopKAccuracy:
    """Streaming top-k accuracy accumulator (sum correct / total over batches)."""

    def __init__(self, ks: tuple[int, ...] = (1, 5)):
        self.ks = ks
        self.correct = {k: 0 for k in ks}
        self.total = 0

    def update(self, logits: Tensor, targets: Tensor) -> None:
        for k, c in topk_correct(logits, targets, self.ks).items():
            self.correct[k] += c
        self.total += targets.shape[0]

    def compute(self) -> dict[int, float]:
        return {k: (self.correct[k] / self.total if self.total else 0.0) for k in self.ks}
