"""A frozen head must be in EVAL mode, not merely requires_grad_(False).

`requires_grad_(False)` leaves BatchNorm updating its running statistics. The ADE20K
probe head carries one (`head.bn`), so on a policy run — where the reward IS the probe's
CE reduction — the scorer chased a probe drifting under its feet, seed-dependently.
Caught by exp27 arm B: policy-independent t0 mIoU 38.50 vs 38.75 on two seeds.
"""
import torch
import torch.nn as nn

from canvit_train.harness.loop import apply_requires_grad
from canvit_train.harness.spec import TrainSpec


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        return self.bn(x)


class _Model(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.canvit = nn.Linear(2, 2)
        self.head = head


def _apply(spec):
    head = _Head()
    apply_requires_grad(model=_Model(head), head=head, joint=None, spec=spec)
    return head


def test_frozen_head_is_in_eval_mode():
    head = _apply(TrainSpec.policy_only(freeze_model=True))
    assert not head.training, "a frozen head must be .eval(), or head.bn keeps drifting"
    assert not any(p.requires_grad for p in head.parameters())


def test_frozen_head_bn_stats_do_not_move():
    """The behaviour the mode flag actually buys — this is what corrupted the reward."""
    head = _apply(TrainSpec.policy_only(freeze_model=True))
    before = head.bn.running_mean.clone()
    for _ in range(5):
        head(torch.randn(8, 4, 6, 6) * 10 + 3)  # wildly off-distribution
    assert torch.equal(head.bn.running_mean, before)


def test_a_trainable_head_stays_in_train_mode():
    """The fix must not freeze the probe/finetune runs that DO train the head."""
    head = _apply(TrainSpec.probe())
    assert head.training
    before = head.bn.running_mean.clone()
    head(torch.randn(8, 4, 6, 6) * 10 + 3)
    assert not torch.equal(head.bn.running_mean, before)
