"""The harness's inline policy loss must carry the SAME gradient scale as the reference.

`ade20k/rl_train.rollout_and_loss` cats every depth into one [horizon*B, A] tensor and
takes a single `F.mse_loss` — one mean over horizon*B. The harness instead accumulates a
per-depth `qreg_loss` into `chunk_loss`, which `run_rollout` then divides by `n_glimpses`.
But only `n_glimpses-1` of those glimpses are POLICY glimpses (t0 is the anchor), so the
policy term came out (n_glimpses-1)/n_glimpses = 0.8x the reference at horizon 4 — a 20%
smaller effective policy LR at the same nominal `policy_lr`.

Found 2026-07-30 while localizing why harness-trained policies sat ~0.0016 CE above the
qband band while `rl_train` landed on it (doc 15 §A5).
"""
import torch
from torch import Tensor

from canvit_train.harness.rl import qreg_loss

H, B, A, N_GLIMPSES = 4, 6, 32, 5  # horizon 4 policy glimpses => t0 + 4 = 5 total


def _rows(w: Tensor) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    scores = [(w * torch.linspace(0.5, 1.5, A)).expand(B, A) * (1 + d) for d in range(H)]
    idx = [torch.arange(B) % A for _ in range(H)]
    tgt = [torch.full((B,), 0.1 * (d + 1)) for d in range(H)]
    return scores, idx, tgt


def _grad(build) -> Tensor:
    # Same weights on both sides — seed HERE, not in the test, or each call draws its own w
    # and the comparison is meaningless.
    w = torch.zeros(A).uniform_(-1, 1, generator=torch.Generator().manual_seed(0))
    w.requires_grad_(True)
    build(w).backward()
    assert w.grad is not None
    return w.grad.clone()


def _reference(w: Tensor) -> Tensor:
    """rl_train: one mse_loss over the concatenated depths."""
    scores, idx, tgt = _rows(w)
    return qreg_loss(torch.cat(scores), torch.cat(idx), torch.cat(tgt))[0]


def _harness(w: Tensor, *, rescale: bool) -> Tensor:
    """run_rollout: per-depth loss summed into chunk_loss, then / n_glimpses."""
    scores, idx, tgt = _rows(w)
    chunk = torch.zeros(())
    for d in range(H):
        ploss = qreg_loss(scores[d], idx[d], tgt[d])[0]
        if rescale:  # the fix
            ploss = ploss * (N_GLIMPSES / (N_GLIMPSES - 1))
        chunk = chunk + ploss
    return chunk / N_GLIMPSES


def test_rescaled_policy_loss_matches_the_reference_gradient():
    torch.manual_seed(0)
    ref = _grad(_reference)
    got = _grad(lambda w: _harness(w, rescale=True))
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_without_the_rescale_the_gradient_is_exactly_0_8x():
    """Pins the defect itself, so a future refactor cannot reintroduce it unnoticed."""
    torch.manual_seed(0)
    ref = _grad(_reference)
    unfixed = _grad(lambda w: _harness(w, rescale=False))
    # rtol here, unlike the exact check above: scaling `ref` after the fact puts the 0.8
    # multiply on the other side of the per-depth summation, which costs ~6e-8 in fp32.
    torch.testing.assert_close(unfixed, ref * ((N_GLIMPSES - 1) / N_GLIMPSES), rtol=1e-6, atol=1e-7)
