"""Task-neutral multi-GPU sync for the unified harness (design §9).

**Uniform MANUAL sync — deliberately no ``nn.parallel.DistributedDataParallel`` wrapper.**
Two of the three tasks reach their backbone as ``model.canvit(...)`` from inside their
bound task (ade20k/in1k), not through the top module, so a DDP wrapper on that module
would be silently BYPASSED and those gradients would never be AllReduced — the exact
silent-mis-train failure §9 says to avoid. Instead every supported cell uses the pattern
``train/joint.py`` already uses for the scorer and IN1k for ``clf.canvit``:

  * :func:`broadcast_parameters` once after construction  => all ranks start identical;
  * :func:`allreduce_grads` once per step, after the (possibly chunked) backwards and
    BEFORE clipping => all ranks apply the same update, so the weights stay identical.

Averaging before the clip matters: clipping rank-local gradients and averaging afterwards
is not the same operation, and train/loop.py averages first (DDP's Reducer runs during
backward, and the scorer is explicitly AllReduced before its clip).

One AllReduce per step is also fewer collectives than a wrapper would issue under chunked
TBPTT, where every chunk's ``backward()`` triggers its own Reducer pass.

The §9 support matrix itself is enforced upstream by ``check_spec(..., is_dist=True)``,
which refuses the coupled ``policy_grad_to_backbone`` cell under DDP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from canvit_pretrain.train import dist as _slurm_dist

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistInfo:
    """Resolved topology for this process."""

    rank: int
    world_size: int
    device: torch.device
    is_dist: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup(*, device: str, rank: int, world_size: int) -> DistInfo:
    """Initialise the process group from the SLURM env (idempotent, no-op when
    ``WORLD_SIZE`` is unset/1) and resolve rank/world_size/device.

    The environment WINS over the passed-in values: under ``srun`` the launcher knows the
    topology and ``RunSettings``' defaults (rank 0 / world 1) would silently make every
    rank think it is alone — training the same data N times instead of sharding it.
    """
    _slurm_dist.init_dist()
    if _slurm_dist.is_dist():
        info = DistInfo(rank=_slurm_dist.rank(), world_size=_slurm_dist.world_size(),
                        device=_slurm_dist.device(), is_dist=True)
        log.info("DDP: rank %d/%d on %s", info.rank, info.world_size, info.device)
        return info
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    return DistInfo(rank=rank, world_size=world_size,
                    device=torch.device(device if use_cuda else "cpu"), is_dist=False)


def broadcast_parameters(*modules: Any, src: int = 0) -> None:
    """Make every rank's parameters AND buffers identical to ``src``'s.

    Called once after construction. Buffers matter as much as parameters here: distill's
    standardizer stats are buffers initialised from a single shard, and a rank whose stats
    differ is training against different targets.
    """
    if not _slurm_dist.is_dist():
        return
    for module in modules:
        if module is None:
            continue
        for tensor in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(tensor.data, src=src)


def allreduce_grads(params: list[Any]) -> None:
    """Average the gradients of ``params`` across ranks, in place.

    Params whose grad is None (frozen, or unused by this step's graph) are skipped — they
    contribute nothing and are consistently absent on every rank for a given spec.
    """
    if not _slurm_dist.is_dist():
        return
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    # SUM-then-divide rather than ReduceOp.AVG: AVG is NCCL-only, and this has to work
    # under gloo too (train/dist.py::all_reduce_mean makes the same choice).
    ws = _slurm_dist.world_size()
    for g in grads:
        dist.all_reduce(g, op=dist.ReduceOp.SUM)
        g /= ws


def all_reduce_mean(value: float) -> float:
    """Mean of a scalar across ranks — so a logged metric describes the global batch
    rather than rank 0's slice (train/loop.py 913). Identity in single-process mode."""
    if not _slurm_dist.is_dist():
        return value
    t = torch.tensor(float(value), device=_slurm_dist.device())
    dist.all_reduce(t, op=dist.ReduceOp.SUM)   # AVG is NCCL-only; see allreduce_grads
    return float(t.item()) / _slurm_dist.world_size()


def barrier() -> None:
    _slurm_dist.barrier()


__all__ = ["DistInfo", "all_reduce_mean", "allreduce_grads", "barrier",
           "broadcast_parameters", "setup"]
