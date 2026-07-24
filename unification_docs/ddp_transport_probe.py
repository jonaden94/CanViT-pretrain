"""Minimal 2-rank NCCL all_reduce probe — is the transport itself correct on this node?

Deliberately dependency-free (no harness, no model): it mirrors the known-good baseline
``plan_dataloading/ddp/minimal_distributed_training.py`` that ran on A100/V100. Its only
knob, ``--device-id``, toggles the ``init_process_group(device_id=...)`` kwarg that
``train/dist.py`` adds on top of that baseline, so its effect can be measured in isolation
rather than inferred. Run under ``srun --ntasks=2`` with WORLD_SIZE=2.
"""

import argparse
import os

import torch
import torch.distributed as dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", action="store_true",
                    help="pass device_id=cuda:local_rank to init_process_group (train/dist.py does)")
    a = ap.parse_args()

    rank = int(os.environ["SLURM_PROCID"])
    world = int(os.environ["WORLD_SIZE"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    local_rank = rank - gpus_per_node * (rank // gpus_per_node)

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    name = torch.cuda.get_device_name(device)
    kwargs = {"device_id": device} if a.device_id else {}
    dist.init_process_group("nccl", rank=rank, world_size=world, **kwargs)

    t = torch.ones(1 << 20, device=device) * (rank + 1)  # 1M floats, so it exercises real transport
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    expected = float(sum(range(1, world + 1)))
    ok = torch.allclose(t, torch.full_like(t, expected))
    print(f"[probe device_id={a.device_id}] rank {rank}/{world} local_rank={local_rank} "
          f"gpu={name} all_reduce_sum={t[0].item():.1f} expected={expected:.1f} "
          f"{'OK' if ok else 'MISMATCH'}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
