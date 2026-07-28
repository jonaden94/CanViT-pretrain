"""End-to-end throughput A/B: harness vs the old loop, same process, same GPU.

Why this shape:
  * The gap is a per-step RATE (measured: present in array task 0, flat thereafter,
    cv 1.0% for the old loop), so a few hundred steps is plenty -- no need for a
    production-length run.
  * Arms are INTERLEAVED (A,B,A,B,...) rather than blocked, so any drift (thermal,
    another job landing on the node, filesystem weather) hits both arms equally.
  * Every arm runs in the SAME process on the SAME device, so the A100 40GB/80GB
    confound that had to be corrected for in the retrospective log analysis cannot
    arise at all.
  * We report the MEDIAN step time per rep and then compare reps pairwise, because
    step time has a long right tail (checkpoint, validation, loader hiccups).

Usage:
  .venv-cu126/bin/python unification_docs/throughput_ab.py --steps 120 --reps 4
"""

from __future__ import annotations

import argparse
import gc
import logging
import statistics
import time
from pathlib import Path

import torch

WDS = Path("/mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/"
           "webdataset-imagenet-21k-with-features")
VAL = Path("/user/henrich1/u25995/jonathan/datasets/imagenet1k-val")


def make_cfg(batch: int, steps_per_job: int, non_blocking: bool):
    from canvit_pretrain.train.config import Config
    return Config(
        webdataset_dir=WDS, val_dir=VAL, batch_size_per_gpu=batch,
        steps_per_job=steps_per_job, num_workers=4, canvas_patch_grid_size=32,
        tracker="none", compile=True, amp=True,
        non_blocking_transfer=non_blocking,
        normalizer_shards=1,   # init cost only; irrelevant to steady-state throughput
        init_backbone_from_teacher=True,
    )


class _StepClock(logging.Handler):
    """Timestamp every per-step log record the harness emits.

    `run.py:358` logs "step %d  loss=..." once per `log_every`; with log_every=1 that is
    one record per training step, so the inter-record interval IS the step time. Timing
    off the existing instrumentation avoids adding a seam to production code purely for
    a benchmark (and avoids a callback that would itself change the timing).
    """

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.intervals: list[float] = []
        self._prev: float | None = None

    def emit(self, record):
        msg = record.getMessage()
        if not msg.startswith("step ") or "loss=" not in msg:
            return
        now = time.perf_counter()
        if self._prev is not None:
            self.intervals.append(now - self._prev)
        self._prev = now


def time_harness(cfg, n_steps: int, warmup: int) -> list[float]:
    from canvit_pretrain.harness.run import RunSettings, run
    from canvit_pretrain.tasks.distill.task import DistillRunTask

    clock = _StepClock()
    root = logging.getLogger()
    root.addHandler(clock)
    try:
        task = DistillRunTask(cfg)
        settings = RunSettings(n_steps=n_steps, device="cuda", amp=cfg.amp, log_every=1,
                               ckpt_dir=None, eval_every=0, grad_clip=cfg.grad_clip,
                               seed=0, compile=cfg.compile, resume=False,
                               log_grad_norms=False)
        run(task=task, spec=task.default_spec(), settings=settings)
    finally:
        root.removeHandler(clock)
    return clock.intervals[warmup:]


def summarize(name: str, xs: list[float]) -> float:
    med = statistics.median(xs) * 1000
    p10 = statistics.quantiles(xs, n=10)[0] * 1000
    print(f"    {name:28s} n={len(xs):4d}  median {med:7.1f} ms  p10 {p10:7.1f} ms")
    return med


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=120, help="timed steps per rep (after warmup)")
    ap.add_argument("--warmup", type=int, default=40, help="steps discarded (torch.compile)")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps-per-job", type=int, default=8192,
                    help="production value; sets shards_per_gpu -> num_workers")
    ap.add_argument("--arm", choices=("sync", "async"), default=None,
                    help="run ONE arm and print 'MEDIAN_MS <v>'. Lets a driver interleave "
                         "arms as separate processes, so each gets a fresh CUDA context "
                         "and no in-process state can leak between them.")
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    print(f"device: {torch.cuda.get_device_name(0)}  SMs={p.multi_processor_count} "
          f"mem={p.total_memory/2**30:.1f} GiB")
    if p.multi_processor_count < 100:
        print("  !! MIG slice / partial GPU: relative deltas are directional, but the")
        print("     magnitude will NOT match a full A100 (compute is slower, so a fixed")
        print("     CPU stall is hidden more effectively).")

    total = args.steps + args.warmup
    # steps_per_job must stay at the PRODUCTION value even though we only run `total`
    # steps: it sets shards_per_gpu, which caps num_workers. A short steps_per_job gives
    # shards_per_gpu=1 -> num_workers=1 -> a data-starved loader (measured data_pct 26%
    # vs ~2% in production), which swamps the effect being measured. We simply stop
    # early; the loader just does not exhaust its shard list.
    spj = args.steps_per_job
    print(f"steps/rep={total} (timing {args.steps} after {args.warmup} warmup), "
          f"steps_per_job={spj} (production value; we stop after {total}), "
          f"batch={args.batch}, reps={args.reps}\n")

    if args.arm is not None:
        cfg = make_cfg(args.batch, spj, non_blocking=(args.arm == "async"))
        xs = time_harness(cfg, total, args.warmup)
        med = summarize(f"harness-{args.arm}", xs)
        print(f"MEDIAN_MS {med:.3f}")
        return

    res: dict[str, list[float]] = {"SYNC (pre-fix)": [], "ASYNC (fixed)": []}
    for rep in range(args.reps):
        print(f"  rep {rep + 1}/{args.reps}")
        for name, nb in (("SYNC (pre-fix)", False), ("ASYNC (fixed)", True)):
            cfg = make_cfg(args.batch, spj, non_blocking=nb)
            xs = time_harness(cfg, total, args.warmup)
            res[name].append(summarize(name, xs))
            gc.collect()
            torch.cuda.empty_cache()

    print("\n" + "=" * 68)
    print("RESULT (median ms/step per rep)")
    for name, v in res.items():
        print(f"  {name:28s} {' '.join(f'{x:7.1f}' for x in v)}   mean {statistics.mean(v):7.1f}")
    a, b = res["SYNC (pre-fix)"], res["ASYNC (fixed)"]
    deltas = [(y / x - 1) * 100 for x, y in zip(a, b)]
    print(f"\n  per-rep delta (fixed vs pre-fix): {' '.join(f'{d:+.1f}%' for d in deltas)}")
    print(f"  mean {statistics.mean(deltas):+.1f}%"
          + (f"  sd {statistics.stdev(deltas):.1f}%" if len(deltas) > 1 else ""))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
