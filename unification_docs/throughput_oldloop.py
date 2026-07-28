"""Third arm: time the OLD loop the same way, so we can say whether the FIXED harness
MATCHES it — not merely that the fix helped.

The old loop has no per-step log record (it drives a tqdm bar and logs to the tracker),
so we run it as a subprocess and read tqdm's own rate readout off stderr, taking the
median over the steady-state tail. Compile warmup and model/data load sit in the leading
portion and are excluded by that tail.

Run this ONLY when nothing else is using the GPU — a concurrent job on the same device
(or MIG slice) contaminates every number here.

  .venv-cu126/bin/python unification_docs/throughput_oldloop.py --steps 130 --reps 3
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WDS = ("/mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/"
       "webdataset-imagenet-21k-with-features")
VAL = "/user/henrich1/u25995/jonathan/datasets/imagenet1k-val"

# tqdm writes "... 123/456 [01:23<02:34,  1.23it/s, loss=...]" (or s/it when slow)
RATE = re.compile(r"[\[,]\s*(\d+\.?\d*)(it/s|s/it)")


def run_once(steps: int, steps_per_job: int, batch: int, tail_frac: float) -> float:
    """Return median ms/step over the steady-state tail of one old-loop run."""
    cmd = [
        str(REPO / ".venv-cu126/bin/python"), "-m", "canvit_pretrain.train",
        "--webdataset-dir", WDS, "--val-dir", VAL,
        "--batch-size-per-gpu", str(batch),
        "--steps-per-job", str(steps_per_job),
        "--num-workers", "4", "--canvas-patch-grid-size", "32",
        "--tracker", "none", "--compile", "--amp",
        "--normalizer-shards", "1",
        "--init-backbone-from-teacher",
        "--val-every", str(10 ** 9),        # no validation inside the timed window
        "--log-every", "1",
        "--run-group", "perf", "--run-name", "oldloop-timing",
        "--logs-dir", "/tmp/oldloop-perf",
    ]
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/user/henrich1/u25995/.cache/huggingface")
    env.setdefault("HF_HUB_OFFLINE", "1")

    rates: list[float] = []
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, text=True, bufsize=1)
    assert proc.stderr is not None
    for line in proc.stderr:
        for chunk in line.replace("\r", "\n").splitlines():
            m = RATE.search(chunk)
            if not m:
                continue
            v = float(m.group(1))
            rates.append(1000.0 / v if m.group(2) == "it/s" else v * 1000.0)
            if len(rates) >= steps:
                proc.terminate()
                break
        else:
            continue
        break
    proc.wait(timeout=120)
    if not rates:
        raise RuntimeError("no tqdm rate readings captured — did the run fail?")
    tail = rates[int(len(rates) * (1 - tail_frac)):]
    return statistics.median(tail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=130)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps-per-job", type=int, default=8192,
                    help="production value: sets shards_per_gpu -> num_workers")
    ap.add_argument("--once", action="store_true",
                    help="run ONE rep and print 'MEDIAN_MS <v>' (for the matrix driver)")
    ap.add_argument("--tail-frac", type=float, default=0.6,
                    help="fraction of readings (from the end) treated as steady state")
    args = ap.parse_args()

    if args.once:
        ms = run_once(args.steps, args.steps_per_job, args.batch, args.tail_frac)
        print(f"    old-loop                     median {ms:7.1f} ms")
        print(f"MEDIAN_MS {ms:.3f}")
        return

    out: list[float] = []
    for rep in range(args.reps):
        ms = run_once(args.steps, args.steps_per_job, args.batch, args.tail_frac)
        out.append(ms)
        print(f"  rep {rep + 1}/{args.reps}: OLD LOOP median {ms:7.1f} ms/step", flush=True)

    print(f"\nOLD LOOP: {' '.join(f'{x:7.1f}' for x in out)}   mean {statistics.mean(out):7.1f} ms/step")
    if len(out) > 1:
        print(f"  sd {statistics.stdev(out):.1f} ms")
    print("\nCompare against unification_docs/throughput_ab.py's ASYNC (fixed) arm:")
    print("  fixed harness ~= old loop  => the non_blocking transfer explained the gap")
    print("  fixed harness  > old loop  => something else remains (metric hooks, engine overhead)")


if __name__ == "__main__":
    sys.exit(main())
