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
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WDS = ("/mnt/lustre-rzg/workspaces/ws/nib00021/u25995-inet21k-feat/"
       "webdataset-imagenet-21k-with-features")
VAL = "/user/henrich1/u25995/jonathan/datasets/imagenet1k-val"

# We parse tqdm's STEP COUNTER ("123/456 [") and take our own wall-clock stamp on each
# redraw, rather than reading tqdm's printed rate. tqdm reports a SMOOTHED EMA over the
# whole run (default smoothing=0.3), which is dominated by the slow torch.compile/warmup
# steps and does not converge within a short run: it reported 1220 ms/step for a loop
# whose steady state is ~470-680 ms on the same device. Own-timestamp deltas give the
# true per-step interval and are directly comparable to how throughput_ab.py times the
# harness (inter-log-record intervals, warmup discarded).
# MUST be anchored to the TRAINING bar (train/loop.py:682 sets desc="Training").
# The teacher weight load renders its own tqdm ("Loading weights: 211/211 [...]"), and an
# unanchored counter regex reads THAT: it saw 0 -> 211, concluded 211 steps had elapsed,
# and killed the run seconds in, before training even started.
COUNT = re.compile(r"Training:.*?(\d+)/\d+\s*\[")


def run_once(steps: int, steps_per_job: int, batch: int, tail_frac: float) -> float:
    """Return median ms/step over the steady-state tail of one old-loop run."""
    # sys.executable, NOT REPO/.venv-cu126: under commit pinning REPO is the `git
    # archive` SNAPSHOT, which contains no venv (git does not track it). The parent is
    # already running the right interpreter, and PYTHONPATH/PYTHONSAFEPATH set by the
    # sbatch make the child import the pinned sources too.
    cmd = [
        sys.executable, "-m", "canvit_pretrain.train",
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

    marks: list[tuple[float, int]] = []
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    assert proc.stderr is not None
    # RAW chunked reads, split on \r AND \n. `for line in proc.stderr` would block
    # forever here: tqdm redraws with a CARRIAGE RETURN and emits no newline until the
    # bar closes, so a newline-delimited iterator yields nothing while the run proceeds.
    # That is why the `old` arm of job 15089880 produced no reading and failed.
    fd = proc.stderr.fileno()
    buf = ""
    # Keep a rolling tail of the child's stderr. The reader consumes stderr to find the
    # counter, so without this a crash in the child is INVISIBLE: we would only see
    # "too few readings" and none of the traceback that explains it.
    tail_lines: list[str] = []
    try:
        while True:
            data = os.read(fd, 4096)
            if not data:
                break
            buf += data.decode("utf-8", "replace")
            parts = re.split(r"[\r\n]", buf)
            buf = parts.pop()                 # keep the incomplete tail
            for chunk in parts:
                if chunk.strip():
                    tail_lines.append(chunk)
            del tail_lines[:-40]
            now = time.perf_counter()
            for chunk in parts:
                m = COUNT.search(chunk)
                if not m:
                    continue
                n = int(m.group(1))
                if not marks or n > marks[-1][1]:
                    marks.append((now, n))
            if marks and marks[-1][1] - marks[0][1] >= steps:
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
    if len(marks) < 5:
        ctx = "\n      ".join(tail_lines[-30:]) or "(no stderr captured)"
        raise RuntimeError(
            f"only {len(marks)} tqdm counter readings captured (rc={proc.returncode}). "
            f"child stderr tail:\n      {ctx}")
    per_step = [(t1 - t0) / (n1 - n0) for (t0, n0), (t1, n1) in zip(marks, marks[1:]) if n1 > n0]
    tail = per_step[int(len(per_step) * (1 - tail_frac)):]   # drop compile/warmup
    return statistics.median(tail) * 1000.0


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
