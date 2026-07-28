# 15 — Why the harness ran ~10% slower than the old loop, and what closed it

Investigated 2026-07-28, after the retrospective log analysis showed the unified harness
consistently slower than `train/loop.py` on identical configs.

## The observation

Per 8192-step array task, hardware controlled (GPU model read from each task's own log):

| pair | 40GB | 80GB |
|---|---|---|
| exp23 uniform16-ti | OLD 4002s → HARNESS 4766s (+19.1%) | 3607s → 3970s (+10.1%) |
| exp23 fovi-ti | 5814s → 6413s (+10.3%) | 5416s → 6051s (+11.7%) |
| exp26 fovi-ti | 5763s → 5973s (+3.6%) | 5398s → 5944s (+10.1%) |

The gap is a per-step **rate**, not an accumulation: present in array task 0 and flat
after. The old loop's per-task time is remarkably stable (cv 1.0% / 0.5%); the harness's
is not (cv 5.9% / 2.7%). That variance asymmetry turned out to be the clue.

## The cause

`tasks/distill/task.py::bind` transferred the teacher targets as

```python
raw_patches.to(device, dtype=torch.float32)          # no non_blocking
```

where `train/loop.py:655-656` always passed `non_blocking=cfg.non_blocking_transfer`.
Measured on the real payload — `[64, 1024, 768]` fp16, 96 MiB, pinned (the loader sets
`pin_memory=True`), with GPU work already queued:

```
.to(dev, float32)                  191.13 ms CPU stall   191.23 ms wall
.to(dev, float32, non_blocking)      0.23 ms              32.71 ms
.to(dev, non_blocking).float()       0.19 ms              32.71 ms    (== exactly)
.to(dev, non_blocking)               0.18 ms              32.13 ms
```

Without `non_blocking`, PyTorch performs the fp16→fp32 cast **on the host**, and the
unpinned intermediate cannot overlap either. The harness also ignored
`cfg.non_blocking_transfer`, a documented ablation flag, entirely.

This explains both secondary observations. The stall is CPU-side, so it costs wall time
only to the extent the GPU has run out of queued work — hence an end-to-end cost
(~40 ms/step) far below 191 ms. And how much is hidden depends on queue depth, which
varies with the **stochastic rollout length** (`continue_prob=0.5` → 2–8 glimpses) —
hence the harness's much larger step-time variance.

## The fix, verified in two hardware regimes

Three arms, interleaved, same device, same method (post-warmup median step time):

| | old loop | harness FIXED | harness PRE-FIX |
|---|---|---|---|
| MIG 3g.40gb (42 SMs) | 478.9 ms | 474.8 ms (**−0.9%**) | 678.2 ms (+41.6%) |
| full A100 (108 SMs) | 249.4 ms | 248.6 ms (**−0.3%**) | 288.8 ms (+15.8%) |

**The fixed harness matches the old loop within run-to-run noise**, and the pre-fix
penalty on production hardware (+15.8%) sits inside the +10.3…+19.1% band the production
logs showed. Nothing measurable remains in the per-glimpse metric hooks or the engine's
Python overhead — those hypotheses are now excluded, not merely untested.

## Measuring this again: the traps

Getting a trustworthy number took seven iterations, **all** in the measurement scaffolding
rather than the code under test. Written down so the next person spends none of that:

1. **`grete:shared` shares the NODE.** A co-tenant contends for CPU and PCIe bandwidth —
   exactly what a host-side transfer stall is sensitive to. Job 15091403 came back
   perfectly bimodal, every arm either ~250 ms or ~465 ms. Aggregate statistics across
   that are meaningless (its own summary reported "+7.8% fixed vs old loop"); splitting at
   the gap gives arms tight to <1% and the result above. Always check for bimodality
   before averaging, and prefer the quiet mode.
2. **A short `steps_per_job` starves the loader.** It sets `shards_per_gpu`, which caps
   `num_workers`: 64 → 1 worker → `data_pct` 26% vs ~2% in production, swamping the
   effect. Keep the production value and stop early instead.
3. **tqdm redraws with `\r` and emits no newline**, so `for line in proc.stderr` yields
   nothing while the run proceeds. Read raw chunks, split on `[\r\n]`.
4. **The rate unit is whatever tqdm was built with** — `train/loop.py:682` passes
   `unit="step"`, so it prints `step/s`, not `it/s`.
5. **tqdm's printed rate is a smoothed EMA** (`smoothing=0.3`) dominated by
   `torch.compile` warmup; it claimed 1220 ms/step for a loop whose steady state is
   479 ms. Timestamp the step counter yourself.
6. **Anchor the counter to `Training:`.** The teacher weight load renders its own tqdm
   (`Loading weights: 211/211`), which an unanchored regex reads as 211 elapsed steps —
   killing the run seconds in, before training starts.
7. **Do not consume stderr and discard it.** A crash in the child then shows up only as
   "too few readings". Keep a rolling tail and report it with the return code; that is
   what finally exposed #6.

Plus two in the sbatch: `#SBATCH` directives after any executable line are silently
ignored, and under `set -euo pipefail` a `grep` with no match kills the job **before** the
branch meant to report which arm failed.

Tools: `unification_docs/throughput_ab.py` (harness, in-process, times the per-step log
record `run.py:358` already emits), `throughput_oldloop.py` (old loop, subprocess, times
the tqdm counter), `slurm_nhr/runs/perf/throughput_matrix.sh` + `submit.sh` (interleaved,
commit-pinned — note it pins the invoked SCRIPTS too, not just the importable package).
