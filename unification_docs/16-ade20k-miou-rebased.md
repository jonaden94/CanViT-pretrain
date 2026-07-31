# ADE20K mIoU was RE-BASED on 2026-07-29 — older numbers are not comparable

**If you are comparing an ADE20K mIoU produced before commit `<this>` with one produced
after, stop. They are measured differently.** Everything before is biased LOW by roughly
0.15–0.20 mIoU. CE is unaffected and stays comparable across the change.

## What changed

The order of `argmax` and upsampling in the metric path.

```python
# BEFORE — argmax at the 64x64 probe grid, then NEAREST-upsample the label map
preds = upsample_preds(logits.argmax(1), 512, 512)

# AFTER — BILINEAR-upsample the logits, then argmax at full resolution
preds = preds_from_logits(logits, 512, 512)
```

64 → 512 is 8×, so taking argmax first locks every label into an 8×8-pixel block and
boundaries can only fall on the coarse grid. Upsampling logits first lets a boundary land
anywhere. The old order is strictly coarser, so it is strictly worse — this was a bug, not
a defensible alternative convention.

## Why it is a bug and not a preference

`canvit_train` was the **only** repo in the stack doing it the old way:

| code | order |
|---|---|
| `canvit_eval/tasks/ade20k_seg.py:92` | bilinear-upsample logits → argmax |
| `canvit_pytorch_rl/scoring.py:44` ("Paper-protocol scoring") | bilinear-upsample logits → argmax |
| `canvit_train/ade20k/metrics.py:35` (before this) | **argmax → nearest-upsample** |

Every published ADE20K number — the paper's Table 4, the qband band, the EG-C2F baselines —
is measured the reference way. So the old path could not be compared to any of them.

## Measured cost

Deploy eval on the **full** ADE20K val split (2000 images), squish protocol, both reductions
computed from the **same** rollout and the **same** logits in one pass, on the two exp27 arm A
checkpoints (script: `scratchpad/measure_miou_order.py`):

| t | s0 old | s0 new | s1 old | s1 new | delta | band |
|---|---:|---:|---:|---:|---:|---:|
| t1 | 42.46 | 42.61 | 42.42 | 42.57 | +0.15 | 42.65 ±.16 |
| t2 | 43.69 | 43.86 | 43.65 | 43.81 | +0.17 | 43.95 ±.16 |
| t3 | 44.29 | 44.47 | 44.17 | 44.35 | +0.18 | 44.62 ±.12 |
| t4 | 44.57 | 44.76 | 44.62 | 44.82 | +0.19 | 44.97 ±.10 |

Reproducible to 0.01 across seeds. The gap grows with `t` because the segmentation sharpens
and the coarse grid costs more as boundaries get finer.

**CE is untouched** — it takes its own bilinear path (`ce_from_logits`), which is why the
symptom was mIoU-only. Measured mean t1–t4 CE 0.6852 (s0) / 0.6862 (s1) vs the band's
0.6853 ± 0.0007, matching the runs' own logged values to bf16 noise. That control is what
proves the measurement isolates the reduction.

## What it does and does not explain

Fixing the order closes **about half** the mIoU gap to the band, not all of it. At t4:
44.60 → 44.79 against a band of 44.97 ± 0.10, i.e. from ~3.7σ low to ~1.8σ low.

The residual ~0.18 is small and unexplained. Plausible sources, none investigated: 2 seeds
vs the band's 8; the two known deliberate deviations (in-graph rollout, BN mode (a),
`p3-notes.md`); bf16. **Do not describe the order fix as closing the gap.**

## What is now stale

Everything below reports ADE20K mIoU under the old reduction. The numbers are internally
consistent with each other and biased low against anything measured after:

- all exp24 ADE20K probe and finetune runs
- exp27 arm A jobs 15097197 / 15097198 (their **CE is valid** — 0.6851 / 0.6865, both in
  band — only mIoU is stale; recompute from `best.pt` with the script rather than re-running)
- exp27 arm B jobs 15097199 / 15097200 if they complete on the old pin
- any earlier ADE20K mIoU in `logs/`, wandb, or the notebooks

## Call sites

Fixed (metric paths): `ade20k/metrics.py::eval_probe_on_batch` (the probe/harness path),
`ade20k/train.py:228` (train-split IoU), `ade20k/rl_train.py:298` (policy deploy eval).
The harness ade20k task goes through `eval_probe_on_batch`, so it is fixed by that.

**Not** fixed, deliberately: `ade20k/viz.py:114`. It renders an already-argmaxed label map
for a figure and has no logits to upsample; nearest is correct for display. `upsample_preds`
stays for that use and its docstring now says so.

Pinned by `ade20k/test_metrics_order.py` (4 tests), including a bit-exact check against
`canvit_eval`'s expression and a regression guard that fails if the old order returns.

## Lesson

Third instance in two days of the same failure, after doc 15 gaps #6 and #7: a difference
that moves a metric survived because nothing ever compared our number to an external
reference. It was found only by putting our mIoU next to a published band and refusing to
accept a mismatch as noise. See `[[documented-drift-still-ships]]`.
