# 14 — Parity coverage: what is actually checked, and what each tool cannot see

Written 2026-07-27, after the exp23 foveated regression showed that the parity apparatus
had two blind spots that together let a real bug reach a 12-hour production run.

## The two blind spots

1. **One configuration.** Every pre-existing check ran the *uniform* patcher on a
   *non-modulated* backbone — `parity_probe.py` builds `create_backbone("vits16")` with
   the default patcher; `harness_realdata_ab.py` hardcodes `is_foveated=False`. The
   foveated path had **zero** same-seed coverage. Uniform passing at scale while foveated
   failed at scale was therefore not bad luck: uniform was the only thing being checked.

2. **Step, not setup.** Those checks compare the per-step rollout *from an already-built
   model state and batch*. The exp23 bug was in **setup** — `init_normalizer_stats_from_tar`
   received `cfg.normalizer_max_samples or 512`, clobbering the documented `0` = "whole
   shard" sentinel, so the two stacks standardized against different statistics. Both
   called the identical function, so every step-level check agreed.

## What now covers them

| tool | question it answers | cost |
|---|---|---|
| `parity_probe.py` | did *this* refactor change the old loop's loss stream? (uniform) | seconds, CPU |
| `parity_configs.py` | does the harness rollout equal `training_step` **per patcher**? | ~6 min, CPU |
| `setup_arg_parity.py` | do both stacks pass shared setup helpers the same **arguments**? | ms, AST only |
| `specialize_equivalence.py` | is the ADE20K port equivalent to `canvit_specialize`? | ~2 min, CPU |
| `capability_matrix.py` | what can each task actually do? (generated, drift-tested) | seconds |

`parity_configs.py` and `setup_arg_parity.py` are complementary and neither alone would
have caught exp23: the rollout math was always correct (it agrees to `0.00e+00` on every
patcher), and the argument was always the bug. `setup_arg_parity` was validated against
commit `24a8500` and does report the historical mismatch.

## Results (2026-07-27)

**Cross-config rollout parity — all exact.** Old `training_step` vs harness `run_rollout`,
same model state, same batch, same RNG:

```
uniform             fov=False  max_reldiff=0.00e+00
uniform+modulated   fov=False  max_reldiff=0.00e+00
foveated            fov=True   max_reldiff=0.00e+00
foveated+film       fov=True   max_reldiff=0.00e+00
foveated+modulated  fov=True   max_reldiff=0.00e+00
square              fov=True   max_reldiff=0.00e+00
square+modulated    fov=True   max_reldiff=0.00e+00
```

Both stacks also derive `is_foveated` identically for `square` — the "square counted as
uniform" bug is gone and now has a regression guard.

**specialize ≡ port (ADE20K probe) — every shared component exact.**

```
SegmentationProbe       same class object (specialize imports core's)
ce_loss                 delta 0.00e+00
mIoUAccumulator         delta 0.00e+00
LR trajectory (40k)     max|dLR| 0.000e+00   (byte-identical builder on both sides)
val transforms          both resize modes, 4 images: 0.00e+00
train augmentation      real pipeline, 3 images: 0.00e+00
uniform viewpoint law   identical centers+scales, 10 glimpses x 2 start modes
```

This **closes the P2 gate's open caveat**. That gate found the port below specialize at
every timestep (-0.0023 -> -0.0068, widening with t) and flagged it as "too consistent to
call pure noise… worth ONE seed-repeat before quoting port numbers in the paper". A
seed-repeat could never have settled it (specialize's probe is unseeded). Component
equivalence does: the gap is seed noise, not a port defect.

Note the P2 gate was like-for-like on resize: at its pin `7e5afac` the port still
hardcoded `"squish"`, exactly as specialize does.

## The one deliberate divergence: val `resize_mode`

`canvit_specialize` **hardcodes** `make_val_transforms(cfg.scene_size, "squish")`. The
port lifted it to a config knob whose default is `center_crop` (commit `1a0b452`, "Lift
val resize_mode into ADE/policy configs, default center_crop").

So a bare-default port run is **not** measuring what specialize measured. Pass
`--cfg.resize-mode squish` to reproduce old numbers. The exp24 launchers already do
(`CFG_RESIZE_MODE=squish`). `specialize_equivalence.py` reports this as a DIFF note
rather than a failure — extending specialize is allowed; extending it *silently* is not.

## What is NOT claimed

- The **foveated** rollout is deliberately NOT identical to specialize: specialize's
  uniform-only viewpoint sampler put every foveated glimpse out of distribution
  (mIoU *decreasing* with glimpses, job 15025338). The port delegates to `RandomSelector`.
  Identity here would mean reproducing a bug.
- `parity_configs.py` uses synthetic targets and identity denorm, so it does not exercise
  the normalizer. That is `setup_arg_parity.py`'s job, by design.
