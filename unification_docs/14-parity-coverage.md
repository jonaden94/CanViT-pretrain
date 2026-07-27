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

## Round 2 (2026-07-27, later): broadening setup-arg parity found two real regressions

`setup_arg_parity.py` initially covered 2 helpers on 1 stack pair. Broadened to **9
helpers across all 3 pairs** (`train/loop.py` <-> distill task + run.py;
`ade20k/train.py` <-> ade20k task; `in1k/train.py` <-> in1k task). That immediately
surfaced two genuine divergences of the same shape — *the harness reads a config/student
value where the old loop reads the real TEACHER*:

**(1) `teacher_dim` was hardwired to 768.** `train/config.py:113` documents the field as a
PLACEHOLDER — "overridden by create_model based on actual teacher" — and `train/loop.py:294`
duly passes `create_model(backbone, teacher.embed_dim, cfg)`, which `create_model:63`
assigns back onto the config. The harness passed `cfg.model.teacher_dim`, making that a
**self-assignment no-op**. Correct by coincidence for dinov3-vitb16 (768); a vitl16 teacher
is **1024**, and 5 exp21 launchers use exactly that. All 5 are still on the old loop, so no
harness run was corrupted — it would have fired on the first ViT-L harness run, which is
precisely what the unification is meant to enable. Fixed: the width now comes from the
teacher's HF config (no full model load), except on resume/HF-seed where the checkpoint's
config is authoritative. Verified: vitb16 -> 768, vitl16 -> 1024.

**(2) `scene_size_px` used the STUDENT's patch size, in two places.** The scene must
tokenize into G x G *teacher* patches — the teacher produces the targets — and
`train/loop.py:307-308` sizes it from `teacher.model.config.patch_size`. The harness used
`model.backbone.patch_size_px` in both the raw-shard path and, separately, in `evaluate()`
(the val scene, i.e. what `val/scene_cos_norm_t*` is computed at). Identical while student
and teacher are both /16 — as in every config to date — so no run was affected. Fixed in
both places.

**(3) `compile_teacher` was never called by the harness.** Confirmed from the exp23 logs:
old loop `"Compiling teacher and model"`, harness `"Compiling model (torch.compile)"`. So
every harness run drove an EAGER teacher for validation targets while the old loop drove a
compiled one. **Measured before fixing** (`teacher_compile_delta.py`, A100): eager vs
compiled teacher features agree to `1-cos = 1.19e-07`. So this could NOT have explained any
observed metric gap — hypothesis raised and empirically rejected. Wired up anyway, for
speed and to remove a gratuitous asymmetry.

Documented non-divergences (encoded as `KNOWN_ABSENT`, so they are decisions rather than
oversights): ade20k's `make_optimizer_and_scheduler` is absent by design (the harness builds
optimizers from `TrainSpec`; numerical equivalence to specialize's `WarmupOneCycleLR` is
proven by `test_onecycle_matches_ade20k_reference_scheduler`), and in1k's
`from_pretrained_with_new_head` only *looks* absent because both stacks go through the
shared `in1k.train.build_classifier`, which lives in `train.py`.

## Settled: the uniform harness-vs-oldloop gap is noise (owner, 2026-07-27)

exp23 uniform showed the harness below the old loop at 15/18 eval points past 32k (mean
-0.0021), largest early (-0.024 at 16k) and decaying. A seed-1 old-loop noise floor was
built to test it; the owner ruled the difference noise and the trainings equally good, so
it was **not submitted** and the launcher was deleted. Do not re-open this.

## What is NOT claimed

- The **foveated** rollout is deliberately NOT identical to specialize: specialize's
  uniform-only viewpoint sampler put every foveated glimpse out of distribution
  (mIoU *decreasing* with glimpses, job 15025338). The port delegates to `RandomSelector`.
  Identity here would mean reproducing a bug.
- `parity_configs.py` uses synthetic targets and identity denorm, so it does not exercise
  the normalizer. That is `setup_arg_parity.py`'s job, by design.
