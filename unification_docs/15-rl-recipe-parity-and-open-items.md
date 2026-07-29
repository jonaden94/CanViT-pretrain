# RL recipe parity + two deferred unifications (2026-07-28/29)

Written after the shared `eval_policy` knob landed (doc 07, "Validation viewpoints").
Two separate things are recorded here: **what still differs** between the validated RL
trainer and the harness path, and **one architectural unification deliberately deferred**.

---

## A. `rl_train.py` vs `harness.run ade20k --preset policy_only` — the recipe gap

**Status: the harness has the CAPABILITY, not the RECIPE.** Do not describe the harness
as reproducing the RL flagship until the table below is closed and a run confirms it.

`ade20k/rl_train.py` is gate-validated on this cluster (jobs 15025279 / 15025337,
mean t1–t4 val CE 0.6855 / 0.6867 vs the qband band 0.6853 ± 0.0007 — see `p3-notes.md`).
The harness path has never been run at scale.

### Already identical (verified by reading both, 2026-07-28)

Candidate grid (`scales=(0.5, 0.25)`, `centers_per_axis=16` → 512 candidates); scorer arch
(`width=128`, `block_layers=3`); `prime_on_policy=0.5`; policy LR 2e-4 / WD 1e-2; grad clip
1.0 with the scorer clipped SEPARATELY from the model; the reward formula
`(prev - cur)/prev.clamp_min(1e-4)` off a full-scene t0 anchor; per-depth `RunningNorm`;
ε-greedy DAgger; BN mode (a).

### CLOSED 2026-07-29 — the harness can now EXPRESS the recipe

| # | knob | `rl_train` | was | fix landed |
|---|---|---|---|---|
| 1 | Adam betas | `(0.9, 0.95)` | every group silently got torch's `(0.9, 0.999)` | **`GroupOptim.betas`** (shared harness type) threaded into `AdamW`; default `(0.9,0.999)` so no existing group moves. Policy default from `JointPolicyConfig.policy_betas=(0.9,0.95)` |
| 2 | LR schedule | ramp over `warmup_frac=0.125`, then **hold** | policy group fell through to `ScheduleSpec()` = `warmup_steps=0` → **no ramp** | `cli.resolve_spec` now builds the policy group from `JointPolicyConfig` (`policy_warmup_frac=0.125`, `warmup_constant`) |
| 3 | train data | **NO augmentation** — `make_val_transforms` on BOTH splits (rl_train.py:349-353) | `make_segmentation_train_transforms` unconditionally | **`Ade20kConfig.augment`** (+ the same-named `In1kConfig.augment`), default `True` |
| 4 | step budget | 8000 = `640_000 // (batch * (1 + train_horizon))` | ade20k default 40000 | plain config value; no code needed |
| 5 | reward resolution | `score_res=128` | probe grid (64) | **STILL OPEN — owner deferred 2026-07-28.** Add `reward_score_res` to `Ade20kConfig`, thread into `BoundAde20kTask.per_image_loss`. Revisit only if a run disagrees with the band |
| 6 | **probe head** | trained probe via `from_pretrained_with_probe` (default `probe-ade20k-40k-s512-c{grid}-in21k`) | `build_model` gated `probe_repo` on `mode=="finetune"`, so a FROZEN (= policy) run built a **fresh RANDOM head** | `probe_repo` now honoured in both modes + `build_policy` WARNS when a policy run has none |

**Gap #6 was the blocker, and it was silent.** The scorer's reward is the fraction of the
PROBE's CE a glimpse removes. `--preset policy_only` runs in `mode="frozen"`, where
`probe_repo` was documented as "Ignored" — so the harness would have trained a policy
against an untrained head, on pure reward noise, with nothing failing and every log
looking healthy. Found 2026-07-29 only by constructing the A/B command end to end rather
than checking the knobs individually. Lesson: knob-by-knob parity is not run-level parity.

Pinned by `harness/tests/test_rl_recipe_knobs.py` (14 tests) against rl_train's own constants.

**Why #3 needed its own flag** (do not "fix" this by tuning the aug knobs): setting
`aug_scale_range=(1.0,1.0)` and `aug_flip_prob=0.0` does NOT disable augmentation.
`make_segmentation_train_transforms` still applies `RandomCropWithLabel` **and** an
unconditional `PhotoMetricDistortion` (colour jitter with no exposed parameter). Identity
knob values give you a *differently* augmented pipeline, not an unaugmented one — exactly
the silent near-miss that would make a harness-vs-`rl_train` comparison unreadable.

**Distill has no config-time run length.** `policy_warmup_frac` resolves against
`cfg.max_steps`, which ade20k and in1k have and **distill does not** — it is
SLURM-array-shaped (`steps_per_job`), so its total is unknown when the spec is built. The
first version of this fix returned 0 there *silently*, i.e. reintroduced gap #2 on the
distill policy path. `cli._policy_warmup_steps` now WARNS in that case, and
`JointPolicyConfig.policy_warmup_steps` (absolute, wins when > 0) is the escape hatch.
Pinned by `test_distill_has_no_config_time_total_so_the_frac_warns_instead_of_vanishing`.

**Unification note (owner asked, 2026-07-29).** #1 and #2 are *shared implementation* —
one `GroupOptim`, one `resolve_spec` — so a policy group on distill/ade20k/in1k gets the
same recipe from a single code path; the test parametrises over tasks to prove it rather
than assume it. #3 is a *shared interface only*: `augment` means the same thing on both
downstream tasks but the bodies differ (dinov3 segmentation transforms vs
`RandomResizedCrop`+flip), so forcing shared code would abstract over things that are not
alike. #5 is genuinely ade20k-only — `score_res` is a segmentation-resolution concept;
in1k's per-image loss is a scalar CE over classes and distill's an MSE over patches.

**Still true after all this:** "can express the recipe" ≠ "reproduces the band". The
harness policy path has still never run at scale. Configuration parity is necessary, not
sufficient — the in-graph rollout and BN mode (a) remain deliberate deviations, and the
harness seeds globally where `rl_train` does not, so the comparison stays statistical
against the band exactly as the P3 gate was.

Horizon mapping, for whoever wires this: `rl_train`'s `train_horizon=4` means t0-full + 4
policy glimpses (the docs' "T=5", band reported over t1–t4) → harness `n_timesteps=5`.

### Gap #7 — RESIZE PROTOCOL (found 2026-07-29 by a run, after this doc had already warned about it)

`rl_train` defaulted to `resize_mode=center_crop` while the band is **squish**. An earlier
revision of this section noted that and was then ignored when exp27 was configured — the
first attempt ran center_crop on both arms and arm A came out at **0.6693**, 0.016 *better*
than the band and ~20× its 0.0007 seed spread. Not a better policy; a different measurement.

Squish is CanViT-PyTorch-RL's *measurement contract*, not a preference: `config.py`'s
docstring ("images and masks squish-resized to `scene_size`"), `CLAUDE.md:30`
("Measurement = the paper's (squish) protocol, **always**"), and a dataset class named
`Ade20kSquish`. At the P3 gate commit `7e5afac`, `rl_train.py:329` hardcoded `"squish"`;
commit `1a0b452` lifted it to a knob defaulting to `center_crop`.

Fix: `PolicyTrainConfig.resize_mode` pinned back to `squish` (it is the frozen reference);
`Ade20kConfig` **deliberately keeps `center_crop`** — aspect-preserving, matches
pretraining, the right default for new work, and the exp24 probe/finetune runs used it.
A policy run under any non-squish mode now logs a not-band-comparable warning.

**Lesson, and it is the same one as gap #6 one day later:** writing the discrepancy into a
doc does not protect a run. Gap #6 was caught by constructing the command end to end; #7
was caught only by *executing* it and disbelieving a good number. A config difference that
moves the metric needs an assertion or a warning at the point of use, not a paragraph.

### Everything else: audited clean against the original repo (2026-07-29)

Read end to end against `canvit_pytorch_rl/{config.py,training/{config,train,eval_loops}.py,data.py}`:
same `model_repo` (`DEFAULT_PRETRAINED_REPO`) and probe rule; the same
`make_val_transforms` function (core's copy is equivalence-tested against specialize's for
**both** modes, `specialize_equivalence.py:133`); full val split (`stride=1, limit=None`);
eval CE at full 512² (`ce_from_logits(...)` with no `score_res` — "full 512^2, sharing the
mIoU logits"); objective = mean CE over t1..t4; and every recipe hyperparameter matching
`TrainConfig`. `rl_train.py`'s only other drift since the gate is additive (per-timestep
mIoU, richer ckpt). Immaterial diffs: eval batch 32 vs 16, workers 4 vs 8 — neither touches
the metric (eval-mode BN, dataset-level mIoU, per-image CE mean).

### Recommended sequence

1. Control: `rl_train` at current HEAD, seed 0 (~65 min, one A100). Re-confirms the gate at
   today's code and produces a LOCAL CE+mIoU reference, so later comparisons are not against
   another machine's numbers. (The 2026-07-23 gate runs predate `845e401`, which added mIoU,
   so they logged CE only.)
2. Close 1–4 above, then run the same recipe through the harness and compare against that
   local control — not against the published band.

---

## A2. OPEN (2026-07-29): the harness's t0 does not match, and t0 is POLICY-INDEPENDENT

**Use t0 as the first check on any policy implementation.** It is the full-scene anchor —
same frozen backbone, same probe, no policy involved — so every implementation must produce
the same number. The reference, measured with `measure_miou_order.py`:

```
t0:  CE 0.7651    mIoU 39.57   (paper Table 4 says 39.6 for c64 — match)
```

The harness reports **t0 mIoU 39.03, CE 0.7886** — off by −0.54 mIoU / +0.024 CE. It is
reproducible locally, identical at step 0 (so not BN drift), and identical under both
`--cfg.eval-policy policy` and `--cfg.eval-policy full` (so not the selector).

This is how the frozen-head BN bug was caught (t0 was 38.50 vs 38.75 across seeds, and a
policy-independent quantity cannot be seed-dependent). Fixing that moved t0 to 39.03 but
did **not** close it, so at least one more difference remains.

### Eliminated, by direct measurement — do not re-check these

| candidate | verdict |
|---|---|
| t0 forward code path (`full_scene_state`+`head_logits` vs `_policy_rollout`+`eval_probe_on_batch`) | **bit-identical**: max\|Δlogits\| = 0.000000, 100% argmax agreement, same canvas |
| the selector's FULL branch | t0 identical via the open-loop `full` generator (0.39030 both) |
| `glimpse_px` None vs 128 | equivalent — `derive_glimpse_px` computes (8−1)·16+16 = 128 for None |
| `NUM_CLASSES` / `IGNORE_LABEL` | both paths import them from `canvit_pretrain.ade20k.data` |
| val loader / transforms | same `make_val_transforms(512,"squish")`, same `ADE20kDataset`, no shuffle/drop_last |
| eval batch size (16 vs 32) | t0 = 39.57 / 39.58 — not batch-size dependent, so head BN is genuinely frozen |
| config | wandb config confirms resize_mode=squish, scene_size 512, canvas_grid 64, n_timesteps 5, augment False, probe loaded ("Initialising head from published probe … mode=frozen") |
| mIoU reduction | pin includes `68b635f`, so both use the paper order |

### Next step

Static reading is exhausted. Dump the harness's t0 logits for a FIXED val batch from inside
`Ade20kRunTask.evaluate` and diff against `measure_miou_order.py`'s t0 logits for the same
batch. Predictions are provably identical on a hand-built batch, so the divergence is in
what reaches the model — bisect the batch tensors (image and mask) before the logits.

---

## B. DEFERRED: unify the eval ROLLOUT (the "layer 2" unification)

The `eval_policy` work unified **which viewpoints** validation takes. It did NOT unify
**how the rollout runs**. Training is already unified (`harness/rollout.py::run_rollout`
drives all three tasks); evaluation still has four separate loops:

| loop | used by |
|---|---|
| `ade20k/rollout.py::rollout_canvas_hidden` | ade20k eval |
| `in1k/rollout.py::rollout_cls_tokens` | in1k eval |
| core `CanViT.forward_reduce` | distill eval |
| `canvit_eval/episode.py:99` | benchmarking (other repo) |

These are the same loop with different readouts, so this is principled to unify. Two things
make it non-trivial, and both are the actual content of this note:

1. **`run_rollout` cannot simply be reused.** It calls `.backward()` and owns BPTT chunking
   and policy-loss accumulation, so it breaks under `no_grad`. Reusing it means splitting it
   into a neutral glimpse-driver + the training concerns — and `run_rollout` carries the
   parity digest `9a0100a1a3de3acd`. Highest-risk refactor in the repo (the digest test is
   the safety net, so it is tractable, but it is not a side quest).
2. **The unified abstraction must be FOLD-based, not list-based.** ade20k/in1k want a list of
   per-timestep readouts, but distill deliberately does not keep one — `ValAccumulator`'s
   docstring records the memory reason ("Metrics computed on full batch -> scalar -> discard
   tensors; PCA viz: sample 0 only -> O(T) not O(B×T)"). A rollout returning per-t tensors
   would reintroduce exactly the O(B×T) cost that comment guards against. `forward_reduce`'s
   `init_fn`/`step_fn` IS the right shape; the list-collecting loops are its degenerate case.

**The concrete blocker is small:** `forward_reduce` takes a viewpoint **list**. If it took a
`next_viewpoint(state, t)` callable, then closed-loop policy eval works for every task with
no extra forward, `rollout_canvas_hidden` and `rollout_cls_tokens` both delete, and distill
keeps its streaming accumulator untouched. But `forward_reduce` lives in **core**
(`canvit_pytorch/model/base/impl.py`), which is the published-model surface and is imported by
CanViT-eval — so this is a cross-repo API change, not a pretrain-local one.

**Visible symptom until then:** distill's policy-deploy eval runs the student backbone rollout
TWICE (select, then replay through the unchanged `forward_reduce`). Teacher forward, IN1k
probe and PCA still run once. ~2,560 extra student glimpse-forwards per validation (256 samples
x 10 glimpses), once per 1000 steps — well under a percent, which is why it was accepted.
ade20k/in1k pay nothing; their eval loops were converted to select-and-step.

**Sequencing (owner, 2026-07-29): do this AFTER the ADE20K policy gate run**, so that a wrong
RL result cannot be confounded between "the port is broken" and "the refactor is broken". The
gate result then becomes the fixed reference to re-verify the refactor against.
