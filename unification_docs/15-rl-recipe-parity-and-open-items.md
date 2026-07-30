# RL recipe parity + two deferred unifications (2026-07-28/29)

Written after the shared `eval_policy` knob landed (doc 07, "Validation viewpoints").
Two separate things are recorded here: **what still differs** between the validated RL
trainer and the harness path, and **one architectural unification deliberately deferred**.

---

## A. `rl_train.py` vs `harness.run ade20k --preset policy_only` — the recipe gap

**Status (2026-07-30): CLOSED. The harness reproduces the recipe.** Every gap below is
closed, the harness eval is bit-identical to the validated eval (§A2), and BN mode (b) —
the last gap — is the default on both paths (§A3). Historical detail kept below because
several of these were silent failures worth recognising again.

`ade20k/rl_train.py` is gate-validated on this cluster (jobs 15025279 / 15025337,
mean t1–t4 val CE 0.6855 / 0.6867 vs the qband band 0.6853 ± 0.0007 — see `p3-notes.md`).
The harness path has never been run at scale.

### Already identical (verified by reading both, 2026-07-28)

Candidate grid (`scales=(0.5, 0.25)`, `centers_per_axis=16` → 512 candidates); scorer arch
(`width=128`, `block_layers=3`); `prime_on_policy=0.5`; policy LR 2e-4 / WD 1e-2; grad clip
1.0 with the scorer clipped SEPARATELY from the model; the reward formula
`(prev - cur)/prev.clamp_min(1e-4)` off a full-scene t0 anchor; per-depth `RunningNorm`;
ε-greedy DAgger.

**BN mode was NOT identical** and was the last real gap — see §A3. Mode (b) (a separate
eval-mode forward chooses the glimpse) is now the default on both paths.

### CLOSED 2026-07-29 — the harness can now EXPRESS the recipe

| # | knob | `rl_train` | was | fix landed |
|---|---|---|---|---|
| 1 | Adam betas | `(0.9, 0.95)` | every group silently got torch's `(0.9, 0.999)` | **`GroupOptim.betas`** (shared harness type) threaded into `AdamW`; default `(0.9,0.999)` so no existing group moves. Policy default from `JointPolicyConfig.policy_betas=(0.9,0.95)` |
| 2 | LR schedule | ramp over `warmup_frac=0.125`, then **hold** | policy group fell through to `ScheduleSpec()` = `warmup_steps=0` → **no ramp** | `cli.resolve_spec` now builds the policy group from `JointPolicyConfig` (`policy_warmup_frac=0.125`, `warmup_constant`) |
| 3 | train data | **NO augmentation** — `make_val_transforms` on BOTH splits (rl_train.py:349-353) | `make_segmentation_train_transforms` unconditionally | **`Ade20kConfig.augment`** (+ the same-named `In1kConfig.augment`), default `True` |
| 4 | step budget | 8000 = `640_000 // (batch * (1 + train_horizon))` | ade20k default 40000 | plain config value; no code needed |
| 5 | reward resolution | `score_res=128` | probe grid (64) | **CLOSED 2026-07-30.** `Ade20kConfig.reward_score_res=128`, and BOTH entry points now compute the reward through ONE function (`ade20k/metrics.py::reward_ce`); `rl_train.ce_from_logits` delegates to it, pinned bit-identical. Falls back to full res with a warn-once when `score_res` does not divide the mask resolution |
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

**Superseded 2026-07-30.** This paragraph used to warn that "can express the recipe" ≠
"reproduces the band" and that the harness path had never run at scale. Both are now
settled: exp27 ran it (see §A3), and the remaining deliberate deviation — the in-graph
rollout — was shown to cost nothing once glimpse SELECTION was fixed to eval-mode BN.
`pooled_policy_loss` (the other half of the in-graph deviation) remains available and
unvalidated, but is not needed for parity.

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

## A2. RESOLVED 2026-07-30: the harness eval was wrong, and the cause was a corrupted probe

**The harness eval is now bit-identical to the validated eval** — 0.0000 on every metric
(mIoU t0..t4 and mean CE), at matched eval batch size. Runnable check:
`unification_docs/eval_equivalence.py`.

Coverage extended 2026-07-30 to **all three model sources**, because "two harness
checkpoints" left open the possibility that the two evals agreed only on models the harness
itself produced. All three sources share an identical 452-key scorer `state_dict`, so a
published or `rl_train` checkpoint loads into the harness scorer unchanged:

| source | ckpt | delta HARNESS vs VALIDATED |
|---|---|---|
| published HF qband | `qpolicy-…-s0` | **0.0000** on t0..t4 + ce_mean |
| ported trainer (`rl_train`) | bneval s0 `last.pt` | **0.0000** |
| unified harness | `step-8000.policy.pt` | **0.0000** |

Independent cross-check on the ported source: the harness eval reproduces `rl_train`'s OWN
logged per-timestep mIoU on its own checkpoints to <=0.01 (s0 42.86/43.84/44.44/44.89 vs its
logged 42.86/43.85/44.44/44.88; s1 identical to 2dp). Note what this does and does not
establish: we never executed CanViT-PyTorch-RL's eval code. The chain is *our validated eval
reproduces their published per-seed numbers to +0.0002 CE / -0.04 mIoU*, and the harness eval
is bit-identical to that. Agreement-with-published-numbers, not code-level identity.

### Root cause: StateEncoder construction polluted the probe's BatchNorm

`StateEncoder.__init__` builds an image-independent template by running the segmentation
probe on a blank canvas (`init_reference`). `harness/run.py` builds the policy at line 277
but freezes the model at line 280 — so that forward ran with the head in **train mode**, on
a batch of **one** synthetic canvas. Two effects, both measured:

1. It **polluted `head.bn.running_mean` by 1.074**. `apply_requires_grad` then froze BN at
   those corrupted values for the entire run, degrading EVERY timestep including t0
   (39.030 instead of 39.579) — and corrupting the policy REWARD, which is that probe's CE
   reduction.
2. The template itself came out **1.621288** off, and that exact number propagated into
   every `ent_delta`/`cos_init` feature, shifting **14/32** of a trained policy's chosen
   glimpses and costing ~0.1 mIoU at each policy step (0.117/0.110/0.101/0.090 at t1..t4,
   reproducible on two checkpoints).

Fixed in **canvit_pytorch `1f5121b`**: `init_reference` forces eval mode for that forward
and restores the caller's mode. Fixed in core rather than by reordering `run.py` because no
caller should have to know that constructing a feature encoder depends on module mode, and
an init template is by definition a property of the weights. Pinned by
`canvit_pytorch/policy/test_init_reference_mode.py` (4 tests).

`ade20k/rl_train.py` was never affected: it calls `seg.eval()` before constructing its
encoder.

### Eval batch size shifts absolute mIoU by ~0.06 — fix it before comparing

Not a bug, but it invalidates careless comparisons. Same checkpoint, same code:

| eval batch | t0 mIoU | t4 mIoU |
|---|---|---|
| 16 (`measure_miou_order` default) | 39.57 | 44.90 |
| 32 (`Ade20kConfig.eval_batch_size`) | 39.58 | 44.84 |

bf16 kernels differ with batch shape, and near-tied candidate scores then flip some
glimpses. **Quote absolute mIoU only with the eval batch size stated.**

Batch size is the ONLY remaining sensitivity. An earlier revision of this section also
blamed cross-process "bf16 nondeterminism" for a ~0.5 mIoU swing — **that was wrong and is
retracted.** After the `1f5121b` fix, two independent processes running
`plot_policy_curves.py` on identical inputs produced BIT-IDENTICAL output: max|Δ| = 0.0000
across 5 baseline curves and 25 policy-seed values. The swing was entirely this bug (whether
a `StateEncoder` had been constructed in train mode before evaluating). An unexplained
~0.5 mIoU difference is a defect, not arithmetic.

### How this was nearly missed — read before debugging anything similar

Two wrong conclusions were published to this doc en route, both from bad methodology:

1. "Localized to `deploy_rollout_viewpoints`" — from comparing 38.948 and 39.579 measured
   in **separate processes**. In one process they were identical.
2. "There is no bug; the harness eval is fine" — from same-process t0 comparisons that
   agreed. They agreed because **both sides shared the corrupted probe**. t0 is also the
   one timestep that cannot expose a glimpse-SELECTION difference.

**Rule: when two implementations agree with each other but neither matches an external
reference, suspect shared upstream state — not the comparison.** And always compare at
t>=1, not just t0, when the suspect is anything policy-driven.
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

---

## A3. BN mode (b) — the last gap, and what actually closed it (2026-07-29/30)

`rl_train` and the harness both used to CHOOSE the training glimpse with the same
train-mode scorer forward that carries the policy loss ("mode (a)"). The scorer holds one
BatchNorm, so that normalizes on BATCH statistics where CanViT-PyTorch-RL uses running
statistics — it could afford a separate eval-mode forward because it collected the rollout
detached. The modes disagree on **45.7%** of chosen glimpses.

Measured on full ADE20K val at the LAST step (best-checkpoint selection adds noise that
hid the effect at n=2), scored by the validated eval:

| | mean(t1–t4) CE | mIoU t4 |
|---|---|---|
| band, last step (published) | 0.6863 | 44.91 |
| `rl_train` mode (b) | **0.6863** | **44.90** |
| `rl_train` mode (a) | 0.6874 | 44.72 |

Mode (b) is now the default on both paths (`PolicyTrainConfig.select_bn_eval`,
`JointPolicyConfig.select_bn_eval`). The `PolicySelector` primitive keeps mode (a) as ITS
default so non-opting callers stay byte-identical and the `run_rollout` parity digest
`9a0100a1a3de3acd` is untouched — the digest is measured with no policy attached.

It is also the train/deploy-consistent choice: deployment always selects under eval-mode
BN, so mode (a) trained on a state distribution the deployed policy never visits.

### Two methodology traps this cost us, worth reading before the next A/B

1. **The first read of mode (b) was a false NULL.** Comparing at each seed's *best-CE*
   checkpoint, one seed's best fell at step 6000 — a less-trained checkpoint — and the
   selection noise swamped a real +0.18 mIoU. **When the effect is smaller than
   checkpoint-selection variance, compare at a FIXED step.** The band publishes a
   last-step row for exactly this reason.
2. **A 45.7% action-flip rate proved the mechanism was ACTIVE, not that it MATTERED.**
   Mechanism tests are for cheaply killing hypotheses (a 0.2% flip rate would have ended
   it), never for confirming them.

## A4. Baselines available for the Figure-4B axis (2026-07-30)

`entropy_coarse_to_fine` (EG-C2F) is ported from `canvit_eval/policies.py` — the
implementation the published row came from — and **validated to max|Δ| = 0.05** against
paper Table 4 (it is deterministic there, so that is a real check). `coarse_to_fine`
matches to 0.07 (a mean of n=10 in the paper, so within CI). Targets live in
`harness/eval_viewpoints.py::PAPER_TABLE4_C64`.

**`random` is NOT the paper's F-IID.** It measures +0.17…+0.42 above that row, growing
with t: t0 matches (it does start full-scene) but the glimpses follow the safe-box AREA
law rather than F-IID's fixed fovea scale. Do not label it F-IID. F-IID and R-IID are
not reachable from this module today.

Plot everything with `unification_docs/plot_policy_curves.py`, which computes all curves
in ONE process at ONE eval batch size — required, since absolute mIoU shifts ~0.06 with
eval batch size (§A2).

### A4 results — the Figure-4B comparison (2026-07-30, 5 harness-trained seeds)

Full ADE20K val, c64, squish-512, ONE process at eval batch 32. Produced by
`plot_policy_curves.py`; artifacts `band_harness_5seed.{png,json}` in the repo root
(PNG is gitignored, so the numbers live here).

| policy | t0 | t1 | t2 | t3 | t4 |
|---|---|---|---|---|---|
| **Viewpoint-Q trained** (n=5, mean) | 39.58 | **42.58** | **43.85** | **44.41** | **44.77** |
| &nbsp;&nbsp;min…max over seeds | 39.58 | 42.45…42.66 | 43.76…43.94 | 44.31…44.62 | 44.66…45.00 |
| Viewpoint-Q **untrained** | 39.58 | 40.79 | 41.33 | 41.81 | 42.19 |
| EG-C2F | 39.58 | 42.22 | 43.31 | 44.05 | 44.67 |
| C2F | 39.58 | 41.23 | 42.54 | 43.54 | 44.71 |
| Random (safe-box IID, NOT F-IID) | 39.58 | 41.37 | 42.18 | 42.84 | 43.42 |
| *paper Table 4 — EG-C2F* | *39.6* | *42.2* | *43.3* | *44.1* | *44.7* |
| *paper Table 4 — C2F* | *39.6* | *41.3* | *42.5* | *43.6* | *44.7* |

Three things worth reading off it:

1. **The learned policy dominates at every t, by the most EARLY** — +0.36 over EG-C2F at
   t1, +0.54 at t2, converging to +0.10 by t4. Same shape as the reference figure: the
   policy's value is getting to a good canvas *fast*, not a higher ceiling.
2. **t0 = 39.58 for EVERY policy.** That is the check that the full-scene anchor is shared
   and the probe is clean; before the probe-BN fix (canvit_pytorch `1f5121b`) it read 39.03
   and varied per seed.
3. **An UNTRAINED scorer is WORSE than random glimpses** (42.19 vs 43.42 at t4). A
   random-init conv net's argmax picks consistently badly rather than diversely, so the
   trained band is measuring a real learned policy and not "any scorer helps".

## A5. The residual arm gap: the ported trainer beats the harness by a little (2026-07-30)

Earlier text here called the t4 shortfall "about 1σ low, worth a look", and rested the
verdict on *best-checkpoint* CE (0.6859 ± 0.0004 vs 0.6853 ± 0.0007, 4/5 seeds inside the
band). That was too generous a reading. Measured **at the last step, both arms through the
SAME eval in ONE process at eval batch 32** (`unification_docs/compare_arms.py`; all 7
checkpoints, full val):

| arm | n | mean(t1–t4) CE | mIoU t4 | t4 per seed |
|---|---|---|---|---|
| band, last step (published) | 8 | 0.6863 | 44.91 | — |
| `rl_train` mode (b) — ported | 2 | **0.6864 ± 0.0008** | **44.91 ± 0.03** | 44.89, 44.93 |
| harness mode (b) — unified | 5 | 0.6880 ± 0.0010 | 44.77 ± 0.13 | 45.00, 44.76, 44.73, 44.70, 44.66 |

**The ported trainer lands exactly on the band; the harness sits 0.0016 CE above it** — about
2.3× the band's own 0.0007 per-seed spread. Both metrics point the same way, which is what
makes it worth taking seriously: a fluke would not move CE and mIoU together.

**But it is not yet statistically established, and cannot be with these group sizes.** Exact
one-sided permutation test over all C(7,2)=21 splits gives **p = 0.095** for both CE and t4 —
and 2/21 is the *floor*: with n=2 vs n=5, even perfect separation cannot reach p < 0.05. A
**third `rl_train` seed** drops the floor to 1/C(8,3) = 0.018 and makes the question
decidable. That is the single cheapest next measurement (~65 min on one A100).

The gap is **training-side, not measurement-side** — §A2's table has both arms' checkpoints
scoring 0.0000 apart under the two evals, and the numbers above come from one process with
one model instance and one loader.

### A5.1 CAUSE FOUND: the harness's policy gradient was exactly 0.8x the reference's

**`rl_train` and the harness are NOT the same training code** — they are two independent
implementations of the same recipe, so a divergence needs no exotic explanation. Reading
both paths end to end turned one up immediately.

`harness/rollout.py` accumulates each glimpse's QReg loss into `chunk_loss`, and the branch
backward divides that by **`n_glimpses`**. But only **`n_glimpses-1`** glimpses are POLICY
glimpses — t0 is the full-scene anchor and carries no policy loss. The reference instead cats
every depth into one `[horizon*B, A]` tensor and takes a single `F.mse_loss`, i.e. **one mean
over `horizon*B`**. So at horizon 4:

```
loss_harness = (1/5) * sum_{t=1..4} mse_t     vs     loss_rl = (1/4) * sum_{t=1..4} mse_t
```

**The harness's scorer gradient was exactly 0.8x the reference's** — measured, not inferred:
ratio `0.800000` on both loss and gradient norm. That is a **20% smaller effective policy LR**
at the same nominal `policy_lr=2e-4`, so the scorer was systematically **under-trained** at a
fixed 8000-step budget. Under-training is the right *direction* for the observed deficit,
which is what makes it the leading candidate.

Nothing compensated: `rl_weight=1.0`, and `policy_lr` equals `rl_train`'s `lr` exactly.
The **VPG path already compensated for this same division** (`* n_glimpses` in the
deferred-credit branch, with a comment explaining why) — only the inline QReg/PG path did not.

Fixed by rescaling the graph term to `ploss * n_glimpses/(n_glimpses-1)`, leaving
`pol_acc["loss"]` on its raw scale so the logged `policy_loss` series stays comparable to
earlier runs and to `rl_train`'s `train_loss`. Pinned by
`harness/tests/test_policy_loss_scale.py`: the fixed gradient is **bit-identical** (atol=0) to
the reference's, and the unfixed one is pinned at exactly 0.8x so a refactor cannot silently
reintroduce it. The `run_rollout` parity digest `9a0100a1a3de3acd` is untouched (it is measured
with no policy attached). Full suite 339 passed.

**This affects every harness QReg/PG policy run, not just ade20k** — including joint distill
policy runs.

**Not yet confirmed as THE cause.** A mechanism being real does not make it the explanation —
see `mechanism-tests-dont-predict-outcomes`. Arm D (`policy-lossfix-s0.sh`, 5 seeds) tests it.

### A5.1b RESULT (2026-07-30): the difference is REAL; the fix explains at most part of it

All 15 checkpoints re-scored at the last step through ONE eval in ONE process at eval batch
32 (`unification_docs/compare_arms.py`, which also runs the exact permutation tests):

| arm | n | mean(t1–t4) CE | mIoU t4 |
|---|---|---|---|
| band, last step | 8 | 0.6863 | 44.91 |
| **arm C** `rl_train` (ported reference) | 5 | **0.6866 ± 0.0008** | **44.854 ± 0.057** |
| arm B harness (0.8× gradient) | 5 | 0.6880 ± 0.0010 | 44.770 ± 0.133 |
| arm D harness + scale fix | 5 | 0.6871 ± 0.0015 | 44.755 ± 0.136 |

Exact one-sided permutation tests (n=5 vs n=5 → 252 splits, attainable floor 0.0040):

| comparison | ΔCE | p | Δt4 | p |
|---|---|---|---|---|
| arm B vs arm C | +0.0014 | **0.0278** | −0.084 | 0.123 |
| arm D vs arm C | +0.0005 | 0.262 | −0.100 | 0.103 |
| arm D vs arm B (improvement) | −0.0009 | 0.151 | −0.016 | 0.583 |

**1. The harness/reference difference is REAL.** Arm B vs arm C on the band's defining metric:
p = 0.0278. This is what n=2 could not establish (its floor was 0.095) and is the answer to
"are they actually different". They are.

**2. Going to n=5 shrank the apparent gap.** Arm C's first two seeds (0.6870, 0.6859) were on
the lucky side of its own distribution; seeds 2–4 came in 0.6857/0.6870/0.6876, moving arm C
from 0.6864 ± 0.0008 to 0.6866 ± 0.0008 and widening its t4 spread. **Do not read an arm's
level off two seeds** — that is what made the first pass of this comparison look cleaner than
the data supported.

**3. The scale fix moved the mean the right way but is NOT individually significant.** Arm D
cuts the CE gap from +0.0014 to +0.0005 (about two thirds), and arm D vs arm C is no longer
detectable (p = 0.26). But arm D vs arm B is only p = 0.151, and **t4 did not improve at all**
(−0.016, p = 0.58). So: "the harness is no longer distinguishable from the reference on CE" is
supported; "the 0.8× was the cause" is **not**. The defect is proven arithmetic; its outcome
effect is at the edge of what 5 seeds can see.

**4. The t4 mIoU shortfall survives in BOTH harness arms and did not respond to the fix.**
Pooling them (they do not differ on t4, p = 0.58) gives n=10 vs n=5: **−0.092, p = 0.078**.
Consistent in direction, never significant, unmoved by the policy-LR change.

**5. The harness is ~2.4× NOISIER on t4** — sd 0.133 / 0.136 vs arm C's 0.057, across both
arms independently. Variance at n=5 is weak evidence, but it is the more distinctive signal
here than the level, and it points at glimpse SELECTION rather than at the optimizer: the
remaining §A5.2 divergence (fp32 vs bf16 logits into the entropy features) flips near-tied
candidates, which is exactly a variance mechanism. **This, not the loss scale, is where I
would look next.**

### A5.4 RETRACTION + RESOLUTION (2026-07-30): the two training paths are BIT-IDENTICAL

**§A5.2 and §A5.3 below are WRONG and are kept only as a record of the reasoning.** There is
no fp32-vs-bf16 feature divergence. Two things I got wrong:

1. `head_logits`'s docstring says "always in fp32", and it does wrap the head in
   `autocast(enabled=False)` — but `get_spatial` is called on the line *before* that guard, so
   it obeys the caller's amp context. I first concluded from the call sites that this created a
   real precision split. **Measured: it does not change the values.** Passing the task's
   already-computed logits into the encoder instead of letting it recompute them is a
   **bit-identical no-op** (measured before the change was reverted). I implemented that
   "fix", measured it, found it changed nothing, and reverted it.
2. Two diagnostic scripts of mine were themselves buggy and manufactured the divergence they
   appeared to detect:
   - the first trace omitted `rollout_and_loss`'s **train-mode scorer forward**, which is what
     updates `frontend.bn`'s running stats before the eval-mode selection reads them. Path A
     therefore selected on stale stats and path B on fresh ones → a fake 14/16 disagreement
     at t1.
   - `diff_training_step.py` (now deleted) reported a 0.36 relative gradient difference from a
     confounded comparison.

**With a correct trace, the paths agree exactly.** Same batch, same scorer, same encoder,
`prime_on_policy=1.0` (no RNG anywhere), BN and reward standardizers restored between runs:

| depth | CE `rl_train` | CE harness | ΔCE | chosen glimpses agreeing |
|---|---|---|---|---|
| t0 | 0.772283 | 0.772283 | +0.000000 | — |
| t1 | 0.753898 | 0.753898 | **+0.000000** | **16/16** |
| t2 | 0.732488 | 0.732488 | **+0.000000** | **16/16** |
| t3 | 0.729524 | 0.729524 | **+0.000000** | **16/16** |
| t4 | 0.725665 | 0.725665 | **+0.000000** | **16/16** |

`reward_frac` matches to all printed digits (+0.015385, and per-depth). Combined with
`test_policy_loss_scale.py` — which pins the loss composition bit-identical (atol=0) — the
forward, the rewards, the targets and the loss are all identical, so the gradient is too.

**Conclusion: after the §A5.1 scale fix there is no remaining code-level divergence in the
training step.** The arm D vs arm C difference (ΔCE +0.0005, p=0.26; Δt4 −0.100, p=0.10) is
seed/stream noise: the two trainers shuffle their data, initialise the scorer and draw
ε-greedy from independent RNG streams, so identical code still gives different runs.

The one thing still unexplained is arm C's *lower variance* (t4 sd 0.057 vs 0.133/0.136). At
n=5 a variance ratio like that is weak evidence (F(4,4) is extremely wide), so it may be
nothing. It cannot be a code difference in the rollout — that is now excluded.

Runnable: `unification_docs/diff_training_trace.py`.

### A5.3 Why §A5.2 was NOT "fixed" while here *(superseded by §A5.4 — the premise was wrong)*

The obvious-looking fix — wrap the t≥1 `sel.select` in `amp_ctx` so the recomputed logits are
bf16 like the reference's — is **wrong**: it would also put the SCORER forward under autocast,
where `rl_train` runs it in fp32. Reference = bf16 logits + fp32 scorer; harness = fp32 logits
+ fp32 scorer; amp-wrapped = bf16 logits + bf16 scorer. That trades one divergence for
another.

The faithful fix is to thread the logits the task already has into `encoder(state, logits=…)`,
matching `rl_train` and removing the redundant probe-head forward. That needs an extra
argument on the `Selector.select` protocol (and so touches `RandomSelector` /
`MixtureSelector`), which is a design change rather than a bug fix — left for the owner rather
than guessed at, especially to chase a p = 0.078 effect.

### A5.2 Two further divergences found, NOT yet fixed *(RETRACTED — see §A5.4: both are no-ops)*

Both concern `PolicySelector.select`, and both are numerical rather than systematic, so they
would act like a seed change rather than a bias:

1. **The selector runs OUTSIDE `amp_ctx` for t>=1.** `run_rollout` wraps the t0 select in
   `with amp_ctx:` (rollout.py:258) but calls the t>=1 select *before* entering the context
   (rollout.py:285-287). So every policy glimpse's encoder + scorer forward runs in **fp32**.
2. **The encoder recomputes the probe logits.** `rl_train` passes the bf16 logits it already
   computed for the CE reward (`encoder(st, logits=logits)`); the harness calls
   `encoder(state)`, and `StateEncoder.__call__` recomputes `head_logits` itself when
   `logits is None`. So the harness's entropy features derive from **fp32** logits where the
   reference's derive from **bf16** ones — and it pays an extra probe-head forward per glimpse
   for the privilege.

Fixing (2) by threading the reward logits through would both match the reference and remove
the redundant forward. Deferred until arm D reports, so that one change is under test at a
time.

Also worth keeping straight: **best-ckpt and last-step tell different stories here.** The
harness is inside the band on best-ckpt CE and outside it on last-step CE. Best-ckpt
selection over 8 evals is a max over noise, so it flatters whichever arm is noisier — and the
harness is (sd 0.133 vs 0.031 on t4). Quote last-step for arm comparisons.

Two earlier observations that still hold: t0 = 39.58 for every policy (the clean-probe
check), and an untrained scorer is worse than random glimpses (42.19 vs 43.42 at t4).
