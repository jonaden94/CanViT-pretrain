# 10 — LR-schedule reproduction + a harness SLURM launcher

Follow-on to `09-cli-and-checkpoint.md`. Closes two of the three items that doc left
open. The third (the ADE20K numeric gate) is recorded here too, since it is what
finally tests the wiring end to end.

Status: **framework completion, not cutover.** Nothing existing was deleted or
repointed; `CanViT-specialize` / `CanViT-PyTorch-RL` and every old launcher still work.

---

## 1. The LR schedules were NOT reproduced (two gaps, not one)

Doc 09 listed "ade20k `warmup_onecycle` raises `NotImplementedError`" as a known hole.
Reading the two standalone recipes end to end turned up a **second, unlisted** one: in1k
had the same class of bug and nobody had noticed, because a wrong-but-running schedule
is silent.

| task | what the standalone actually does | what the harness `default_spec` did | verdict |
|---|---|---|---|
| ade20k | `AdamW` + `WarmupOneCycleLR` (`ade20k/data.py:66`) | `warmup_constant` | **wrong shape** |
| in1k | `AdamW` + `warmup_cosine_scheduler` (`in1k/train.py:131`) | `warmup_constant`, `warmup_steps=0` | **wrong shape + no warmup** |
| distill | warmup → constant | `warmup_constant` | correct |

Neither would have crashed. Both would have quietly trained a different recipe than the
one the published numbers came from — exactly the failure mode the owner's "old stuff
should be 100% reproducible" requirement is about.

### 1.1 `warmup_onecycle`

Ported from `dinov3.eval.segmentation.schedulers.WarmupOneCycleLR` into
`harness/optim.py::_onecycle_factor`, under the parametrization ADE20K actually fixes
(`anneal_strategy='cos'`, `final_div_factor=inf` ⇒ `min_lr=0`, momentum untouched).

Two quirks of the original are reproduced deliberately rather than "cleaned up":

* the warmup ramp is scaled by an extra `(1 - step/total_steps)` decay;
* the cosine anneal's progress is `(step+1)/total_steps` measured **from step 0**, not
  from the end of warmup — so the LR never actually returns to the peak after warmup.

A tidier schedule would be a different experiment. The point is bit-reproduction.

### 1.2 in1k `warmup_cosine`

The harness's existing `warmup_cosine` already had the right shape; in1k just wasn't
selecting it. `In1kRunTask` now takes `total_steps` (passed down by `In1kCmd.build()`
from `--opts.n-steps`, which in1k already requires) and derives
`warmup = warmup_epochs/epochs * total_steps` — algebraically the same number
`in1k/train.py` gets from `warmup_epochs * batches_per_epoch`. Constructed without a
runner, `In1kRunTask` has no step budget and holds at peak rather than inventing a decay
horizon.

### 1.3 How this is verified

Applying the lesson from `08`/`09` (a fixture I write myself tests my *belief* about the
format, not the format): the tests do not hard-code expected LR values. They instantiate
**the real schedulers the standalone entry points build** and compare step for step.

* `harness/tests/test_optim.py` — the primitives, 400 steps, `rel_tol=1e-9`, over three
  warmup regimes (`warmup_ratio` truthy / zero / no warmup).
* `tasks/tests/test_run_wrappers.py` — the *wiring*: that each task's `default_spec`
  selects and parameterizes the right primitive.

Both pass. A hard-coded-expectation test would have passed against the broken version too.

---

## 1b. Best-checkpoint selection was missing entirely

Found while auditing what else the standalones do that the harness didn't. **Both**
task standalones keep the best checkpoint, not just the last one:

* `ade20k/train.py:177-182` — `probe.update_best(mious)`, saves on improvement, and logs
  a `best_val_miou_t*` series;
* `in1k/train.py:210-214` — `best_top1`, saves `best.pt` + a `best-hf/` export.

The harness had **no best tracking at all** (`grep best canvit_train/harness/` was
empty), so a 40 000-step probe would have published its *last* head rather than its best.
That is a silent quality regression, not a crash.

Added to `harness/loop.py`: a task sets `best_metric` (the eval key to MAXIMIZE) and the
loop writes `ckpt_dir/best.pt` whenever it improves; `run.py` passes an `on_best` callback
that mirrors the running max to the tracker. `Ade20kRunTask.best_metric = "miou_final"`,
`In1kRunTask.best_metric = "top1"`. Distill sets none — `train/loop.py` has no
best-checkpoint selection either, so that stays faithful.

Two bugs caught while writing it, both by asking "what does this touch that already
exists?" rather than by the tests:

1. **DDP deadlock.** My first version tracked best in `run.py`'s `on_eval`; moving it to
   the loop meant `_track_best` runs on *every* rank, but validation is rank-0-only
   (`on_eval` returns `{}` elsewhere). Reusing `_save`, which barriers, would have hung
   every multi-GPU run. Best-tracking now writes directly with no barrier.
2. **`latest.pt` corruption.** `save_checkpoint(update_latest=True)` is the default, so
   writing `best.pt` would have repointed `latest.pt` at it — and the next array task
   would have silently resumed from the *best* step instead of the newest, quietly
   replaying training. Fixed with `update_latest=False`; `test_loop_e2e.py` now asserts
   `find_latest()` still resolves to `step-4.pt` after a best write.

## 2. `slurm_nhr/harness_train.sbatch`

One launcher for all three tasks (`TASK=distill|ade20k|in1k`) — the operational payoff of
having one entry point. Modeled directly on `base_train.sbatch`: same commit pinning
(offline `git archive` → `$TMPDIR/canvit_src` → `PYTHONPATH` + `PYTHONSAFEPATH=1`), same
`/local`-full TMPDIR fallback, same per-job Inductor/Triton/CUDA caches, same DDP env.

Env → flag mapping:

    CFG_FOO_BAR=value  →  --cfg.foo-bar value      (task config)
    OPT_FOO_BAR=value  →  --opts.foo-bar value     (harness knobs, e.g. OPT_N_STEPS)
    EXTRA_ARGS         →  verbatim

`EXTRA_ARGS` remains the only route to **nested** trees (`--cfg.model.patcher-name
foveated`) and to boolean flags, because `FOO_BAR → foo-bar` cannot encode a dot. That
limitation is inherited from `base_train.sbatch`, not new.

`submit.sh` gained one line — `SBATCH_SCRIPT=${SBATCH_SCRIPT:-slurm_nhr/base_train.sbatch}`
— so the same config can be sent through either path. Default behaviour is unchanged.

One real bug was caught while writing it: ade20k/in1k must **not** receive the
`--cfg.run-group`/`--cfg.logs-dir`/data flags that only distill's config defines, so the
per-task flag set is a `case`. (I also converted three `[ -n "$X" ] && arr+=(...)` lines
to if-statements believing `set -e` would kill the job on a false test — testing it
showed bash exempts the left operand of `&&`, so that was a non-bug. The if-form is kept
because it is also safe in last-statement position, but it fixed nothing.)

---

## 3. ADE20K numeric gate (job 15041762)

The standalone probe (`ade20k/train.py`) has **no seeding at all** — no `cfg.seed`, no
`torch.manual_seed`. Probe init, augmentation and viewpoint sampling all differ run to
run, so bit-parity against it is impossible by construction and a single
harness-vs-standalone number would be unfalsifiable: any gap reads as either a port bug
or ordinary noise.

So the gate runs four arms in one array under an identical environment:

| arm | what |
|---|---|
| 0,1,2 | standalone replicates — their spread is the **measured** noise floor |
| 3 | harness (`python -m canvit_train.harness.run ade20k`) |

PASS = the harness curve sits inside the standalone band at every eval point. 6000 steps
(which also sets the LR horizon, so every arm anneals identically), val every 1000 ⇒ six
comparison points.

All arms share the same eval primitives (`ade20k.metrics.eval_probe_on_batch`,
`mIoUAccumulator`), so what this gates is the **training** path: spec → optimizer →
schedule wiring, rollout, loss, eval cadence.

**A third bug, found only by reading the log rather than the final number.** The harness
arm of the 6000-step run (15042059_3) reported 0.3592 at step 2000 against a standalone
band of [0.3908, 0.3957] — apparently a fail. It was not a quality regression: its log
*starts at step 2000*, because `RunSettings.resume=True` made it resume from the previous
gate's `latest.pt` in the same `--probe-ckpt-dir`. The standalone has no resume and cannot
show this, so the two arms were not running the same experiment at all. The gate now wipes
each arm's OWN dir (never the shared root — concurrent array tasks would race) and takes a
`GATE_OUT` override so a single arm can be re-run without disturbing arms still writing.
Symptom worth remembering: **the harness arm's first eval is not at step 0.**

A first pass (job 15041762 → 15041857) ran 3 arms × 2000 steps and produced exactly one
usable comparison point, because both loops evaluate at `step % val_every == 0` over
`range(0, max_steps)` and so never evaluate at `max_steps` itself. At that point the two
standalone reps bracketed `t9 ∈ [0.2740, 0.2885]` and the harness scored **0.2873**
(and `t0`: standalone `[0.2441, 0.2600]`, harness **0.2597**) — inside the band on both
metrics. Since each arm only cost ~7 min, the gate was re-run wider rather than reported
off a single point with n=2.

Result: see §4.

---

## 4. Results — PASS

**The harness ade20k probe is deterministically seeded; the standalone is not.** The
three same-seed harness replicates (15042308 / 15042639 / 15042640) came back
**byte-identical** — `RunSettings.seed=0` → `torch.manual_seed(0)` at run.py:173, and
`Ade20kCmd` had no way to change it. So a same-seed rerun measures nothing about spread.
To get the harness's own run-to-run band I added a seed override (`--opts.seed`, §4a) and
re-ran at seeds 1 and 2 (15042927 / 15042928).

`miou_final` (last-timestep val mIoU), n=3 each:

| step | standalone (unseeded) min/mean/max | harness (seeds 0/1/2) min/mean/max |
|---|---|---|
| 1000 | 0.3193 / 0.3233 / 0.3302 | 0.3099 / 0.3221 / 0.3332 |
| 2000 | 0.3908 / 0.3941 / 0.3957 | 0.3871 / 0.3928 / 0.3977 |
| 3000 | 0.4011 / 0.4049 / 0.4112 | 0.4065 / 0.4093 / 0.4120 |
| 4000 | 0.4101 / 0.4156 / 0.4203 | 0.4078 / 0.4113 / 0.4134 |
| 5000 | 0.4154 / 0.4186 / 0.4208 | 0.4090 / 0.4157 / 0.4197 |

The two ranges overlap at every step; the mean gap is ≤0.0044 mIoU, alternates sign
(harness higher at 3000, lower elsewhere), and is smaller than each group's own spread
(~0.005–0.010). No systematic, practically-significant direction.

**This is what "reproduces within noise" looks like when one side is unseeded and the
other is deterministic.** Note the earlier false alarm: seed 0 alone (0.4090 at step
5000) reads as a fail against the standalone band [0.4154, 0.4208]; it is just the low
end of the harness's own distribution, whose other two seeds (0.4185, 0.4197) sit inside
the standalone band. A single deterministic sample vs an unseeded band is not a valid
comparison — the whole reason for the seed knob.

The full fidelity audit behind this (each axis checked against the standalone, all
faithful): matmul precision (`high` both), backbone frozen+`eval()`, head dropout off in
validation (`was_training` guard), the LR schedule (§1, unit-verified step-for-step), amp
dtype, the eval viewpoint law (same `make_random_viewpoints`), the **training** viewpoint
law (`RandomSelector` vs `random_viewpoints` — sampled 40 960 draws each, identical
scale/center distributions), and the per-timestep loss aggregation
(`chunk_loss / n_glimpses` = mean over T, matching `torch.stack(losses).mean()`).

### 4a. `--opts.seed`

`HarnessOpts.seed` (None => the task config's seed; distill/in1k already had one, ade20k
did not, so None => 0 there). Threaded into `RunSettings.seed` in all three `build()`
methods with the override taking precedence. Being locked to seed 0 with no override was
a real limitation independent of this gate; covered by `test_opts_seed_overrides_task_seed`.

---

## 5. `NameError` on the foveated ADE20K path

The ADE20K task's foveated OOD warning (ported from `ade20k/train.py:97`) called
`log.warning(...)` in a module that never defined `log`. It sits behind
`if consumes_full_image(seg)`, so **uniform** runs skip it entirely — which is every GPU
smoke and every gate arm run so far. The first foveated ADE20K probe would have died with
`NameError: name 'log' is not defined` inside `build_model`.

Fixed (`import logging` + a module logger) and covered by
`test_ade20k_build_model_warns_on_foveated_view_scale`, which stubs the HF load and forces
`consumes_full_image` True so the branch actually executes. `ruff --select F821` over
`tasks/` + `harness/` is now clean.

Worth generalizing: **branches that only the foveated path takes are invisible to a
uniform smoke test.** The same asymmetry produced the earlier `is_foveated`
"square-as-uniform" bug and the view-scale footgun.

---

## Still open

* **A behavioural difference, needing an owner decision, not a fix:** the harness
  **resumes by default** (`RunSettings.resume=True`); the standalone probes cannot resume
  at all. So `python -m canvit_train.ade20k` twice = two independent runs, while
  `harness.run ade20k` twice into the same `--cfg.probe-ckpt-dir` = the second *continues*
  the first. That is the right default for distill's 245-job arrays and a footgun for the
  single-job probe tasks. Options: leave it (documented), or default `resume` per task
  (distill True, ade20k/in1k False). **Not changed unilaterally** — it is a semantic
  choice about how the unified repo should behave, not a port bug.
* The big-bang cutover (deleting the old loop / repointing the production launchers) —
  owner-gated, untouched.
* `harness/task.py` is a drifted duplicate protocol imported nowhere; noted, not deleted.
* Pre-existing lint left alone: `run.py:160` UP028, and `I001` import order in
  `tasks/{ade20k,distill,in1k}/task.py`.
* Everything here is uncommitted, per the standing guardrail.
