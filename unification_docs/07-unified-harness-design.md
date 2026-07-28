# 07 — Unified training harness: one harness, three peer tasks, orthogonal grad control

**Status:** DESIGN APPROVED (owner, 2026-07-23) — IMPLEMENTATION IN PROGRESS. This doc
completes master-plan §4.2 ("one loop, task + selector + grad-regime injected") and
*corrects* it: the harness must be **task-neutral**, with distill / ADE20K / IN1k as
three **equal** tasks built on top of shared training machinery — not "pretrain's
`loop.py`+`step.py` substrate that the other two borrow." All GPU work (parity re-gate,
real multi-GPU, numeric acceptance) is deferred to the very end; everything here is
implementable and CPU-validatable now.

## LOCKED DECISIONS (owner-approved 2026-07-23) — do not relitigate

- **D-A** neutral `harness/` + peer `tasks/{distill,ade20k,in1k}/`; distill demoted to a peer.
- **D-B** §3.1 Task interface (implementer's call); **composed** config (shared
  HarnessConfig/TrainSpec/EpisodeConfig + per-task config), NOT one mega-dataclass.
- **D-C** `none/full/chunked` rollout; ADE20K/IN1k use fixed horizon (stochastic
  `continue_prob` stays distill-only).
- **D-D** branch machinery **generalized to all tasks**; defaults collapse to today's
  behavior (distill N-full+N-random; ADE20K/IN1k single branch).
- **D-E** **full per-group** optimizer/scheduler spec (backbone/head/policy each own
  lr/wd/schedule; small schedule registry).
- **D-F** reward = fractional per-image **task-loss** reduction (default); `task_metric`
  opt-in.
- **D-G** persist **all locally** (backbone+head+policy sidecar + `train_spec` +
  `pretrain_view_scale` metadata). **NEVER auto-push to HF** — training writes local
  files only, exactly as CanViT-pretrain/specialize do today. HF publishing stays a
  separate **manual** `python -m canvit_pretrain.checkpoint.to_hf` step.
- **Loop model** — **step-based only** (owner approved). The harness has ONE step-driven
  loop (owns SLURM-array `steps_per_job` resume + commit-pinning). IN1k's "epochs" are
  derived: its WebDataset `with_epoch(N)` fixes `steps_per_epoch`, so epoch boundaries
  become periodic LR/eval hooks and cosine-over-epochs maps to cosine-over-steps. No
  second (epoch) loop shape.
- **Migration** — **big-bang**: ONE new unified entry point
  `python -m canvit_pretrain.train --task {distill,ade20k,in1k}`; the three old entry
  modules (`ade20k.train`, `ade20k.rl_train`, `in1k.train`, the distill loop) and their
  SLURM launchers are rewritten to it and the old loops deleted.
- **PARITY SAFEGUARD (owner approved)** — big-bang deletes the old distill loop, so the
  digest `9a0100a1a3de3acd` becomes the *sole* reference. Mitigation: **keep the old
  distill loop locally during development to A/B old-vs-new until the new path reproduces
  the digest byte-for-byte; only then delete it.** Do not fly blind. (Recorded here + in
  memory so it is not forgotten across context compaction.)
- **Process** — implement stages 1–3, CPU-validate at each boundary (unit + parity digest
  + 2-rank gloo DDP smoke), **stop before any commit** for owner review; dataclass config,
  no tyro.

## IMPLEMENTATION STATUS (2026-07-23) — spine DONE + VERIFIED, additive

All work so far is **additive** (new `canvit_pretrain/harness/` package); no existing
file touched, so the live distill/ade20k/in1k trainers are untouched and the tree is
unbroken. Nothing committed (owner review pending).

**DONE + CPU-verified:**
- `harness/spec.py` — `TrainSpec` / `BpttSpec` / `GroupOptim` / `ScheduleSpec` / `TaskCaps`
  + `check_spec` validation (reject/warn, design §8) + presets (probe/finetune/policy_only/
  joint). **22 unit tests green** (`tests/test_spec.py`).
- `harness/rollout.py` — the generalized, task-agnostic, TrainSpec-driven rollout engine
  (`run_rollout`, design §3/§6): grad regimes none/full/chunked, generalized branch list
  (D-D), Task+Selector+JointPolicy seams. **Reproduces the distill parity digest
  `9a0100a1a3de3acd` BYTE-FOR-BYTE** via `tests/test_rollout_parity.py` (drives the engine
  with a distill adapter over the parity-probe setup) — proven a byte-exact superset of the
  historical distill rollout, on the first try. **+5 new-axes tests** (`tests/test_rollout_engine.py`:
  fixed/stochastic horizon, none-regime freezes backbone/trains external head, full-regime
  trains backbone).
- `harness/task.py` — the full `Task` protocol (run-level seam: caps/default_spec/build_model/
  build_loaders/build_selector/bind→RolloutTask/evaluate/checkpoint_payload), `runtime_checkable`.
- `tasks/{distill,ade20k,in1k}/task.py` — the three peers' **engine-facing cores** (`BoundDistillTask`/
  `BoundAde20kTask`/`BoundIn1kTask`: `forward_glimpse`/`step_loss`/`per_image_loss`, reusing the
  ported `consumes_full_image`/`derive_glimpse_px`/`ce_loss` helpers). **Smoke-tested driving the
  unified engine** (`tasks/tests/test_task_rollout.py`, 6 tests): distill finetune trains backbone;
  ADE20K probe (frozen bb → head) AND **finetune (NEW capability)**; IN1k frozen AND finetune;
  **distill joint task+policy** (P4b mechanism through the new engine — scorer + backbone both
  train, `policy_metrics` populated). Grad routing lands in exactly the right modules per spec.

- `harness/optim.py` — per-group optimizer/scheduler builder (design D-E): one AdamW with one
  param group per trainable module (own lr/wd), one `LambdaLR` whose per-group `lr_lambda`
  realizes that group's `ScheduleSpec` (warmup_constant / warmup_cosine; onecycle stubbed with a
  loud `NotImplementedError` until ADE's exact `WarmupOneCycleLR` is ported). **5 unit tests green**
  (`tests/test_optim.py`). Not parity-gated (probe runs at constant LR).
- `harness/policy.py` — task-agnostic joint-policy builder `build_policy` (design §7 D-D crown
  jewel): generalizes `train/joint.py::build_joint_policy` by splitting `canvit` (`.cfg`) from
  `encode_model` (what `StateEncoder` featurizes: distill shim / seg / clf) + task `feature_groups`
  (INTRINSIC for distill+in1k; full set w/ ent-features for ade20k's spatial probe). Reuses
  `JointPolicy`/objectives/selectors/scorer. **Joint task+policy now works for ALL THREE tasks** —
  smoke-tested through the engine: **ADE20K joint (probe+scorer train, backbone frozen — NEW)** and
  **IN1k joint (head+scorer train — NEW)**, plus distill joint. The capability you flagged as the
  point of the refactor.
- `harness/checkpoint.py` — unified LOCAL checkpoint I/O: `save_checkpoint` / `load_checkpoint` /
  `restore_into` / `find_latest` (+ `latest.pt` symlink). Persists model (backbone+head), optimizer,
  scheduler, step, `asdict(TrainSpec)`, model_config, metadata; policy in a `.policy.pt` sidecar.
  **Never touches the network** (D-G: publishing stays the manual `to_hf` step).
- `harness/loop.py` — the task-neutral **step-based driver** `run_training_loop` + `apply_requires_grad`
  (harness sets requires_grad from the spec, not the task). Thin: per-step rollout → grad-clip →
  opt/sched step, with log/checkpoint/eval-cadence hooks; task-specifics stay behind the `Task` seam.

Baseline reconfirmed on this machine/venv (`.venv-cu126`): parity probe prints
`9a0100a1a3de3acd`. **42 harness/task tests green** — the full config cross-product (task-only
probe/finetune, policy-only, joint-with-freeze-combos) across all three tasks, PLUS an end-to-end
CPU vertical (`tasks/tests/test_loop_e2e.py`): the loop trains an ADE20K probe on synthetic data
through the engine + per-group optimizer and the local checkpoint round-trips into a fresh model.

**REAL-DATA GPU VALIDATION (2026-07-24, A100 + bf16 AMP).** `unification_docs/harness_realdata_ab.py`
A/Bs the old `training_step` vs new `run_rollout` per step on REAL IN21k feature-webdataset batches
(batch 32, 512px, same model/RNG). Result: **max reldiff = 0.00e+00 over 20 steps** (byte-exact) across
n_glimpses 2–14 — the engine preserves distill exactly on production data/numerics, not just synthetic
CPU. Fully offline (random backbone, precomputed features, no teacher/HF).

**RUN-LEVEL WRAPPERS + SINGLE ENTRY — DONE + REAL-DATA VALIDATED (2026-07-24, A100 offline):**
- `harness/run.py` — the single orchestration `run(*, task, spec, settings)` (build_model →
  build_policy(joint) → apply_requires_grad → build_optimizer → build_loaders → build_selector →
  run_training_loop + eval/log/ckpt hooks) + `RunSettings` (composed config, D-B) + `RunTask` Protocol
  + an ADDITIVE CLI `python -m canvit_pretrain.harness.run --task {distill,ade20k,in1k} [--preset ...]`.
  The CLI does NOT replace `python -m canvit_pretrain.train` (that repoint is the gated cutover).
- `tasks/{distill,ade20k,in1k}/task.py` gained the run-level `*RunTask` (full seam: caps/default_spec/
  build_model[from_pretrained_with_new_probe|head / create_model]/build_loaders/build_selector/
  build_policy/trainable_param_groups[in1k head = norm+head]/bind/evaluate[mIoU / top-1,5 / distill
  validate()]/model_config). Config is composed (task holds its config; policy config passed in as `rl`).
- `spec.task_weight` now threaded into `run_rollout` (design §4), guarded `== 1.0` so **parity digest
  `9a0100a1a3de3acd` is preserved**; policy loss stays scaled by `joint.rl_weight` (== policy_weight).
- Validation: **8/8 configs PASS** through `run()` on real cached models —
  ade20k{probe,finetune,joint}, in1k{frozen,finetune,joint}, distill{finetune,joint}
  (`unification_docs/harness_run_integration.py`); joint on ade20k/in1k = the flagship NEW capability,
  head/probe + scorer training with `reward_frac` populated. **59 harness/task CPU tests green**
  (42 + 9 run-wrapper + 2 task_weight-scaling + 6 loop-ops).

**SINGLE-GPU OPERATIONAL FEATURES — CORE SET DONE + VALIDATED (2026-07-24 pass 2):** ported the CORE
task-neutral operational features into the harness (single-GPU resume / preemption / crash-safety): resume
(`find_latest`→`restore_into` before loaders, `start_step` via a `task.resume_start_step` hook), SIGUSR1
checkpoint-on-signal, FAILED-marker + `cancel_slurm_array` crash-loop guard (opt-in), provenance snapshot
+ `latest.pt` symlink, EMA-smoothed loss + per-module grad-norms. Optuna DROPPED (deprecated).
Validated: run()-level resume PASSES on real data (`harness_run_resume.py`: leg1→5, resume→9) + 6 CPU
ops tests (`test_loop_ops.py`) + 8/8 integration re-run (no regression). Built DDP-aware (rank-0-guarded).
**PASS 3 — the rest of the single-GPU fidelity gap (2026-07-24).** (Pass 2 had planned from a targeted
grep rather than reading `training_loop` end to end, so several features were missed and the result was
overclaimed as "feature-complete"; pass 3 read it top-to-bottom.) Ported: **seed-mode** start
(`RunSettings.seed_ckpt` weights-only @ step 0, priority resume>seed>fresh; distill also honors
`cfg.hf_seed_ckpt` in `build_model` with the HF config winning); **`torch.compile`**; **config +
provenance HISTORY accumulated across resumes** + distill's **`pretrain_view_scale`** stamping (what
`to_hf` reads — the foveated footgun); **run_dir layout**; **data%/gpu% timing**; and the **distill PCA
viz** (owner: distill only). Viz uses a new engine seam `run_rollout(collect_viz=, viz_task=)` +
`task.viz_init/viz_frame/render_viz` — branch-0 only, **off by default so the parity path is untouched** —
reusing `extract_sample0_viz`/`plot_multistep_pca`/`save_figure` so figures are identical and land
**LOCALLY** under `{run_dir}/visualization/pca_train/` (never uploaded). GPU-verified rendering.

**Pass 4 (2026-07-24) — WebDataset multi-job resume, the last CORRECTNESS gap. DONE + verified.**
Production distill runs are SLURM arrays where each task trains exactly `steps_per_job` steps over its
own shard slice; the next task must read the NEXT slice. The harness previously derived `start_step`
from the scheduler and passed a **hardcoded `job_index=0`** to `create_loaders`, so a multi-job resume
would have silently re-read job 0's shards. Now:
- distill `resume_start_step` derives `start_step = (saved job_index + 1) * steps_per_job` on the
  WebDataset path (scheduler only on the sharded path), and **raises** on a checkpoint with no
  `job_index` or one whose `scheduler.last_epoch` disagrees with the job boundary (a mid-job / SIGUSR1
  save, or a leg that ran a step count other than `steps_per_job`).
- `build_loaders` passes the real `job_index` and re-checks the schedule invariants
  (`world_size`/`batch_size_per_gpu`/`steps_per_job`/`samples_per_shard`) against the checkpoint,
  raising rather than silently shifting the slice offset.
- new task seam **`resume_state() -> dict`** (parallel to `resume_start_step`), stored at checkpoint
  `metadata["resume_state"]`; `{}` for ade20k/in1k (map dataset / `with_epoch` reshuffle — no cursor).
`SLURM_ARRAY_TASK_ID` itself stays with the launcher rewrite: the index comes from the checkpoint, not
the environment, so array tasks chain correctly without it.
Verified on the real IN21k webdataset (`unification_docs/harness_run_wds_resume.py`, `ALL PASS`): leg 1
ran steps 0-127 on `shard-001751.tar` saving `job_index=0`; leg 2 resumed at 128, advanced to
`job_index=1`, and read the **disjoint** `shard-002991.tar`; all refusal cases raised. Plus 11 CPU tests
(`tasks/tests/test_wds_resume.py`).

**Pass 5 (2026-07-24) — the four gaps pass 4's end-to-end read surfaced, ALL CLOSED.** None were on an
earlier list: **(a) raw/no-feature WebDataset shards (on-the-fly teacher targets) were unsupported and
CRASHED** — the exp21 path; `build_loaders` now dispatches on `train.has_features` and `bind()` makes
targets via `_teacher_targets` when the batch has none (verified on the real no-feature IN1k
webdataset). **(b)** `cfg.reset_normalizer` now forces a re-init. **(c)** `run()` resolves the resume
checkpoint BEFORE `build_model` and passes `prior_model_config=`, so the checkpoint's arch wins over
CLI defaults (distill rebuilds via `dacite`; its `model_config` gained a lossless `"canvit"` entry;
RESUME now correctly beats `hf_seed_ckpt`). **(d)** the scorer is clipped **separately** from the model
(after `allreduce_grads()`), matching `train/loop.py` 874-878 — one joint norm had been coupling their
magnitudes on every RL run.

**Pass 6 (2026-07-24) — full wandb metric richness + the distill validation phase, DONE.** Per-branch
`full/…` / `random/…` series via two OPTIONAL task hooks (`glimpse_metrics` per glimpse,
`final_metrics` on the last readout) + a neutral `loop.branch_metrics()` that groups by t0 type and
averages (hookless tasks just get `{type}/loss`); EMA over every series logged under the plain names
(instantaneous total kept as `total_loss_raw`); the `train/` namespace + `lr`/`grad_norm`/
`continue_prob`/`prime_on_policy`; `log_parameters` (flattened config + spec + param counts). The
distill validation phase had been silently gutted — `evaluate()` used a throwaway `tracker="none"`
and a temp run dir, DISCARDING every `val/…` series and figure; the `evaluate` seam now carries
`tracker=`/`run_dir=` and distill passes the probe, curve/PCA cadences, `foveated_eval_scale` and
spatial stats. Verified by `unification_docs/harness_metric_parity.py` (ALL PASS, records rather than
uploads): 92 metric keys incl. `val/in1k_tts_top1_t0..t9`, 171 hyperparameters, PCA figure on disk.

**NOT yet a full drop-in** — remaining: **DDP** (multi-GPU node), which also owns the
`ddp.all_reduce_mean` on each logged scalar. That is the only item left before the cutover.

**REMAINING:**
- `harness/ddp.py` — manual grad-sync for in-rollout modules (backbone/scorer) per the §9 matrix
  (loop already calls `joint.allreduce_grads()` when dist); assert unsupported cells. **Owner: SKIP until a multi-GPU node.**
- (optional) port ADE20K's `WarmupOneCycleLR` into `optim.py` (onecycle currently raises); distill
  `evaluate()` full viz/PCA (currently best-effort `validate()` reuse) — both land naturally at the cutover.
- **Big-bang cutover (owner GREEN LIGHT required — destructive):** repoint `python -m canvit_pretrain.train`
  at `harness.run`, flip `train/step.py`→`run_rollout` (re-confirm the REAL parity probe +
  93-test pretrain suite), delete old loops + rewrite SLURM launchers. Then the GPU gate. The scaffolding
  above makes this mechanical.

## 0. Why (the problem this closes)

The three trainers today cover an arbitrary, non-orthogonal subset of the possible
configurations, purely because each was ported from a different origin repo at a
different time and bolted on where convenient:

| | task: probe | task: full-finetune | policy: frozen model | policy: joint w/ task | BPTT |
|---|---|---|---|---|---|
| **distill** (`train/`) | n/a (heads are projections) | ✅ (the only path) | ❌ | ✅ (`cfg.rl.use_rl`) | stochastic chunked |
| **ADE20K** (`ade20k/`) | ✅ (only path) | ❌ not ported | ✅ (`rl_train.py`) | ❌ | none (backbone frozen) |
| **IN1k** (`in1k/`) | ✅ | ✅ (`mode=finetune`) | ❌ | ❌ | full-graph |

None of the ❌ are principled. All three are **CanViT backbone → recurrent glimpse
rollout → readout of canvas state → loss**, with an optional `ViewpointScorer` policy
whose reward is the per-image reduction of *that task's* loss. The differences that are
real (readout, target, loss, metric, data) are exactly the things a `Task` object
should own; everything else (rollout, BPTT, grad routing, DDP, checkpointing,
optimizer groups, tracking) is shared and should live in one neutral harness.

The evidence this was always the intent: `train/task.py` already declares a `Task`
Protocol "for the unified harness (master-plan §4.2)" and says the ADE20K/IN1k tasks
"arrive in P2/P5". They arrived as *parallel loops* instead of `Task` implementations.
This doc finishes the job.

## 1. Principle: one neutral harness, three peer tasks

- **Harness** owns everything task-agnostic: the rollout engine, BPTT/grad regime,
  loss composition, optimizer/param-group construction, DDP sync, checkpoint/HF I/O,
  tracking, the validation layer. It knows nothing about DINOv3, segmentation, or
  classification.
- **Task** owns everything task-specific behind a fixed interface: data, targets,
  the head module (or none), readout, loss, per-image reward signal, eval/metrics,
  checkpoint payload.
- **`DistillTask` is demoted to a peer** of `Ade20kTask` / `In1kTask`. Distill is not
  the substrate; it is one `tasks/distill.py` sitting beside the other two, all three
  calling the same `harness.run(task, spec, cfg)`.

The repo keeps the name `CanViT-pretrain` (renaming a repo is out of scope), but its
*internals* stop privileging pretraining.

## 2. Target package layout

```
canvit_pytorch (core)                    ← already done in P1
  policy/                                ViewpointScorer, StateEncoder, candidates
  data/ade20k.py, metrics.py             shared eval primitives (P6)

canvit_pretrain/
  harness/                               ← NEW: task-neutral training machinery
    run.py           run(task, spec, cfg): the single entry the 3 tasks call
    rollout.py       the one rollout engine (§6): no-grad / full / chunked TBPTT
    trainstep.py     loss composition, per-glimpse task+policy loss accumulation
    spec.py          TrainSpec (the orthogonal flags, §4) + validation (§8)
    optim.py         per-group optimizer/scheduler builder (§ D-E)
    ddp.py           manual broadcast + grad-AllReduce helpers + support matrix (§9)
    checkpoint.py    unified ckpt/HF format (§10), policy sidecar, view-scale metadata
    loop.py          val cadence, logging, commit-pinning glue (moved from train/)
    policy.py        JointPolicy assembly (folded from train/joint.py), selectors, rl.py
  tasks/                                 ← the three EQUAL tasks
    distill/         __main__.py (defaults) + task.py  (moved from train/{step,task}.py)
    ade20k/          __main__.py (defaults) + task.py  (readout/loss/eval from ade20k/)
    in1k/            __main__.py (defaults) + task.py
  slurm_nhr/         launchers unchanged in shape (one per task, or one param'd by --task)
```

Each `tasks/<name>/__main__.py` is thin: it assembles the task's *default* `Config` +
`TrainSpec` and calls `harness.run(...)`. Per-task defaults (e.g. distill's stochastic
horizon vs. ADE20K's fixed 10 glimpses) live here as data, not as forked control flow.

> Migration note: the pretrain parity digest (`9a0100a1a3de3acd`) is computed on the
> loss stream and is **import-path independent**, so moving `step.py`/`task.py` into
> `tasks/distill/` and `harness/` does not by itself change it. The digest is the CPU
> regression guard that the move + refactor preserved distill exactly.

## 3. The three seams

### 3.1 Task interface (extends master-plan §4.2)

```python
class Task(Protocol):
    def build_model(self, cfg) -> tuple[CanViT, nn.Module | None]   # backbone, head|None
    def build_loaders(self, cfg) -> tuple[DataLoader, DataLoader]   # train, val
    def build_targets(self, batch, device) -> Targets               # feats | mask | label
    def readout(self, state) -> Any                                 # tokens+cls | hidden | cls
    def step_loss(self, pred, targets) -> LossOutput                # per-glimpse task loss
    def per_image_loss(self, pred, targets) -> Tensor               # [B], reward raw material
    def reward_signal(self) -> Literal["task_loss", "task_metric"]  # default "task_loss"
    def policy_feature_groups(self) -> tuple[str, ...]              # INTRINSIC vs probe-aware
    def evaluate(self, model, head, val_loader, cfg) -> dict        # task metrics (mIoU/top1/…)
    def checkpoint_payload(self, model, head) -> dict               # what to persist/publish
```

The Task supplies the head *module* but does **not** decide what is trainable — the
harness sets `requires_grad` on {backbone, head, scorer} from the `TrainSpec`. This is
the key inversion: freezing is a harness/TrainSpec concern, uniform across tasks, not
baked into each task's loop (today ADE20K hard-codes `requires_grad_(False)`).

### 3.2 Selector interface — already exists

`RandomSelector` / `PolicySelector` (+ `MixtureSelector` for ε-curriculum) from
`train/selector.py`, consulted inside the rollout. No change needed beyond relocation
into `harness/policy.py`. Patcher-aware action space (fixation grid for foveated/square,
safe-box for uniform) is already handled in `build_joint_policy`.

### 3.3 TrainSpec — the orthogonal grad/training control

This is the heart of the request. The full cross product you enumerated is generated by
a small orthogonal set (see §5):

```python
@dataclass
class TrainSpec:
    # which parameters get gradients (any subset; empty => error)
    train_backbone: bool
    train_head: bool
    train_policy: bool

    # which losses are active (both zero => error)
    task_weight: float          # 0 => task loss not backpropped (still computed for reward)
    policy_weight: float        # 0 => no policy loss (=> policy unused unless deploying fixed)

    # loss -> backbone routing (only meaningful when train_backbone=True)
    task_grad_to_backbone: bool     # False + train_head => "probe"; True => "finetune"
    policy_grad_to_backbone: bool   # == not feats_detached; True needs BPTT into trunk

    # rollout gradient regime (§6)
    bptt: BpttSpec                  # none | full | chunked(chunk_size, continue_prob)

    # per-module optimizer/schedule (only groups whose train_* is True are built) (§D-E)
    optim: dict[str, GroupOptim]    # {"backbone":…, "head":…, "policy":…}
```

`feats_detached` (P4b) becomes `not policy_grad_to_backbone`. `mode ∈ {frozen,finetune}`
(IN1k) becomes `train_backbone` + `task_grad_to_backbone`. `use_rl` (P4b) becomes
`train_policy` + `policy_weight>0`. All three collapse into one spec.

## 4. Loss composition

Single scalar, single optimizer step (unchanged from §4.2):

```
loss = task_weight * task_loss + policy_weight * policy_loss(QReg | PG)
```

with per-glimpse accumulation inside the rollout (§6). The reward for the policy is
formed from `Task.per_image_loss` (or a task metric — see D-F), standardized per depth
by the existing `RunningNorm`. Stop-grad boundaries are applied by the harness according
to `task_grad_to_backbone` / `policy_grad_to_backbone` before the single backward.

## 5. The combinatorics (your enumerated configs as points in TrainSpec)

Same table applies to **all three tasks** (distill's "head" = its recon/CLS projections):

| your config | train_bb | train_head | train_policy | task_w | policy_w | task→bb | pol→bb |
|---|---|---|---|---|---|---|---|
| task only, full model | ✅ | ✅ | – | >0 | 0 | ✅ | – |
| task only, frozen bb (probe) | ❌ | ✅ | – | >0 | 0 | – | – |
| policy only, everything frozen | ❌ | ❌ | ✅ | 0* | >0 | – | ❌ |
| policy only, bb NOT frozen | ✅ | opt | ✅ | 0* | >0 | – | ✅ |
| joint: probe task + full-net policy | ✅ | ✅ | ✅ | >0 | >0 | ❌ | ✅ |
| joint: full task + full policy | ✅ | ✅ | ✅ | >0 | >0 | ✅ | ✅ |
| joint: full task + policy-net-only | ✅ | ✅ | ✅ | >0 | >0 | ✅ | ❌ |
| joint: probe task + policy-net-only | ✅ | ✅ | ✅ | >0 | >0 | ❌ | ❌ |

`*` policy-only still **computes** the task forward/loss to produce the reward; it just
does not backprop it. BPTT (`none/full/chunked`) is an independent axis crossed with
every row. So ~6 knobs generate the whole space, identically for all three tasks — this
is exactly your claim, and it holds.

## 6. Rollout engine: unifying the three BPTT regimes

Today: distill = stochastic chunked TBPTT (`chunk_size=2`, `continue_prob=0.5`, detach at
chunk boundary); IN1k finetune = one graph over the whole rollout; ADE20K/IN1k-frozen =
backbone under `no_grad`. One engine covers all three:

```
BpttSpec = none | full | chunked(chunk_size:int, continue_prob:float|None)
```

- **none**: backbone forward under `torch.no_grad()`; only head/scorer carry graph.
  (probe, frozen-policy). Horizon fixed.
- **full**: single graph over the whole rollout; one backward at the end.
  (= `chunked` with `chunk_size = horizon`, `continue_prob = None`.)
- **chunked**: backward + detach at each chunk boundary; `continue_prob` gives the
  stochastic horizon (distill) or `None` + fixed horizon gives deterministic length.

Per-timestep task losses are produced by `Task.step_loss` at each glimpse; the engine
decides *when* to call backward from `BpttSpec`. The policy loss node is accumulated per
glimpse regardless. Frozen-backbone tasks simply never put backbone params in the graph.
**One mechanism, three configs** — no per-task rollout code.

## 7. Open sub-decisions (the real review surface — veto individually)

These are genuine design choices, each with a recommendation. Everything above is
settled; these are what to argue about.

- **D-A Package neutrality.** *Rec:* physically move shared machinery to `harness/` and
  make `tasks/distill/` a peer (§2). Cost: large file move + import churn; parity digest
  survives (path-independent). *Alt:* leave distill in `train/` and have others import it
  — rejected, that's the privileged-substrate anti-pattern you called out.
- **D-B Task signature.** *Rec:* §3.1. Head module supplied by Task, `requires_grad` set
  by harness. Open detail: whether `readout` returns a task-opaque object or a typed union.
- **D-C Rollout regime.** *Rec:* the `none/full/chunked` union in §6, chunked as the
  general case. Open detail: whether ADE20K/IN1k ever want stochastic horizon (probably
  not — fixed is fine; `continue_prob=None`).
- **D-D Branch structure.** Pretrain runs N branches per image (`n_full_start_branches`
  + `n_random_start_branches`; policy operates on FULL-anchored branches). ADE20K/IN1k
  run one. *Rec:* generalize to a list of branch specs; default `[FULL]*n_full +
  [RANDOM]*n_random`, ADE/IN1k default to a single branch. Adds (opt-in) config surface
  for the two simpler tasks. *Alt:* keep branches distill-only — simpler but less uniform.
- **D-E Optimizer groups.** Up to 3 trainable groups, each wanting its own LR/WD/schedule
  (finetune bb LR ≪ head LR; policy 2e-4). *Rec:* `TrainSpec.optim: {group: GroupOptim}`;
  build only trainable groups; a small schedule registry (warmup+const/cosine,
  WarmupOneCycleLR). Defaults per task reproduce today's recipes. Fiddly bit: OneCycle
  needs total steps.
- **D-F Reward source.** *Rec:* default reward = fractional reduction of
  `Task.per_image_loss` (generic, already in `DistillTask`); allow `reward_signal="task_metric"`
  (e.g. per-image mIoU delta) where meaningful. Keep task_loss default — per-image metrics
  are noisier/costlier.
- **D-G Checkpoint / HF format.** *Rec:* main ckpt = CanViT (+head if trained) in the HF
  `config.json`+safetensors format, with `metadata.pretrain_view_scale` (P6 footgun) and
  a new `metadata.train_spec`; policy in a `.policy.pt` sidecar (P4b). Extend
  `checkpoint/to_hf.py` to record task + spec. Open detail: for finetuned ADE/IN1k, do we
  publish the backbone too or only the head? *Rec:* persist both, converter chooses.

## 8. Validation layer (allow-with-warning, per your call)

Every combo is *allowed*; the layer only rejects incoherent specs and warns on vacuous
ones (we trust the user to choose deliberately).

- **Reject (error):** nothing trainable (`train_* all False`); `task_weight==0 and
  policy_weight==0`; `policy_weight>0 and not train_policy`; `task_grad_to_backbone and
  not train_backbone` (routing into a frozen module — contradiction); a DDP run selecting
  an unsupported cell (§9).
- **Warn (run anyway):** frozen-backbone distill (recon heads on frozen features —
  near-vacuous); backbone trained *only* by the policy loss with `task_weight==0`
  (unusual but valid); probe on a non-pretrained backbone.

## 9. DDP: best-effort, documented support matrix (per your call)

DDP is **not** a must-have for every cell. Single-GPU supports the full cross product;
DDP supports a subset, and unsupported cells **assert-and-refuse under DDP** (never
silently mis-train). Any trainable module called *inside* the per-glimpse loop
(backbone, scorer) needs manual broadcast + grad-AllReduce (the pattern already in
`joint.py` for the scorer and in IN1k for `clf.canvit`); heads applied outside the loop
DDP-wrap normally.

| config | single-GPU | DDP |
|---|---|---|
| task-only probe (head only) | ✅ | ✅ (trivial) |
| task-only finetune (bb in rollout) | ✅ | ✅ (manual bb sync — IN1k pattern) |
| policy-only, frozen bb | ✅ | ✅ (manual scorer sync — P4b/RL pattern) |
| joint, `policy_grad_to_backbone=False` | ✅ | ✅ (P4b) |
| joint, `policy_grad_to_backbone=True` (coupled) | ✅ | ❌ **unsupported** (assert) |
| policy-only with `policy_grad_to_backbone=True` | ✅ | ❌ unsupported initially |

The coupled-into-backbone-under-DDP cell is the one P4b already asserts UNSUPPORTED;
generalizing the manual multi-module grad-sync to lift it is deferred and explicitly
optional. Flag it in code + here; do not block the rest of the framework on it.

## 10. Eval

Eval stays substantially per-task (distill: cosine/recon on a fixed val subset; ADE20K:
`mIoUAccumulator` over val; IN1k: top-1/5 over ImageFolder val) — but behind
`Task.evaluate`, so the harness owns *cadence* (val_every, viz cadence, rank-0-only) and
the Task owns *what is measured*. No attempt to unify the metrics themselves.

## 11. Staged implementation plan (revised per your comments)

GPU is needed only at the final gate. Stages 1–3 are CPU-implementable and
CPU-validatable (unit tests + the parity digest + CPU-gloo 2-rank DDP smoke).

1. **Neutral harness + Task seam, three peers.** Create `harness/` and
   `tasks/{distill,ade20k,in1k}/`. Move distill's step/loss behind `DistillTask`; port
   ADE20K and IN1k readout/loss/eval behind their `Task`s. Keep three entry points, all
   calling `harness.run`. Re-establish the parity digest for distill (CPU). *No behavior
   change intended.*
2. **Unified TrainSpec + rollout + validation (single-GPU semantics).** Land the
   orthogonal flags (§3.3), the `none/full/chunked` rollout (§6), loss composition, the
   validation layer (§8), per-group optimizer (D-E). Every existing config becomes a
   TrainSpec point; unit-test the mapping. Parity digest unchanged.
3. **DDP grad-sync generalization.** Manual broadcast/AllReduce for every supported cell
   (§9); assert on unsupported cells. CPU-gloo 2-rank smoke.
4. **GPU gate (deferred, at the end).** Re-gate each task numerically + real multi-GPU;
   confirm parity, joint runs, finetune vs canvit_eval baseline, foveated re-gate.

Stages 1–3 unblock the new configs you want (joint on ADE20K/IN1k, policy on IN1k,
full-finetune on ADE20K, arbitrary freeze combos) on single-GPU without waiting on the
GPU campaign.

## 12. Testing tiers

- **CPU unit:** Task conformance (each implements the protocol), TrainSpec→param-group
  mapping, validation reject/warn rules, rollout regime equivalence (chunked with
  `chunk_size=horizon` == full), reward standardization.
- **CPU parity:** distill loss-stream digest `9a0100a1a3de3acd` (guards stage 1–2).
- **CPU DDP:** 2-rank gloo grad-AllReduce equivalence for each supported cell.
- **GPU (final):** numeric acceptance per task; deferred.

## 13. Risks

- **Blast radius.** Touches the rollout, all three trainers, DDP, checkpointing, and the
  parity guard at once. Mitigated by staging: stage 1 is a pure move+extract with the
  digest as a tripwire.
- **Parity preservation.** The distill path must reproduce `9a0100a1a3de3acd` through the
  move and the rollout refactor. This is a strict, CPU-checkable guard — good, but it
  makes stage 1–2 exacting.
- **Optimizer/schedule generalization (D-E).** The subtlest non-DDP piece; per-group
  schedules with different total-step semantics need care.
- **Deferred GPU gate.** The framework can be *built and unit-correct* without GPU, but is
  not *proven* until the stage-4 runs. Accepted.

## BPTT: the one rule that must not be broken (2026-07-28)

**A FROZEN backbone ALWAYS takes `bptt.mode="none"`.** Never pair `train_backbone=False`
with `"full"` or `"chunked"`.

`bptt` moves the **backbone and nothing else**. The head reads the canvas state at t and
never feeds back into it, so no head parameter influences a later timestep and there is no
cross-timestep path to propagate. Measured (`harness/tests/test_bptt_chunking.py`): head
gradients are **bit-identical** between `none` and `full` — whether the backbone is frozen
*or* trainable. Building a graph over a frozen backbone therefore changes no number and
holds activations for the whole rollout.

Three layers enforce this, so it should be impossible to get wrong:

1. `harness.spec.fixed_horizon_bptt` — the single mapping used by every task's
   `default_spec` AND by `cli.resolve_spec`. `frozen=True` returns `none`, ignoring
   `chunk_size`. The config path cannot express the mistake.
2. `check_spec` emits a warning if a spec pairs a frozen backbone with a graph regime.
   That catches hand-built `TrainSpec`s, the only remaining route. Warning, not error —
   it is wasteful, not incorrect.
3. `BpttSpec`'s docstring states the rule where anyone configuring it will read it.

**Not covered:** `train_policy` runs. The policy's cross-timestep path has not been
measured, so `check_spec` deliberately does not warn there. Measure before assuming.

### Chunk length is otherwise free

`horizon` need NOT be divisible by `chunk_size`. `run_rollout` flushes the trailing
partial chunk, and every chunk normalises by `n_glimpses` rather than by its own length,
so the accumulated gradient is identical however the rollout is split. 7 glimpses at chunk
3 runs `[0,1,2][3,4,5][6]`. Prime horizons are fine — verified for 6@3, 6@2, 7@3, 7@2,
5@4, 11@3. `chunk_size >= horizon` collapses to `full`.

| task | length | regimes reachable from a launcher |
|---|---|---|
| distill | stochastic (`CFG_CHUNK_SIZE`, `CFG_CONTINUE_PROB`) | `chunked` |
| ade20k / in1k, `mode=frozen` | fixed `n_timesteps` | `none` (forced) |
| ade20k / in1k, `mode=finetune` | fixed `n_timesteps` | `full` (default) or `chunked` via `CFG_BPTT_CHUNK_SIZE` |
