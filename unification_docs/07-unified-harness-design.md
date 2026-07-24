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
**NOT yet a full drop-in** — the old loop can't be deleted until these ALSO land (beyond DDP): seed-mode
start (`seed_ckpt`/`hf_seed_ckpt` weights-only @ step 0); full WebDataset shard-aligned `job_index`
multi-job resume (only the hook seam exists); `pretrain_view_scale` footgun metadata + config/provenance
history in distill checkpoints (needed by `to_hf`); `torch.compile`; run_dir layout; full wandb metric
set + distill val viz/PCA/IN1k-probe.

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
