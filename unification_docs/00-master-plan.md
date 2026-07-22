# Unification master plan — merging specialize + RL into CanViT-pretrain

**Written:** 2026-07-22 · **Status:** approved design, implementation not started
**Prerequisite reading:** `CanViT-specialize/docs/unification-status.md` (the divergence
inventory; this plan builds on it and does not repeat it).

---

## 1. Goal

One training repo (`CanViT-pretrain`) that supports every cell of this matrix:

| axis | values |
|---|---|
| task | `distill` (DINOv3 feature regression) · `ade20k` (seg) · `in1k` (cls, new CUDA impl) |
| locations | `random` (today's behavior) · `policy` (learned, closed-loop) |
| trainable parts | any subset via stop-grad config: backbone / task head / policy (e.g. policy-only on a frozen model = today's RL repo; backbone+head with policy-off = today's pretrain/specialize) |
| patcher | uniform · foveated · square (policy support for foveated is **new design**, see §6) |

Today's three-repo functionality maps to three specific configurations:

1. `task=distill, locations=random, train=backbone+heads` → current CanViT-pretrain
2. `task=ade20k, locations=random, train=head-only (frozen backbone)` → current CanViT-specialize/ADE
3. `task=ade20k, locations=policy, train=policy-only (frozen backbone+probe)` → current CanViT-PyTorch-RL

After the merge, `CanViT-specialize` and `CanViT-PyTorch-RL` are **archived** (read-only,
docs mined first — see §9). The RL repo's `flow/` module (continuous-action RealNVP) is
**not ported** (parked experiment; its negative results are preserved in docs).

## 2. Locked decisions (user, 2026-07-22)

| # | decision | choice |
|---|---|---|
| D1 | Policy net home | **`canvit_pytorch` (core)**: `ViewpointScorer` + `StateEncoder` + action-space builders move next to the model classes. HF-hub loading stays intact; `canvit_eval` can deploy policies without depending on the training repo. Losses/harness go to pretrain. |
| D2 | IN1k | **TPU/XLA path left behind entirely** (deprecated, never used). A fresh **CUDA IN1k task** is built inside the new harness, modeled on the ADE20K task, scoped as a late phase. |
| D3 | `recon_normalized` | **Dropped.** ADE training moves fully onto the stable `CanViTForSemanticSegmentation` wrapper. The pretraining-head bypass (root cause of the 3-month silent breakage) disappears. |
| D4 | Tooling | **Pretrain conventions everywhere**: pretrain's dataclass configs, wandb + disk viz, commit-pinned SLURM launchers. Ported RL code drops MLflow/tyro/justfile/uv. Parity with old RL runs is checked statistically (§8). |

## 3. Asserted defaults (flagged, not asked — object if wrong)

- **Viewpoint distribution:** canonical = pretrain's existing **per-patcher** sampling,
  reproduced exactly by `RandomSelector`: uniform → safe-box law (`p(s) ∝ (1-s)`,
  `train/viewpoint.py`); foveated/square → full-field centers + `FoveatedScaleConfig`
  scales (no safe-box — foveation always sees the whole image). Core's `policies/`
  generators remain for eval only. A test asserts **both** branches so neither can
  silently change (unification-status §5.4).
- **Foveated is first-class, not a variant.** Every interface (Selector, action space,
  viz, tests) is patcher-aware from day one; each phase gate includes a foveated case
  where applicable. Structurally cheap on the policy *input* side: `StateEncoder` reads
  the canvas, which is a uniform grid regardless of patcher — the whole policy stack
  (features → frontend → U-Net → readout) is patcher-agnostic. Patcher-dependence is
  confined to the action space (§6.1) and glimpse extraction (already in the patchers).
- **ADE20K data:** one train-time pipeline (consolidating specialize's `ADE20kDataset` and
  the RL repo's `Ade20kSquish` — currently *three* implementations exist incl. canvit_eval's).
  Validation always follows `canvit_eval`'s protocol for paper-comparability.
- **Policy feature groups are task-configurable.** ADE/IN1k: all six groups (probe entropy
  exists). Distill: the four intrinsic groups (`cos_prev`, `cos_init`, `ln_feat`,
  `feat_delta`) — probe-entropy groups don't exist, and teacher-error features would make
  the policy undeployable without the teacher (privileged). A teacher-error feature group
  can be added later as an explicit opt-in with a loud "not standalone-deployable" marker.
- **Policy checkpoints carry compatibility metadata** (feature groups, action space,
  canvas grid, patcher) with strict asserts on load — cross-task/stage reuse is allowed
  exactly when these match (e.g. distill-pretrained 4-group policy reusable on ADE;
  6-group ADE policy not runnable under distill; c64 policies don't load onto c32 runs).
- **Reward, unified:** per-glimpse fractional task-loss reduction
  `r_t = (L_t − L_{t+1}) / L_t` where `L` = per-image task loss (seg CE / cls CE /
  distill MSE), computed detached, standardized by per-depth `RunningNorm`. The RL repo's
  128px `score_res` scoring trick ports as-is for seg.
- **Repo name stays `CanViT-pretrain`** for now. Renaming (e.g. `canvit-train`) is a
  cosmetic decision for later; nothing in the plan depends on it.

## 4. Target architecture

### 4.1 What moves where

```
canvit_pytorch (core)                          ← D1
  model/policy/
    net.py        ViewpointScorer (from RL policy/net.py; HF mixin kept,
                  repo_url updated; loads existing published checkpoints)
    features.py   StateEncoder + FEATURE_GROUPS (task-configurable subsets)
    actions.py    candidate-viewpoint builders: uniform (scales × safe-box grid,
                  from RL canvas_ops) + foveated variant (new, §6)

canvit_pretrain (the merged training repo)
  train/
    loop.py, step.py, …    existing pretrain loop — REFACTORED into the harness (§4.2)
    selector.py            RandomSelector / PolicySelector / MixtureSelector (ε-schedule)
    rl.py                  QReg + PG (+ optional Q-Prop critic), RunningNorm,
                           entropy-floor (from RL training/{train,stats}.py)
    tasks/
      distill.py           DINOv3 teacher targets + MSE heads (today's objective)
      ade20k.py            seg probe on the stable wrapper (D3), from specialize
      in1k.py              CUDA classification task (new, late phase; D2)
  datasets/                webdataset (existing) + unified ADE20K (+ IN1k later)
  slurm_nhr/               launchers; commit pinning simplifies to PRETRAIN/PYTORCH/FOVI
  unification_docs/        this plan + per-phase implementation notes

canvit_eval                unchanged in v1; later gains policy benchmarking
                           (ArgmaxPolicy episodes) importing ViewpointScorer from core
```

### 4.2 The harness: one loop, task + selector + grad-regime injected

The existing pretrain loop (`loop.py` + `step.py`) is the substrate — it already has DDP,
TBPTT chunking, checkpointing, wandb, disk viz, commit pinning. The refactor parameterizes
it along three seams; **with defaults, behavior is bit-identical to today** (§8 gate).

**Task interface** (each task supplies):

```python
class Task(Protocol):
    heads: nn.Module                      # distill heads / SegmentationProbe / cls head
    backbone_trainable: bool              # + finer stop-grad flags
    def build_data(cfg) -> DataLoader
    def build_targets(batch) -> Targets   # teacher features / masks / labels
    def step_loss(model_out, targets) -> Tensor   # per-image, per-glimpse
    def policy_feature_groups() -> tuple[str, ...]
    def reward(loss_t, loss_t1) -> Tensor # default: fractional reduction
```

**Selector interface** (consulted *inside* the rollout, closed-loop):

```python
class Selector(Protocol):
    def select(state, image, t, rng) -> tuple[Viewpoint, SelAux]
# RandomSelector      → reproduces today's schedule exactly (incl. FULL-t0 branches,
#                       fixation-style viewpoints for foveated, FoveatedScaleConfig)
# PolicySelector      → StateEncoder feats → ViewpointScorer → sample (PG) or
#                       ε-greedy (QReg); emits log-prob / taken-index in SelAux
# MixtureSelector     → ε-schedule between the two = warmup curriculum + DAgger
#                       prime_on_policy generalized (autoreg_tryout's rl_warmup analogue)
```

**Gradient regime** (per-part stop-grad config, autoreg_tryout's `rl_grad_mode` generalized):

```python
train_parts: set[str]          # {"backbone", "head", "policy"} — any subset
policy_feats_detached: bool    # True  = RL loss confined to policy net ("heads_only";
                               #         default, and the only mode the old RL repo had)
                               # False = RL loss flows through StateEncoder → canvas →
                               #         backbone within the TBPTT chunk ("all")
```

**Loss composition** (single optimizer step, single scalar):

```
loss = task_weight * task_loss + rl_weight * rl_loss(QReg | PG)
```

Either weight may be zero: `rl_weight=0, random selector` = today's pretrain/specialize;
`task_weight=0, frozen model` = today's RL repo; both nonzero = joint training (new).

### 4.3 In-graph rollout replaces collect-detached-then-reforward (verified feasible)

The old RL rollout collects under `torch.no_grad()` and re-forwards saved features — the
policy loss *cannot* reach the backbone by construction, and every state is forwarded
twice. The harness uses a single pattern instead (autoreg_tryout's): featurize + score
**with grad inside the rollout**, accumulate loss nodes, one backward at rollout end.
Verified compatible with all three ported algorithms:

- **PG**: log-prob kept from the sampling forward (exactly on-policy — more correct than
  the re-forward, whose train-mode forward differed from the sampling one).
- **QReg**: the late-arriving reward is no obstacle — `q[taken]` graph nodes persist;
  the MSE against the standardized target is formed post-hoc per depth and summed.
- **Q-Prop**: `crit_all`/`logp_all` come from rollout-time forwards; the analytic term
  is a plain sum over them.

Compute: one forward per state instead of two. Memory: with `policy_feats_detached=True`
only the small scorer's activations × horizon; with backbone-trainable, backbone
activations enter the graph bounded by the TBPTT chunk (pretrain's existing machinery).
Sampling remains non-differentiable (no selection-BPTT — not needed for QReg/PG; flow is
out of scope).

**BatchNorm mode — DECIDED (user, 2026-07-22): (a) train-mode rollout forward**
(autoreg_tryout's choice). One forward serves both action selection and training; BN
running stats keep training; the strict DAgger property (selection under exactly the
deployed eval-mode policy) is knowingly given up — selection happens under batch-stats
BN, a slight deploy/train mismatch. Accepted as a try-first default; if the P3 parity
run drifts out of the qband band and the drift traces to this, fall back to
(b) eval-mode-with-grad + explicit stat updates. Recorded here; no separate 04 doc.

## 5. What is NOT ported

- `flow/` (RealNVP) — parked; negative results preserved in the migrated docs.
- MLflow, tyro CLI, justfile, uv tooling (D4).
- The TPU/XLA IN1k pipeline (D2) — not even copied; it stays in the archived specialize repo.
- The RL repo's sweep infrastructure (`search/`, `perpetual_sweep`) — both sweeps ended on
  plateaus at/below seed noise; re-evaluate need after the merge rather than porting.
- `recon_normalized` ADE feature type (D3).

## 6. New design work (not ports) — each gets its own doc before code

1. **Foveated/square action space — DECIDED (user, 2026-07-22):** actions = fixation
   points only, no scale dimension (foveation always sees the full field). Candidates =
   one per canvas grid position (e.g. 32×32 = 1024 actions; density configurable).
   Simplifications that follow: candidates coincide with score-map pixels, so the
   readout reads the map directly (no `grid_sample` interpolation, no per-scale
   embeddings); no safe-box constraint. Scale, where a foveated model uses one, comes
   from `FoveatedScaleConfig` as in random sampling — not from the policy. Policies for
   foveated models have never been trained — treat first runs as research.
2. **Distill-task policy** (`02-distill-policy.md`): intrinsic-only feature groups; reward
   from per-glimpse distill-MSE reduction; interaction with multi-branch rollouts
   (FULL-start vs RANDOM-start branches) and geometric trajectory lengths.
3. **CUDA IN1k task** (`03-in1k-task.md`, late): classification probe/fine-tune in the
   harness; data via canvit_eval's IN1k plumbing; DDP epochs; policy support included.
4. ~~Scorer BatchNorm mode~~ — decided in §4.3 (train-mode forward); no separate doc.

## 7. Phase plan — each phase independently useful, gated, reversible

| phase | content | acceptance gate |
|---|---|---|
| **P0** ✅ | **DONE 2026-07-22** (see `p0-notes.md`): ADE 2-step CPU smoke tests (specialize), uniform-sampler distribution test (pretrain; foveated already covered), deterministic parity probe + baseline digest `9a0100a1…`. Ckpt compat asserts moved to P3 (need the new ckpt format). | ✅ all 3 suites green on current code (61/24/9) |
| **P1** ✅ | **DONE 2026-07-22** (see `p1-notes.md`): `selector.py` + `task.py` seams injected into `training_step` (defaults = historical behavior); spy-injection test added. Rollout-engine extraction deferred to P3/P4 by design. | ✅ parity digest byte-identical (`9a0100a1…`); 61+1 tests green |
| **P2** 🟡 | **CODE COMPLETE 2026-07-22** (see `p2-notes.md`): `canvit_pretrain/ade20k/` package (wrapper-only, patcher-aware rollout, 3 smoke tests incl. foveated), commit-pinned launcher. Core gained `from_pretrained_with_new_probe` (`2759e18`). | 🟡 gate run pending (user submits; command in p2-notes) — CPU/code gates green (65 tests, parity digest unchanged) |
| **P3** | Policy machinery: `ViewpointScorer`/`StateEncoder`/actions → core; `rl.py` + `PolicySelector`/`MixtureSelector` → pretrain; frozen-model policy-only config | **statistical parity** vs old RL repo: one seed of the default QReg recipe lands inside the 8-seed `qband` mean-CE band (t1–t4) |
| **P4** | Joint modes: `task_weight>0 ∧ rl_weight>0`, `policy_feats_detached` flag, ε-curriculum, distill-policy (doc §6.2) | DDP-correctness tests (incl. RunningNorm per-rank behavior); joint ADE run trains stably; distill+policy run trains stably |
| **P5** | CUDA IN1k task (doc §6.3) | linear-probe/fine-tune numbers sane vs canvit_eval's frozen-probe baseline |
| **P6** | Archival: mine RL repo docs (26 session docs, results, preserved-checkpoints) into `unification_docs/migrated/`; archive specialize + RL on GitHub; update session CLAUDE.md + git_status_all.sh | nothing references the archived repos; published HF policy checkpoints still load via core |

Risks inherited from unification-status §7 all stand; the two I re-emphasize:
P1's bit-for-bit gate is **the** protection for the 2M-step production recipe, and P3's
foveated-policy work is research, not engineering — budget accordingly.

## 7b. Testing tiers (available hardware)

1. **CPU smoke tests** — tiny configs, every task/patcher path; run anywhere (P0 onward).
2. **20 GB GPU (interactive)** — inference-scale only: load real checkpoints, single-batch
   rollouts, policy selection, foveated patching on GPU. Not sufficient for training.
3. **SLURM runs via the repo launchers** — anything real: the P1 bit-for-bit reference
   run, P2 probe reproduction, the P3 seed run. Prepared as launcher configs; submission
   is the user's call (session guardrail).

## 8. Parity criteria (what "nothing regressed" means)

- **Pretrain path (P1):** same seed + pinned data order ⇒ identical loss curves to float
  tolerance over ≥500 steps, before/after refactor. Any divergence is a bug, full stop.
- **ADE path (P2):** probe val-mIoU/CE within run-to-run noise of specialize's current
  exp22 ADE numbers on the same checkpoint.
- **RL path (P3):** re-tooling (wandb, config system, seeds) makes bit parity impossible;
  the gate is landing inside the existing 8-seed `qband` band (mean val CE t1–t4).
- **In-flight safety:** merge copies *from* specialize/RL — those repos stay untouched
  until P6. Pretrain edits are safe for pinned jobs by construction; specialize's
  (unpinned!) running ADE jobs are safe because specialize is never edited.

## 9. Knowledge preservation (P6 detail)

The RL repo's `docs/` (26 session docs, qband/head-band results, sweep post-mortems,
preserved checkpoints, the negative results on pathwise training) and specialize's
`docs/unification-status.md` are copied into `unification_docs/migrated/` **before**
archival. Hub-published policy checkpoints keep loading: `ViewpointScorer` in core
retains the HF mixin and (if needed) a repo-rename shim.
