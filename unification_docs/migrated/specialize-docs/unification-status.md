# CanViT-specialize vs CanViT-pretrain — divergence report & unification notes

**Written:** 2026-07-22 · **Audience:** whoever plans the pretrain/specialize unification
**Status of facts:** every claim below was checked against the working tree on 2026-07-22.
File:line references are given so you can re-verify. Code moves fast — **re-check before
acting on anything here**, especially the "broken" claims.

---

## 1. Purpose and scope

The stated goal is to eventually collapse `CanViT-pretrain` and `CanViT-specialize` into a
single training repo with as little duplicated code as possible. This document answers the
prerequisite question: **where does specialize actually stand relative to pretrain today, and
what will bite during a merge?**

It is deliberately *not* a merge design. It is the inventory you need before choosing a design.

Out of scope: the RL integration (`CanViT-PyTorch-RL`) — it is a separate decision, and its
own repo has a `CLAUDE.md` that governs work inside it.

---

## 2. TL;DR

**The two repos have not diverged by design. They diverged by neglect.** Specialize's training
code is frozen at roughly April–May 2026; pretrain's is July 2026; the shared core moved on
through July. Nearly everything in the catalogue below is pretrain gaining a capability and
specialize simply never being updated.

Three consequences that matter for planning:

1. **Specialize's ADE20K training was broken until today.** It died on the first forward call
   against the current core (`TypeError: ... unexpected keyword argument 'glimpse'`). Fixed
   2026-07-22, but this means "specialize works today" was not true, and any merge plan built
   on that assumption needs revisiting.
2. **Specialize has no foveated support at all.** Not "partial" — its training path is
   uniform-only, and it does not even declare the `[fovi]` extra. Pretrain's entire exp21/exp22
   line is foveated. This is the single biggest functional gap.
3. **The IN1k/TPU split is real** and is the one genuine dependency fault line
   (`torch_xla`, SPMD, TPU v6e). It is *not*, however, the main obstacle people assume —
   see §7.

---

## 3. Snapshot

| | CanViT-pretrain | CanViT-specialize |
|---|---|---|
| Python LOC (excl. venv) | ~8,700 | ~3,800 |
| HEAD (2026-07-22) | `fe24aa1` (2026-07-12) | `e53e27b` (2026-06-26) |
| Training code last touched | **2026-07-12** | **2026-05-06** (ADE), 2026-05-04 (IN1k) |
| Core dep | `canvit-pytorch[fovi]` | `canvit-pytorch` (**no fovi extra**) |
| Patchers supported | uniform, foveated, square | **uniform only** |
| Multi-GPU | DDP (`train/dist.py`, `loop.py:554`) | **none** (single-GPU) |
| Gradient discipline | TBPTT, `chunk_size`, per-chunk backward | ADE: backbone frozen, no grad through recurrence |
| Viewpoints | own `train/viewpoint.py` (+`random_fixation`) | delegates to `canvit_pytorch.policies` |
| Figures | **disk sink** (`viz/disk.py`) | **wandb images** (`tracker.log_image`) |
| Commit pinning | yes (`PRETRAIN/PYTORCH/FOVI_COMMIT` + `git archive`) | **none** |
| Launchers | `slurm_nhr/runs/` — 6 experiment groups | `exp01` + `jon_exp22_full_runs` (added today) |
| Test files | 6 | **1** (`tests/test_metrics.py`, no training tests) |

---

## 4. Root cause: specialize bypasses the core's stability façade

This is the most useful structural finding, and it explains today's breakage.

The core deliberately exposes **task wrappers** whose signatures were kept stable across the
patcher refactor:

- `CanViTForSemanticSegmentation.forward(*, glimpse, state, viewpoint)` — `segmentation/__init__.py:96`
- `CanViTForImageClassification.forward(*, glimpse, state, viewpoint)` — `classification/__init__.py:128`

Both still accept `glimpse=` and internally call `self.canvit(image=glimpse, ...)`. The uniform
patcher's docstring states this explicitly: the `glimpse_size_px=None` path exists to *"keep the
existing `clf(glimpse=...)` / `seg(glimpse=...)` call sites working unchanged"*.

Now look at which class each specialize path actually uses:

| path | model class used | hit by the API break? |
|---|---|---|
| IN1k (`gcp_in1k_clf_ft`) | `CanViTForImageClassification` (the wrapper) | **No** — façade held |
| ADE20K (`training/ade20k`) | `CanViTForPretrainingHFHub` (**the raw pretraining model**) | **Yes** — broke |

**ADE20K reaches past the task wrapper into the raw pretraining model, so it took the refactor
head-on while IN1k was insulated.** That is the actual root cause, not "old code."

*In fairness, there is a real reason:* ADE20K supports a `recon_normalized` feature type that
needs `predict_teacher_scene` — a **pretraining** head that `CanViTForSemanticSegmentation`
deliberately discards. So it can't switch to the wrapper for free. But note that
`canvit_eval`'s own ADE20K task *does* use the wrapper
(`from_pretrained_with_probe`, `tasks/ade20k_seg.py:121`) and only ever uses canvas features.

**Recommendation:** during unification, make the wrapper the single supported entry point for
task training, and treat `recon_normalized` as the one feature that needs an explicit escape
hatch. This converts a recurring breakage class into a maintained contract.

---

## 5. Divergence catalogue

Ordered by how much pain each will cause during a merge.

### 5.1 🔴 Foveated / square patcher support — absent in specialize

Pretrain has foveated awareness in **11 files**, including a dedicated `train/viz/foveated_plot.py`
and `train/test_viewpoint_scale.py`. Specialize has **zero** (the only two greps that hit are a
`"squish"` resize-mode string in `datasets/ade20k.py` and an assert added today).

Concretely missing in specialize:
- **Fixation-style viewpoints.** Pretrain has `Viewpoint.random_fixation` — for the foveated path
  scales are ignored (foveation always covers the full image) and only the fixation center
  matters. Specialize has no such concept.
- **The `FoveatedScaleConfig` surface** (`fixed` / `per_rollout` modes, `fixed_scale`,
  distribution, min/max) that exp22's fovi runs depend on
  (`--foveated-scale.fixed-scale 2.0`). Specialize has no equivalent knob.
- **Patcher-aware routing.** Pretrain branches on
  `is_foveated = cfg.patcher_name in ("foveated", "square")` (`train/step.py:128`) — note
  `"square"` is included, which was a fixed bug. Specialize never branches.
- **The `[fovi]` extra.** Specialize declares plain `canvit-pytorch` (`pyproject.toml:7`).
  (`import fovi` happens to succeed in `.venv-cu126` via the editable install, so this is a
  latent declaration bug rather than an immediate failure — but don't rely on it.)

**Merge impact:** you cannot train an ADE20K probe on any exp22 *fovi* checkpoint today. Since
the foveated models are the active research line, this is the gap that most limits the merged
repo's usefulness. It is also the largest chunk of genuinely new work.

> Guard added 2026-07-22: `features.py` now asserts `patcher_name == "uniform"`, so a foveated
> checkpoint fails loudly instead of silently **double-cropping** (pre-crop, then re-foveate).
> `canvit_eval` handles this correctly via its `consumes_full_image` branch — that is the
> reference implementation to copy.

### 5.2 🔴 Core API drift — ADE20K was broken (fixed today)

`extract_canvas_features` called `model(glimpse=..., state=..., viewpoint=...)`, but
`CanViTForPretraining.forward` is now keyword-only on `image=`. Job 15022968 died ~90s in.

Fixed by renaming the kwarg (`features.py:47`) — **the pre-crop stays**, because downstream apps
load with `glimpse_size_px=None`, which makes `UniformPatcher` treat its input as an
already-cropped glimpse (`patcher/uniform.py:50-55`). Verified: `canvas_hidden (B,32,32,1024)`,
`recon_normalized (B,32,32,768)`, probe logits `(B,150,32,32)`, all finite.

**Merge impact:** low now, but it is the canary. Specialize had **no test** that would have
caught this (§5.8), and it went unnoticed for ~3 months.

### 5.3 🟠 Gradient discipline / TBPTT — structurally different

- **Pretrain:** genuine TBPTT — `ChunkState`, `chunk_size`, per-chunk `backward()`, explicit
  `RecurrentState(canvas.detach(), recurrent_cls.detach())` at boundaries
  (`train/step.py:71-76, 290-311`). Variable-length, multi-branch rollouts.
- **Specialize/ADE:** backbone frozen (`requires_grad_(False)`, `train_canvit.py:196`),
  viewpoints precomputed, one averaged `backward()`. No gradient through the recurrence at all.
- **Specialize/IN1k:** full BPTT over its 4 glimpses, but glimpses are sampled **in the
  dataloader workers**, so it is structurally unable to be state-conditioned.

**Merge impact:** these are not variants of one loop — they are three different gradient
regimes. A unified driver must express all three, and must reproduce pretrain's chunked
behavior *bit-for-bit* or you regress the expensive production recipe. This is the single
highest-risk refactor in the whole project.

### 5.4 🟠 Viewpoint generation — three implementations

1. `canvit_pytorch/policies/` — `random_viewpoints`, `coarse_to_fine_viewpoints`, etc.
2. `canvit_pretrain/train/viewpoint.py` — own `Viewpoint` wrapper over `CoreViewpoint`, plus
   `PixelBox`, `ViewpointType`, `random_fixation`, `viewpoint_to_pixel_box`, and a documented
   L²-uniform safe-box scale sampler (`p(s) ∝ (1-s)`).
3. `canvit_specialize/training/utils.py` — thin `make_viewpoints` delegating to (1).

Note pretrain's sampler is **not** the same distribution as core's — it deliberately weights by
safe-box geometry. Merging must pick one and be explicit about which distribution wins, because
this silently changes what every run trains on.

**Merge impact:** medium. Real duplication, but the pieces are small and well documented.

### 5.5 🟡 Visualization — opposite directions

You already removed wandb figure logging from pretrain; specialize still has it.

- **Pretrain:** figures go to **disk**, `run_dir/visualization/{subdir}/step-N.png`
  (`viz/disk.py`), whose docstring notes it *"cleans up matplotlib state aggressively to prevent
  the leaks that the previous wandb-backed `log_figure()` had to defend against."*
- **Specialize:** `tracker.log_image` → `wandb.Image`, and since wandb has no native curve
  logging, `log_curve` **renders a PNG and uploads that too** (`ade20k/tracker.py:59-71`).
  `log_viz` fires every `viz_every=500` steps for **both** train and val splits.

Also: specialize's `viz.py` has **no patcher awareness whatsoever**, so it never received the
foveated-plot treatment pretrain has. It silently assumes uniform glimpses.

**Merge impact:** low technically, but pick one policy deliberately — otherwise the merged repo
re-introduces the matplotlib/wandb leak pretrain already escaped.

### 5.6 🟡 Commit pinning — absent in specialize

Pretrain's `slurm_nhr/base_train.sbatch` pins `PRETRAIN_COMMIT` / `PYTORCH_COMMIT` /
`FOVI_COMMIT` and `git archive`s a frozen snapshot into `$TMPDIR`, so in-flight jobs are immune
to later edits. **Specialize has none of this** — its jobs run straight off the editable
working tree.

**Merge impact:** this cuts *for* merging. A merged repo means one commit hash pins a whole run,
which simplifies the 3-hash machinery. But note the corollary: **the reproducibility guarantee
you rely on for pretrain does not currently exist for any specialize run**, including the
ADE20K job launched today.

### 5.7 🟡 Scale-out — DDP vs single-GPU

Pretrain: NCCL + `DistributedDataParallel` (`dist.py:69`, `loop.py:554`), and the DDP
constraint visibly shapes `step.py` (heads run inside `forward` so the Reducer sees them).
Specialize: **no DDP anywhere**.

**Merge impact:** a shared loop must be DDP-correct, which is a stronger constraint than
anything specialize's code currently satisfies. Porting ADE20K *into* a DDP-aware loop is
mostly free; the reverse is not.

### 5.8 🟡 Test coverage — 6 files vs 1

Pretrain tests: `train/test.py`, `test_step.py`, `test_viewpoint_scale.py`, `viz/test.py`,
`checkpoint/test.py`, `datasets/test_indexed_image_folder.py`.
Specialize: `tests/test_metrics.py` only — **nothing covering training**.

**Merge impact:** this is why §5.2 went unnoticed for three months. Before refactoring anything,
add a CPU smoke test that runs a 2-step ADE20K rollout end-to-end; it would have caught the
break in seconds and will catch the next one.

### 5.9 🟢 Launcher hygiene (fixed today)

`slurm_nhr/exp01/train_ade20k_canvit.sbatch` hardcodes `VENV=".venv"`, which does not exist
(only `.venv-cu126`) — it would exit immediately. The new
`slurm_nhr/jon_exp22_full_runs/train_ade20k_uniform16_best.sbatch` uses the correct venv.
The exp01 script is still stale.

### 5.10 🟢 Probe → eval handoff (fixed today)

Specialize trains a probe to a raw `.pt`; `canvit_eval` loads via
`SegmentationProbe.from_pretrained` (a `PyTorchModelHubMixin`). The only bridge was
`scripts/push_probes.py`, which **only publishes to the HF Hub**. Added
`scripts/probe_ckpt_to_hf_format.py` — a local converter (`save_pretrained`, no network),
verified with an offline round-trip (`HF_HUB_OFFLINE=1`, bit-identical weights).

---

## 6. What is actually duplicated vs actually different

This is the calculus that should drive the merge design.

**Genuinely duplicated (unify these):**
- rollout loop / state threading / viewpoint plumbing
- checkpoint save-resume, scheduler/warmup, tracker plumbing
- per-timestep validation and metric aggregation
- SLURM launcher scaffolding and (should-be) commit pinning

**Genuinely different (keep behind an interface, do not force together):**
- **objective** — DINOv3 feature-regression MSE vs labeled CE
- **data** — in21k webdataset + teacher vs ADE20K / IN1k TFRecords
- **backbone trainability** — pretrain trains it; ADE freezes it
- **gradient regime** — chunked TBPTT vs frozen vs full-BPTT (§5.3)
- **compute target** — CUDA/SLURM vs TPU/XLA (IN1k only)

The genuinely-different list is short and mostly *data + loss*, which is exactly the shape that
factors cleanly behind a small task interface. The hard part is §5.3, not the packaging.

---

## 7. Merge risk register

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Refactoring pretrain's chunked TBPTT loop regresses the 2M-step production recipe | **High** | Require bit-for-bit equivalence vs a current run *before* enabling anything new. Pin and diff. |
| 2 | Foveated support must be built for the task path, not just ported | **High** | Copy `canvit_eval`'s `consumes_full_image` routing; it already solves this correctly. |
| 3 | Viewpoint distribution silently changes (§5.4) | Medium | Choose one sampler explicitly; assert the distribution in a test. |
| 4 | No specialize tests to catch regressions | Medium | Add the CPU rollout smoke test first (§5.8). |
| 5 | `torch_xla`/TPU deps collide with the CUDA stack | Low–Medium | See below — smaller than it looks. |
| 6 | Specialize runs are not commit-pinned, so "what code produced this probe" is unrecoverable | Medium | Adopt pinning at merge time; treat pre-merge probes as provenance-weak. |

**On risk 5 (the CUDA-vs-XLA argument).** This was treated in the design chat as the main reason
not to fully merge. It is real — IN1k is genuinely PyTorch/XLA SPMD on TPU v6e, with `torch_xla`
imports and a SkyPilot launcher — but I'd weight it lower than the chat did:

- It is confined to **one subpackage** (`training/gcp_in1k_clf_ft/`), which shares almost nothing
  with ADE20K beyond the model class.
- It is the **only** path with that dependency, and it targets GCP, not your NHR cluster.
- It is insulated from the drift precisely *because* nobody touches it (§4).

So the fault line is **IN1k**, not pretrain-vs-specialize. The clean cut is: merge
**pretrain + ADE20K** (both CUDA/SLURM, both actively developed), and leave IN1k/TPU isolated as
its own extra, subdir, or repo. That gets ~all the anti-duplication win without the dependency
union, and it means the CUDA-vs-XLA objection should **not** block the merge you actually want.

---

## 8. Suggested sequencing

Ordered so that each step is independently useful and reversible:

1. **Stop the bleeding.** Add a CPU smoke test for the ADE20K rollout (§5.8) and adopt commit
   pinning for specialize launchers (§5.6). Cheap; prevents recurrence of §5.2.
2. **Close the wrapper gap.** Move ADE20K onto `CanViTForSemanticSegmentation` where possible,
   with an explicit escape hatch for `recon_normalized` (§4). Converts a breakage class into a
   contract.
3. **Add foveated support to the task path** (§5.1), copying `canvit_eval`'s routing. This is
   the biggest *capability* win and is independent of any merge.
4. **Only then** unify the loop — and make bit-for-bit reproduction of a pinned pretrain run the
   acceptance criterion (risk 1).
5. Decide IN1k's fate separately (§7). It does not need to be resolved before steps 1–4.

Steps 1–3 are worth doing **whether or not** you ever merge, which is the main argument for
front-loading them: they are not merge-contingent, and they de-risk step 4.

---

## 9. Open questions

- **Is the IN1k/TPU path still live?** If dormant, §7 gets much simpler and it may be droppable.
  It has been untouched since 2026-05-04 and is insulated only by neglect.
- **Should `recon_normalized` survive?** It is the only reason ADE20K uses the raw pretraining
  model. If it is unused in practice, the wrapper migration (§4) becomes trivial.
- **Which viewpoint distribution is canonical** — core's or pretrain's safe-box sampler (§5.4)?
- **Do you want probe training to stay uniform-only** in the interim, or is foveated ADE20K a
  blocker for the current research line?

---

## Appendix — how to re-verify

```bash
# drift: when was each side last touched?
git -C CanViT-pretrain   log -1 --date=short --format='%ad %h' -- canvit_pretrain/train/step.py
git -C CanViT-specialize log -1 --date=short --format='%ad %h' -- canvit_specialize/training/ade20k/

# foveated awareness, per repo
grep -rl "foveated\|square\|patcher_name" CanViT-pretrain/canvit_pretrain   --include=*.py | wc -l
grep -rl "foveated\|square\|patcher_name" CanViT-specialize/canvit_specialize --include=*.py | wc -l

# does the task wrapper still expose the stable façade?
grep -n -A3 "def forward" CanViT-PyTorch/canvit_pytorch/model/segmentation/__init__.py

# ADE20K rollout smoke test (CPU, ~1 min) — the test that should exist
#   see scripts/probe_ckpt_to_hf_format.py header for the local probe→eval loop
```
