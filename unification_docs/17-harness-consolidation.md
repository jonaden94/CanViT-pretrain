# 17 — The harness consolidation: one entry point (2026-07-31)

The big-bang cutover docs 00/07 gated on the owner's green light. This is it, scoped to
what the goal actually needed: **delete the duplicate entry points so a newcomer sees one
interface.** Owner's constraint: remove nothing we still need, and *report* anything
about to be removed that the harness had not absorbed.

`python -m canvit_train.harness.run <task> --preset <preset>` is now the only trainer
in the repo.

## Deleted

| file | lines | was |
|---|---|---|
| `train/__main__.py`, `train/loop.py`, `train/step.py` | 1,492 | the distill trainer |
| `train/test_step.py`, `test_joint.py`, `test_seams.py` | 727 | tests OF that trainer |
| `ade20k/__main__.py`, `ade20k/train.py` | 271 | the standalone ADE20K probe |
| `ade20k/rl_train.py`, `ade20k/test_rl_train.py` | 618 | the ported CanViT-PyTorch-RL policy reference |
| `in1k/__main__.py`, `in1k/train.py` | 346 | the standalone IN1k trainer |

`train/` itself STAYS — `config`, `viewpoint`, `selector`, `rl`, `joint`, `task`, `ema`,
`dist`, `tracker`, `scheduler`, `probe`, `data/`, `viz/` are the harness's own substrate
(~15 import sites). It is a library now, not a trainer; its `__init__` docstring says so.

The deleted trainers' coverage maps onto harness tests that already existed:
`test_step.py`→`harness/tests/test_bptt_chunking.py` + `test_rollout_engine.py`,
`test_joint.py`→`tasks/tests/test_task_rollout.py` + `harness/tests/test_vpg.py`,
`test_seams.py`→`test_rollout_engine.py`.

## Moved, not lost

`in1k/train.py` held two functions the harness *imported from it* — deliberately, so the
two entry points could not diverge on head init (that sharing was the fix for the
random-head finetune bug, `8f780ba`). Extracted verbatim:

- `build_classifier` + `_resolve_probe_repo` → **`in1k/model.py`**
- `evaluate` + `_policy_rollout_cls` → **`in1k/eval.py`**

## Two things the harness had NOT absorbed

**1. IN1k HF-format classifier export — FIXED.** `in1k/train.py` wrote
`clf.save_pretrained(run_dir/"best-hf")`, and that directory is what
`CanViT-eval/canvit_eval/tasks/in1k_clf.py:68` loads. The harness never called
`save_pretrained` anywhere, and `to_hf` handled only pretraining checkpoints — so this was
a **pre-existing gap**, not one the deletion created: the exp25 finetunes (82.4 / 81.9 /
84.0 top-1) had no path into canvit_eval. `to_hf` now auto-detects an in1k checkpoint
(`model_config["task"] == "in1k"`) and emits the classifier layout, rebuilding the module
and reusing the class's own `save_pretrained` so the layout cannot drift from what
`from_pretrained` expects. `from_pretrained_with_new_head` is a safe architecture-only
reconstruction for BOTH modes — finetune trained from a probe-fused head, but the
architecture either constructor yields is identical and every weight is loaded over.

**2. ADE20K train-mIoU logging — STILL OPEN, owner's call.** The standalone logged
`train_miou_mean` (accumulated over each `log_every` window) and a `best_val_miou_t{t}`
series. The harness logs eval mIoU only. An earlier estimate of "~15 lines via the
existing `glimpse_metrics` hook" was **wrong**: that hook receives only the scalar
`TaskLoss.combined`, so it cannot see predictions, and mIoU is a ratio of *accumulated*
counts — averaging per-batch mIoU is a different quantity. A faithful port needs a
window-accumulator seam plus an on-log reset hook the engine does not have. Not done;
train loss (every `log_every`) and per-timestep val mIoU (every `val_every`) are unaffected.

Verified safe to drop: the legacy probe checkpoint format
(`canvas_hidden_best_t*_miou*.pt`, keys `probe_state_dict` / `best_mious_per_t`) has
**zero readers** in all three repos — `ade20k/train.py` was both its only writer and its
only mention. Existing files stay loadable by hand.

## Guards preserved by re-pointing at UPSTREAM

Three live tests pinned the harness against our own *port*. Since the port is gone, they
now cite `CanViT-PyTorch-RL` itself — strictly stronger, because a mis-transcription of
the reference is now detectable where comparing against our port never could be:

- `ade20k/test_reward_ce_shared.py` — was `import ce_from_logits from rl_train` (which by
  then was a one-line delegation to `reward_ce`, i.e. asserting a wrapper equals what it
  wraps). Now transcribes `canvas_ops.py::ce_from_logits` + `scoring.py::per_image_ce`
  inline and asserts bit-identity. **Passes.**
- `harness/tests/test_rl_recipe_knobs.py` — constants now cited to
  `canvit_pytorch_rl/training/config.py` (lr 2e-4 :81, wd 1e-2 :85, betas .9/.95 :87-88,
  warmup_frac 0.125 :92). The squish test now pins *our* center_crop default and defers to
  the not-band-comparable WARNING test, which is the real guard.
- `ade20k/test_ade20k.py` — gained `test_square_patcher_also_routes_the_full_image`.
  "Square counts as foveated" was a real bug and its only same-seed check lived in
  `parity_configs.py`, which retired here.

## Retired scaffolding (and the one honest cost)

`parity_probe.py` was **re-pointed at the harness** rather than retired: it drives
`run_rollout` through the same distill adapter as `test_rollout_parity.py` and still
prints `9a0100a1a3de3acd`, so the digest stays regenerable and diffable per commit.
**Cost, stated plainly:** it can no longer be re-derived from the original
implementation, because that implementation is gone. The recorded constant in
`test_rollout_parity.py` is now the sole reference — the trade docs 00/07 sanctioned,
conditional on the harness reproducing it byte-for-byte first, which it did and does.

Genuinely un-repointable, because their whole purpose was comparing two implementations:
`parity_configs.py`, `harness_realdata_ab.py`, `setup_arg_parity.py` +
`tasks/tests/test_setup_arg_parity.py`. That last one caught two real bugs (the `or 512`
normalizer sentinel; the `teacher_dim` placeholder) — but it is a *static old-vs-new source
diff*, and all three of its pairs lose their old side. This is not a safety regression: it
guarded two implementations drifting apart, which is the hazard consolidation removes.

Also retired (one-shot exp27 forensics that imported `rl_train`): `eval_equivalence.py`,
`diff_training_trace.py`, `diff_training_multistep.py`, `diff_optimizer_path.py`,
`diff_data_pipeline.py`, `measure_miou_order.py`. Their findings are recorded in doc 15.

## Unrelated hardening done in the same pass (owner-approved)

`--rl.objective vpg` with `select_bn_eval=True` now **raises** instead of warning. The knob
defaults True (it reproduces the qband checkpoints for QReg/PG, which do not sample), so
a bare `--rl.objective vpg` was silently getting the biased off-policy REINFORCE gradient
with only a log line to say so. Opting in is one explicit flag; there is no legitimate
reason to want the biased gradient. Pinned by
`test_vpg_refuses_the_biased_select_bn_eval_default`.

## Launchers: left in place ON PURPOSE

166 scripts reference `slurm/archive/base_train.sbatch`. `base_train.sbatch`,
`slurm/{ade20k,in1k}/`, and the `*-oldloop*.sh` / `policy-{bneval,oldloop,pooled}-s0.sh`
runs invoke entry points that no longer exist — and still work, because each pins
pre-consolidation commits that offline `git archive` restores into the job's `TMPDIR`. That
is exactly how the exp22/exp23/exp27 comparisons stay reproducible. Moving them would break
166 paths for no functional gain, so they stay and the README labels them historical.

## Verification

- full suite green (count in the commit message); `parity_probe.py` prints
  `9a0100a1a3de3acd`
- `capability_matrix.md` **regenerates byte-identical** — no task's capabilities or
  preset→spec resolution moved
- `default_spec()` for all three tasks was already verified byte-identical to `8f780ba`
  before this pass
