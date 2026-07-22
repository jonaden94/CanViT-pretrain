# P0 — smoke tests + parity baseline (DONE 2026-07-22)

## What was added

1. **Uniform-sampler distribution test** — `canvit_pretrain/train/test_viewpoint_scale.py::
   test_named_random_uniform_branch_safebox_law`. The foveated branch was already pinned by
   the existing tests in that file; this adds the missing uniform branch (`Viewpoint.random`:
   scale bounds, `p(s) ∝ (1-s)` skew, per-sample safe-box center coupling). Both canonical
   per-patcher distributions (master plan §3) are now asserted.
2. **ADE20K CPU smoke tests** — `CanViT-specialize/tests/test_ade20k_smoke.py` (2 tests):
   - 2-step rollout end-to-end on a tiny locally-constructed `CanViTForPretrainingHFHub`
     (vits16, G=8, CPU): random viewpoints → `extract_canvas_features` → `SegmentationProbe`
     → `ce_loss` → backward; asserts head gets gradients, frozen backbone gets none.
     This is the test whose absence let the `glimpse=` breakage go unnoticed for 3 months
     (unification-status §5.2/§5.8).
   - The foveated double-crop guard fails loudly (`AssertionError` mentioning "uniform").
3. **Parity probe** — `unification_docs/parity_probe.py`: 25 deterministic CPU
   `training_step` calls (tiny model, synthetic batches, pinned `random` + torch seeds,
   `torch.use_deterministic_algorithms(True)`), records every loss as full-precision hex to
   `unification_docs/parity/record_<rev>.json`. **This is the P1 bit-for-bit gate**: after
   any loop refactor, re-run and byte-diff. Config exercises both branch types
   (`n_full_start_branches=1, n_random_start_branches=1`), TBPTT `chunk_size=2`, geometric
   trajectory lengths (`continue_prob=0.5`) — the full structural surface of `training_step`.

## Baseline state (all green, pre-refactor)

| repo | suite | result |
|---|---|---|
| CanViT-pretrain | full (`pytest canvit_pretrain`) | **61 passed** (incl. new test) |
| CanViT-PyTorch-RL | full (`pytest`) | **24 passed** |
| CanViT-specialize | `pytest tests` | **9 passed** (incl. 2 new) |

Parity baseline: `parity/record_fe24aa1-dirty.json`, loss-stream
`sha256[:16] = 9a0100a1a3de3acd`. Verified deterministic (two runs, identical digest).
Note the `-dirty` suffix: the working tree contained the P0 test additions (additive
only — no training-path file was modified; the probe exercises the same `training_step`
a clean `fe24aa1` checkout would). Re-record on a clean commit once P0 lands in git.

## Deviations from the plan

- "Policy ckpt compat asserts" (P0 row) **moved to P3**: the compatibility metadata
  (feature groups / action space / canvas grid / patcher) doesn't exist in checkpoints
  yet — the asserts are part of the new checkpoint format introduced when the policy
  stack moves to core, and can't be written against the old format meaningfully.
- Distill-path smoke tests were NOT newly written: pretrain's existing `test_step.py`
  already covers 1/2/3-glimpse gradients, chunk-boundary detach semantics, foveated
  scale modes, and same-seed determinism on CPU. Existing coverage was better than
  unification-status §5.8's framing suggested (its gap claim was about specialize,
  which is now closed).

## P1 acceptance restated (now concrete)

Refactor `training_step`/`loop.py` behind the Selector/Task seams with everything
default-off, then: `parity_probe.py` digest must equal `9a0100a1a3de3acd` (or the
clean-commit re-record), and all three suites stay green.
