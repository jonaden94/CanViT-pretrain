# P6 — archival & knowledge preservation (IN PROGRESS 2026-07-23)

## Done

- **Docs mined** into `unification_docs/migrated/` (commit `bd7efc1`): verbatim
  `CanViT-PyTorch-RL/docs/` (36 md incl. the 26 session docs, qband/head bands,
  preserved-checkpoints, sweeps) + specialize's `unification-status.md`, with a
  provenance `README.md`. Knowledge is preserved ahead of archival.
- **`git_status_all.sh`** (session root) now covers all **six** repos (added
  `CanViT-PyTorch-RL`). Not committed — the session root isn't a git repo.
- **mIoU in policy deploy eval** (commit `845e401`, listed under P3 loose ends but
  done now): `rl_train.evaluate` → `EvalResult(ce_mean, miou_per_t)`.

## Done 2026-07-23 (session 2 — post-lunch)

- **eval → specialize dependency REMOVED (archival prereq cleared).** Owner chose
  *shared-primitives-to-core* over an eval→pretrain dep. The ~6 shared val-protocol
  symbols now live in **core**: `canvit_pytorch/data/ade20k.py` (`ADE20kDataset`,
  `make_val_transforms`, `ResizeMode`, `IGNORE_LABEL`, `NUM_CLASSES`; ImageNet mean/std
  inlined so core carries no timm dep) and `canvit_pytorch/metrics.py` (`mIoUAccumulator`).
  pretrain's `ade20k/{config,data,metrics}.py` re-export from core (train pipeline stays
  in pretrain); eval's 5 task files import from core; `collect_metadata` → eval-local
  `canvit_eval/run_metadata.py`. eval's `pyproject.toml` drops `canvit-specialize`
  (uv.lock −469 lines: dinov3/comet-ml/submitit orphans gone → eval got *leaner*, torch
  cu128 intact). Tests: core 41, eval 22, pretrain 87; parity `9a0100a1a3de3acd` unchanged.
  **specialize now has no runtime importers.**
- **Foveated view-scale footgun FIXED (write + read).** Write: converter ported into the
  unified repo as `canvit_pretrain/checkpoint/to_hf.py` (`python -m`, replaces the
  archived specialize script) and now records an explicit `metadata.pretrain_view_scale`
  ({patcher_name, mode, fixed_scale, …}) extracted from `training_config_history`
  (`None` for uniform / older ckpts). Read: eval `config.resolve_view_scale` +
  pure `resolve_scale_from_metadata` auto-set the eval view-scale (fixed foveated/square
  → pin; multi-scale/uniform/none → policy scales; explicit `override_scale` always wins),
  wired once at `runner.eval_batches`. **No-op for every pre-fix repo** (no metadata field),
  so it's safe without GPU. Unit-tested (6 converter + 7 resolver CPU tests). Forward-looking:
  existing prod repos benefit only after a re-push (`python -m ...to_hf` → HF), an owner action.

## Deliberately NOT done autonomously (need a decision or GPU — flagged for the owner)

- **Port `baselines.figure4b` → canvit_eval.** On inspection, `figure4b.py` is a
  *plotting / paper-comparison* script (reads `runs/*/manifest.json` + `summary.json`,
  overlays on `paper_reference.py`'s Table-4 mIoU/CI), **not** a baseline *measurer* —
  the measuring is done by `policy.eval` episodes over the baseline policies
  (coarse_to_fine / entropy_coarse_to_fine / random / …), which `canvit_eval`'s episode
  machinery already runs. So the real port is: (a) confirm canvit_eval can run the EG-C2F
  policy, (b) bring over `paper_reference.py` + a figure4b-style overlay adapted to
  canvit_eval's output format. Deferred: it's an analysis/plotting port tangled with
  canvit_eval's run-output format and only exercisable on GPU (running eval on a checkpoint).
- **Update the session `CLAUDE.md`** (mark specialize/RL archived, four-repo world) —
  it's the owner's operating doc + depends on the archival actually happening.
- **Archive `specialize` + `CanViT-PyTorch-RL` on GitHub** — outward/irreversible; owner's
  explicit call. Prereqs before doing it: converter ported out of specialize; docs mined
  (done); nothing imports the archived packages at runtime (specialize is still imported by
  `canvit_eval` — see below).

## Note surfaced during P6 — RESOLVED 2026-07-23

`canvit_eval` used to import `canvit_specialize` (`collect_metadata`, the ADE20K
dataset/transforms, `mIoUAccumulator`). That was the last runtime dependency on
specialize and a hard archival prerequisite. **Now removed** (see "Done" above:
shared primitives → core, `collect_metadata` → eval-local). Nothing imports the
archived packages at runtime anymore → specialize + RL are archival-ready pending
the owner's explicit go (and the RL repo's own runtime-importer check).
