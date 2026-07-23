# Migrated docs (P6 knowledge preservation)

Verbatim snapshots of the docs from the repos being **archived** by the unification
(master plan §9), copied here on **2026-07-23** *before* archival so nothing is lost
when the source repos go read-only. These are historical records — not live docs; do
not edit them here. The live unification docs are the `../*.md` files and `../00-master-plan.md`.

## `rl-docs/` — from `CanViT-PyTorch-RL/docs/` (36 markdown files, verbatim)

The RL repo's empirical record and session history. Highlights the rest of the stack
still depends on:
- `qband_results.md` — the 8-seed QReg band (mean val CE t1–t4 = 0.6853 ± 0.0007) and
  the EG-C2F-c64 baseline (0.6949). This is the reference the unified policy trainer's
  P3 gate was validated against.
- `head_band_results.md`, `old_frontend_band_results.md` — earlier scorer-architecture bands.
- `preserved_checkpoints.md` — which policy checkpoints to keep (HF-published).
- `sweep_sets.md`, `dataset-findings.md`, `archive-synthesis.md`, `milestones.md`,
  `paper-tables.md`, `benchmarks.md` — sweeps, negative results, and paper material.
- `sessions/` — the per-session working docs (the bulk of the history).

## `specialize-docs/` — from `CanViT-specialize/docs/`

- `unification-status.md` — specialize's own status doc at the time of the merge.

## Provenance

Sources (at migration time, all on `main`):
- `github.com/m2b3/CanViT-PyTorch-RL` (upstream clone) → `rl-docs/`
- `CanViT-specialize` (jonaden94 fork) → `specialize-docs/`

The actual GitHub archival of those two repos is a separate, deliberate step (owner's call).
