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
- **Foveated view-scale → HF `config.json` (the footgun).** The mechanism EXISTS —
  the pretraining HF export (`canvit_pytorch/model/pretraining/hub/__init__.py`) already
  merges an `extra_metadata` dict into `config.json` as `"metadata"`. But a real fix
  spans: (1) the checkpoint→HF **converter** writing the pretraining `foveated_scale`
  into that metadata — and the converter still lives in the to-be-archived
  `CanViT-specialize/scripts/pretrain_ckpt_to_hf_format.py`, so it should be **ported
  into the unified repo first** (itself a P6 task); (2) `canvit_eval` **reading** the
  metadata to auto-set the view-scale (else the footgun persists). This intersects the
  converter-relocation + archival decisions → left for the owner. Interim mitigation is
  already in place: `Ade20kConfig`/`In1kConfig`/`PolicyTrainConfig` all take an explicit
  `foveated_scale`, and the ADE probe logs a loud warning naming the scale.
- **Update the session `CLAUDE.md`** (mark specialize/RL archived, four-repo world) —
  it's the owner's operating doc + depends on the archival actually happening.
- **Archive `specialize` + `CanViT-PyTorch-RL` on GitHub** — outward/irreversible; owner's
  explicit call. Prereqs before doing it: converter ported out of specialize; docs mined
  (done); nothing imports the archived packages at runtime (specialize is still imported by
  `canvit_eval` — see below).

## Note surfaced during P6

`canvit_eval` still imports `canvit_specialize` (e.g. `from canvit_specialize.training.utils
import collect_metadata` in `tasks/in1k_clf.py`). So **specialize cannot be archived until
canvit_eval's dependency on it is removed** (port `collect_metadata` + any other used bits
into core/eval). This is a concrete archival prerequisite.
