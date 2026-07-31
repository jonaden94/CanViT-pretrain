# CanViT-train

All training for [CanViT](https://github.com/m2b3/CanViT-PyTorch) ([arXiv:2603.22570](https://arxiv.org/abs/2603.22570)), behind one harness: passive-to-active dense latent distillation from [DINOv3](https://github.com/facebookresearch/dinov3) ([arXiv:2508.10104](https://arxiv.org/abs/2508.10104)) (`distill`), ADE20K segmentation and ImageNet-1k probes/finetunes (`ade20k`, `in1k`), and reinforcement learning of the viewpoint-selection policy on any of them.

> Formerly `CanViT-pretrain`. Renamed 2026-07-31: downstream training (from `CanViT-specialize`) and viewpoint-policy RL (from `CanViT-PyTorch-RL`) were merged in, so "pretrain" no longer described the scope.

Originally designed to run on [the Nibi SLURM cluster](https://docs.alliancecan.ca/wiki/Nibi) using its [hosted ImageNet-21k `winter21_whole` replica](https://docs.alliancecan.ca/wiki/ImageNet).

## Setup

```bash
cp .envrc.example .envrc && direnv allow
# Edit .envrc to adapt to your environment.
```

Please ensure that `HF_TOKEN`, `COMET_API_KEY`, and `COMET_WORKSPACE` are set.

## Environment & local multi-repo setup

CanViT-train is developed together with four sibling repos and depends on
`CanViT-PyTorch[fovi]`:

```
repos/
├── fovi/               # leaf — no internal deps
├── CanViT-PyTorch/     # depends on fovi
├── CanViT-specialize/  # depends on CanViT-PyTorch
├── CanViT-train/       # this repo; depends on CanViT-PyTorch[fovi]
└── CanViT-eval/        # depends on CanViT-PyTorch[fovi]
```

Each repo has its **own** uv-managed venv. Clone all five **as siblings in the
same parent folder**, then create the env:

```bash
# Default env (.venv) — H100 (sm_90), CUDA-13.x torch
uv sync

# V100 + A100 env (.venv-cu126) — cu126 torch (Grete V100 + A100 partitions)
UV_PROJECT_ENVIRONMENT=.venv-cu126 uv sync --no-group cuda --group cu126
```

The two envs are **conflicting, separately-locked resolutions** of the same
project: torch is pinned in the `cuda` (default) and `cu126` dependency groups in
`pyproject.toml`, so each `uv sync` is fully reproducible. cu126 wheels still
include sm_70 (V100) support, which the default CUDA-13.x wheels dropped. Both
envs share the same `[tool.uv.sources]`.

The cross-repo link is committed in `pyproject.toml` under `[tool.uv.sources]`
as a **relative-path editable install** (`canvit-pytorch = { path =
"../CanViT-PyTorch", editable = true }`; `fovi` comes in transitively, also
editable). Relative paths resolve on any machine as long as the repos are
siblings, and the editable installs mean edits in the local `CanViT-PyTorch` /
`fovi` clones are picked up immediately — no reinstall, no manual
`uv pip install -e`. To install without the siblings present, swap that line back
to the remote fork
(`canvit-pytorch = { git = "https://github.com/jonaden94/CanViT-PyTorch.git" }`)
and `uv sync`.

### Pinning code for long runs

A training run can take days, during which you may keep editing the local clones
— but a single run should use **one** fixed version of the code. The run scripts
therefore pin each repo to an exact commit:

```bash
# in slurm_nhr/runs/<group>/<run>.sh
PRETRAIN_COMMIT=bc2db02
PYTORCH_COMMIT=d864b83
FOVI_COMMIT=763bf7a
```

`slurm_nhr/base_train.sbatch` extracts those commits from the local clones with
offline `git archive` (reads the local object store only — no network, no SSH,
works with private repos) into the job's `TMPDIR` and prepends them to
`PYTHONPATH` (with `PYTHONSAFEPATH=1`), so they **override** the venv's editable
install for that job. The array job is thus snapshotted against the pinned
commits and is unaffected by any later edits or `git pull` on the originals while
it runs. The three vars are optional and independent; omit them to fall back to
the venv's (editable, local) install. See
`slurm_nhr/runs/jon_exp21_modulation/*.sh` for examples.

## Run

**There is one training entry point.** Every task and every training configuration goes
through it:

```bash
python -m canvit_train.harness.run <task> --preset <preset> [--cfg.* ...] [--opts.* ...]
```

| `<task>` | what it trains | data |
|---|---|---|
| `distill` | passive→active dense latent distillation from DINOv3 (the pretraining objective) | IN21k webdataset shards |
| `ade20k` | ADE20K semantic-segmentation probe / finetune | ADE20K (`$ADE20K_ROOT`) |
| `in1k` | ImageNet-1k linear probe / full finetune | IN1k webdataset + val ImageFolder |

`--preset` picks *what trains*, orthogonally to the task:

| preset | trains |
|---|---|
| `default` | the task's own recipe (task-tuned LR schedule) |
| `probe` | head only, backbone frozen |
| `finetune` | backbone + head |
| `policy_only` | the viewpoint-selection policy only, everything else frozen |
| `joint` | task + policy together |

Not every cell is meaningful (`distill` has no head, so `--preset probe` is refused).
**`unification_docs/capability_matrix.md` is generated from the live task objects** and
lists exactly which task/preset combinations exist and what spec each resolves to — read
it rather than guessing, and regenerate it with
`unification_docs/capability_matrix.py` after touching a `default_spec`.

On SLURM, use `slurm_nhr/harness_train.sbatch` and copy an existing launcher from
`slurm_nhr/runs/<group>/` as your template.

The in21k feature webdataset is already built (`$WEBDATASET_DIR`), so nothing below is
needed for normal runs. To build one from scratch:

```bash
uv run python scripts/build_shuffled_index.py \
  --image-root $IN21K_IMAGE_DIR --index-dir $INDEX_DIR --dataset in21k
uv run python scripts/export_in21k_features.py --help   # then drive it as an array job
```

The SLURM wrapper that used to drive the export (`slurm/export_features.sh`) was deleted
2026-07-31 with the rest of the Nibi-cluster tooling; the Python script it called remains.
Write a `slurm_nhr`-style launcher for it if you need to re-export.

Publish a trained checkpoint to the local HF layout (never automatic — always explicit):

```bash
python -m canvit_train.checkpoint.to_hf --pt-path <run>/checkpoints/best.pt --out-dir <dir>
```

It detects the checkpoint type: a `distill` checkpoint becomes the pretraining layout
(`CanViTForPretrainingHFHub`), an `in1k` one becomes the classifier layout that
CanViT-eval's `in1k_clf` task loads.

### Historical launchers — do not use for new work

`slurm_nhr/base_train.sbatch`, `slurm_nhr/ade20k/`, `slurm_nhr/in1k/`, and the
`*-oldloop*.sh` / `policy-{bneval,oldloop,pooled}-s0.sh` run scripts drive entry points
that **no longer exist in this repo** (`canvit_pretrain.train`, `canvit_pretrain.ade20k`,
`canvit_pretrain.in1k`, `canvit_pretrain.ade20k.rl_train`). They still run correctly,
because each pins pre-consolidation commits that `git archive` restores into the job's
`TMPDIR` — that is how the exp22/exp23/exp27 comparisons stay reproducible. Keep them for
reproduction; start new work from a `harness_train.sbatch` launcher.

**Those `canvit_pretrain` module paths are deliberate, not stale.** The package was
renamed `canvit_pretrain` → `canvit_train` on 2026-07-31, but a launcher pinning a
pre-rename commit gets a snapshot whose package is still `canvit_pretrain` — so that is
the only name under which those entry points exist. Do not "fix" them.
`harness_train.sbatch` is shared between new runs and pre-rename pinned arms (exp23,
exp27 B/D/E), so it *detects* which package the snapshot contains and dispatches
accordingly; see `_PKG` in that file.

## Citation

```bibtex
@article{berreby2026canvit,
  title={CanViT: Toward Active-Vision Foundation Models},
  author={Berreby, Yoha{\"i}-Eliel and Du, Sabrina and Durand, Audrey and Krishna, B. Suresh},
  year={2026},
  eprint={2603.22570},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2603.22570}
}
```

## License

MIT. See [LICENSE](LICENSE) for details.
