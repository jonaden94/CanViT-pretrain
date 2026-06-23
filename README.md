# CanViT-pretrain

Passive-to-active dense latent distillation of [CanViT](https://github.com/m2b3/CanViT-PyTorch) ([arXiv:2603.22570](https://arxiv.org/abs/2603.22570)) from [DINOv3](https://github.com/facebookresearch/dinov3) ([arXiv:2508.10104](https://arxiv.org/abs/2508.10104)).

Originally designed to run on [the Nibi SLURM cluster](https://docs.alliancecan.ca/wiki/Nibi) using its [hosted ImageNet-21k `winter21_whole` replica](https://docs.alliancecan.ca/wiki/ImageNet).

## Setup

```bash
cp .envrc.example .envrc && direnv allow
# Edit .envrc to adapt to your environment.
```

Please ensure that `HF_TOKEN`, `COMET_API_KEY`, and `COMET_WORKSPACE` are set.

## Environment & local multi-repo setup

CanViT-pretrain is developed together with four sibling repos and depends on
`CanViT-PyTorch[fovi]`:

```
repos/
├── fovi/               # leaf — no internal deps
├── CanViT-PyTorch/     # depends on fovi
├── CanViT-specialize/  # depends on CanViT-PyTorch
├── CanViT-pretrain/    # this repo; depends on CanViT-PyTorch[fovi]
└── CanViT-eval/        # depends on CanViT-PyTorch[fovi] + CanViT-specialize
```

Each repo has its **own** uv-managed venv. Clone all five **as siblings in the
same parent folder**, then create the env:

```bash
uv sync   # dev venv (.venv); installs CanViT-PyTorch + fovi editable
```

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

On the Grete V100 partition, build the dedicated venv with
`bash _setup_v100/setup.sh` instead (creates `.venv-v100` from the lock file,
then overwrites torch with a V100-compatible cu126 build). It uses the same
committed `[tool.uv.sources]`, so it also picks up the local editable clones.

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

Export DINOv3 teacher features once:

```bash
uv run python scripts/build_shuffled_index.py \
  --image-root $IN21K_IMAGE_DIR --index-dir $INDEX_DIR --dataset in21k
sbatch --array=0-99%20 slurm/export_features.sh
```

Pretraining:

```bash
sbatch slurm/train.sbatch [--flag value ...]
```

Ablations:

```bash
bash slurm/ablations/baseline.sh
bash slurm/ablations/no-bptt.sh
# ...
```

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
