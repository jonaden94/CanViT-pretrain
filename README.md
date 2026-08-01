# CanViT-train

All training for [CanViT](https://github.com/m2b3/CanViT-PyTorch)
([arXiv:2603.22570](https://arxiv.org/abs/2603.22570)) — a recurrent dual-stream
vision transformer that builds up a persistent *canvas* representation from a
sequence of glimpses.

Three training objectives, one unified framework and one entry point:

| task | what it trains | data |
|---|---|---|
| `distill` | passive→active dense latent distillation from [DINOv3](https://github.com/facebookresearch/dinov3) ([arXiv:2508.10104](https://arxiv.org/abs/2508.10104)) — the pretraining objective | ImageNet-21k WebDataset shards + a validation image folder |
| `ade20k` | ADE20K semantic-segmentation probe / finetune | ADE20K (`$ADE20K_ROOT`) |
| `in1k` | ImageNet-1k linear probe / full finetune | ImageNet-1k WebDataset + validation image folder |

On top of any of the three, the *viewpoint-selection policy* — the network that
decides where the model looks next — can be trained by reinforcement learning,
either alone against a frozen model or jointly with the task.

## Repository layout

CanViT-train is one of four repos that make up the project:

```
repos/
├── fovi/               # foveated-vision library: cortical-magnification patch geometry
├── CanViT-PyTorch/     # the model: canvas ViT, patchers, HF-hub model classes
├── CanViT-train/       # this repo: all training
└── CanViT-eval/        # evaluation / benchmarking and analysis notebooks
```

The dependency direction is `fovi` → `CanViT-PyTorch` → {`CanViT-train`,
`CanViT-eval`}. Each repo has its **own** uv-managed virtual environment.

## Setup

Clone the repos **as siblings in one parent folder**, then create an environment:

```bash
# Default env (.venv) — CUDA-13.x torch, sm_90 (H100) and newer
uv sync

# Alternative env (.venv-cu126) — cu126 torch, keeps sm_70 (V100) support
UV_PROJECT_ENVIRONMENT=.venv-cu126 uv sync --no-group cuda --group cu126
```

The two are **conflicting, separately-locked resolutions** of the same project:
torch is pinned in the `cuda` (default) and `cu126` dependency groups in
`pyproject.toml`, so each `uv sync` is fully reproducible. The cu126 wheels still
ship sm_70 kernels, which the CUDA-13.x wheels dropped. Both share the same
`[tool.uv.sources]`.

`[tool.uv.sources]` links the siblings as a **relative-path editable install**
(`canvit-pytorch = { path = "../CanViT-PyTorch", editable = true }`; `fovi`
arrives transitively via `canvit-pytorch[fovi]`, also editable). Relative paths
resolve on any machine as long as the repos are siblings, and the editable
installs mean edits in the local `CanViT-PyTorch` / `fovi` clones take effect
immediately. To install *without* the siblings present, point that entry at the
remote instead — `canvit-pytorch = { git = "https://github.com/jonaden94/CanViT-PyTorch.git" }`
— and `uv sync`.

### Environment variables

Training needs the dataset roots, the cache directories, and a place to write run
artifacts. **For compute project `nib00021` on the Grete cluster (GWDG, Göttingen)
these are already filled in, in `.envrc.grete`** — its paths are that project's
storage, so they apply to that project and not to Grete in general. `slurm/env.sh`
sources the file directly (along with the GWDG proxy that compute nodes need for
HuggingFace Hub access), so submitting a run needs no `direnv` and no path hunting.
The launchers likewise hardcode `#SBATCH -A nib00021` and the `grete:shared`
partition; change both if you belong to a different project.

For **interactive** work on a login node — running the tests, publishing a
checkpoint, inspecting one, or launching training outside SLURM — nothing sources
that file for you, so load it into your shell. With
[direnv](https://direnv.net/), which auto-loads `.envrc` on entering the
directory:

```bash
cp .envrc.grete .envrc && direnv allow   # project nib00021 on Grete
cp .envrc.example .envrc && direnv allow # elsewhere, then edit for your machine
```

Or just `source .envrc.grete` in the shell you are working in.

The variables that matter:

| variable | used for |
|---|---|
| `LOGS_DIR` | root for run artifacts: `$LOGS_DIR/<run_group>/<run_name>/` |
| `WEBDATASET_DIR`, `VAL_DIR` | `distill` training shards and validation images |
| `VAL_INDEX_DIR` | cache for the validation-image index (the directory scan runs once, then is reused) |
| `ADE20K_ROOT` | ADE20K `ADEChallengeData2016` directory |
| `IN1K_TRAIN_DIR`, `IN1K_VAL_DIR` | ImageNet-1k shards and validation images |
| `HF_HOME`, `HF_TOKEN` | HuggingFace cache and access (teacher weights, pretrained CanViT and probe checkpoints) |
| `WANDB_DIR` | where Weights & Biases writes its run files |
| `WANDB_PROJECT`, `WANDB_ENTITY` | fallback project / entity for the tracker |

In `.envrc.grete` the dataset paths are **shared**: they live under the project
directory (`/mnt/vast-nhr/projects/nib00021`) or the project's lustre workspace,
both group-readable, so any member of the project uses them unchanged. The caches
derive from `$HOME`, so they are already per-user.

#### First-time setup for a new member of project `nib00021`

Two things are personal and must be set up once:

1. **Point `LOGS_DIR` and `WANDB_DIR` at a directory you own.** These are run
   *outputs*, and the defaults in `.envrc.grete` are group-readable but writable
   only by their owner. The project root `/mnt/vast-nhr/projects/nib00021` *is*
   group-writable, so create your own subdirectory there and point both at it.
2. **Log in to the two services**, whose credentials are per-user:

   ```bash
   hf auth login   # writes $HF_HOME/token, which .envrc.grete picks up as HF_TOKEN
   wandb login
   ```

Nothing else needs adapting: with an empty `WANDB_ENTITY` runs land in your own
W&B account, and every dataset path is already shared.

`distill` requires `WEBDATASET_DIR` and `VAL_DIR`. The `ade20k` root comes from
`ADE20K_ROOT`, and the `in1k` roots fall back to a shared path when
`IN1K_TRAIN_DIR` / `IN1K_VAL_DIR` are unset — so outside project `nib00021`, set
all of them.

Metrics go to Weights & Biases by default; `--cfg.tracker none` is also possible.

## Datasets

`distill` and `in1k` read [WebDataset](https://github.com/webdataset/webdataset)
tar shards, which are assumed to have been built already and are pointed at with
`--cfg.webdataset-dir`. Distillation shards come in two flavours and both are
handled automatically:

- **with precomputed features** — each sample carries the DINOv3 teacher's `cls`
  and dense patch targets, so no teacher forward is needed at train time;
- **raw images** — a frozen DINOv3 teacher is loaded and produces the targets on
  the fly, both for the target standardizers and for every batch.

ADE20K is read from the extracted `ADEChallengeData2016` directory
(`$ADE20K_ROOT`), and ImageNet-1k validation from a standard class-per-folder
image directory.

## Package layout

One rule: **`harness/` holds the entry point and everything shared by more than
one task; each task folder holds only what is specific to that task.**

```
canvit_train/
├── harness/          the entry point + every shared primitive
│   ├── run.py        process entry point
│   ├── cli.py        tyro CLI; task × --preset → TrainSpec
│   ├── loop.py       training loop: cadence, validation, checkpointing
│   ├── spec.py       TrainSpec / BpttSpec / GroupOptim + validation
│   ├── config.py     config every task shares (foveated view-scale law, RL recipe)
│   ├── rollout/      batch → glimpse sequence: engine, viewpoint, selector, eval viewpoints
│   ├── policy/       learned viewpoint policy: builder, joint task+policy, RL objectives
│   ├── optim/        optimizer construction, LR schedules, EMA
│   ├── infra/        checkpoint I/O, best-metric tracker, DDP, SLURM shard schedule, utils
│   └── viz/          task-agnostic PCA / figure-writing / metric leaves
├── distill/          DINOv3 latent distillation: loss, student model, probe, data/, viz/
├── ade20k/           ADE20K segmentation: data, metrics, rollout, viz
├── in1k/             ImageNet-1k: data, metrics, model, eval, rollout
└── checkpoint/       checkpoint I/O and `to_hf` — publishing to the HF-hub layout
```

The five flat files in `harness/` are the ones to read first; the subpackages are
the machinery. Every task folder has the same shape: `task.py` (the adapter the
framework calls — build model, build loaders, evaluate, visualize), `config.py`
(that task's knobs), plus its own data / metrics / rollout helpers.

## Running a training job

**There is one training entry point.** Every task and every configuration goes
through it:

```bash
python -m canvit_train.harness.run <task> --preset <preset> [--cfg.* ...] [--opts.* ...]
```

`<task>` is `distill`, `ade20k`, or `in1k`. `--preset` picks *what trains*,
orthogonally to the task:

| preset | trains |
|---|---|
| `default` | the task's own recipe (its tuned LR schedule) |
| `probe` | head only, backbone frozen |
| `finetune` | backbone + head |
| `policy_only` | the viewpoint-selection policy only, everything else frozen |
| `joint` | task and policy together |

Not every combination is meaningful (`distill` has no head, so `--preset probe`
is refused). **`unification_docs/capability_matrix.md` is generated from the live
task objects** and lists exactly which task/preset combinations exist and what
each resolves to — read it rather than guessing, and regenerate it with
`unification_docs/capability_matrix.py` after changing a `default_spec`.

`--cfg.*` are the task's own knobs (see the `Config` dataclass in the task's
`config.py`); `--opts.*` are the framework's (run identity, cadence, checkpoint
paths). `--help` on any task prints the full, documented flag surface.

### Run artifacts

Each run writes everything under one directory derived from its identity:

```
$LOGS_DIR/<run_group>/<run_name>/
├── checkpoints/      step-<N>.pt, best.pt, latest.pt (symlink to the newest)
├── log/              SLURM job logs
└── visualization/    figures, written locally and never uploaded
```

Start-up mode is decided by priority **resume > seed > fresh**. `--opts.resume`
continues a run with its optimizer, scheduler and data schedule intact (the
default for `distill`, whose array tasks must chain; off by default for the
downstream tasks). Seeding instead loads *weights only* and starts a new run at
step 0 with a fresh optimizer and schedule: `--cfg.seed-ckpt` from a local
checkpoint or `--cfg.hf-seed-ckpt` from the HuggingFace Hub for `distill`, and
`--cfg.model-repo` for `ade20k` / `in1k`, which always start from a published
pretrained model. With neither, training starts from scratch.

### On SLURM

`slurm/harness_train.sbatch` runs any task on one node with one or more GPUs.
All variation comes from environment variables, so the same script serves every
run:

| variable | meaning |
|---|---|
| `TASK` | `distill` \| `ade20k` \| `in1k` |
| `RUN_GROUP`, `RUN_NAME` | run identity → the artifact directory and the tracker run name |
| `NGPU` | GPUs per node (`ade20k` is single-GPU only) |
| `CFG_FOO_BAR=v` | becomes `--cfg.foo-bar v` |
| `OPT_FOO_BAR=v` | becomes `--opts.foo-bar v` |
| `EXTRA_ARGS` | verbatim extra flags — the only way to reach nested config trees such as `--cfg.model.patcher-name foveated`, since `FOO_BAR` cannot encode a dot |

Jobs run in `.venv-cu126`, and the launcher fails immediately with the `uv sync`
command to run if that environment is missing.

**Set the W&B project per run, not globally.** `$WANDB_PROJECT` is only a
fallback; each launcher sets `CFG_WANDB_PROJECT` to its own run group, so a
campaign's runs stay together and separate from everything else:

```bash
CFG_WANDB_PROJECT=my_ablation_study   # set in slurm/runs/<group>/*.sh
```

Note the W&B entity may be a **shared** one. If so, pick a project name nobody
else is using, and keep figures off the tracker — the framework writes them to
`run_dir/visualization/` on local disk rather than uploading them.

Long runs are split into a SLURM **array**, each task resuming where the last
left off. The array is a *budget*, not a schedule: the step count advances only
for tasks that succeed, so a failed task costs steps rather than skipping data.

To launch a run, copy an existing launcher from `slurm/runs/<group>/` and edit
it. Each is a small self-documenting shell script that sets the variables above
and calls `sbatch`.

### Pinning code for long runs

A training run can take days, during which the local clones may keep changing —
but a single run must use **one** fixed version of the code. The launchers
therefore pin each repo to an exact commit:

```bash
TRAIN_COMMIT=<sha>     # CanViT-train
PYTORCH_COMMIT=<sha>   # CanViT-PyTorch
FOVI_COMMIT=<sha>      # fovi
```

`harness_train.sbatch` extracts those commits with an offline `git archive`
(local object store only — no network, no SSH, works for private repos) into the
job's `TMPDIR` and prepends them to `PYTHONPATH` with `PYTHONSAFEPATH=1`, so the
snapshot **overrides** the editable install for that job. A submitted job is
therefore immune to later edits or pulls of the clones. The three variables are
optional and independent; omit them to use the environment's editable install.

`TRAIN_COMMIT` was called `PRETRAIN_COMMIT` before this repo was renamed, and
`harness_train.sbatch` still accepts that spelling — around 48 launchers under
`slurm/` use it to reproduce older experiments, and dropping it would leave them
running the editable install with no error at all.

## Publishing a checkpoint

Conversion to the HuggingFace-hub layout is always explicit, never automatic:

```bash
python -m canvit_train.checkpoint.to_hf --pt-path <run>/checkpoints/best.pt --out-dir <dir>
```

It detects the checkpoint type: a `distill` checkpoint becomes the pretraining
layout (`CanViTForPretrainingHFHub`), an `in1k` one the classifier layout that
CanViT-eval loads.

## Tests

```bash
.venv/bin/python -m pytest .
```

The suite is CPU-only and covers the rollout engine, the specification
resolution, the RL objectives, each task's adapter, and checkpoint round-trips.
It includes **digest tests** that hash a short training run's loss stream and
parameter fingerprint, so any change that perturbs training numerics fails
loudly instead of silently.

## Further documentation

[`readme_docs/`](readme_docs/) holds procedures for specific training campaigns —
what they train, how to launch them, and how to judge the results:

- [`readme_docs/verification_runs.md`](readme_docs/verification_runs.md) — the
  exp32–exp35 campaign: pretraining with a learning-rate drop, ImageNet-1k
  finetunes, ADE20K probes, and 10 seeds of viewpoint-policy training, each
  checked against an established expected result.
- [`readme_docs/q_policy_foveated.md`](readme_docs/q_policy_foveated.md) —
  training the Q viewpoint policy for a **foveated** model, and the three
  settings that fail silently if they do not match the backbone/probe pair.
  Not yet run end to end: every Q-policy result so far is on a uniform
  backbone.

`unification_docs/` holds design notes and the generated capability matrix.

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
