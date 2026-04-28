# WebDataset Loader + Multi-GPU/Multi-Node DDP + Pluggable Experiment Tracker

This document describes three additions on top of the original single-GPU,
sharded-features, Comet-only training pipeline.

## Goal

The original repo supported only one training data path (precomputed `.pt`
feature shards via `canvit_pretrain/train/data/shards.py::ShardedFeatureLoader`,
plus raw images for validation), only single-GPU execution, and only Comet ML
for experiment tracking.

Three orthogonal additions were made:

1. **WebDataset loader** for a different on-disk format —
   `<dir>/{train-shuffled,val}/shard-NNNNNN.tar` with `info.json` metadata,
   per-sample tar entries `jpg` / `json` (label) / `cls.npy` / `ptch.npy`,
   pre-shuffled at dataset-creation time.
2. **DDP** training over multiple GPUs and multiple nodes on the Grete
   SLURM cluster (`SLURM_PROCID` / `WORLD_SIZE` / `MASTER_ADDR` / `MASTER_PORT`).
3. **Pluggable experiment tracker** — `cfg.tracker = "comet" | "wandb" | "none"`.
   The training loop calls `exp.log_*` against a thin `Tracker` wrapper that
   fans out to whichever backend is active. Selecting `"none"` makes every
   log call a no-op, which is also what non-main DDP ranks get.

Guiding principle for all three: change as little of the existing code as
possible. The existing `ShardedFeatureLoader` path keeps working byte-for-byte
unchanged on single GPU; the new data path activates only when
`cfg.webdataset_dir` is set; DDP activates only when `WORLD_SIZE > 1`; the
tracker swap is a single dispatch in `loop.py`.

## Design overview

Four independent new modules:

1. **`canvit_pretrain/train/dist.py`** — DDP singleton.
   * `init_dist()` reads SLURM env vars, sets the CUDA device to `local_rank`,
     and calls `dist.init_process_group("nccl")`. Idempotent.
   * Also exposes `is_main`, `rank`, `world_size`, `local_rank`, `device`,
     `barrier`, `broadcast_module_buffers`, `all_reduce_mean`.
   * **Single-process fall-back:** when `WORLD_SIZE` is unset or 1, every
     helper returns rank=0 / world_size=1 / `is_dist()=False`. Single-GPU
     paths stay unchanged.

2. **`canvit_pretrain/train/data/schedule.py`** — shard schedule helpers.
   * `compute_shards_per_gpu(steps_per_job, batch_size, samples_per_shard)`
     enforces the divisibility constraint
     `steps_per_job * batch_size % samples_per_shard == 0`.
   * `load_schedule(path)` reads the precomputed `shard_schedule.npz` (numpy
     arrays for `shards`, `meta_keys`, `meta_vals`).
   * `slice_for_job(schedule, job_index, shards_per_gpu, world_size, rank)`
     returns the rank's per-job slice of shard paths.

3. **`canvit_pretrain/train/data/webdataset.py`** — train/val loaders.
   * `WebDatasetTrainLoader` matches the existing `ShardedFeatureLoader`'s
     `.next() -> (images, raw_patches, raw_cls, labels)` contract — `loop.py`
     does not need to change at the call site.
   * `WebDatasetValLoader` matches `InfiniteLoader.next_batch_with_labels()`
     and cycles forever (so the loop can call it once per validation step).
   * `init_normalizer_stats_from_tar(shard_path, scene_norm, cls_norm,
     device, max_samples)` is the WebDataset analogue of
     `init_normalizer_stats_from_shard`. It reads `.cls.npy` and `.ptch.npy`
     entries directly from one tar via the stdlib `tarfile` module and calls
     `set_stats` exactly the same way as the legacy path.

4. **`canvit_pretrain/train/tracker.py`** — pluggable experiment tracker.
   * `Tracker` class holds an *optional* Comet experiment and an *optional*
     wandb run. Each `log_*` method fans out to whichever backend is set;
     when both are `None`, every method is a silent no-op.
   * `make_tracker(...)` factory builds the right `Tracker` for the active
     job: `tracker="comet"` → real Comet experiment; `tracker="wandb"` →
     real `wandb.Run`; `tracker="none"` or non-main rank → empty `Tracker()`.
   * `Tracker.get_comet_id()` / `Tracker.get_wandb_id()` return the
     resume-key for whichever backend is active; both are stored separately
     in the checkpoint so resumes pick up on the right backend.

Standalone scripts:

* **`scripts/build_shard_schedule.py`** — reads `info.json`, drops the partial
  last shard, builds `n_epochs` permutations with
  `np.random.default_rng(seed=cfg.seed)`, writes `shard_schedule.npz` to
  `<webdataset_dir>/train-shuffled/shard_schedule.npz`. Refuses to overwrite
  without `--force`.

SLURM:

* **`slurm_jonathan/train_ddp.sbatch`** — multi-node template (sets
  `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE` per the
  `plan_dataloading/ddp/minimal_distributed_training.sh` reference; uses
  `srun uv run python -m canvit_pretrain.train`). Existing
  `slurm_jonathan/train.sbatch` retained; both pass `--wandb-project /
  --wandb-entity / --wandb-dir` from env vars.

## Decode pipeline

Inside `_build_pipeline`:

```python
ds = wds.WebDataset(shards, shardshuffle=False, empty_check=False)
if use_worker_split:
    ds = ds.compose(wds.split_by_worker)
ds = (
    ds.to_tuple("jpg", "json", "cls.npy", "ptch.npy")
      .map_tuple(_decode_jpg, _decode_label, _decode_npy_fp16, _decode_npy_fp16)
      .batched(batch_size, partial=False)
)
```

* `shardshuffle=False` — schedule already provides global shuffle, no in-memory
  shuffle buffer (which would blur shard boundaries and break clean resume).
* `_decode_jpg` applies `canvit_pytorch.preprocess.preprocess(scene_resolution)`
  to match the existing convention.
* fp16 features stay fp16 in CPU; cast to fp32 only at GPU transfer time
  (`loop.py`'s existing `load_train_batch` does this).

DataLoader settings: `num_workers = shards_per_gpu` (one shard per worker),
`persistent_workers=False`, `pin_memory=True`, `prefetch_factor=2`.

**No `split_by_node` is used inside WebDataset.** Instead, each rank receives
a pre-sliced list of shard paths from `slice_for_job`. This is equivalent to
`split_by_node` but more explicit about resume semantics.

## Sharding & job-array semantics

The schedule is a flat list of shard paths produced by tiling per-epoch
permutations of the training shards (excluding the partial last shard). Each
job consumes `shards_per_job = shards_per_gpu * world_size` contiguous shards
from the schedule, starting at offset `job_index * shards_per_job`.

```
shards_per_gpu = (steps_per_job * batch_size_per_gpu) // samples_per_shard
shards_per_job = shards_per_gpu * world_size
start          = job_index * shards_per_job
shards         = schedule[start : start + shards_per_job]
# rank r within the job gets shards[r*shards_per_gpu : (r+1)*shards_per_gpu]
```

`compute_shards_per_gpu` asserts the divisibility constraint at startup with a
clear error message.

The constructor also asserts
`samples_per_shard % batch_size_per_gpu == 0` so each worker yields a clean
number of batches (no `partial=False` drops).

## Job-index resume protocol

* Each end-of-job checkpoint stores `job_index = start_job_index` (the index of
  the job that wrote it).
* On the WebDataset path, the next job derives:
  `start_job_index = ckpt["job_index"] + 1` (or 0 if no checkpoint or
  seed-mode).
* `start_step = start_job_index * steps_per_job`. The scheduler's
  `last_epoch` is restored from the checkpoint and should agree, but the
  WebDataset path treats `job_index` as the single source of truth for which
  shards to feed.
* For SIGUSR1 mid-job saves, the saved `job_index` equals
  `start_job_index` of the running job. If a job is killed mid-stream and the
  SIGUSR1 save was the last save, the next job re-runs that job's shards.
  Acceptable corner case (slightly wasteful, never inconsistent).
* The sharded-features path (legacy) does not use `job_index` and continues to
  derive `start_step` from `scheduler.last_epoch`.

## Checkpoint integration

`CheckpointData` (TypedDict in `canvit_pretrain/checkpoint/__init__.py`)
gained two new fields:

```python
job_index: int | None       # WebDataset shard cursor (see above)
wandb_run_id: str | None    # mirror of comet_id for the wandb backend
```

* `save(...)` accepts `job_index: int | None = None` and
  `wandb_run_id: str | None = None`. The sharded-features path passes
  `job_index=None`; the WebDataset path passes the running job's index.
  Whichever tracker is active fills its own ID slot via
  `exp.get_comet_id()` / `exp.get_wandb_id()` (the inactive one is `None`).
* `load(...)` reads `raw.get("job_index")` and `raw.get("wandb_run_id")`
  (NOT asserted) so existing pre-change checkpoints load without error.

## Standardizer initialisation under DDP

* Rank 0 runs the appropriate `init_normalizer_stats_from_*` to fill the
  standardizers' buffers from a single shard.
* DDP's constructor (`broadcast_buffers=True` default) then broadcasts those
  buffers from rank 0 to all other ranks when `model = DDP(model, ...)` is
  constructed.
* This avoids a separate explicit broadcast and keeps the dispatch simple:
  one call site for the per-path tar/`.pt` choice, then one barrier.

## DDP wrap order

`compile-then-DDP`:

```python
compile_model(model)        # in-place model.compile()
load_state_dict_flexible(model, weights_to_load)
core_model = model
if ddp.is_dist():
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[ddp.local_rank()],
        find_unused_parameters=False,
    )
```

* `core_model` keeps a direct reference to the unwrapped CanViTForPretraining
  for state-dict access and validate-time forward.
* The DDP-wrapped `model` is passed into `training_step(...)`. DDP's
  `__getattr__` proxy transparently forwards method calls (`model.predict_*`,
  `model.init_state`) to the inner module, while `model(...)` calls go through
  DDP's forward to set up the reducer for backward-time all-reduce.
* `find_unused_parameters=False` is the default — every CanViT parameter is
  reached every step. If a future change introduces conditionally-unused
  params, switch to `True`.

## Validation under DDP

Each rank's `WebDatasetValLoader` reads its non-overlapping slice of val
shards (round-robin partition). The existing `validate(...)` function runs on
every rank, called with the unwrapped `core_model` so DDP forward is
bypassed (validation is forward-only; no all-reduce hook setup needed).

Per-rank metrics are all-reduced via `ddp.all_reduce_mean` at training-log
time:

```python
metrics = {
    f"train/{k}": ddp.all_reduce_mean(v).item()
    for k, v in ema.items()
}
```

A `ddp.barrier()` after each validation block keeps non-main ranks (which run
validate but log into a no-op `Tracker`) synchronised with rank 0.

## Rank-0 guards in loop.py

Factored through `ddp.is_main()`:

* Tracker creation. Non-main ranks pass `is_main=False` to `make_tracker(...)`,
  which returns a no-op `Tracker()` (both backends `None`). All
  `exp.log_parameters` / `exp.log_metric` / `exp.log_metrics` /
  `exp.log_image` / `exp.log_curve` / `exp.get_comet_id` /
  `exp.get_wandb_id` / `exp.end()` calls in the loop stay unconditional.
* `tqdm(..., disable=not ddp.is_main())` for the progress bar.
* Both `save_checkpoint(...)` + `update_symlink(...)` call sites (SIGUSR1 and
  end-of-job). Followed by `ddp.barrier()`.
* `failed_marker.write_text(...)` and `cancel_slurm_array()` on the crash
  path, plus the `run_dir.mkdir()` on first run.
* Per-module grad norms use `core_model` (so names don't get the `module.`
  DDP prefix).

## Tracker abstraction

The training loop never imports `comet_ml` or `wandb` directly. Instead it
imports `make_tracker` from `canvit_pretrain.train.tracker` and dispatches
on `cfg.tracker`:

```python
exp = make_tracker(
    tracker=cfg.tracker,                  # "comet" | "wandb" | "none"
    is_main=ddp.is_main(),
    is_seeding=is_seeding,
    run_name=run_name,
    wandb_project=cfg.wandb_project,
    wandb_entity=cfg.wandb_entity,
    wandb_dir=cfg.wandb_dir,
    prev_comet_id=ckpt_data["comet_id"]    if ckpt_data else None,
    prev_wandb_id=ckpt_data.get("wandb_run_id") if ckpt_data else None,
)
```

The returned object is the only logging surface the loop ever touches. Its
methods mirror the comet-ml `Experiment` surface that the loop and `viz/*`
historically depended on:

| Method | Comet behavior | wandb behavior |
|---|---|---|
| `log_parameters(dict)` | `experiment.log_parameters(dict)` | `run.config.update(dict, allow_val_change=True)` |
| `log_metric(name, value, step=)` | native | `run.log({name: value}, step=step)` |
| `log_metrics(dict, step=)` | native | `run.log(dict, step=step)` |
| `log_image(image, name=, step=)` | native (accepts PIL) | `run.log({name: wandb.Image(image)}, step=step)` |
| `log_curve(name, x=, y=, step=)` | native (`exp.log_curve`) | render line plot via matplotlib → `wandb.Image` (no native equivalent) |
| `get_comet_id()` / `get_wandb_id()` | wraps `get_key()` | wraps `run.id` |
| `end()` | `experiment.end()` | `run.finish()` |

Two backend-specific notes:

* **Curves on wandb** are emitted as PNG images via `_curve_as_pil` in
  `tracker.py` — wandb has no native `log_curve` analogue. This means each
  curve update lands as a new image (versioned by step), same as PCA figures.
  The 900-curve budget enforced in `viz/comet.py::log_curve` still applies
  uniformly.
* **Figure handling.** `viz/comet.py::log_figure` now converts the matplotlib
  figure to a `PIL.Image` once (single decode), then hands it to
  `Tracker.log_image`. Both backends accept PIL natively, so there is no
  per-backend `isinstance` plumbing inside `Tracker`.

### Resume semantics

* The checkpoint stores both `comet_id` and `wandb_run_id` independently. On
  load, `make_tracker` is given both. It uses whichever ID matches the
  active backend; the other is ignored.
* If you switch backends across resumes (e.g. saved with `tracker=comet`,
  resume with `tracker=wandb`), `make_tracker` simply starts a fresh wandb
  run and logs `"Creating NEW wandb run"`. No error, no stale ID dereference.
* SEED mode (`--seed-ckpt`) always starts a new tracker run regardless of
  the IDs in the seed checkpoint.

## Configuration

Seven new fields in `canvit_pretrain/train/config.py` (three from the
WebDataset path, four from the tracker path):

```python
# WebDataset
webdataset_dir: Path | None = None
shard_schedule_path: Path | None = None  # default: <webdataset_dir>/train-shuffled/shard_schedule.npz
seed: int = 0

# Tracker
tracker: Literal["comet", "wandb", "none"] = "wandb"
wandb_project: str | None = None
wandb_entity: str | None = None
wandb_dir: Path | None = Path("/mnt/vast-nhr/projects/nib00021/jonathan")
```

Setting `webdataset_dir` activates the new data path. The runtime asserts
that exactly one of `feature_base_dir` / `webdataset_dir` is set.

`tracker` defaults to `"wandb"` (the migration target). Pass `--tracker comet`
to use Comet; `--tracker none` to disable logging entirely (useful for smoke
tests). When `tracker="wandb"`, `wandb_project` is required and asserted at
startup.

## File-by-file summary

### New

| File | Purpose |
|---|---|
| `canvit_pretrain/train/dist.py` | DDP singleton. |
| `canvit_pretrain/train/data/schedule.py` | `load_schedule`, `slice_for_job`, `compute_shards_per_gpu`. |
| `canvit_pretrain/train/data/webdataset.py` | `WebDatasetTrainLoader`, `WebDatasetValLoader`, `init_normalizer_stats_from_tar`. |
| `canvit_pretrain/train/tracker.py` | `Tracker` class + `make_tracker` factory (Comet / wandb / none). |
| `scripts/build_shard_schedule.py` | Standalone schedule builder. |
| `slurm_jonathan/train_ddp.sbatch` | Multi-node SLURM template. |
| `claude_docs/webdataset-ddp-and-tracker.md` | This document. |

### Modified (additive, backward-compatible)

| File | Change |
|---|---|
| `pyproject.toml` | Added `webdataset` and `wandb` dependencies (kept `comet-ml`). |
| `.envrc.grete` | Added `WANDB_PROJECT` / `WANDB_ENTITY` / `WANDB_DIR` plus `~/wandb_api_key.txt` loader. |
| `canvit_pretrain/train/config.py` | Added `webdataset_dir`, `shard_schedule_path`, `seed`, `tracker`, `wandb_project`, `wandb_entity`, `wandb_dir`. |
| `canvit_pretrain/train/data/__init__.py` | Wider `Loaders` types; `create_loaders` accepts `job_index`/`world_size`/`rank`; dispatches on `cfg.webdataset_dir`; new `_create_webdataset_loaders` helper. |
| `canvit_pretrain/checkpoint/__init__.py` | `job_index` and `wandb_run_id` fields on `CheckpointData`; matching `save` kwargs; `load()` uses `.get()` for both (backward compat). |
| `canvit_pretrain/train/loop.py` | `ddp.init_dist()`; `cfg.device` override; rank guards on FAILED marker / mkdir / tracker / tqdm / save / symlink; normalizer-init dispatch; DDP wrap (compile-then-DDP); `core_model` separation; `start_job_index` threading; metric all-reduce on log; `grad_norms_by_module(core_model, ...)`; `make_tracker(...)` replaces the inline Comet block; both checkpoint saves pass `comet_id` and `wandb_run_id`. |
| `canvit_pretrain/train/viz/comet.py` | `log_figure` converts to PIL once (works for both backends); type hints retargeted to `Tracker`. |
| `canvit_pretrain/train/viz/validate.py` | Type hint retargeted to `Tracker`; dropped unused `comet_ml` import. |
| `slurm_jonathan/train.sbatch` | Pass `--wandb-project` / optional `--wandb-entity` / `--wandb-dir` from env vars. |
| `slurm_jonathan/train_ddp.sbatch` | Same wandb flags as legacy script. |

### Untouched

`scripts/bench_dataloader.py` continues to use `ShardedFeatureLoader`
directly. No changes to `step.py`, `viz/{plot,pca,sample,image,metrics}.py`,
`model.py`, `scheduler.py`, `probe.py`, `ema.py`, `viewpoint.py`, or any
test.

## How to run

### Build the shard schedule once per dataset (WebDataset path only)

```bash
uv run python scripts/build_shard_schedule.py \
    --webdataset-dir "$WEBDATASET_DIR" \
    --seed 0
# writes $WEBDATASET_DIR/train-shuffled/shard_schedule.npz
```

### Single-GPU WebDataset (no DDP)

```bash
uv run python -m canvit_pretrain.train \
    --webdataset-dir "$WEBDATASET_DIR" \
    --steps-per-job 16 --batch-size 256 \
    --ckpt-dir /tmp/canvit-test \
    --tracker wandb --wandb-project canvit-pretrain
```

`WORLD_SIZE` is unset → `init_dist()` is a no-op → single-process behaviour.

### Multi-GPU / multi-node DDP

```bash
sbatch slurm_jonathan/train_ddp.sbatch
# multi-node:
sbatch --nodes=2 --ntasks-per-node=2 --gpus-per-node=A100:2 \
    slurm_jonathan/train_ddp.sbatch
```

The sbatch template sets `MASTER_ADDR`, `MASTER_PORT`, and `WORLD_SIZE`, then
launches one process per GPU via `srun`. Each process calls `init_dist()`,
which reads `SLURM_PROCID` / `SLURM_GPUS_ON_NODE` / `WORLD_SIZE` to compute
`(rank, local_rank, device)` and joins the NCCL process group. Tracker flags
(`--wandb-project`, `--wandb-entity`, `--wandb-dir`) are passed through from
the env vars defined in `.envrc.grete`.

### Legacy sharded-features path (unchanged data; tracker still selectable)

```bash
sbatch slurm_jonathan/train.sbatch                        # default: tracker=wandb
sbatch slurm_jonathan/train.sbatch --tracker comet        # opt back into Comet
sbatch slurm_jonathan/train.sbatch --tracker none         # no logging at all
```

`webdataset_dir` is unset → existing data behaviour preserved.

## Risk notes & known caveats

* **`torch.compile` × DDP order**: compile-then-wrap is the documented
  PyTorch 2.x recommendation but has had bugs historically. If you see DDP
  hooks failing to fire under compile, swap to wrap-then-compile (one-line
  change in `loop.py`).
* **`persistent_workers` + WebDataset**: explicitly disabled. Workers hold
  open tar streams; persistence does not compose cleanly across our finite
  per-job iteration. Re-enabling needs explicit verification.
* **Run-name with multi-rank fresh runs**: `cfg.run_name` defaults to a
  per-rank `datetime.now()` if not provided. Always pass `--run-name` under
  DDP (the SLURM template does this from `SLURM_ARRAY_JOB_ID`). If absent,
  ranks would land in different `run_dir`s.
* **Standardizer broadcast** relies on DDP's default `broadcast_buffers=True`
  at construction time. Do not change this default unless you also add an
  explicit `broadcast_module_buffers` call after rank-0 init.
* **Last (partial) shard** is excluded from training by
  `build_shard_schedule.py`. The val side keeps it (every val sample
  evaluated).
* **`has_features=False`** (decode `jpg` only and run the teacher on-the-fly)
  is intentionally **not implemented** in v1. The training loader asserts
  `"cls.npy" in info["keys"]` at startup. Adding the on-the-fly path later is
  small: skip the cls/ptch decoding and call `compute_raw_targets()` inside
  `load_train_batch` (the closure already exists in the loop).
* **wandb requires `WANDB_API_KEY`** in the environment (the `.envrc.grete`
  loader sources it from `~/wandb_api_key.txt`). On Grete, network access
  goes through `HTTPS_PROXY=http://www-cache.gwdg.de:3128`, which is
  exported in `slurm_jonathan/env.sh`.
* **Switching trackers across resumes** silently starts a new run on the new
  backend. If you depend on continuity, stick with one backend per run.
* **Curve image cost on wandb**: each `Tracker.log_curve` call on the wandb
  backend renders a PNG and uploads it. The 900-curve budget in
  `viz/comet.py::log_curve` keeps this bounded but the constant is
  Comet-derived; lower it via that file if wandb storage becomes a concern.

## Verification plan

1. **Single-GPU sharded path is byte-identical**: run an existing ablation
   (`slurm_jonathan/ablations/no-bptt.sh`) for a few steps; confirm logs and
   checkpoint match the pre-change behaviour.
2. **Build the schedule**:
   `uv run python scripts/build_shard_schedule.py --webdataset-dir $WEBDATASET_DIR --seed 0`;
   confirm the file lands at
   `$WEBDATASET_DIR/train-shuffled/shard_schedule.npz` and contains
   `1000 * (n_shards - 1)` entries with the last shard absent.
3. **Single-GPU WebDataset path**:
   `uv run python -m canvit_pretrain.train --webdataset-dir $WEBDATASET_DIR --steps-per-job 16 --batch-size 256 --ckpt-dir /tmp/canvit-test --tracker none`.
   Confirm: shard divisibility log, normalizer stats log, no tracker
   network calls, one job's worth of steps, checkpoint with `job_index=0`
   saved, second invocation resumes with `job_index=1` and slices the next
   shards.
4. **Tracker swap (single-GPU)**: re-run step 3 with `--tracker wandb
   --wandb-project canvit-pretrain`; confirm a new wandb run appears with
   matching params and metrics. Then with `--tracker comet`; confirm a new
   Comet experiment appears. Then resume each — confirm the same run / key
   continues (`prev_wandb_id` / `prev_comet_id` flow into `make_tracker`).
5. **Smoke DDP (1 node, 2 GPUs)**: submit `slurm_jonathan/train_ddp.sbatch`
   with `--nodes=1 --gpus-per-node=A100:2 --ntasks-per-node=2 --steps-per-job 16`.
   Confirm: both ranks initialise, only rank 0 logs / saves / writes failed
   marker, val all-reduce produces consistent metrics, checkpoint includes
   `job_index` and `wandb_run_id`.
6. **Multi-node DDP (2 nodes × 2 GPUs)**: same as above with `--nodes=2`.
   Confirm `MASTER_ADDR` resolves, no NCCL errors, end-of-job barrier
   completes cleanly, resume from checkpoint works.
7. **Compile × DDP smoke**: confirm `cfg.compile=True` works under DDP. If it
   fails, swap order (wrap-then-compile) and re-test.
8. **Backwards-compat checkpoint load**: load a pre-change checkpoint with
   the new `load()` — must not error on missing `job_index` or
   `wandb_run_id`.
