# WebDataset training-data path in CanViT-pretrain

This document describes the WebDataset-based training-data pipeline as it
exists in the current codebase: the on-disk shard format the pipeline
expects, the deterministic per-rank shard scheduling, and the loader code
that consumes everything. The goal is reproducibility — a reader who reads
this doc and the code it points to should understand the full data path
end-to-end.

The codebase uses **WebDataset** (the `wds.*` library) over **pre-shuffled
tar shards** on shared NFS, with **precomputed teacher features** baked into
each sample. Training is **stateless** in the sense that there is no on-disk
schedule artifact: each rank reconstructs its slice of the shard sequence
deterministically from `(seed, job_index, world_size, rank)` plus the shards
already on disk.

---

## 1. Shard layout on disk

A WebDataset dataset directory contains:

```
<webdataset_dir>/
├── train/
│   ├── info.json
│   ├── shard-000000.tar
│   ├── shard-000001.tar
│   └── ...
├── train-shuffled/        ← what the loader actually consumes
│   ├── info.json
│   ├── shard-000000.tar
│   └── ...
└── val/
    ├── info.json
    ├── shard-000000.tar
    └── ...
```

The training loader is wired to **`<webdataset_dir>/train-shuffled/`**, not
`train/` (see §8). The shuffled directory holds the same data with samples
randomly permuted before sharding so each shard contains a class-mixed
sample. The loader does **no further per-sample shuffling** — it relies on
this build-time shuffle.

### 1.1 `info.json`

A small JSON file at the root of each split directory. Required fields the
loader actually reads:

```json
{
  "n_images": 1281167,
  "n_shards": 313,
  "images_per_shard": 4096,
  "keys": ["jpg", "json", "cls.npy", "ptch.npy"]
}
```

- `n_images`: total samples in the split. Used only for log messages on the
  val loader.
- `images_per_shard`: number of samples per tar (assumed equal across all
  shards except possibly the last). The train loader reads this and asserts
  it divides `batch_size_per_gpu` so each worker emits whole batches.
- `keys`: WebDataset extension keys present in every sample. The train
  loader asserts `"cls.npy"` is present (it's part of the precomputed-feature
  contract) and the pipeline `to_tuple()`s in this exact order.

The other fields (resize, crop, jpeg_quality, teacher.*, etc.) are
informational — not consumed by the loader, but recorded for provenance.

### 1.2 Tar shard contents

A tar shard is a flat archive of grouped samples. WebDataset uses the
**filename stem** as the sample key — files sharing a stem belong to the
same sample. Each sample contributes four files:

```
000000484.jpg            # JPEG-encoded image
000000484.json           # original metadata (label, source path, etc.)
000000484.cls.npy        # teacher CLS feature: [embed_dim] fp16 (L2-normalized)
000000484.ptch.npy       # teacher patch features: [n_patches, embed_dim] fp16 (L2-normalized)
```

The pipeline decodes:
- `jpg` → `Tensor[3, H, W]` via `PIL.Image.open` + `canvit_pytorch.preprocess`
  at `image_size=cfg.scene_resolution`.
- `json` → integer label via `json.loads(...)["label"]`.
- `cls.npy` and `ptch.npy` → `Tensor` from `np.load` (the `.npy` header
  encodes shape and dtype).

Sample order **inside** a shard is untouched. WebDataset streams samples in
tar order from each shard; the `train-shuffled/` directory's build step is
what guarantees mixed-class shards.

### 1.3 Last shard is treated as partial

`compute_schedule_slice` excludes the last shard (`all_shards[:-1]`). If
your build script produced an exact integer multiple of
`images_per_shard` samples (so the last shard is full), this just means the
loader sees one less shard than physically exists; no harm. If it was
partial, the loader would never emit a partial batch.

The val loader does **not** exclude the last shard — validation must touch
every sample.

---

## 2. Deterministic per-rank shard scheduling

**File: `canvit_pretrain/train/data/schedule.py`** (87 lines)

The schedule is a pure function of `(seed, job_index, world_size, rank)` and
the on-disk shard list. No prebuilt schedule file, no NCCL collectives.

### 2.1 `compute_shards_per_gpu`

```python
def compute_shards_per_gpu(steps_per_job, batch_size_per_gpu, samples_per_shard) -> int:
    samples_per_gpu = steps_per_job * batch_size_per_gpu
    assert samples_per_gpu % samples_per_shard == 0
    return samples_per_gpu // samples_per_shard
```

Each rank consumes exactly `shards_per_gpu` shards per job. The divisibility
constraint means shard boundaries align with job/step boundaries — no
partial last batch. Ensure your `steps_per_job * batch_size_per_gpu` is a
multiple of `samples_per_shard` (4096 in the typical setup).

### 2.2 `compute_schedule_slice`

```python
def compute_schedule_slice(*, seed, train_dir, job_index, shards_per_gpu,
                           world_size, rank) -> list[Path]:
```

The schedule is conceptually an infinite flat sequence of shard paths,
generated by repeating per-epoch permutations:

```
[π₀(s₀), π₀(s₁), ..., π₀(s_{n-1}),  # epoch 0 (permutation π₀ from rng)
 π₁(s₀), π₁(s₁), ..., π₁(s_{n-1}),  # epoch 1 (permutation π₁)
 ...]                                # epochs are generated lazily
```

where `s_i` are the sorted training shards (last excluded) and `π_e` is
a permutation drawn from `np.random.default_rng(seed)`.

Job `j` consumes the slice `[j*shards_per_job, (j+1)*shards_per_job)` where
`shards_per_job = shards_per_gpu * world_size`. Within that window, rank
`r` gets the `r`-th `shards_per_gpu`-sized contiguous chunk:

```
job j, rank r → slice[r*shards_per_gpu : (r+1)*shards_per_gpu]
```

Properties this gives you:

- **Deterministic**: same `(seed, job_index, rank, world_size)` always
  produces the same shard list.
- **Disjoint across ranks within one job**: rank 0 and rank 1 see different
  shards, so they cover different samples each step.
- **Crosses epoch boundaries cleanly**: if a job's slice spans
  `[shards_per_job * j, shards_per_job * (j+1))` and that range straddles
  the n-shard epoch boundary, the function generates exactly the right
  permutations and concatenates the relevant chunks.
- **Effectively infinite**: there's no `n_epochs` cap; the function will
  generate as many epoch permutations as needed to reach the requested
  `job_index`.

`job_index` comes from the caller (typically `start_step / steps_per_job`
on a fresh run, or `ckpt_data.job_index + 1` on resume) and is what makes
the data path stateless: nothing about past jobs is kept on disk; the
schedule is always recomputed from `(seed, job_index)`.

---

## 3. Pipeline construction (`_build_pipeline`)

The same helper is used by both train and val loaders:

```python
def _build_pipeline(shards, *, image_size, batch_size, use_worker_split):
    workersplitter = wds.split_by_worker if use_worker_split else None
    ds = wds.WebDataset(
        shards,
        shardshuffle=False,
        empty_check=False,
        nodesplitter=None,
        workersplitter=workersplitter,
    )
    return (
        ds.to_tuple("jpg", "json", "cls.npy", "ptch.npy")
        .map_tuple(
            lambda d: _decode_jpg(d, image_size),
            _decode_label,
            _decode_npy_fp16,
            _decode_npy_fp16,
        )
        .batched(batch_size, partial=False)
    )
```

### 3.1 The four splitter / shuffle kwargs

These four kwargs together fully specify how a WebDataset distributes work,
and **all four matter**:

- **`shardshuffle=False`**: the `train-shuffled/` directory was built with
  shuffled samples baked in; per-run shard reshuffling is not needed and
  would interact badly with the deterministic schedule.

- **`empty_check=False`**: the val loader and the normalizer-init helper
  build single-shard pipelines; WebDataset's default empty check warns when
  fewer shards than workers, which is fine here.

- **`nodesplitter=None`**: tells WebDataset to do no node-level splitting
  inside the dataset. Shards are already pre-sliced per-rank by
  `compute_schedule_slice` upstream — applying another node-level split
  would either duplicate that work or, worse, conflict with it. Note: the
  WebDataset default is `single_node_only`, which raises `ValueError`
  whenever `WORLD_SIZE > 1` — so under DDP, leaving this at default fails
  outright.

- **`workersplitter=wds.split_by_worker`**: tells WebDataset to distribute
  the rank's shards across DataLoader workers at the **shard level** —
  each worker gets a disjoint subset of shards. The loader sets `num_workers
  = shards_per_gpu / k` (for some integer `k`), so this just means each
  worker streams `k` whole shards sequentially.

### 3.2 Subsequent operations

- **`.to_tuple("jpg", "json", "cls.npy", "ptch.npy")`**: extracts the
  four expected keys per sample, in this exact order.
- **`.map_tuple(...)`**: applies decoders position-wise. Each decoder
  operates on one element of the tuple.
- **`.batched(batch_size, partial=False)`**: groups consecutive samples into
  batches; `partial=False` discards any incomplete trailing batch (which
  cannot happen here because the divisibility constraint in §2.1 ensures
  whole-batch boundaries).

The result is an iterable that yields tuples of
`(images, labels, cls_feats, patch_feats)` per batch.

---

## 4. `WebDatasetTrainLoader`

**File: `canvit_pretrain/train/data/webdataset.py`, ~line 106**

The training loader wraps `_build_pipeline` for the train path. Its
`__init__` does five things:

### 4.1 Resolve `samples_per_shard` from `info.json`

```python
info = _read_info(train_dir)
assert "cls.npy" in info["keys"], "..."
self.samples_per_shard = int(info["images_per_shard"])
assert self.samples_per_shard % batch_size_per_gpu == 0
```

The first assert encodes that this loader requires precomputed features.
The second assert ensures each worker emits whole batches.

### 4.2 Compute `shards_per_gpu` and the rank's shard list

```python
self.shards_per_gpu = compute_shards_per_gpu(
    steps_per_job, batch_size_per_gpu, self.samples_per_shard
)
self.shard_files = compute_schedule_slice(
    seed=seed, train_dir=train_dir, job_index=job_index,
    shards_per_gpu=self.shards_per_gpu,
    world_size=world_size, rank=rank,
)
```

### 4.3 Resolve `num_workers`

```python
requested = max(1, num_workers)
capped = min(requested, self.shards_per_gpu)
nw = capped
while self.shards_per_gpu % nw != 0:
    nw -= 1
self.num_workers = nw
```

`num_workers` is cap-and-round-down-to-divisor of `shards_per_gpu`. Two
constraints:

- **Cap at `shards_per_gpu`**: more workers than shards means some workers
  get zero shards. With `partial=False` on `.batched()`, a worker with zero
  shards yields nothing — fine, just wasteful processes.
- **Divisibility**: each worker needs to stream the same integer number of
  shards (`shards_per_gpu / num_workers`) so per-worker batch counts are
  uniform; otherwise some workers would run out before others, causing
  uneven epoch ends.

This cap is what keeps memory bounded on long jobs (where `shards_per_gpu`
grows linearly with `steps_per_job`). The `cfg.num_workers` default is `4`
in `config.py`.

### 4.4 Sanity-assert sample count

```python
total_samples = len(self.shard_files) * self.samples_per_shard
assert total_samples == steps_per_job * batch_size_per_gpu
```

Catches any drift between the schedule logic and the divisibility
contract early.

### 4.5 DataLoader construction (lazy)

The actual `torch.utils.data.DataLoader` is built lazily on first `next()`:

```python
ds = _build_pipeline(
    [str(p) for p in self.shard_files],
    image_size=self.image_size,
    batch_size=self.batch_size,
    use_worker_split=True,
)
self._loader = DataLoader(
    ds,
    batch_size=None,           # WebDataset already batches
    num_workers=self.num_workers,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=2 if self.num_workers > 0 else None,
)
```

`batch_size=None` because `_build_pipeline` already calls `.batched()`. With
two layers of batching enabled, you'd get nested batches.

`persistent_workers=False` because the loader is created fresh for every
job and is meant to terminate cleanly when shards are exhausted.

### 4.6 The `next()` contract

```python
def next(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Returns (images, raw_patches, raw_cls, labels)."""
```

Output ordering: `(images, raw_patches, raw_cls, labels)` — note that this
**reorders** the pipeline's `(images, labels, cls_feats, patch_feats)` into
the order the rest of the codebase expects. Labels are converted to a long
tensor (WebDataset's `.batched()` returns a list of ints).

`first_shard_path()` exposes the rank's first scheduled shard for
normalizer initialization (§6).

---

## 5. `WebDatasetValLoader`

**Same file, ~line 227**

Simpler counterpart, with three differences from the train loader:

### 5.1 Round-robin partition (no schedule)

```python
all_shards = sorted(val_dir.glob("shard-*.tar"))
rank_shards = [s for i, s in enumerate(all_shards) if i % world_size == rank]
```

Validation is deterministic and exhaustive — just round-robin partition
the shards across ranks. No `seed`, no `job_index`. Each rank covers
roughly `1/world_size` of the val set.

### 5.2 Last shard kept

Unlike train, the val loader includes the partial last shard (validation
must touch every sample).

### 5.3 Cycles forever

```python
def _next_with_cycle(self):
    try:
        return next(self._iter)
    except StopIteration:
        # rebuild and start over
        self._loader = self._build_loader()
        self._iter = iter(self._loader)
        return next(self._iter)
```

So `next_batch_with_labels()` always returns a batch — convenient because
the training loop calls validation at arbitrary step intervals and
shouldn't have to reason about val-epoch boundaries.

The same `_build_pipeline` is used. `num_workers = min(len(rank_shards), 4)`.

---

## 6. Normalizer initialization (`init_normalizer_stats_from_tar`)

**Same file, ~line 303**

Helper that reads samples directly from a single tar via the **stdlib
`tarfile` module** — bypassing the WebDataset pipeline entirely. Used to
initialize standardizer (`PatchStandardizer`, `CLSStandardizer`) buffers
before the first training step.

```python
with tarfile.open(shard_path, "r") as tf:
    for member in tf:
        ...
        if name.endswith(".cls.npy"):
            key, kind = name[:-len(".cls.npy")], "cls"
        elif name.endswith(".ptch.npy"):
            key, kind = name[:-len(".ptch.npy")], "ptch"
        else:
            continue
        arr = np.load(io.BytesIO(tf.extractfile(member).read()))
        ...
```

Why bypass the pipeline:

- We only need `cls.npy` and `ptch.npy`, not jpg/json — saves decoding cost.
- We need exactly one shard's worth of samples for stats; the WebDataset
  pipeline's batching/cycling semantics are unnecessary and make this
  awkward.
- This runs once on rank 0 before DDP wrap, so worker-process spawning
  would be wasteful overhead.

Output: in-place call to `scene_norm.set_stats(patches)` and
`cls_norm.set_stats(cls.unsqueeze(1))`. Caller is responsible for then
broadcasting these buffers to other ranks (see DDP doc, §2.2).

---

## 7. Decoder helpers

Three small functions at the top of `webdataset.py`:

```python
def _decode_jpg(data: bytes, image_size: int) -> Tensor:
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return preprocess(img, image_size)

def _decode_label(data: bytes) -> int:
    return int(json.loads(data)["label"])

def _decode_npy_fp16(data: bytes) -> Tensor:
    arr = np.load(io.BytesIO(data))
    return torch.from_numpy(arr.astype(np.float16))
```

`_decode_npy_fp16` casts to fp16 explicitly. The `.npy` header in the shard
already encodes fp16 (per `info.json`'s `feature_dtype`), but the explicit
cast is defensive: if the build pipeline ever changes dtype, the loader
still outputs the contract type.

`preprocess` (from `canvit_pytorch`) handles resize-to-`image_size` +
ImageNet-normalize.

---

## 8. Wiring into `create_loaders`

**File: `canvit_pretrain/train/data/__init__.py`**

```python
def create_loaders(
    cfg: Config, *, job_index: int, world_size: int, rank: int,
) -> Loaders:
    ...
    if cfg.webdataset_dir is not None:
        return _create_webdataset_loaders(cfg, job_index=job_index,
                                          world_size=world_size, rank=rank)
    ...

def _create_webdataset_loaders(...) -> Loaders:
    train_dir = cfg.webdataset_dir / "train-shuffled"
    val_dir_wds = cfg.webdataset_dir / "val"
    train_loader = WebDatasetTrainLoader(
        train_dir=train_dir,
        seed=cfg.seed,
        job_index=job_index,
        batch_size_per_gpu=cfg.batch_size_per_gpu,
        steps_per_job=cfg.steps_per_job,
        image_size=cfg.scene_resolution,
        world_size=world_size,
        rank=rank,
        num_workers=cfg.num_workers,
    )
    val_loader = WebDatasetValLoader(
        val_dir=val_dir_wds,
        batch_size_per_gpu=cfg.batch_size_per_gpu,
        image_size=cfg.scene_resolution,
        world_size=world_size,
        rank=rank,
    )
    return Loaders(train=train_loader, val=val_loader)
```

The dispatch is a single `if cfg.webdataset_dir is not None:` check at the
top of `create_loaders`. The function is called from the training loop
once, at `job_index` derived from the checkpoint or zero (fresh run).

Notice: train uses `<webdataset_dir>/train-shuffled/`, val uses
`<webdataset_dir>/val/`. `train/` (un-shuffled) is *not* used by the
loader — it would normally be the canonical pre-shuffle build artifact and
is kept on disk only as a checkpoint of the build pipeline.

---

## 9. Configuration

**File: `canvit_pretrain/train/config.py`**

The relevant fields:

```python
webdataset_dir: Path | None = None  # set this to activate the WebDataset path
seed: int = 0                       # drives compute_schedule_slice
batch_size_per_gpu: int = 64
steps_per_job: int = 128
num_workers: int = 4                # capped at shards_per_gpu inside the loader
scene_resolution: int = 512         # passed as image_size to the pipeline
```

`webdataset_dir` should point to the directory containing
`train-shuffled/`, `val/`, etc. (the parent of the split directories).

Step-and-batch must satisfy
`steps_per_job * batch_size_per_gpu % samples_per_shard == 0` (typically
`samples_per_shard = 4096`).

In the launch sbatch, these are passed as CLI args:

```
--webdataset-dir "$WEBDATASET_DIR" \
--batch-size-per-gpu 64 \
--steps-per-job 128 \
```

---

## 10. Reproduction checklist

Starting from a codebase without WebDataset:

**On disk** (build artifact, outside this repo):

- [ ] Build a `train-shuffled/` directory of `shard-NNNNNN.tar` files where:
      - Each tar contains samples grouped by filename stem.
      - Each sample provides four extensions: `jpg`, `json`, `cls.npy`,
        `ptch.npy`.
      - All shards (except optionally the last) contain exactly
        `images_per_shard` samples.
      - Samples are shuffled at build time so each shard is class-mixed.
      - Drop `info.json` at the directory root with at least
        `images_per_shard` and `keys` populated.
- [ ] Build a `val/` directory with the same per-shard format. The
      schedule treats this independently (round-robin per-rank).

**New module**:

- [ ] `canvit_pretrain/train/data/schedule.py` — `compute_shards_per_gpu`
      and `compute_schedule_slice` from §2.

**New module**:

- [ ] `canvit_pretrain/train/data/webdataset.py` with:
      - `_decode_jpg`, `_decode_label`, `_decode_npy_fp16` decoders (§7).
      - `_read_info` helper.
      - `_build_pipeline` with `nodesplitter=None`, `shardshuffle=False`,
        `empty_check=False`, `workersplitter=wds.split_by_worker` (§3).
      - `WebDatasetTrainLoader` (§4).
      - `WebDatasetValLoader` (§5).
      - `init_normalizer_stats_from_tar` (§6).

**Edits to `canvit_pretrain/train/data/__init__.py`**:

- [ ] Import the WebDataset classes.
- [ ] Branch `create_loaders` on `cfg.webdataset_dir is not None` and call
      `_create_webdataset_loaders` (§8).

**Edits to `canvit_pretrain/train/config.py`**:

- [ ] Add `webdataset_dir: Path | None = None`, `seed: int = 0`.
- [ ] Set `num_workers: int = 4` (or any small default; gets capped per
      §4.3).
- [ ] Use `batch_size_per_gpu` (renamed from `batch_size` for DDP clarity).

**Add `webdataset` to `pyproject.toml`** dependencies.

**Edits to the training loop** (already covered by the DDP doc; relevant to
WebDataset in:
- `start_step = start_job_index * cfg.steps_per_job` (WebDataset-only resume
  logic, see `loop.py`).
- `init_normalizer_stats_from_tar(train_loader.first_shard_path(), ...)` for
  fresh-init normalizer stats.

After all of the above, training reads from your WebDataset shards
deterministically per `(seed, job_index)`, with stable per-rank,
per-worker shard slicing, and known-divisible batch sizes throughout.
