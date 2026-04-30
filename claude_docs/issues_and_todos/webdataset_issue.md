# WebDataset + PyTorch DataLoader: split_by_worker investigation

**Status:** Working hypothesis (2026-04-29). The analysis below reflects our current
understanding. It may turn out to be partially or fully incorrect. The code change
described at the bottom was made based on this hypothesis — if problems recur,
revisit this document first.

---

## Observed symptom

With `steps_per_job=256`, `batch_size=64`, `shards_per_gpu=4`, `num_workers=4`:
training crashes with `StopIteration` from the train DataLoader after exactly **64
batches** — one shard's worth — instead of the expected 256.

The crash is reproducible and the number 64 is stable across runs. It does **not**
depend on `val_every` (confirmed by running with `val_every=1000`).

---

## What the code does (before the fix)

`_build_pipeline` in `webdataset.py`:

```python
ds = wds.WebDataset(shards, shardshuffle=False, empty_check=False)
if use_worker_split:
    ds = ds.compose(wds.split_by_worker)   # ← our explicit call
return (
    ds.to_tuple(...)
    .map_tuple(...)
    .batched(batch_size, partial=False)
)
```

---

## Why this is broken (our hypothesis)

`wds.WebDataset.__init__` has the following default parameter:

```python
workersplitter=shardlists.split_by_worker,
```

This means `split_by_worker` is **automatically appended to the internal pipeline
at construction time**, operating on the **shard URL stream** (before any tar files
are opened). This is the correct level — it assigns whole shards to workers.

Our `.compose(wds.split_by_worker)` call then adds `split_by_worker` a **second
time**, but now it operates on the stream of **decoded individual samples** (after
tars have been opened and samples decoded).

With 4 workers and 4 shards, the two applications interact as follows:

| Stage | Input | Worker 0 sees |
|-------|-------|---------------|
| 1st `split_by_worker` (built-in, shard level) | [s0, s1, s2, s3] | [s0] |
| Tar open + decode | shard s0 → 4096 samples | 4096 samples |
| 2nd `split_by_worker` (ours, sample level) | 4096 samples | every 4th → 1024 samples |
| `batched(64, partial=False)` | 1024 samples | **16 batches** |

16 batches × 4 workers = **64 batches total** — exactly the observed crash point.

Workers 1–3 are also affected symmetrically, each producing 16 batches from their
respective shard. The DataLoader correctly collects from all 4 workers, giving 64
total instead of 256.

---

## The fix

In `_build_pipeline`:

1. Remove the explicit `ds = ds.compose(wds.split_by_worker)` call — the built-in
   `workersplitter` already handles per-worker shard splitting at the right level.

2. Pass `nodesplitter=None` to `wds.WebDataset` — the default
   `nodesplitter=shardlists.single_node_only` would raise `ValueError` for DDP
   (`world_size > 1`). Since we pre-slice shards per rank before constructing the
   dataset, no node-level splitting is needed inside WebDataset.

Result: the pipeline has exactly one `split_by_worker` (built-in, at shard level),
which correctly assigns each worker its own subset of shards.

---

## Why this went unnoticed with shards_per_gpu=1

With `steps_per_job=64`, `batch_size=64`, `shards_per_gpu=1`, `num_workers=1`:

- WebDataset built-in `split_by_worker`: 1 shard, 1 worker → worker 0 gets [s0].
- Our explicit `split_by_worker`: `islice(4096 samples, 0, None, 1)` → all 4096
  samples (since `num_workers=1`, the `if num_workers > 1` branch in
  `split_by_worker` is skipped, all samples pass through).
- Result: 64 batches. Correct — so the bug was invisible.

The double-split only manifests when `num_workers > 1`.
