# Follow-up concerns: WebDataset loader + DDP

These are open questions / improvements raised after implementing
`webdataset-loader-and-ddp.md`. Captured here for a future work session.
None of them block correctness on a single fixed configuration, but several
prevent silent degradation as you scale or change settings.

Items are ordered by what I'd act on first.

---

## 1. Add LR scaling for DDP and `world_size` consistency check on resume

### 1a. Linear LR scaling under DDP

**Current state.** `cfg.peak_lr` is tuned for `batch_size_per_gpu=64` on one
GPU. Running with `world_size=4` keeps the per-GPU batch the same but
quadruples the effective batch via gradient averaging. The current LR is then
likely too small.

**Proposal.** Add a config flag (default off, so legacy runs are unchanged):

```python
scale_lr_by_world_size: bool = False
"""Linear LR scaling rule (Goyal et al., 2017). Multiplies peak_lr and
start_lr by world_size before constructing the scheduler."""
```

If on, in `loop.py`:

```python
if cfg.scale_lr_by_world_size and ddp.is_dist():
    cfg.peak_lr *= ddp.world_size()
    if cfg.start_lr is not None:
        cfg.start_lr *= ddp.world_size()
    log.info(f"DDP LR scaling: peak_lr={cfg.peak_lr}, start_lr={cfg.start_lr}")
```

Optionally also support square-root scaling via a `lr_scaling_rule:
Literal["none", "linear", "sqrt"]` field.

### 1b. Resume-with-different-world-size silent misbehaviour

**Current state.** Resume slicing is
`start = job_index * shards_per_gpu * world_size`. If a run is killed at
job_index=N with world_size=2 and resumed with world_size=4, job N+1 reads
from offset `(N+1) * 4 * shards_per_gpu`, but the previous run was at
`N * 2 * shards_per_gpu`. Shards either get re-processed or skipped silently.
The same hazard exists if `batch_size`, `steps_per_job`, or
`samples_per_shard` change.

**Proposal.** Store the relevant fields in the checkpoint and assert on
resume:

```python
# In checkpoint TypedDict:
ddp_world_size: int | None
batch_size_per_gpu: int | None
samples_per_shard: int | None
steps_per_job_at_save: int | None

# In loop.py at resume time:
if ckpt_data is not None and not is_seeding and cfg.webdataset_dir is not None:
    saved_ws = ckpt_data.get("ddp_world_size")
    if saved_ws is not None and saved_ws != ddp.world_size():
        raise RuntimeError(
            f"Cannot resume with world_size={ddp.world_size()}; "
            f"checkpoint was saved with world_size={saved_ws}. "
            f"Same constraint applies to batch_size and steps_per_job."
        )
    # ... similar checks for batch_size, steps_per_job, samples_per_shard
```

A more permissive alternative: track total samples consumed
(`samples_consumed = job_index * world_size * batch_size * steps_per_job`)
and recompute the shard offset from that on resume. Allows changing
world_size mid-run, which is occasionally useful (e.g., switching between
debug and full configurations).

**Effort.** Strict-assertion version: ~30 min. Permissive
samples-consumed version: ~2 hours.

---

## 2. `compile × DDP` order should be smoke-tested before any serious run

**Current state.** I implemented compile-then-DDP-wrap (PyTorch ≥ 2.2
documented recommendation) but PyTorch has had bugs in both orderings
depending on version.

**Proposal.** Before the first long run, do a deliberate smoke test:

```bash
# 100 steps with cfg.compile=True under DDP (1 node, 2 GPUs).
sbatch --array=0-0%1 --time=00:20:00 \
  --gpus-per-node=A100:2 --ntasks-per-node=2 \
  slurm_jonathan/train_ddp.sbatch --steps-per-job 100
```

If this hangs or produces NaN gradients, swap the order in `loop.py` (move
`compile_model(model)` to AFTER `model = DDP(model, ...)`). One-line change.

**Effort.** A 20-minute test slot on the cluster.

---

## 3. `broadcast_buffers` optimization

**Current state.** DDP defaults to `broadcast_buffers=True`, which broadcasts
all module buffers from rank 0 to other ranks on **every forward**. After the
standardizer is initialized once at training start, our buffers don't change.

**Proposal.** Pass `broadcast_buffers=False` to the DDP constructor (relying
on the construction-time broadcast for initial sync):

```python
model = nn.parallel.DistributedDataParallel(
    model,
    device_ids=[ddp.local_rank()],
    find_unused_parameters=False,
    broadcast_buffers=False,  # standardizers don't change after init
)
```

**Risk.** If we ever introduce a buffer that mutates during training (e.g.,
running stats in BatchNorm), this would silently desync ranks. Document the
assumption.

**Effort.** 5 minutes. Low priority; the per-step cost is small.

---

## 4. Per-sample fp16 `.npy` entries are storage-inefficient

**Current state.** Each WebDataset sample contains `cls.npy` and `ptch.npy`
as separate tar entries. Per-sample overhead:

- 2 tar headers × 512B = 1024B
- 2 numpy headers × ~80B = 160B
- 2 `np.load(io.BytesIO(...))` calls per `next()` (parse + alloc + copy).

For ImageNet-1k (1.28M samples) the storage overhead is ~1.5GB — small.
But the **per-sample CPU work** (numpy parsing) is paid every step. The
existing `.pt` shard format stores all `patches` and `cls` in two big
contiguous tensors per shard with zero per-sample parsing.

**Proposal (only relevant if you scale beyond IN-1k or hit dataloader-bound
training).**

A hybrid format: keep `jpg`/`json` per sample inside the tar, but move
features into a sidecar `<shard-name>.features.pt` next to each tar with two
tensors: `patches: [N, 1024, 768]` and `cls: [N, 768]` (both fp16). Loader
opens the tar for images and mmaps the .pt for features, indexed by
in-shard sample index.

Pros: zero per-sample numpy parsing; mmap for free.
Cons: two files per shard instead of one; a custom loader instead of pure
WebDataset.

**Effort.** ~1 day if you decide to do it; not worth it for IN-1k.

---

## 5. Per-rank pre-slicing vs `wds.split_by_node`

**Current state (deviation from spec).** The original `dataloading.md` plan
prescribes `wds.WebDataset(...).split_by_node().split_by_worker()`. I instead
pre-slice in Python via `slice_for_job(...)` and pass each rank only its
shards, using `split_by_worker` inside WebDataset.

The two are equivalent in effect, but the deviation should be on the record:

- **Pre-slice (current)**: more explicit about what each rank reads; easier
  to log and debug; no dependency on WebDataset's understanding of the dist
  group; integrates naturally with the `slice_for_job` resume helper.
- **`split_by_node` (spec)**: more idiomatic WebDataset; one less thing for
  us to maintain; assumes WebDataset reads `dist.get_rank()` correctly inside
  workers.

**Proposal.** Either is fine; flag the deviation in any future review and
decide whether to align with the spec. If aligning, replace the per-rank
slicing with `split_by_node` after the slice is computed for the whole job
block.

**Effort.** A few lines either way. No urgency.

---

## Bonus: validation strategy worth revisiting

The chosen path is "split_by_node + all-reduce metrics on log". For a 50k-sample
val set this is overkill — a full val pass on rank 0 alone takes seconds, and
the all-reduce machinery has to be right (which it currently is, but it's
extra surface area). If the val set ever shrinks further or simplicity becomes
worth more than throughput, switching to "rank 0 runs validate(); other ranks
barrier" is a one-screen change in `loop.py`.

---

## Suggested order to tackle these

1. **#3 (compile×DDP smoke test)** — cheapest, prevents wasted training time.
2. **#2 (LR scaling + resume guards)** — prevents silent training degradation.
3. **#1 (deterministic-on-demand schedule)** — quality-of-life, removes a step.
4. **#4 (broadcast_buffers)** — micro-optimization, low priority.
5. **#5 (storage format)** — only if we scale beyond IN-1k or become I/O-bound.
6. **#6 (deviation alignment)** — purely cosmetic; defer indefinitely.
