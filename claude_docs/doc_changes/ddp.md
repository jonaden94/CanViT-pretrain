# Reproducing DDP in CanViT-pretrain

This document describes everything that had to be added to the codebase to make
distributed data-parallel (DDP) training work end-to-end (training + validation
+ wandb logging + PCA plotting) on a multi-GPU node. It assumes a starting
point with no DDP at all (e.g. commit `1a36eec`, the "Prepare pretrain code for
anonymous release") and walks to the current HEAD.

The goal is reproducibility: a reader who applies the changes described here
to such a pre-DDP codebase should end up with a working DDP setup. The body
of the document (§1–§8) covers the essentials; §9 documents the changes that
*looked* essential during the original DDP debugging session but turned out
to be non-essential once the matplotlib NFS-cache bug (§4) was identified
and fixed. Knowing what's *not* needed is just as important as knowing what
is — several of those non-essentials carry their own complexity costs, and
without §9 a future maintainer would re-introduce them out of caution.

The codebase uses **PyTorch DDP** with **NCCL** backend, **`torch.compile`**
for the model, **truncated BPTT** with `chunk_size=2` (two forwards per
backward), and a **WebDataset** training data path on shared NFS shards.
Several of the fixes are needed precisely because of that combination —
they are not generic DDP requirements but TBPTT- or compile- or NFS-specific.

---

## 1. The `dist.py` singleton

**File: `canvit_pretrain/train/dist.py` (new module)**

A small module that owns rank/world_size/local_rank/device state and exposes
helpers. Idempotent `init_dist()` reads from SLURM env vars; behaves as a
no-op on single-GPU runs (`WORLD_SIZE` unset or 1).

```python
def init_dist() -> None:
    global _initialized, _rank, _world_size, _local_rank, _device
    if _initialized:
        return
    _initialized = True

    world_size_env = os.environ.get("WORLD_SIZE")
    if world_size_env is None or int(world_size_env) <= 1:
        # single-process fallback — pick best available device
        if torch.cuda.is_available():
            _device = torch.device("cuda")
        ...
        return

    _world_size = int(world_size_env)
    _rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    _local_rank = _rank - gpus_per_node * (_rank // gpus_per_node)

    torch.cuda.set_device(_local_rank)
    _device = torch.device("cuda", _local_rank)

    dist.init_process_group(backend="nccl", rank=_rank, world_size=_world_size)
```

Public helpers used elsewhere:

- `init_dist()` — idempotent init.
- `is_dist()`, `is_main()`, `rank()`, `world_size()`, `local_rank()`, `device()` — accessors.
- `barrier()` — wraps `dist.barrier()` with single-process no-op.
- `broadcast_module_buffers(module, src=0)` — broadcasts every buffer/param of
  a `nn.Module` from `src` to all ranks. Used to manually propagate state that
  DDP's auto-broadcast no longer handles (see §4).
- `all_reduce_mean(tensor)` — averages a tensor across ranks; identity in
  single-process mode. Used for metric aggregation at log time.

The module assumes single-node DDP — `device_id=` is **not** passed to
`init_process_group`, so NCCL infers the device from `local_rank` (which
matches `SLURM_PROCID` on single node). For **multi-node** DDP, add
`device_id=torch.device("cuda", _local_rank)` back to the
`init_process_group` call.

---

## 2. Training-loop integration (`loop.py`)

### 2.1 Init at function entry

```python
def train(cfg: Config, trial: optuna.Trial) -> float:
    signal.signal(signal.SIGUSR1, _handle_sigusr1)
    ddp.init_dist()
    cfg.device = ddp.device()
    ...
```

`cfg.device` is overwritten with the rank-local device so every downstream
`.to(cfg.device)` lands on the correct GPU.

### 2.2 Standardizer (normalizer) initialization

The model has buffers (`scene_norm.mean/std`, `cls_norm.mean/std`) that must
be identical across ranks before training. Fresh runs compute them from the
first WebDataset shard.

```python
if need_init:
    if ddp.is_main():
        if cfg.webdataset_dir is not None:
            init_normalizer_stats_from_tar(
                train_loader.first_shard_path(),
                scene_norm, cls_norm,
                cfg.device, cfg.normalizer_max_samples,
            )
        else:
            ...
    ddp.barrier()
    if ddp.is_dist():
        ddp.broadcast_module_buffers(scene_norm)
        ddp.broadcast_module_buffers(cls_norm)
```

Why explicit broadcast: the DDP wrap is configured with
`broadcast_buffers=False` (see §3), so DDP no longer auto-broadcasts buffers
on each forward. Without this explicit step, rank 1's normalizer stays
uninitialized and the first `scene_normalizer(...)` call asserts.

Resume-from-checkpoint paths skip this whole block — the checkpoint state_dict
is identical across ranks because every rank loads the same file.

### 2.3 DDP wrap

```python
core_model = model
if ddp.is_dist():
    log.info(f"Wrapping model in DDP (local_rank={ddp.local_rank()})")
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[ddp.local_rank()],
        find_unused_parameters=False,
        broadcast_buffers=False,
    )
    model._set_static_graph()
```

Two non-default kwargs and one explicit call:

- **`broadcast_buffers=False`** (§3.1) — disables DDP's per-forward in-place
  buffer broadcast, which collides with TBPTT.
- **`_set_static_graph()`** (§3.2) — required because TBPTT reuses parameters
  across the two forward chunks before backward. Also makes
  `find_unused_parameters=True` redundant (per PyTorch docs), so it stays
  `False`.

`core_model` (the non-DDP-wrapped reference) is kept around because:

- DDP only proxies `forward()`. Custom methods (`init_state`,
  `predict_teacher_scene`, `predict_scene_teacher_cls`, `get_spatial`) must be
  called on the unwrapped module.
- `validate()` is called with `core_model`, which avoids triggering DDP
  collectives during validation.
- `save_checkpoint` writes `core_model.state_dict()`, not the DDP-wrapped one.

### 2.4 Per-rank guards in the training-loop body

Throughout `training_loop`, side-effecting work that must happen exactly once
per training run is wrapped in `if ddp.is_main():`. This includes:

- Run-directory creation, `FAILED` marker writing, `cancel_slurm_array` calls
  (in `train()` and `cancel_slurm_array()` helpers).
- `tqdm` progress bar (`disable=not ddp.is_main()`).
- Checkpoint save + symlink update on SIGUSR1 and at end of job.
- Setting up the wandb experiment tracker (handled inside `make_tracker`,
  which returns a no-op `DummyExperiment` on non-main ranks so all the
  `exp.log_*` call sites can stay unconditional).

Validation, however, runs on **all ranks** in HEAD's form (see §5).

### 2.5 Metric aggregation

Per-step metrics are averaged across ranks before logging via
`ddp.all_reduce_mean(metric_value)`. Tensors created for this must be on
CUDA (NCCL constraint, §3.4).

### 2.6 Checkpointing semantics under DDP

Two save sites (SIGUSR1-triggered and end-of-job) both:

1. Compute the EMA loss (already aggregated across ranks).
2. Wrap the actual save in `if ddp.is_main():` (only rank 0 writes to disk).
3. Follow with `ddp.barrier()` so non-main ranks don't race ahead.

The `job_index` field is threaded through both save calls so deterministic
WebDataset shard scheduling resumes correctly (§7).

---

## 3. Training-step DDP correctness fixes

These are essential at every step of training and were the bulk of the DDP
debugging effort.

### 3.1 `broadcast_buffers=False` (TBPTT version mismatch)

**Symptom**: backward crashes with `[torch.cuda.FloatTensor [...]] is at
version 3; expected version 2 instead`.

**Why**: DDP's default `broadcast_buffers=True` calls `_sync_module_states`
*at the start of every forward*, which broadcasts buffers in-place from
rank 0 to other ranks. TBPTT does two forwards before each backward
(`chunk_size=2`); the in-place broadcast between forward 1 and forward 2
mutates tensors that AOT autograd (used by `torch.compile`) saved at version
2, bumping them to version 3 — and the backward's version-counter check then
fails.

**Fix**: pass `broadcast_buffers=False` when constructing
`DistributedDataParallel`. With this disabled, anything that needs to be
synced across ranks must be synced manually (§2.2 for normalizers; nothing
else mutates buffers in this codebase).

### 3.2 `_set_static_graph()` (TBPTT chunk-shared param hook collision)

**Symptom**: backward crashes with `Parameter at index N has been marked as
ready twice` (in the original session, this was on
`scene_cls_head.proj.weight`).

**Why**: DDP installs an autograd hook on every parameter to mark it ready
for allreduce when its gradient is computed. With TBPTT's two-chunk forward,
the same parameters appear in both chunks' graphs; one backward triggers the
hook twice.

**Fix**: call `model._set_static_graph()` immediately after constructing
the DDP-wrapped model. This tells DDP that the computation-graph structure is
identical every iteration — on the first backward DDP records the
parameter-ready order and reuses that record afterward, so the hook only
fires once per parameter per iteration.

### 3.3 Pass `core_model` (not the DDP wrapper) to custom methods

**Symptom**: `AttributeError: 'DistributedDataParallel' object has no
attribute 'get_spatial'` (or `init_state`, `predict_teacher_scene`,
`predict_scene_teacher_cls`).

**Why**: `DistributedDataParallel` only proxies `forward()` / `__call__()`.
Anything else needs the unwrapped module.

**Fix**: in both `loop.py::training_loop` and `step.py::training_step`, keep a
reference `core_model = getattr(model, "module", model)` and route all custom
method calls through it. In particular, `extract_sample0_viz` is called with
`core_model`, not `model`, in `step.py`:

```python
viz_data.viz_samples.append(extract_sample0_viz(out, glimpse, L.scene_pred, core_model))
```

The same `core_model` is passed to `validate()` from `loop.py` so validation
also bypasses the DDP wrapper (which avoids unintended NCCL collectives
during validation).

### 3.4 `device=cfg.device` on the `n_glimpses` EMA tensor

**Symptom**: `RuntimeError: No backend type associated with device type cpu`
in `ddp.all_reduce_mean(...)`.

**Why**: `torch.tensor(scalar, dtype=torch.float32)` defaults to CPU. NCCL
backend requires CUDA tensors for its collectives.

**Fix**: pass `device=cfg.device` when wrapping `step_metrics.n_glimpses`
into a tensor for the EMA tracker:

```python
ema.update("n_glimpses", torch.tensor(step_metrics.n_glimpses,
                                      dtype=torch.float32, device=cfg.device))
```

(All other EMA-tracked tensors come from model output and are already on the
right device.)

### 3.5 Synchronize trajectory length across ranks (`step.py`)

**Symptom**: hang at "Starting training loop" — no further output. Eventually
NCCL watchdog timeout.

**Why**: `step.py::training_step` samples trajectory length stochastically:

```python
n_glimpses = chunk_size
while random.random() < continue_prob:
    n_glimpses += chunk_size
```

If two ranks sample different `n_glimpses`, they end up with different
numbers of `backward()` calls, and the next NCCL allreduce deadlocks waiting
for a peer that never enters the collective.

**Fix**: broadcast rank-0's sampled value to all ranks:

```python
import torch.distributed as dist

n_glimpses = chunk_size
while random.random() < continue_prob:
    n_glimpses += chunk_size

if dist.is_available() and dist.is_initialized():
    n_glimpses_t = torch.tensor(n_glimpses, device=device)
    dist.broadcast(n_glimpses_t, src=0)
    n_glimpses = int(n_glimpses_t.item())
```

(In practice, identical Python `random` seed + identical code path makes
both ranks sample the same value anyway, but the broadcast is a robustness
guard against any future code change that could perturb one rank's random
state.)

---

## 4. Matplotlib NFS-cache safety (`__main__.py`)

**Symptom**: ~45 % of multi-rank runs hang at the first PCA plot — rank 0
(which calls `plot_multistep_pca`) freezes inside the first `pyplot` figure
render; rank 1 proceeds to the next training step and deadlocks at the next
NCCL collective. Stochastic; no error message.

**Why**: matplotlib's font-cache file (`~/.cache/matplotlib/fontList.json`)
and config dir (`~/.config/matplotlib`) live on shared NFS by default. With
multi-rank DDP, both ranks (and any concurrent SLURM job sharing the same
node) race on those files at matplotlib import time, leaving the cache in a
state where subsequent matplotlib calls hang indefinitely.

**Fix**: redirect matplotlib's config + cache to a per-rank, per-job `/tmp`
directory **before any matplotlib import**, including transitive imports
through `train.viz` and `train.tracker`. The redirect is done at the very top
of `canvit_pretrain/train/__main__.py`:

```python
"""Entry point for CanViT pretraining."""

# CRITICAL (DDP-safety): redirect matplotlib config + cache to a per-rank,
# per-job /tmp directory BEFORE any matplotlib import happens anywhere in the
# dep tree. ...
import os as _os
_slurm_rank = _os.environ.get("SLURM_PROCID", "0")
_slurm_job = _os.environ.get("SLURM_JOB_ID", "nojob")
_mpl_dir = f"/tmp/mpl_config_rank{_slurm_rank}_job{_slurm_job}"
_os.makedirs(_mpl_dir, exist_ok=True)
_os.environ["MPLCONFIGDIR"] = _mpl_dir
import matplotlib as _matplotlib  # noqa: E402
_matplotlib.use("Agg")

import logging
... # rest of __main__.py
```

Two pieces:

- `MPLCONFIGDIR` to a per-rank-per-job local-disk path eliminates the NFS
  race entirely.
- `matplotlib.use("Agg")` forces the non-interactive raster backend so
  matplotlib never probes for a display on headless compute nodes.

**This was empirically the dominant cause of "DDP doesn't work" hangs.** All
other DDP work depends on it; without it, ~half of runs hang regardless of
the other fixes.

---

## 5. Validation under DDP

The `validate()` function in `canvit_pretrain/train/viz/validate.py` is the
**original `1a36eec`-form code**:

```python
model_was_training = model.training
model.eval()
try:
    with torch.inference_mode():
        ...validation body...
        return acc.scene_cos_raw[-1]
finally:
    if model_was_training:
        model.train()
```

— byte-identical to its pre-DDP form. No DDP-specific changes were needed.

Two important conventions:

1. **`validate()` is called with `core_model`** (not the DDP-wrapped `model`)
   from `loop.py`. This avoids triggering NCCL collectives during validation.
2. **All ranks run validation.** There's no `if ddp.is_main():` gate around
   the validate call site; both ranks consume the same val batch and produce
   their own metrics. This is benign because `core_model` doesn't engage DDP
   anyway, and the wandb tracker is a no-op on non-main ranks.

The matplotlib fix (§4) is what makes the PCA plotting inside validation
reliable.

---

## 6. WebDataset DDP integration (`data/`)

The data path uses WebDataset tar shards on shared NFS. Three DDP-relevant
pieces:

### 6.1 Shard splitting across ranks and workers

`canvit_pretrain/train/data/webdataset.py::WebDatasetTrainLoader` constructs
the dataset with explicit splitter callbacks rather than the default
`split_by_worker`/`split_by_node`:

```python
ds = wds.WebDataset(
    shard_files_for_this_rank,
    workersplitter=wds.split_by_worker,
    nodesplitter=None,
    ...
)
```

- `nodesplitter=None` disables WebDataset's default `single_node_only`
  guard, which raises `ValueError` under any multi-rank context. Ranks have
  already been pre-assigned disjoint shard lists upstream, so a node-level
  splitter would be redundant.
- `workersplitter=wds.split_by_worker` distributes the rank's shards across
  DataLoader workers at the **shard level** — each worker streams an integer
  number of shards. Without this kwarg the default applies a sample-level
  filter on top of WebDataset's already-shard-level split, silently cutting
  throughput.

### 6.2 Deterministic per-rank shard scheduling

`canvit_pretrain/train/data/schedule.py` provides `compute_schedule_slice` and
`compute_shards_per_gpu`. Each rank receives a **deterministically computed**
list of shard paths — function of `(steps_per_job, batch_size_per_gpu,
job_index, world_size, rank)`. No NCCL involved.

The constraint `steps_per_job * batch_size_per_gpu * world_size %
samples_per_shard == 0` is asserted at construction time so shard boundaries
align cleanly with job boundaries (no partial last shard, no skipped
samples).

### 6.3 `num_workers` cap

To avoid OOM on long jobs (where `shards_per_gpu` becomes large), the loader
caps `num_workers` at `shards_per_gpu`, rounded down to a divisor:

```python
requested = max(1, num_workers)
capped = min(requested, self.shards_per_gpu)
nw = capped
while self.shards_per_gpu % nw != 0:
    nw -= 1
self.num_workers = nw
```

The default `cfg.num_workers = 4` in `config.py` keeps memory bounded for
typical step-per-job values. Each worker streams `shards_per_gpu /
num_workers` shards sequentially.

### 6.4 `batch_size_per_gpu` (clarity)

The config field is named `batch_size_per_gpu` (not `batch_size`) to make it
obvious that the per-rank batch count is independent of `world_size`. The
global batch is `batch_size_per_gpu * world_size`.

---

## 7. SLURM / sbatch infrastructure

**File: `slurm_nhr/tmux_resubmit/train_ddp_v100_tmux_resubmit.sbatch`**

The DDP launch script is responsible for two non-trivial things:

### 7.1 Per-job `MASTER_PORT`

Multiple DDP jobs can be packed onto the same node on a `grete:shared`-style
partition. A hardcoded `MASTER_PORT` causes `EADDRINUSE` collisions. Compute
a unique-per-job port from the job ID:

```bash
export MASTER_PORT=$((20000 + (SLURM_JOB_ID % 10000)))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
```

### 7.2 Per-node cache wipe at job start

Cancelled prior jobs can leave half-written Triton kernels, stale lock files,
KMP shared-memory entries, and NVIDIA module-cache state on a compute node.
On reuse this poisons subsequent runs. Wipe cleanly at the start of each
job:

```bash
srun --ntasks-per-node=1 --ntasks=$SLURM_NNODES bash -c '
    rm -rf /tmp/triton_cache_rank* /tmp/torchinductor_* /tmp/nvrtc_cache_* /tmp/.nv 2>/dev/null || true
    rm -rf $HOME/.triton/cache 2>/dev/null || true
    rm -rf $HOME/.nv/ComputeCache 2>/dev/null || true
    find /dev/shm -maxdepth 1 -user $USER \( -name "__KMP_REGISTERED_LIB_*" -o -name "torch_*" -o -name "python_*" -o -name "pymp-*" \) -delete 2>/dev/null || true
'
```

### 7.3 The launch line

The script's training invocation:

```bash
exec srun "$VENV/bin/python" -m canvit_pretrain.train \
    --webdataset-dir "$WEBDATASET_DIR" \
    --ckpt-dir "$CHECKPOINTS_DIR" \
    --wandb-project "${WANDB_PROJECT:-canvit-pretrain}" \
    ${WANDB_ENTITY:+--wandb-entity "$WANDB_ENTITY"} \
    ${WANDB_DIR:+--wandb-dir "$WANDB_DIR"} \
    --batch-size-per-gpu 64 \
    --steps-per-job 128 \
    --val-every 512 \
    "$@"
```

`srun` provides `SLURM_PROCID`, `SLURM_GPUS_ON_NODE`, etc. that `dist.py`
reads. `--ntasks-per-node` must equal `--gpus-per-node` (one task per GPU).

The relevant `#SBATCH` directives:

```
#SBATCH -p grete:shared
#SBATCH --nodes=1
#SBATCH --gpus-per-node=A100:2
#SBATCH --ntasks-per-node=2      # MUST equal --gpus-per-node
#SBATCH --cpus-per-task=16
#SBATCH -C inet                  # required for HuggingFace Hub access
```

### 7.4 Hardware capability fallback (`loop.py`)

For nodes without sm_80+ (e.g. V100 = sm_70), FlashAttention and bfloat16
must be disabled — otherwise import-time crashes. Guard at the top of
`loop.py`:

```python
if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
# else: keep PyTorch defaults (mem_efficient + math SDPA fallback)
```

The same code path handles AMP dtype selection (bf16 on sm_80+, fp16
otherwise) elsewhere in the loop.

---

## 8. Reproduction checklist

Starting from `1a36eec` (no DDP at all), the minimal set of additions to make
DDP work end-to-end:

**New file:**

- [ ] `canvit_pretrain/train/dist.py` (§1) — singleton with `init_dist`,
      rank/device accessors, `barrier`, `broadcast_module_buffers`,
      `all_reduce_mean`.

**Edits to `canvit_pretrain/train/__main__.py`:**

- [ ] Prepend matplotlib `MPLCONFIGDIR` redirect + force `Agg` backend (§4).

**Edits to `canvit_pretrain/train/loop.py`:**

- [ ] Call `ddp.init_dist()` at the top of `train()`, set `cfg.device =
      ddp.device()` (§2.1).
- [ ] Rank-0 normalizer init + barrier + explicit
      `broadcast_module_buffers(scene_norm)` and `broadcast_module_buffers(cls_norm)`
      (§2.2).
- [ ] DDP wrap with `broadcast_buffers=False`, `find_unused_parameters=False`,
      followed by `model._set_static_graph()` (§2.3, §3.1, §3.2).
- [ ] Keep `core_model` reference; pass it to `validate()` and to
      `save_checkpoint` (§2.3).
- [ ] Rank-0 guards on FAILED marker, run_dir creation, tqdm,
      `cancel_slurm_array`, all checkpoint save sites — followed by
      `ddp.barrier()` (§2.4, §2.6).
- [ ] `device=cfg.device` on the `n_glimpses` EMA tensor (§3.4).
- [ ] sm_80+ capability check around FlashAttention enable (§7.4).
- [ ] `all_reduce_mean` on metric values before logging (§2.5).

**Edits to `canvit_pretrain/train/step.py`:**

- [ ] `core_model = getattr(model, "module", model)` at the top of
      `training_step` (§3.3).
- [ ] Route `init_state`, `predict_teacher_scene`, `predict_scene_teacher_cls`,
      `get_spatial` calls through `core_model` (§3.3).
- [ ] Call `extract_sample0_viz(..., core_model)` instead of `extract_sample0_viz(..., model)`
      at the two viz sites in `run_branch` (§3.3).
- [ ] Broadcast `n_glimpses` from rank 0 right after sampling (§3.5).

**Edits to `canvit_pretrain/train/config.py`:**

- [ ] Rename `batch_size` → `batch_size_per_gpu` (§6.4).
- [ ] Default `num_workers = 4` (§6.3).
- [ ] Add `webdataset_dir`, `seed`, and any other DDP-relevant fields used by
      the loaders.

**Edits to `canvit_pretrain/train/data/__init__.py`,
`canvit_pretrain/train/data/webdataset.py`,
`canvit_pretrain/train/data/schedule.py`:**

- [ ] WebDataset path with explicit `workersplitter=wds.split_by_worker,
      nodesplitter=None` (§6.1).
- [ ] Deterministic per-rank shard scheduling (§6.2).
- [ ] `num_workers` cap at `shards_per_gpu` rounded to divisor (§6.3).

**Edits to `canvit_pretrain/train/viz/validate.py`:**

- [ ] None — keeps the original `1a36eec` form (`inference_mode` +
      `model.eval()` toggle inside try/finally) (§5).

**SLURM script (e.g. `slurm_nhr/tmux_resubmit/train_ddp_v100_tmux_resubmit.sbatch`):**

- [ ] `#SBATCH --nodes=1 --gpus-per-node=A100:2 --ntasks-per-node=2 --cpus-per-task=16` (§7.3).
- [ ] Per-job `MASTER_PORT=$((20000 + (SLURM_JOB_ID % 10000)))` (§7.1).
- [ ] `MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)`.
- [ ] `WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))`.
- [ ] Per-node cache-wipe srun step at job start (§7.2).
- [ ] Launch via `exec srun "$VENV/bin/python" -m canvit_pretrain.train ...`.

After all of the above, DDP training + validation + wandb logging + PCA
plotting works end-to-end. Empirical evidence: 30/30 DDP sanity runs and
20/20 single-GPU sanity runs all reached `step=256` with `val_loss ≈ 2.0019`
on clean HEAD, no errors of any kind.

---

## 9. What's *not* needed (and why)

During the original DDP debugging effort, several modifications were
introduced because they appeared to fix DDP failures. After the matplotlib
NFS-cache bug (§4) was identified and fixed, a controlled re-test removed
each of these modifications independently — keeping all other fixes in
place — and re-ran the standard DDP sanity test (3 jobs per modification,
each running training + validation + wandb logging + PCA plotting on clean
HEAD; pass criterion was reaching `step=128` with checkpoint saved within
15 minutes). The modifications listed here all passed 3/3 in that re-test:
**they are not required for correctness**, even though earlier observations
seemed to suggest they were.

The reason most of these "looked essential" originally is that the
matplotlib hang was **stochastic** (~45 % of multi-rank runs) and silent
(no error message — rank 0 froze inside a `pyplot` call, rank 1 deadlocked
at the next NCCL collective). When a hang is the only signal you have, you
can't easily distinguish "removing fix X caused the hang" from "the run
would have hung anyway because matplotlib was unlucky this time". With
matplotlib fixed, the silent-hang signal cleared up entirely and each of
these modifications could be tested cleanly.

This section documents each one: why it was introduced, what would
hypothetically go wrong if you didn't have it, and the empirical evidence
that — at least in this codebase as currently structured — you actually
don't need it.

### 9.1 NCCL `timeout=timedelta(minutes=30)` on `init_process_group`

**Originally introduced because**: in early DDP runs, validation appeared
to hang for >10 minutes on cold-cache nodes. The hypothesis was that
`torch.compile`'s first-time Triton kernel compilation during validation
(under `inference_mode`, see §9.6) takes 15–20 minutes on a cold cache,
which exceeds NCCL's default 10-minute watchdog timeout. Bumping the
timeout to 30 minutes was thought to give compilation time to finish.

**Why this turned out wrong**: the apparent hang was matplotlib, not
Triton compilation. Real Triton compilation completes in seconds-to-tens-of-
seconds, well within the default 10-minute NCCL watchdog. Empirical
re-test: removing the 30-minute override (so `init_process_group` uses the
default 10-minute timeout) → 3/3 PASS.

**Conclusion**: don't pass `timeout=` to `init_process_group`. The default
is fine.

### 9.2 Triton + Inductor cache redirected to per-rank `/tmp`

**Originally introduced because**: cold-cache validation appeared to hang
forever, and the suspicion was NFS lock contention on the default
`~/.triton/cache` (which lives on shared NFS). The proposed fix redirected
both `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` to per-rank-per-job
paths under `/tmp` to localize the cache to node-local disk:

```python
# (NOT NEEDED — kept here only as a record of what was tried)
job_id = os.environ.get("SLURM_JOB_ID", "nojob")
os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_rank{rank}_job{job_id}"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"/tmp/torchinductor_rank{rank}_job{job_id}"
```

**Why this turned out wrong**: the apparent NFS-lock hang was matplotlib.
The default NFS cache works fine for Triton/Inductor compilation under DDP,
particularly because the sbatch script's cache-wipe step (§7.2) clears any
stale state at job start. Empirical re-test: removing the redirect → 3/3
PASS with default `~/.triton/cache`.

**Conclusion**: don't redirect Triton/Inductor caches in `dist.py`. The
sbatch-level cache wipe is sufficient.

### 9.3 All-ranks normalizer initialization

**Originally introduced because**: an early attempt at
"rank-0-only init + explicit broadcast" (the form documented in §2.2)
appeared to hang, and the diagnosis was that issuing any NCCL collective
(`broadcast_module_buffers`) *before* `DistributedDataParallel.__init__`
corrupts NCCL state in a way that makes the next collective inside the DDP
constructor fail or hang. The proposed workaround was to have all ranks
read the same first shard independently and compute identical stats —
deterministic input + deterministic compute → identical buffers without any
NCCL traffic before DDP wrap.

**Why this turned out wrong**: the apparent NCCL-state-corruption hang was
matplotlib. There's no real NCCL state issue; pre-DDP collectives are fine.
Empirical re-test: rank-0-only init + explicit broadcast (the §2.2 form)
→ 3/3 PASS.

The §2.2 approach is preferable to all-ranks init because:
1. Only rank 0 reads the shard — single NFS read, not `world_size` reads.
2. Bit-identical buffers across ranks (NCCL broadcast guarantees byte
   equality; all-ranks-compute relies on floating-point determinism, which
   is *almost* always identical but isn't formally guaranteed).
3. Smaller code surface — no need to restructure the init block to remove
   the rank-0 guard.

**Conclusion**: use the §2.2 form (rank-0-only init + explicit
`broadcast_module_buffers`).

### 9.4 `find_unused_parameters=True` in the DDP constructor

**Originally introduced because**: without it, an early DDP run crashed
with "Parameter indices which did not receive grad for rank 0: 0 1 2".
This is the standard PyTorch DDP error you get when some parameters don't
receive gradients in every backward pass — multi-branch TBPTT (the codebase
uses 1 full-start branch + 1 random-start branch per step) means some
parameters are only used in one branch's computation graph, so the all-
reduce on those parameters never fires and DDP complains.

**Why this turned out wrong (kind of)**: the *crash* without
`find_unused_parameters=True` is real — but only when `_set_static_graph()`
(§3.2) is *not* set. With `_set_static_graph()` set (which we do, for the
TBPTT chunk-shared param hook collision), DDP records on the first backward
which parameters are used and which aren't, and stops complaining.
PyTorch's own docstring on `_set_static_graph` actually says it makes
`find_unused_parameters=True` redundant. Empirical re-test: with
`_set_static_graph()` in place, removing `find_unused_parameters=True`
(i.e. leaving it at the default `False`) → 3/3 PASS.

**Conclusion**: pass `find_unused_parameters=False` (the default) to the
DDP constructor. `_set_static_graph()` does the equivalent work and is
already required for an unrelated reason.

### 9.5 Rank-0 gate around the validation block

**Originally introduced because**: the early form of the codebase ran
`validate()` on all ranks. There were two concerns: (a) two ranks doing
`torch.compile`-driven Triton compilation simultaneously could thrash the
shared NFS Triton cache, slowing or hanging compile; (b) two ranks running
validation could write duplicate metrics to wandb. The proposed fix wrapped
the entire validation block in `if ddp.is_main():`, with a `ddp.barrier()`
afterward so non-main ranks waited.

**Why this turned out wrong**: (a) matplotlib was the actual hang cause;
the Triton concurrent-compile concern is empirically not a problem with
the per-job cache wipe in §7.2. (b) The wandb tracker uses `make_tracker`,
which returns a no-op `DummyExperiment` on non-main ranks — so the call
sites `exp.log_*` are always safe to invoke on rank 1, they just no-op.
There's no double-logging because rank 1 has no real wandb run. Empirical
re-test: removing the rank-0 gate so both ranks run validation → 3/3 PASS,
no doubled metrics.

**Conclusion**: don't gate validation on rank 0. The current code (§5)
intentionally calls `validate()` from both ranks — which costs a little
extra compute on rank 1 but simplifies the control flow and matches the
pre-DDP single-rank behavior.

### 9.6 `torch.no_grad()` instead of `torch.inference_mode()` in `validate.py`

**Originally introduced because**: `torch.inference_mode()` is treated by
`torch.compile` as a *distinct compilation context* — entering it for the
first time triggers compilation of an inference-mode-specific kernel set,
separate from the training-mode kernels already compiled. This compilation
appeared to hang for >30 minutes on cold-cache nodes, masking validation
entirely. The proposed fix replaced `torch.inference_mode()` with
`torch.no_grad()`, which is *transparent* to `torch.compile` (it doesn't
change traced values, so the same compiled kernels are reused).

**Why this turned out wrong**: the >30-minute "compile hang" was
matplotlib. Real `inference_mode` compilation does cost a few extra seconds
of one-time work — measured ~4 seconds in the redistillation — but
completes well within the NCCL watchdog. Empirical re-test: keeping
`torch.inference_mode()` (the original `1a36eec` form) → 3/3 PASS.

**Conclusion**: keep `torch.inference_mode()` in `validate.py`. It's the
correct PyTorch idiom for validation (stricter than `no_grad`, see below)
and it works fine in this codebase. The one-time compile cost is
negligible.

`inference_mode()` does more than `no_grad()`: it disables version
counters and view tracking on tensors created inside it, giving lower
runtime overhead and memory savings. Tensors made under it can't be mixed
back into a gradient-tracked computation later — but validation doesn't
need to do that.

### 9.7 Removing the `model.eval()` / `model.train()` toggle in `validate.py`

**Originally introduced because**: similar reasoning to §9.6. Calling
`model.eval()` flips `model.training` from `True` to `False`, and
`torch.compile` traces `model.training` as a flag — flipping it produces a
*different* compiled graph than training mode, requiring fresh Triton
compilation on first call. This compilation appeared to hang. The proposed
fix removed the eval/train toggle entirely, leaving the model in training
mode throughout validation.

**Why this turned out wrong**: same as §9.6 — the hang was matplotlib.
The eval/train toggle does cost a one-time compile (measured ~18 seconds
in the redistillation, the dominant compile-cost contributor) but completes
fine. Empirical re-test: keeping the `model_was_training; model.eval();
try: ... finally: model.train(model_was_training)` block (the original
`1a36eec` form) → 3/3 PASS.

**Conclusion**: keep the eval/train toggle in `validate.py`. It's the
correct PyTorch idiom — it ensures Dropout doesn't drop units during
validation (deterministic metrics) and BatchNorm uses its stored running
stats rather than corrupting them with validation batches. The one-time
compile cost is amortized over the rest of the job.

The Dropout/BatchNorm semantics matter even though *this particular* model
may not currently use either — the discipline of eval/train toggling
during validation is the kind of thing that quietly breaks future code if
it's ever removed and someone adds a Dropout layer two months later.

### 9.8 `device_id=torch.device("cuda", _local_rank)` in `init_process_group`

**Originally introduced because**: PyTorch's `init_process_group` accepts
a `device_id` kwarg that pins NCCL to a specific CUDA device. Without it,
NCCL infers the device from the global rank — which under naive launch
patterns might not match local rank.

**Why this turned out partially wrong**: under SLURM single-node DDP, the
launch script sets `torch.cuda.set_device(_local_rank)` before
`init_process_group`, and `SLURM_PROCID` (= global rank) equals local rank
because there's only one node. NCCL's inference therefore picks the right
device. Empirical re-test: removing `device_id` → 3/3 PASS on single node.

**However**: in **multi-node** DDP, global rank ≠ local rank. NCCL would
then guess wrong, and the kwarg becomes essential.

**Conclusion**: omit `device_id` for single-node DDP (the current form).
**Re-introduce it for multi-node DDP** — add
`device_id=torch.device("cuda", _local_rank)` to the
`init_process_group(...)` call before scaling out beyond one node.

### 9.9 `torch.distributed.broadcast` of `n_glimpses` (kept anyway as robustness)

This one is in HEAD (and documented as essential in §3.5), but the redistill
showed empirically that it can be removed without immediate breakage:
both ranks happen to have identical Python `random` state and execute
identical code paths, so `random.random()` produces the same `n_glimpses`
value deterministically on both. Empirical re-test (broadcast removed) →
3/3 PASS.

**Why we keep it anyway**: the determinism is fragile. *Anything* that
perturbs one rank's random state but not the other (per-rank seeding for
data augmentation, a rank-conditional `if ddp.is_main()` block that calls
`random.random()`, a third-party library that draws from the global
`random` instance differently per rank) would silently desync the trajectory
length, and the symptom would be a hang at the next NCCL collective with
no useful traceback. The broadcast is 3 lines of code and costs sub-
millisecond per step; the robustness it buys is worth it.

This is the **only** case in this section where we deliberately keep code
that the empirical re-test classified as non-essential. Every other §9
modification is genuinely absent from HEAD.

### 9.10 Summary table

| Modification | Verdict | Action in HEAD |
|--------------|---------|----------------|
| §9.1 NCCL 30-min timeout | non-essential | omitted |
| §9.2 Triton/Inductor cache → /tmp | non-essential | omitted |
| §9.3 All-ranks normalizer init | non-essential | omitted (§2.2 used instead) |
| §9.4 `find_unused_parameters=True` | non-essential (with §3.2 in place) | omitted |
| §9.5 Rank-0 gate on validation | non-essential | omitted |
| §9.6 `no_grad()` instead of `inference_mode()` | non-essential | original `inference_mode` kept |
| §9.7 Removing `model.eval()` toggle | non-essential | original toggle kept |
| §9.8 `device_id` kwarg in init_pg | non-essential single-node | omitted (re-add for multi-node) |
| §9.9 `n_glimpses` broadcast | non-essential, but robustness guard | **kept** — see §3.5 |
