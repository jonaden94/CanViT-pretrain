# How to check whether dataloading is the bottleneck

The training loop already logs the data-vs-GPU breakdown every
`cfg.log_every` steps (see `canvit_pretrain/train/loop.py`, the metric block
that ends in `exp.log_metrics(...)`). This document explains what to look at
and what to do if you find a bottleneck.

## TL;DR

After a real (not smoke-test) job has been running long enough for warm-up
to settle (say, 1000+ steps past JIT compile), look at:

```
train/data_pct   — % of cumulative wall time spent waiting for the loader
train/gpu_pct    — % of cumulative wall time spent on the GPU step
```

| `data_pct` | Verdict |
|---|---|
| < 30% | GPU-bound. Dataloader is not the bottleneck. Do nothing. |
| 30–60% | Mostly GPU-bound. Dataloader has slack but isn't hurting much. Optional: revisit if you scale GPUs further (per-GPU step gets faster → ratio shifts). |
| 60–80% | Dataloader-bound at the margin. Worth tuning. |
| > 80% | Dataloader-bound. GPU sitting idle. Fix this. |

These percentages are cumulative-since-job-start, not instantaneous, so
they're robust to single slow shards (e.g. cold filesystem cache on the very
first one).

## Where the numbers come from

In `loop.py`:

```python
t_data_total += time.perf_counter() - t_data_start  # before load_train_batch
t_gpu_total  += time.perf_counter() - t_gpu_start   # around training_step + optimizer
...
metrics["train/data_pct"] = t_data_total / (t_data_total + t_gpu_total) * 100
metrics["train/gpu_pct"]  = t_gpu_total  / (t_data_total + t_gpu_total) * 100
```

Important caveats:

- The first batch includes `torch.compile` JIT cost (logged once as "First
  training_step took Ns (includes compile)"). **Ignore the first ~50–100
  steps** when reading `data_pct`.
- Worker startup is also slow on the first iteration (cold filesystem cache,
  shard `mmap`/tar header parse). The cumulative ratio absorbs this within a
  few hundred steps.
- These timers are CPU-side only. They do not include GPU kernels that
  overlap async with data prefetch. As long as `data_pct < 50%`, you have
  headroom.

## What to do if `data_pct` is high

In order of cost:

### 1. Bump `prefetch_factor`

Currently 2 in `webdataset.py`. Try 4. Costs RAM proportionally; helps if I/O
is bursty (slow shards followed by fast ones).

```python
# canvit_pretrain/train/data/webdataset.py — _ensure_iter / _build_loader
prefetch_factor=4 if self.num_workers > 0 else None,
```

### 2. Decouple `num_workers` from `shards_per_gpu`

The current design pins `num_workers = shards_per_gpu` (one shard per worker).
If you have spare CPU cores per rank and shards are big, you can run more
workers than shards by switching from `wds.split_by_worker` (shard-level) to
sample-level distribution. Moderate refactor in `webdataset.py`.

### 3. Switch JPEG decode to GPU

JPEG decode on CPU is the dominant per-sample cost. PyTorch has
`torchvision.io.decode_jpeg(..., device="cuda")` and NVIDIA's `nvjpeg`. The
decode happens on the GPU between forward passes; CPU is freed up.

Drop-in replacement for `_decode_jpg` in `webdataset.py`:

```python
import torch
from torchvision.io import decode_jpeg

def _decode_jpg_gpu(data: bytes, image_size: int, device: torch.device) -> Tensor:
    raw = torch.frombuffer(data, dtype=torch.uint8)
    img = decode_jpeg(raw, device=device)  # [C, H, W] uint8
    # then resize/normalize on GPU
    ...
```

Caveats: GPU decode requires sending raw JPEG bytes (small) instead of
decoded tensors (big) to GPU; total transfer cost goes *down*. But it
serializes with the model forward unless you use a separate CUDA stream.
Worth it if CPU JPEG decode is genuinely the bottleneck.

### 4. Profile to find the real culprit

If 1–3 don't move the needle, instrument per-step:

```python
import torch.profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=10, warmup=10, active=20, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profile"),
) as prof:
    for step in range(50):
        ...
        prof.step()
```

Open the trace in `chrome://tracing` or TensorBoard. Look for: long PIL
calls, idle GPU gaps, worker queue starvation.

Existing per-shard timing logs in `shards.py:158–168` (the legacy path) show
read/transform/patch times per worker; the WebDataset loader could be
extended with similar instrumentation if you want shard-level granularity
without a full profiler run.

### 5. Last resort: storage-format rework (concern #5 in followup doc)

Move features into a sidecar `.pt` per shard, mmap them. Speeds up the
feature-load step but not the JPEG decode. Worth it only if you've
exhausted 1–4 and `data_pct` is still high.

## What to look at in `nvidia-smi`

A quick orthogonal sanity check during a live job:

```bash
nvidia-smi dmon -s u -i 0 -d 1
```

If GPU utilisation hovers below ~80% with periodic dips to 0%, the loader is
starving the GPU. Confirms what `data_pct` already tells you, but useful as a
visual.

## Concrete first step

Once your first real job has run for 5–10 minutes:

```bash
# In Comet / W&B, plot train/data_pct and train/gpu_pct over step.
# Or if you want a quick local read:
grep "data_pct\|train/gpu_pct" logs/train-*.out | tail -20
```

If `data_pct < 50%`: stop worrying. Move on.
If `data_pct > 50%`: try step 1 (prefetch_factor=4), re-measure, then step 3
(GPU JPEG decode) if needed.
