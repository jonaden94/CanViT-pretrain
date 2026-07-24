# DDP validation (unified harness) — results

Validated on **2× A100 (grete:shared, NCCL)** — the real production target. The jupyter
V100/RTX5000 partition was abandoned as a playground: its NCCL segfault (below) is a
partition artifact, and A100 is what production runs on.

## What was proven

| Property | Result | Evidence |
|---|---|---|
| NCCL transport on A100 (with & without `device_id`) | ✅ | `ddp_transport_probe.py`: all_reduce sum correct both ways |
| Cross-rank parameter identity after training | ✅ exact (`0.000e+00`) | `harness_ddp_smoke.py` — task_only **and** joint |
| Standard distill (`--preset default`) trains under DDP | ✅ | real run, 40 steps, exit 0 (job 15035672) |

**Cross-rank identity is the property that makes DDP safe.** The two ranks train on
*different* data yet end bit-identical — only possible if `allreduce_grads` averages every
step and all ranks apply the same update (no silent per-rank drift). The real distill run
confirms it end-to-end: reduced metrics + `grad_norm` identical across ranks, only the
rank-local `total_loss_raw` differs, and the stochastic `n_glimpses` is broadcast so both
ranks agree.

**Directly observed, not inferred** (`ddp_data_grad_check.py`, real distill webdataset path,
2× A100). Per rank, from the production loader + `allreduce_grads`:

| | rank 0 | rank 1 |
|---|---|---|
| shards assigned (8 each) | `82,610,1318,1751,2455,2563,2991,3090` | `85,1230,1331,1344,1525,2717,2870,3018` — **disjoint** |
| batch fingerprint | `+1.32e5` | `−8.37e5` — **different data** |
| grad norm BEFORE AllReduce | `5.0446e-2` | `6.8718e-2` — **differ** |
| grad norm AFTER AllReduce | `4.257518e-2` | `4.257518e-2` — **identical** |

The full chain is confirmed: different shards → different data → different per-rank
gradients → averaged to identical. (The averaged norm `4.26e-2` is below *both* pre-average
norms, i.e. the rank gradients point in partly different directions — real per-rank grads,
not a scale artifact.) Shard partitioning is `compute_schedule_slice`: rank r gets
`job_block[r*shards_per_gpu:(r+1)*shards_per_gpu]` of a `shards_per_gpu*world_size` block.

The averaging **scale** is `dist.all_reduce(g, SUM); g /= world_size` (harness/ddp.py) —
identical to the battle-tested `train/dist.py::all_reduce_mean` the old DDP trainer used.

## Why "DDP == single-GPU-2B" does NOT hold here (and why that's fine)

A synthetic 2×(batch B) vs 1×(batch 2B) equivalence check fails, but it is **not a sync
bug** — it fails for two model/loss reasons, both single-process facts with no DDP involved:

1. **The rollout draws a stochastic glimpse count** (`continue_prob=0.5`) that DDP
   broadcasts from rank 0, so a 1-GPU run takes a different trajectory by construction.
2. **The loss is not batch-linear.** The ade20k loss normalizes per-sample by valid-pixel
   count and is reduced across the batch/glimpses; the seg forward is per-sample independent
   (verified: logits batch-4 vs batch-2 match to 3e-7) but the batch reduction is not a
   plain per-sample mean, so `grad(2B) ≠ mean(grad(B), grad(B))`.

The old repo's DDP wrapper averaged gradients identically, so the harness **matches the old
repo's DDP semantics** (the relevant baseline), and per-rank behaviour matches the parity
digest. A single-GPU-large-batch reference was never the target.

## DDP support matrix (enforced by `check_spec(..., is_dist=True)`)

- ✅ **Supported:** every spec with `policy_grad_to_backbone=False` — distill standard,
  `probe`, `finetune`, `policy_only`, and `joint` (policy detached from the backbone).
- ❌ **Rejected upfront (hard error):** `policy_grad_to_backbone=True` — the policy→backbone
  path runs through the unwrapped core model and would bypass AllReduce (silent per-rank
  drift). This is the one cell that cannot train under DDP; the harness refuses it with a
  clear message rather than mis-training. (Single-GPU it is unaffected.)

## Harness CLI (superseded 2026-07-24 — now full tyro parity)

The stopgaps used during DDP validation (env-driven `VAL_DIR`/`VAL_INDEX_DIR` and a
hand-rolled `--steps-per-job`) are **gone**. `canvit_pretrain/harness/cli.py` now runs
`tyro` over each task's own config dataclass — the same idiom as the three standalone
entry points — so every field is reachable, including `--cfg.model.patcher-name` (fovi)
and the `--cfg.foveated-scale.*` tree. Invocation changed to subcommands:

```
python -m canvit_pretrain.harness.run distill --cfg.webdataset-dir ... --cfg.val-dir ...
```

`RunSettings` is DERIVED from the task config (`cfg.steps_per_job` → `n_steps`,
`cfg.val_every` → `eval_every`, plus compile/amp/grad_clip/tracker/wandb), so there is no
second place to set the same knob. See `unification_docs/09-cli-and-checkpoint.md`.

## The jupyter-partition NCCL segfault (resolved as a non-issue)

Earlier multi-GPU tests on the `jupyter` partition (V100/RTX5000) segfaulted on the first
NCCL collective. This does **not** reproduce on A100 with the *same* NCCL 2.28.9 — it was a
partition artifact, not harness code and not the `device_id` kwarg (both probe variants pass
on A100). The temporary `CANVIT_DIST_BACKEND=gloo` escape hatch added to `train/dist.py` for
that partition has been **reverted** (V100 is only a playground; production is A100/NCCL).

## Scripts (unification_docs/)

- `ddp_smoke_a100.sbatch` — transport probes + cross-rank identity smoke (task_only + joint).
- `harness_ddp_smoke.py` / `harness_ddp_compare.py` — the smoke and its verdict.
- `ddp_transport_probe.py` — dependency-free NCCL all_reduce probe (± `device_id`).
- `ddp_distill_a100.sbatch` — the real distill DDP run through the harness entry point.

## Still gated (not done here)

The **big-bang cutover** (repoint `python -m canvit_pretrain.train` at the harness, delete
the old loops) remains owner-gated and destructive. Before it: parity probe prints
`9a0100a1a3de3acd`, full suite green, launchers set `eval_every == cfg.val_every`.
