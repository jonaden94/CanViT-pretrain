# Old-frontend viewpoint-Q — 8-seed reliability band vs EG-C2F (2026-06-17/18)

> Naming: this band's frontend is the **old-frontend** variant (shared concat→proj→32-d bottleneck). Its
> frozen artifacts keep their original `curated` names (git tag `result/curated-8seed-band`, run dirs
> `seedband_curated_s*`, `preserved_ckpts/seedband_curated/`) — those are immutable, so "curated" == "old-frontend"
> wherever it appears below. HEAD's frontend (per-group proj + per-group LayerNorm) is the current one.

The matched-budget seed band that turns the single-seed "beats EG-C2F" / "≈ anchor" claims into mean±std.
**Headline: the old-frontend viewpoint-Q policy beats EG-C2F-c64 on CE (the optimized metric) at every horizon by
many σ of the seed band.** Reproducible numbers, git tags, and checkpoints below.

## Provenance (git tags — recover exactly)
- **`result/curated-8seed-band`** → commit `55fda96`. The band's ARCH = the old-frontend
  (BN(2052) + per-group gate + SHARED 2052→32 proj). **The band ckpts ONLY load at this tag** — HEAD has since
  moved to a per-group-LN frontend (different params; not migratable).
- **`result/egc2f-c64-baseline`** → commit `e805277`. EG-C2F-c64 T=5 deterministic baseline.
- Recipe = the `QConfig` defaults at the band commit: `input_mode=curated`, c64 (`canvas_grid=64`),
  `scales=(0.5,0.25)`, `train_horizon=4` (T=5 = t0..t4), `lr=3e-4`, `wd=1e-2`, `width=128`, `block_layers=3`,
  `curated_dim=32`, 1M `budget_forwards` (→ 12500 steps), `prime_on_policy=0.5`, `warmup_frac=0.1`. seed 0..7.

## EG-C2F-c64 (the bar, deterministic) — `result/egc2f-c64-baseline`
| t | t0 | t1 | t2 | t3 | t4 |
|---|---|---|---|---|---|
| mIoU | 39.60 | 42.22 | 43.30 | 44.04 | 44.65 |
| CE | 0.7649 | 0.7258 | 0.7004 | 0.6828 | 0.6707 |

Repro: `uv run python -m canvit_pytorch_rl.baselines.evaluate --policy entropy_coarse_to_fine --n-timesteps 5 --canvas-grid 64 --run-name egc2f_c64_t5_ce` (per-t CE+mIoU in the run's `summary.json`).

## Old-frontend 8-seed band — `result/curated-8seed-band`
Per seed (matched 1M endpoint = final eval; deploy = best val_ce_t4 ckpt):

| seed | final ce_t4 | final mIoU_t4 | best ce_t4 @step | best mIoU_t4 |
|---|---|---|---|---|
| 0 | 0.6663 | 44.72 | 0.6645 @9k | 44.94 |
| 1 | 0.6658 | 44.70 | 0.6644 @7k | 45.19 |
| 2 | 0.6646 | 44.83 | 0.6640 @8k | 44.93 |
| 3 | 0.6644 | 44.88 | **0.6636 @9k** | 44.91 |
| 4 | 0.6664 | 44.61 | 0.6641 @10k | 44.90 |
| 5 | 0.6658 | 44.91 | 0.6649 @12k | 44.84 |
| 6 | 0.6667 | 44.81 | 0.6660 @12k | 44.85 |
| 7 | 0.6660 | 44.64 | 0.6651 @9k | 44.67 |

**Bands (n=8):**
- **Matched endpoint (final eval — the rigorous claim):** CE t1–t4 = `0.7151±0.0012 / 0.6887±0.0009 /
  0.6751±0.0009 / 0.6657±0.0008`; mIoU = `42.59±0.12 / 43.76±0.12 / 44.35±0.09 / 44.76±0.11`.
- **Deploy (best.pt, optimistic eval-selection):** ce_t4 `0.6646±0.0008`, mIoU_t4 `44.91±0.15`.

Repro: at the tag, deploy → `nohup setsid uv run python throwaway/seed_band.py`; aggregate with
`uv run python -m canvit_pytorch_rl.tools.seed_report`. Per-step trajectories: `tools.sweep_report`-style
via `throwaway/run_traj.py --glob 'runs/*seedband_curated_s<N>*/metrics.jsonl'`.

## The result: old-frontend CE beats EG-C2F at every t, robustly
| t | EG-C2F CE | old-frontend endpoint CE (n=8) | margin | in σ of band |
|---|---|---|---|---|
| t1 | 0.7258 | 0.7151±0.0012 | **+0.0107** | ~9σ |
| t2 | 0.7004 | 0.6887±0.0009 | +0.0117 | ~13σ |
| t3 | 0.6828 | 0.6751±0.0009 | +0.0077 | ~8σ |
| t4 | 0.6707 | 0.6657±0.0008 | **+0.0050** | ~6σ |

The CE win is largest at **t1** (the first glimpse) and shrinks by t4. On the noisier mIoU the margin is
smaller (+0.11 endpoint / +0.26 deploy at t4) — the documented CE↔mIoU misalignment; **judge by CE**. The old
single-seed "44.94 peak" was best-of-trajectory optimism; the seeded endpoint is mIoU_t4 44.76±0.11.

## Checkpoints (`crockett:~/projects/CanViT-PyTorch-RL/preserved_ckpts/`, gitignored; load at the tag)
- `seedband_curated/s3_best_step9000.pt` (best seed by val_ce_t4: 0.6636), `s3_last_step12500.pt`.
- All 8 seed runs persist in `runs/*seedband_curated_s{0..7}` (full per-step `metrics.jsonl`).
- These are the old-frontend arch — **`git checkout result/curated-8seed-band` before loading** (current code's
  frontend differs). All other preserved ckpts likewise load only at their git tags (see
  `docs/preserved_checkpoints.md`).
