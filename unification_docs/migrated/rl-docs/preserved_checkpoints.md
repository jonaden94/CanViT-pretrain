# Preserved checkpoints + analysis (2026-06-16)

**Frozen pre-Hub record.** From 2026-07-04, headline checkpoints are preserved by publication to the
HF Hub (`tools.publish_policy`; published sets are recorded in their results docs, e.g.
`qband_results.md`). This doc covers the earlier era only.

Backup of the checkpoints analyzed this session →
`crockett:~/projects/CanViT-PyTorch-RL/preserved_ckpts/` (gitignored; THIS doc is the tracked record).
Each `<stem>.pt` original (pre-2026-06-17, old `RichAuxNet`/twin-era schema) has a `<stem>_unified.pt` copy
[2026-06-17, via `throwaway/migrate_qnet_keys.py --apply`]. **NOTE [2026-06-18]: these no longer load under
HEAD** — the single-`Frontend` refactor deleted `CanvasFrontend` (so canvas-mode anchor ckpts) and changed
the frontend params (so the old-frontend ckpts). Load any of them at its recorded git tag / `git_rev`, not
under current code. **mIoU** = full-val 512²
paper protocol (×100); **val CE** = the optimized objective (lower better).
[user 2026-06-16: always report BOTH metrics; keep best-by-CE AND best-by-mIoU per horizon.]

## Preserved checkpoints (headline numbers)

| ckpt | run / step | git_rev | t1 | t2 | t3 | t4 (mIoU) | CE@t4 | remark |
|---|---|---|---|---|---|---|---|---|
| **richaux best** | richaux_q_20k / 18000 | 503d3cd | 42.42 | 43.79 | 44.52 | 44.76 | **0.6676** | best val_ce_t4; the DEFAULT recipe; beats EG-C2F at every t; ≈ anchor at t4, anchor marginally better at t1 |
| richaux last | richaux_q_20k / 20000 | 503d3cd | 42.47 | 43.80 | 44.40 | 44.70 | 0.6676 | final endpoint (step 20000, 1.6M fwd — over the 1M cap); CE 0.6676 ties step 18000, does not beat it |
| **anchor @1M** | grid_t5_aligned_2scale_c64_20k / 10000 | ed3e8e9 | 42.94 | 44.05 | 44.55 | 44.94 | ~0.667 | the "bar" (960k-fwd budget pt) — but t1=42.94 is this seed's PEAK (its 20-eval t1 mean 42.63, max 42.91); single-critic, entropy-channel ON |
| anchor curve | grid_t5_… / 5k·15k·20k | ed3e8e9 | 42.73·42.81·42.75 (t1 only) | — | — | — | — | scaling-curve ckpts preserved; multi-horizon eval'd only @10k (others log t1 `val_miou_t1_mode` only) |
| EG-C2F-c64 | baseline (deterministic) | — | 42.22 | 43.30 | 44.04 | 44.65 | — | the baseline rich-aux beats at every t1–t4 |

Reference: best-of-K true-CE t1 **oracle** (c32) = 44.05 vs EG-C2F 41.0 — BOUNDS t1 headroom (not achievable by a generalizable policy). [docs/milestones.md, CLAUDE.md "The target".]

## richaux_q_20k — full eval trajectory (single seed; the default recipe)

mIoU (×100) and val CE per timestep, every 1000 steps (the run logs full-val t1..t4):

| step | t1 | t2 | t3 | t4 mIoU | ce_t1 | ce_t2 | ce_t3 | ce_t4 |
|---|---|---|---|---|---|---|---|---|
| 1000 | 42.46 | 43.71 | 44.44 | 44.82 | 0.7200 | 0.6939 | 0.6805 | 0.6724 |
| 2000 | 42.51 | 43.65 | 44.20 | 44.57 | 0.7174 | 0.6923 | 0.6797 | 0.6730 |
| 3000 | 42.55 | 43.67 | 44.20 | 44.65 | 0.7204 | 0.6950 | 0.6810 | 0.6710 |
| 4000 | 42.44 | 43.70 | 44.10 | 44.64 | 0.7199 | 0.6938 | 0.6818 | 0.6737 |
| 5000 | 42.61 | 43.79 | 44.34 | 44.69 | 0.7210 | 0.6931 | 0.6797 | 0.6704 |
| 6000 | 42.20 | 43.66 | 44.29 | 44.67 | 0.7202 | 0.6930 | 0.6783 | 0.6687 |
| 7000 | 42.53 | 43.75 | 44.23 | 44.47 | 0.7189 | 0.6947 | 0.6830 | 0.6782 |
| 8000 | 42.62 | 43.99 | 44.56 | 44.85 | 0.7181 | 0.6933 | 0.6783 | 0.6699 |
| 9000 | 42.56 | 43.85 | 44.45 | 44.82 | 0.7203 | 0.6931 | 0.6774 | 0.6686 |
| 10000 | 42.37 | 43.45 | 44.06 | 44.43 | 0.7180 | 0.6936 | 0.6788 | 0.6704 |
| 11000 | 42.27 | 43.50 | 44.29 | 44.64 | 0.7179 | 0.6921 | 0.6769 | 0.6691 |
| 12000 | 42.54 | 43.56 | 44.37 | 44.63 | 0.7186 | 0.6916 | 0.6777 | 0.6686 |
| 13000 | 42.53 | 43.86 | 44.39 | 44.84 | 0.7174 | 0.6901 | 0.6764 | 0.6678 |
| 14000 | 42.64 | 43.87 | 44.57 | 44.86 | 0.7170 | 0.6914 | 0.6778 | 0.6693 |
| 15000 | 42.56 | 43.90 | 44.34 | 44.61 | 0.7158 | 0.6887 | 0.6766 | 0.6687 |
| 16000 | 42.36 | 43.79 | 44.33 | 44.77 | 0.7168 | 0.6899 | 0.6777 | 0.6698 |
| 17000 | 42.46 | 43.76 | 44.29 | 44.54 | 0.7167 | 0.6898 | 0.6768 | 0.6683 |
| 18000 ** | 42.42 | 43.79 | 44.52 | 44.76 | 0.7165 | 0.6899 | 0.6759 | 0.6676 (best CE@t4)** |
| 19000 | 42.46 | 43.72 | 44.41 | 44.71 | 0.7160 | 0.6891 | 0.6753 | 0.6678 |
| 20000 | 42.47 | 43.80 | 44.40 | 44.70 | 0.7163 | 0.6898 | 0.6760 | 0.6676 |

**Best-by-metric × horizon** [user standing convention — they land at different steps]:

| horizon | best val CE @ step | best mIoU @ step |
|---|---|---|
| t1 | 0.7158 @ 15000 | 42.64 @ 14000 |
| t2 | 0.6887 @ 15000 | 43.99 @ 8000 |
| t3 | 0.6753 @ 19000 | 44.57 @ 14000 |
| t4 | 0.6676 @ 18000 | 44.86 @ 14000 |

Remarks: on **CE** (the optimized, less-noisy metric) richaux improves ~monotonically across all t
(ce_t1 0.7200→0.7163, ce_t4 0.6724→0.6676); the t1 *mIoU* by contrast wanders in a noisy
~42.2–42.6 band. Beats EG-C2F at every horizon. Vs the anchor: ≈ at t4 (mIoU+CE), anchor marginally better
at t1 (its t1 CE mean 0.7159 vs richaux ~0.719). **Caveat:** only `best.pt` (min val_ce_t4) + `last.pt` were
saved (`keep_every=0`) — best-by-mIoU / other-horizon checkpoints at other steps were never written.

Provenance: run dirs on crockett `runs/`; manifests record git_rev + config. mIoU/CE numbers from each run's
`metrics.jsonl` (full-val) except the anchor @1M multi-horizon, from a `q.eval` deploy rollout.

## Old-frontend 8-seed band (2026-06-17/18) — `seedband_curated/`

The matched-budget reliability band (8 seeds, 1M each). Full table + EG-C2F comparison + repro:
**docs/old_frontend_band_results.md**. Git tag **`result/curated-8seed-band`** (`55fda96`) — the band's arch is the
old-frontend (BN + per-group gate + shared 2052→32 proj); **`git checkout` that tag before loading
these ckpts** (current HEAD frontend is per-group-LN, different params, NOT migratable).

- `seedband_curated/s3_best_step9000.pt` — best seed by val_ce_t4 (**0.6636** deploy / 0.6644 endpoint).
- `seedband_curated/s3_last_step12500.pt` — its 1M endpoint.
- All 8 seeds persist in crockett `runs/*seedband_curated_s{0..7}` (full per-step metrics.jsonl).

Band: matched-endpoint ce_t4 **0.6657±0.0008** / mIoU **44.76±0.11**; deploy ce_t4 0.6646±0.0008 / 44.91±0.15.
Beats `result/egc2f-c64-baseline` on CE at every t by ~6–13σ.
