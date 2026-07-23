# 2026-06-14 — Grid value-policy: build, the gridcorr objective, and tuning

Continuation of the dense-Q-grid direction. This session BUILT the grid value-policy as a
real trainer and got the first trustworthy HP tuning going. Single entry point to pick up cold.
The precise best-config record is in `docs/milestones.md` ("2026-06-14 — Dense value-grid
policy: best result to date"); the keystone technique in memory `fractional-ce-target-unlocks-k1`.

## WHAT THIS IS
Grid value-policy (NOT the flow): U-Net (`grid_net.GridValueNet`) over t0 canvas features ->
value for every viewpoint in a `{0.5,0.25} × 16×16` overlap grid; **argmax = the t1 policy**
(self-contained, no separate critic). Frozen perception. Files: `grid_net.py`, `grid_train.py`,
`reward_maps.py`, `grid_optuna.py`. Trains on full ADE20K; reports full val.

## THE RECIPE (validated, current)
- **K=1 sparse supervision**: one sampled viewpoint/scene/step, masked MSE at that grid cell.
  Unbiased SGD; t0+t1 share [B,...] shape (compile-friendly). K removed as a knob entirely.
- **Target** = fractional CE `(t0_ce-vp_ce)/t0_ce` + online global-z via **bias-corrected EMA**
  (`RunningNorm`) — no per-scene stats. THE keystone (memory `fractional-ce-target-unlocks-k1`).
- **Reward scored at 128²** (not 512²): ~2× throughput, ranking preserved (memory
  `scoring-resolution-bottleneck`). Eval mIoU at full 512² (paper protocol).
- **riid t0** during train (random start viewpoint = free start-state augmentation); eval always
  full_scene. **replicate conv padding** (zero-pad caused spurious corner artifacts in PRED grid).
- **OBJECTIVE = `val_gridcorr`** — per-image Pearson(predicted grid, PRECOMPUTED true reward
  landscape), mean over 2000 val imgs. NOT mIoU. See "the methodological win" below.
- Fixed: width 128 (3.68M), grad_clip 1.0, batch 16, warmup_frac 0.5. Searched: lr, wd, β2, mom.

## THE METHODOLOGICAL WIN (most important learning)
mIoU is argmax-discrete -> ~1pp jitter on the FIXED val set (a near-tie flips the argmax). It
literally **could not distinguish a 30× LR range** (grid_s1 era thought lr 1.7e-3 was great; it's
the WORST). Fix: precompute the true reward landscape over the whole grid for all of val+train-
slice ONCE (`reward_maps.py`, ~24min/1.5M forwards, reusable while action space fixed), then each
trial correlates its predicted grid against it -> smooth, argmax-free, low-noise. Validated:
`corr(gridcorr, CE) = -0.81`, `corr(gridcorr, mIoU) = +0.49` across trials — fit and deploy track.
**Always monitor Spearman gridcorr too** [user] (logged; closer to what argmax cares about).

## WHAT WE LEARNED (tentative, mostly @ 2000-step horizon, n=6-9, single-eval)
- **LOW LR wins** (best ~5e-5), BUT horizon-dependent: lr 1.4e-5 LOST at 2k (too low to converge
  in the steps). So very-low LR needs the longer horizon -> motivates the 5k+ baseline.
- **High WD (up to 6e-2) is tolerable** at 2k (flat gridcorr); seed defaults to wd 5e-2 [user]
  for the longer-horizon regime where the overfit gap (train_gridcorr-val ~0.13-0.15) grows.
- **β2 0.95 / mom 0.98** mildly beat 0.99/0.997. **Padding fix lifted gridcorr/CE/mIoU together.**
- **The bar is central-crop 41.1** (scale 0.625, measured through OUR eval; our grid EXCLUDES 0.625,
  best in-grid constant 40.74). t0 floor 38.53. Best run mIoU 41.18 — just clears the bar.
- **Scale COLLAPSES** to one value (position varies) across ALL configs — unexplained, parked [user].
- Throughput: training ~18 steps/s (~640 glimpses/s, overhead-bound, CAN'T hit the 1300 pure-
  forward ceiling); precompute ~1094/s (pure forward stream). t0 is a SUPERVISION-efficiency tax,
  NOT a throughput one.

## MISTAKES MADE THIS SESSION — guards so future-you doesn't repeat
1. **Bogus throughput "7 steps/s" + "3.5h/40 trials"** — parsed the log ACROSS trials whose step
   counters reset to 1, mixing them + the inter-trial gap. Real rate ~18 steps/s. GUARD: parse a
   SINGLE trial (break on counter reset); a per-trial completion time ("done in Ns") is the truth.
2. **"grid_s3 below baseline" from n=2** — with 6 trials it spanned 40.79-41.36. RULE ONE: n=2≠trend.
3. **Over-claimed the train_corr 0.2->0.64 jump** — confounded (riid landscape is easier to fit
   in-sample than full-scene); the deploy number (val_gridcorr 0.28) was modest. Compare like-for-like.
4. **Wrong baseline framing** — reported "gain over t0 (+2.75pp)" as success; the real bar is
   central-crop t1 41.1, NOT t0 38.5. User caught it.
5. **Re-litigated CE-vs-IoU** — CE is the SETTLED reward [CLAUDE.md]; don't redesign toward IoU.
6. **Insane disk-cache idea** — proposed caching ~80GB of t0 feats to disk; recompute (12ms) is
   far cheaper and it doesn't fit in RAM/VRAM. Don't cache big feats.
7. **Watcher race** — the auto-launch watcher fired the OLD single-horizon grid_s4 before I updated
   it. GUARD: kill the watcher BEFORE changing launch config.
8. **ssh flakiness** — repeated exit-255 on combined `pkill; git checkout; launch` over ssh left
   checkouts incomplete (wrong HEAD) and orphan procs. GUARD: split into separate ssh calls; `git
   fetch origin` explicitly (deploy push alone doesn't update the checkout's origin tracking); verify
   HEAD after each; `pgrep -f X` self-matches the command — confirm kills with a clean check.

## CURRENT RUN (live at compaction)
- **grid_s8**, commit **a5bb099**, seed 6, study `grid_s8_h{5000,10000,20000,40000}` (horizon
  ladder: 5k base × 2^r, 10 trials/rung, carry top-2 configs forward; each rung is its OWN optuna
  study so gridcorr is comparable within a horizon).
- Search: lr [5e-6, 5e-4] (cut high LR — always loses, worse at long horizons), wd [1e-4, 1e-1],
  β2 {0.9,0.95,0.99}, mom {0.98,0.99,0.997}. Seed: lr 5e-5, wd 5e-2, β2 0.95, mom 0.98.
- ETA: ~5min/trial @5k, ~38min @40k; full ladder to 40k ~11.7h (cap `--rungs 3` -> ~5.4h to 20k).
- Maps: `runs/reward_maps/grid_{validation,trainslice}_s0.5-0.25_g16_r128_c32.pt` (reusable).
- Watch: http://localhost:5500/#/experiments/4 (`canvit-grid`); check latest BEFORE any relaunch [rule].

## OPEN / NEXT
- **warmup_frac=0.5 is too high for the long rungs** (40k -> 20k warmup steps); decide a smaller
  fraction or fixed warmup-step-count BEFORE the long rungs land. FLAGGED, not yet changed.
- **Data augmentation** — we use `Ade20kSquish` val transforms = NONE. `canvit_specialize` has
  train transforms (`make_train_transforms`, not yet inspected — file at
  ~/Downloads/.../CanViT-specialize/canvit_specialize/datasets/ade20k.py). Open: add it? It's a
  generalization lever and the long rungs are where overfit grows.
- Does gridcorr keep climbing at 10k-40k, and do the very-low LRs (lost at 2k) win there?
- Does mIoU push CLEARLY past 41.1 at long horizons (the gridcorr↔mIoU coupling holding)?
- Scale collapse — parked.
- Checkpoints: each trial saves best.pt(==last.pt)+last.pt on disk; NOT logged to mlflow as
  artifacts (could log only best.pt if browsability wanted). No study-best pointer.
