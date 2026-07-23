# Sweep sets — interesting + diverse HP configs and their mIoU curves

Hand-picked, *diverse* HP configurations from retired sweeps, kept as reference points (not just the
top-by-metric, which cluster on a flat landscape). For each: the HPs, the val mIoU curve, and a behavior
label. The per-eval metrics live in the named run dirs on crockett (their checkpoints were pruned 2026-07-03
with all other superseded-era ckpts; metrics.jsonl + manifests remain). Curves are FROZEN (the studies are retired). Regenerate the deploy table with
`python -m canvit_pytorch_rl.tools.sweep_report --study <study>`; full per-eval rows in
`runs/<run>/metrics.jsonl`.

## Overnight sweep — `t5_c64_ce_f1000000` (c64 / T=5 / 1M forwards; broad 12-dim search; 7 trials)

Single-seed, FLAT landscape — top-5 best `val_ce_t4` span only **0.6656–0.6706**, and CE vs mIoU disagree on
the winner. Judge by val CE; mIoU always reported. These 7 span the swept space (full_scene vs riid, WD
6e-4..8.6e-2, width 64..256, entropy on/off, prime 0/0.5/1.0, warmup 0.05..0.5) **and** the behavior modes
(clean-monotone / overfit / underfit). Glimpse-forwards = step × 96 (batch 16 × (1+5)); 10416 steps ≈ 1M.
`t0` mIoU = 39.60 for every trial (the frozen full-scene floor). Run dirs:
`runs/t5_c64_ce_f1000000__trial000N/` (`best.pt` = the lowest-`val_ce_t4` checkpoint).

### HPs + best checkpoint (sorted by best val_ce_t4)

| trial | behavior | t0 | wd | warmup | prime | ec | fm | blk | width | lr | best val_ce_t4 @step | deploy mIoU t1/t2/t3/t4 @best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0004 | **best CE — but OVERFITS** | full | 6.3e-4 | 0.05 | 0.0 | off | on | 3 | 128 | 6.8e-5 | **0.6656** @6000 | 42.33 / 43.77 / 44.51 / 44.86 |
| 0003 | **best mIoU**; riid + on-policy + big | riid | 5.8e-4 | 0.5 | 1.0 | off | on | 3 | 256 | 1.0e-4 | 0.6662 @10416 | 42.48 / 43.75 / 44.51 / **44.99** |
| 0001 | compact (w64) + high WD; plateaus | full | 1.4e-2 | 0.1 | 1.0 | off | off | 2 | 64 | 6.3e-5 | 0.6675 @10000 | 42.46 / 43.78 / 44.27 / 44.83 |
| 0000 | **clean monotone, best-at-last** (seed) | full | 1.2e-2 | 0.5 | 0.0 | on | off | 2 | 128 | 8.3e-5 | 0.6678 @10416 | 42.77 / 43.81 / 44.55 / 44.86 |
| 0002 | riid + prime0.5 + **ec-ON** + big | riid | 1.8e-3 | 0.5 | 0.5 | on | on | 3 | 256 | 5.5e-5 | 0.6706 @10416 | 42.59 / 43.75 / 44.53 / 44.94 |
| 0005 | **UNDERFIT** (lr 10× too low) | full | 1.3e-4 | 0.05 | 0.0 | off | on | 3 | 128 | 5.5e-6 | 0.6860 @2000 (pruned) | 42.55 / 43.44 / 43.78 / 43.92 |
| 0006 | incomplete (killed ~2k) | full | 8.6e-2 | 0.25 | 0.0 | off | on | 3 | 128 | 5.0e-4 | 0.6708 @2000 | 42.39 / 43.72 / 44.41 / 44.69 |

### val mIoU t4 over training (the behavior signal; step → mIoU t4)

- **0000** clean monotone, best-at-last: 42.18 → 43.88 → 44.50 → 44.66 → 44.70 → 44.80 → **44.86** (still rising at 1M)
- **0003** best, monotone: 41.76 → 44.56 → 44.77 → 44.71 → 44.71 → 44.99 → **44.99**
- **0002** riid, rises to last: 41.71 → 44.25 → 44.60 → 44.54 → 44.77 → 44.86 → **44.94**
- **0001** fast then plateau: 41.37 → 44.60 → 44.86 → 44.79 → 44.90 → 44.83 → 44.82
- **0004** OVERFITS: 41.96 → 44.67 → 44.80 → **44.86 (peak @6k)** → 44.77 → 44.77 → 44.72 (declines)
- **0005** underfit (pruned): 41.96 → 43.92
- **0006** incomplete: 41.96 → 44.69

Behavior modes worth keeping: **0000 / 0003 monotone best-at-last** (well-regularized — the shape the new
defaults aim for; both would take more than 1M); **0004 overfit** (wd 6e-4 + warmup 0.05 → CE bottoms
0.6656 @6k then rises to 0.6671, mIoU t4 peaks @6k then falls — *this is why the winner-by-objective is a
trap and why the new defaults keep wd high*); **0005 underfit** (lr 5.5e-6, pruned). trial0006 was earlier
mislabeled "diverging": `0.7258` was its step-0 baseline, not a diverged value — its only trained eval (2k)
was a healthy `0.6708 / 44.69`; it was killed incomplete when the sweep was retired (inconclusive, NOT a
divergence example).

Figure (val mIoU t4 vs training glimpse-forwards, and deploy mIoU vs t at each best ckpt) delivered to the
user; regenerate from `runs/t5_c64_ce_f1000000__trial*/metrics.jsonl`.
