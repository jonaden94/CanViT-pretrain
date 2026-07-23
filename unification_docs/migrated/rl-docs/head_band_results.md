# HEAD-frontend 8-seed band (c64, T=5) — the 1M-forward reference band

The per-group-LN frontend (HEAD; per-channel BN → per-group Conv→LN(affine)→gate→sum → token-MLP) under
the 2026-06-18 recipe (that day's `QConfig` defaults), matched 1M-glimpse-forward budget, 8 seeds (s0–s7).
Superseded as the validated result by the 640k-forward qband (`qband_results.md`, 2026-07-04), which
matches it within seed noise; stays the like-for-like 1M reference. Supersedes the old-frontend band
(`old_frontend_band_results.md`) for the live arch. Tag: `result/qpolicy-8seed-band`. Reproduce:
`python -m canvit_pytorch_rl.tools.seed_report --prefix seedband_s --final-step 12500`.

## Deploy band — per-seed best-mean(t1–t4)-CE checkpoint, mean ± std over 8 seeds

```
   t            t1        t2        t3        t4
 val_ce    0.7141    0.6881    0.6743    0.6654     (±0.0005 / 0.0003 / 0.0005 / 0.0005)
 val_miou   42.74     43.85     44.51     44.87      (±0.15 / 0.13 / 0.06 / 0.12)
```

LAST-step (12500) ckpt band, for context: ce 0.7144/0.6891/0.6755/0.6670, miou 42.74/43.84/44.46/44.80.
Judge by val CE (the optimized objective, less noisy); mIoU is the reporting protocol.

## vs EG-C2F-c64 (measured, `egc2f_c64_t5_ce`) — beats it at every t

```
   t            t1        t2        t3        t4
 EG-C2F ce 0.7258    0.7004    0.6828    0.6707
 HEAD   ce 0.7141    0.6881    0.6743    0.6654   (lower=better; ~10σ at t4 given band σ 0.0005)
 EG-C2F mi  42.22     43.30     44.04     44.65
 HEAD   mi  42.74     43.85     44.51     44.87   (higher=better; small but positive at every t)
```

## Per-seed deploy ckpt (best-mean step) + provenance

```
 seed  run dir                         git_rev    best-mean@  ce_t4    mi_t4   step-ckpt on disk
 s0    20260618-183641_seedband_s0     2e5b22a3   10000      0.6651   44.72   no  (best.pt/last.pt only)
 s1    20260618-203757_seedband_s1     5acc349a    8000      0.6651   45.00   no  (best.pt/last.pt only)
 s2    20260618-223826_seedband_s2     87c436d4    8000      0.6658   44.95   yes (step_008000.pt)
 s3    20260619-004019_seedband_s3     87c436d4    5000      0.6650   44.85   yes (step_005000.pt)
 s4    20260619-024215_seedband_s4     51576885    8000      0.6663   44.96   yes (step_008000.pt)
 s5    20260619-045518_seedband_s5     1b4b69a3    6000      0.6649   44.67   yes (step_006000.pt)
 s6    20260619-071204_seedband_s6     cdc47661    9000      0.6657   44.94   yes (step_009000.pt)
 s7    20260619-092137_seedband_s7     58003716    9000      0.6656   44.87   yes (step_009000.pt)
```

The band spans git_revs (seeds launched across ~a day); the constant is the recipe = `QConfig` defaults at
each launch (the only relevant changes in that window were `keep_every`'s default→1000 — so s0,s1 predate
the per-eval step ckpts — and the inert `dump_init` flag; neither alters training). s0/s1's best-mean step
ckpts were not saved; for behavioral re-analysis use their `last.pt` (the scale signature is stable late).

## Behavioral analysis (where it looks, and is the value function sensible)

Full record: `docs/sessions/2026-06-19-action-analysis-coarse-to-fine.md`. Summary:
- Coarse→fine emerges in 6/8 seeds (fine-scale fraction rises ~0→11–21% over t1–t4) but is NOT causal —
  2/8 seeds (s5, s6) win staying ~pure-coarse, and masking the fine scale at deploy costs the zoomers ~0
  (s3/s4: ΔCE ≤0.0004). The t1 reward landscape is coarse- and center-biased (best candidate coarse ~90%),
  which explains it. Performance comes from coarse-glimpse PLACEMENT, not the scale schedule.
- Q is sensible but weakly calibrated (t1 vs best-of-512 oracle): Spearman ~0.33, chosen ~70th percentile,
  ~⅓ of the random→oracle gap; untrained ≈ 0. Replicates across seeds.
