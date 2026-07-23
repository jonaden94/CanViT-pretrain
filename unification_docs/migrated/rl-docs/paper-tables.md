# Paper reference tables (arXiv:2603.22570, appendix)

Verbatim targets for harness verification, provided by the user 2026-06-11.
Frozen CanViT-B, ADE20K val, paper (squish-512 "scene") protocol, T=21.
Stochastic policies: mean over n=10 runs ± 95% bootstrap CI half-width.
† = deterministic.

## Table 4 — mIoU (%) by policy and timestep, canvas 64² (the rows we verify against)

| Policy | t=0 | t=1 | t=2 | t=3 | t=4 | t=9 | t=16 | t=20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EG-C2F† | 39.6 | 42.2 | 43.3 | 44.1 | 44.7 | 45.6 | 45.8 | 45.7 |
| C2F | 39.6±.00 | 41.3±.10 | 42.5±.08 | 43.6±.06 | 44.7±.03 | 45.1±.05 | 45.6±.03 | 45.9±.02 |
| F-IID | 39.6±.00 | 41.2±.10 | 42.0±.09 | 42.5±.10 | 43.0±.09 | 44.1±.06 | 44.7±.08 | 44.9±.07 |
| R-IID | 18.3±.30 | 27.1±.37 | 31.6±.29 | 34.3±.19 | 36.3±.16 | 40.5±.17 | 42.6±.12 | 43.1±.13 |
| RFS† | 39.6 | 41.1 | 41.1 | 41.0 | 40.9 | 40.3 | 39.6 | 39.3 |
| F2C | 11.0±.38 | 18.1±.32 | 22.6±.29 | 25.9±.33 | 28.3±.29 | 35.4±.36 | 40.1±.19 | 42.9±.15 |

Canvas 32² rows (same table): EG-C2F† 38.5 / 41.1 / 42.0 / 42.7 / 43.2 /
43.9 / 44.2 / 44.1; C2F t20 44.2±.02; F-IID t20 43.3±.06; R-IID t0 18.1±.53,
t20 41.6±.09; RFS† 38.5 / 40.0 / 40.0 / 39.9 / 39.8 / 39.1 / 38.5 / 38.2;
F2C t0 10.8±.14, t20 41.2±.10.

## Table 5 — best-t mIoU by canvas grid (one probe per grid)

| Policy | c=8 | c=16 | c=32 | c=64 |
|---|---:|---:|---:|---:|
| EG-C2F† | 31.6 | 39.3 | 44.2 | 45.8 |
| C2F | 31.6±.03 | 39.1±.04 | 44.2±.02 | 45.9±.02 |
| F-IID | 30.7±.04 | 38.1±.09 | 43.3±.06 | 44.9±.07 |
| R-IID | 29.5±.10 | 37.2±.07 | 41.6±.09 | 43.1±.13 |
| RFS† | 30.3 | 36.8 | 40.0 | 41.1 |
| F2C | 29.2±.14 | 36.8±.11 | 41.2±.10 | 42.9±.15 |

Notes:

- The released 45.9% headline = C2F c64 best-t (t=20), a STOCHASTIC mean
  (C2F randomizes visitation order within each scale); EG-C2F peaks 45.8 at
  t=16 deterministically.
- Stochastic-policy verification = our single run should sit within/near
  the n=10 CI band; deterministic (EG-C2F, RFS) should match to bf16 noise.
- Already verified 2026-06-11 (`ff29289`, run `egc2f_t2_full`): EG-C2F c64
  t0 39.602 vs 39.6, t1 42.219 vs 42.2.
