# qband: 8-seed band of the 2026-07-03 recipe (c64, T=5, 640k forwards)

The recipe defaults as of 2026-07-03 (`QConfig` at the band's revs; the class is `TrainConfig` since
the 2026-07-05 sum-type refactor — dueling head, deploy-mode rollouts, per-depth reward z, lr 2e-4,
640k glimpse-forwards = 8000 steps, 1000-step warmup then hold), 8 seeds (`qband_s0..7`), run
2026-07-03/04. Confirms the recipe: **matches the 1M-forward HEAD band
(`head_band_results.md`) on CE and mIoU at 64% of its training compute** (~80 min/seed on the
4090, including nine full evals). Judge by mean(t1–t4) val CE. Reproduce:
`python -m canvit_pytorch_rl.tools.seed_report --prefix qband_s --final-step 8000`.

## Deploy band — per-seed best-mean(t1–t4)-CE checkpoint, mean ± std over 8 seeds

```
   t              t1             t2             t3             t4
 val_ce    0.7143±.0012   0.6878±.0009   0.6741±.0007   0.6652±.0008
 val_miou   42.65±.16      43.95±.16      44.62±.12      44.97±.10
```

mean(t1–t4) val CE = **0.6853 ± 0.0007** (per-seed means below).
LAST-step (8000) band, for context: ce 0.7148/0.6886/0.6751/0.6665, miou 42.71/43.94/44.61/44.91 —
the deploy rule's early selection (best-mean steps 4k–8k) buys ~0.001 CE over just taking the end.

## vs the 1M-forward HEAD band and EG-C2F-c64

```
                        mean(t1-4) CE   ce per-t (t1..t4)                    mi_t4      forwards
 qband (this)           0.6853±0.0007   0.7143 0.6878 0.6741 0.6652         44.97±.10   640k
 HEAD band (2026-06-19) 0.6855±0.0004   0.7141 0.6881 0.6743 0.6654         44.87±.12   1M
 EG-C2F-c64 (det.)      0.6949          0.7258 0.7004 0.6828 0.6707         44.65       —
```

Identical to the HEAD band within seed noise on both metrics; beats EG-C2F on CE at every t
(t4 gap 0.0055 ≈ 7σ of the band spread) and on mIoU at every t.

## Per-seed deploy ckpt + provenance

All step ckpts on disk (`keep_every=1000`), so every deploy checkpoint is a saved `step_*.pt`.
Run dirs on crockett, `~/projects/CanViT-PyTorch-RL/runs/`.

```
 seed  run dir                       git_rev    best-mean@  mean(t1-4) ce   ce_t4    mi_t4
 s0    20260703-191825_qband_s0     bcb9742f      5000        0.6858       0.6654   45.15
 s1    20260703-204905_qband_s1     04bd99c1      5000        0.6852       0.6644   44.94
 s2    20260703-221130_qband_s2     007f7173      6000        0.6845       0.6649   44.96
 s3    20260703-233058_qband_s3     007f7173      5000        0.6848       0.6650   44.83
 s4    20260704-005023_qband_s4     007f7173      4000        0.6848       0.6648   44.96
 s5    20260704-021003_qband_s5     007f7173      4000        0.6851       0.6645   45.04
 s6    20260704-032949_qband_s6     007f7173      7000        0.6860       0.6658   44.87
 s7    20260704-044909_qband_s7     007f7173      8000        0.6865       0.6667   45.04
```

The rev span bcb9742f→007f7173 contains docs, `throwaway/`, and a `seed_report` display fix
only — zero training-code changes, so the band is one code version for training purposes.

## Published checkpoints (HF Hub, public)

Every seed's deploy ckpt (`best.pt`) is published for reproduction:
`canvit/qpolicy-ade20k-c64-t5-qband-2026-07-04-s{0..7}` (weights + model card + the run's
manifest/metrics; `python -m canvit_pytorch_rl.tools.publish_policy`). The flagship
(`DEFAULT_QPOLICY_REPO` in `q/net.py`) is the band-best seed s2; performance claims cite the BAND,
never the flagship's own number (best-of-8 selection sits ~1σ below the band mean by construction).
`canvit/qpolicy-ade20k-c64-t5-2026-07-03` (`qlr2e4_s0`, mean CE 0.6856) predates the band and stays
up as-is.
