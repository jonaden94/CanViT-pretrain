# 2026-06-24 — perpetual HP sweep post-mortem (study `qpolicy_c64_t5_ce_f1000000`)

User forgot a sweep was running on crockett (launched 2026-06-19 when the 8-seed band freed the GPU); asked what
days of compute bought. **Stopped the sweep** this session (supervisor `perpetual_sweep.py` killed first so it
couldn't respawn, then the orphaned `q.optuna` trial by PID; GPU → 0 MiB confirmed). All numbers below
[read from crockett `runs/q_optuna.db` + per-trial `runs/*__trial*/metrics.jsonl` on 2026-06-24, crockett @ commit `2b19e37`].

## What was searched

`q.optuna` study **`qpolicy_c64_t5_ce_f1000000`** (base name `qpolicy_c64_t5_ce` + `_f1000000`), objective =
**minimize `val_ce_t4`**, train_horizon 4, 1M forwards/trial, eval-every 2000 steps (→ 8 evals/trial), search =
`lr, weight_decay, adam_beta1, adam_beta2, width, block_layers`. All other HPs from the on-disk `QConfig` defaults.
Repro: `optuna.load_study("qpolicy_c64_t5_ce_f1000000", "sqlite:///runs/q_optuna.db")`.

Trials: **245 total — 30 COMPLETE, 202 PRUNED (87% prune rate), 12 FAIL, 1 RUNNING (killed)**.

**Objective mismatch [user 2026-06-24]:** the user cares ONLY about **mean(t1–t4) val CE** (the deploy rule), but
the sweep minimized **`val_ce_t4`** (endpoint only) — verified via `trial.value == min(val_ce_t4)` for 30/30. So
the overnight tuner ranked trials on a metric the user doesn't use. It didn't change the verdict (both metrics are
noise blobs — see below) but any future `q.optuna` sweep must set the objective to mean(t1–t4) CE.

**On mean(t1–t4) val CE (the metric that matters):** best **#173 ≈ #31 = 0.6844**, worst **#30 = 0.6861**, spread
0.0017, **cross-trial std 0.00040 — *below* the 0.0005 single-config seed σ.** I.e. across this metric the 30
configs vary *less* than re-seeding one config does → HP choice is below the noise floor; the plateau is flatter
here than on t4. The near-twins #31 (rank 2) and #30 (rank 30/30, last) bracket the whole field on near-identical
configs — the sharpest noise demonstration in the data, and it needs no winner's-curse simulation.

## Verifications passed (before trusting any number)

- dir↔trial glob unique for 30/30 completed (first attempt `*trial0000` wrongly matched an old `grid_s1__trial0000` —
  fixed to `*qpolicy_c64_t5_ce_f1000000__trial####`).
- `trial.value == min(val_ce_t4 over the 8 evals)` for 30/30 → objective semantics confirmed.
- `val_ce_t0` spread across all 30 trials = **0.0000** (shared frozen full-scene t0 state) → metric pipeline consistent.

## Headline: the deployed recipe is on a plateau

- **`val_ce_t4` across 30 completed trials: best 0.6640 (trial #31), median 0.6653, worst 0.6659.** The honest
  noise comparison is **std-to-std, not range-to-σ**: the **cross-trial std (0.00050) EQUALS the 8-seed band σ
  (0.0005)** (`docs/head_band_results.md`) → ratio 1.0. I.e. 30 *different* configs vary by the same amount as
  re-seeding *one* config — the HP effect is **indistinguishable from seed noise**. (An earlier framing here called
  the 0.0019 best-to-worst spread "3.8× band-σ"; that compared a range to a σ and oversold it — corrected.)
- **Winner's curse:** best trial #31 beats the median by 0.0013; if the HP effect were exactly zero, the luckiest
  of 30 noise draws typically beats the average by 0.0010 (5–95%: 0.0007–0.0015). 0.0013 sits inside that band →
  #31 is consistent with being the luckiest draw, not a better config. (Resolving it needs a seed-replication of
  #31 vs the default — **user declined 2026-06-24**; so "real ±0.002 win vs winner's curse" stays unresolved, and
  the sweep cannot settle it by construction.) Note: 0.0019 is ~36% of the 0.0053 EG-C2F t4 margin, so it is not
  negligible *in absolute terms* — the point is only that it is not *attributable to HPs*.
- Best trial **#31** (lr 3.35e-4, wd 1.92e-2, width 64, block_layers 3, β1 0.9, β2 0.99) ≈ the deployed default
  (lr 3e-4, wd 1e-2, width 128, bl 3, β2 0.95). The difference is within seed noise — **no searched config beats
  the default by more than ~noise.** Searched ranges that all land within 0.002 CE: lr 6e-5–5e-4, wd 5e-3–8e-2,
  width 64–256, bl 2–3.
- Tiny spread at **every** horizon — per-t `val_ce` spread: t1 0.0035, t2 0.0027, t3 0.0027, t4 0.0019. mIoU same
  story (t4 mIoU best 0.4465 / worst 0.4503; mIoU-selected-step best 0.4474–0.4511).
- **Capacity is irrelevant:** width 64 and 256 both appear among the best; bl 2 and bl 3 tie. fANOVA importances
  (wd 0.64, lr 0.19, β2 0.12, width 0.05, bl 0.01) are fitting a 0.0019 range — i.e. ranking noise.
- Weak, noise-adjacent HP signals: β2↑ → slightly lower CE (Pearson −0.29); wd↑ → slightly less late-degradation
  (−0.24). Nothing actionable.

## Actionable positive: convergence by ~16% of budget

Mean `val_ce_t4` vs forwards (n=30): **0.7015 (step 0, untrained) → 0.6670 @160k fwd → 0.6665 @1M.** The first
**160k forwards (16% of the 1M budget) reach within 0.0005 of the final**; the back 840k buy ~0.0005. Best-eval
step median ≈ 800k (idx5/7), mild late-degradation (29/30 trials' final eval is ≤ +0.001 above their own best).

→ This concretizes the standing "explore short, confirm long" rule with a number: **future HP search for this
recipe can run at ~250k forwards/trial (~4× more trials per GPU-hour) and rank configs just as well; reserve the
full 1M only for final confirmation.** The perpetual sweep ran every trial to 1M — most of that compute was spent
re-confirming a plateau already visible by 160k, then pruning 87% of trials anyway.

## CE↔mIoU decoupling persists inside the plateau

Cross-trial Pearson `val_ce_t4` vs `val_miou_t4` (same step) = **−0.12** (weakly negative, as direction expects,
but near-zero). Within this tiny spread, the CE-best trial is **not** the mIoU-best — consistent with the
documented ρ≈0.26 CE/mIoU alignment. Judge-by-CE stands; mIoU adds little ranking power here.

## No reliably-bad / unstable region (the pruned + failed trials)

User asked whether the sweep flags anything reliably bad/unstable. Looked at the 202 PRUNED + 12 FAIL trials (the
30 completed are the plateau). Answer: **no.**

- **Pruning is uniform, not targeted.** Prune rate ~80–91% for *every* value of every HP (width 64/128/256 →
  79/82/91%; bl 2/3 → 89/80%; β2 0.9/0.95/0.99 → 82/86/82%). **189/202 pruned trials died at the FIRST eval**,
  spanning the full lr (5e-6–5e-4) and wd ranges → the 87% prune rate is the MedianPruner being aggressive on a
  flat landscape, not a bad-region detector. The best `val_ce_t4` any pruned trial reached was **0.6662** ≈ the
  completed-worst (0.6659) — nothing the sweep touched is dramatically bad.
- **Failures (~5%) are sporadic infra, not an HP mode.** 12 fails scatter across normal lr/wd, mostly width 64
  (10)/128 (2), **zero at width 256** (opposite of an OOM signature; the supervisor's `expandable_segments` env
  var that targets width-256 OOMs evidently works). Optuna captured no fail_reason — not guessing beyond
  "not HP-driven" (crockett pause/resume or transient CUDA likely).
- **Instability uniformly tiny.** Converged std (last 3 evals): width 64 → 0.00050, 256 → 0.00062 (n=6), 128 →
  0.00036 (n=1) — all ≈ seed σ. Late-degrade ≤+0.001 everywhere, slightly larger at width 64. No "big nets
  unstable" story.
- **One weak, confounded flag:** `adam_beta1=0.95` went **0 completed / 18 pruned / 2 failed** — the only value
  that never completed (β1=0.9 ≈ 13% complete). But TPE drew it rarely (20/245), an aggressive first-eval pruner
  kills rarely-drawn values against an established median, and β1's fANOVA importance among completed ≈ 0.
  Binomial 0/20 under a 13% base rate is p≈0.06 — suggestive, not conclusive; the sweep never gave β1=0.95 a fair
  shot. Do not record as "reliably bad."

Net: the searched box (lr 5e-6–5e-4, wd 1e-4–1e-1, width 64–256, bl 2–3, β2 0.9–0.99) is **uniformly benign** —
no cliff, no reliably-bad corner, no instability hotspot. Consistent with the plateau: no bad region because
there is barely any region-dependence at all.

## The policy itself works (unchanged by HP)

Mean per-t trajectory at the deployed endpoint (final eval, ±across trials): CE
**0.765 → 0.714 → 0.689 → 0.675 → 0.667** (±0.0008); mIoU **0.396 → 0.427 → 0.438 → 0.444 → 0.448** (±0.001).
Monotone, diminishing-returns per glimpse; identical across all 30 configs.

## Bottom line for the user's question

Days of GPU produced a **clear confirmatory-negative result** — the recipe sits on an HP plateau; the overnight
tuner cannot improve it because the optimizer/capacity HPs are not the bottleneck (consistent with the action
analysis: the gain is coarse-glimpse *placement*, and Q is only weakly calibrated). The one genuinely useful
positive is the convergence-speed measurement above. Headroom, if any, is in the *method* (calibration / where to
look), not these HPs.

## Housekeeping flagged

- **crockett disk at 99% (33 GB free).** `runs/` = 44 GB; sweep checkpoints = 38 GB across 363 trial dirs
  (best.pt+last.pt+step_*.pt, ~84 MB each at width 256); `runs/mlflow.db` = 4.2 GB. The supervisor self-stops at
  <5 GB free. Pruned-trial checkpoints are the obvious reclaim target (not yet deleted — pending user OK).
- Plots: `outputs/sweep_findings_qpolicy_c64_t5_ce.png` (convergence + per-t trajectory + objective-vs-band-σ),
  `outputs/sweep_qpolicy_c64_t5_ce.png` (objective vs each HP).
