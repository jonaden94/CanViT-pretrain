# 2026-07-03 (cont.) — deep study: BN train/deploy argmax gap; per-depth supervision; pipeline verifications

Follow-on to the same-day audit session. Goal [user]: study the algorithm/data flow/training pipeline in
depth and check whatever is insightful, on crockett where useful. New probes: `throwaway/bn_mode_agreement.py`,
`throwaway/objective_rule_delta.py` (committed at `14a9a59`). All crockett numbers below
[CC measured 2026-07-03, crockett @ `14a9a59`].

## Finding 1 — the DAgger rollout does NOT follow the deployed policy: BN-mode argmax agreement is ~25–33%

The training rollout picks its on-policy (`prime_on_policy`) actions with the net in **train() mode**
(Frontend BatchNorm uses batch statistics); deploy/eval uses **eval() mode** (running statistics). Measured
on val batches at the train batch size (B=16, mirrors the rollout's same-depth batches), 128 images/seed,
`bn_mode_agreement.py` on three HEAD-band `best.pt` ckpts:

| seed | t0 agree | t1 agree | Q-map Spearman t0/t1 | signed Δreward (eval−train choice) t0/t1 | abs Δreward on disagreements |
|---|---|---|---|---|---|
| s3 | 32.8% | 23.4% | 0.925 / 0.891 | −0.009 / +0.005 | 0.051 / 0.040 |
| s5 | 25.8% | 28.1% | 0.931 / 0.900 | −0.012 / +0.009 | 0.050 / 0.043 |
| s7 | 22.7% | 27.3% | 0.930 / 0.891 | +0.002 / +0.004 | 0.053 / 0.044 |

(t1 states reached by the eval-mode argmax = the deploy trajectory; Δreward = true fractional-CE of the
chosen action, scored at `score_res` like training.)

Readings, carefully separated:

- **The two modes' Q MAPS agree well (Spearman ~0.9) but their ARGMAX agrees on only ~1/4–1/3 of images.**
  The top of the Q landscape is a broad near-tie: a small BN perturbation flips which of many
  near-equivalent candidates wins. Direct corroboration: on disagreements the two chosen actions' true
  rewards differ by |Δ|≈0.04–0.05 with signed mean ≈0 — different picks, same quality on average. The
  deployed argmax's IDENTITY is fragile; its reward level is not.
- **Consequence for training semantics:** `prime_on_policy=0.5` is documented as DAgger (rollout follows
  the net's own policy), but the policy it follows at train time is a batch-stat perturbation of the
  deployed one that picks a different viewpoint ~70% of the time. The state distributions are likely still
  close in quality terms (signed Δ≈0), so this is a *semantics* gap more than an obvious performance bug —
  but it is unmeasured territory: whether an exact-DAgger rollout (action selection under eval-mode BN,
  a one-line change in `rollout_samples`) trains better/worse is an open, cheap A/B (1M-forward run).
  Everything else is mode-consistent: `evaluate_q`, `rollout_eval`, `q_map_figure`, and deploy all run
  under `eval_mode(net)`; the grad-bearing supervised forward runs in train mode like the rollout's argmax.
- Not over-claiming a mechanism for the known per-t selection-quality decline: agreement does drop t0→t1
  (s3) but not in s5/s7; no clean depth trend in n=3.

## Finding 2 — per-depth supervision: target spread halves d0→d3; the fit follows corr×σ exactly

From the band runs' own logged diagnostics (`tools.metric_stats`, last 60 windows ≈ end of training;
s0/s3/s5 all agree):

- **Raw within-depth reward spread `fracstd_d` halves along the rollout**: ~0.085 (d0) → ~0.060 (d1) →
  ~0.049 (d2) → ~0.044 (d3). Under the single pooled global-z, the within-depth variance shares are
  ~49/23/15/13% — the MSE weights depth-0 discrimination ~4× more than depth-3. (Shares computed from
  within-depth spreads; between-depth mean differences add a further static component.) Depth-3 spread is
  still ~10× the bf16 CE-recompute wobble (~0.004 frac), so late supervision is downweighted, not
  noise-dominated.
- **Per-depth fit declines only mildly with depth**: `corr_d0` 0.41–0.48 → `corr_d3` 0.34–0.39. The
  deploy-side per-t selection-quality decline (Spearman 0.28→0.11 vs the true landscape,
  2026-06-19 session) is steeper than this train-side decline — but the two measures aren't directly
  comparable (train batch K=1 mixed-policy actions vs full 512-candidate val landscape), so this
  bounds rather than settles "is late-t weakness a training-weight problem".
- **No pathological output compression**: `predstd_d ≈ corr_d × (fracstd_d/σ_pooled)` at every depth,
  i.e. exactly the spread an MSE-optimal predictor of that accuracy should have. `target_std` ≈ 0.95–0.98:
  the online global-z is calibrated.
- If anyone attacks the late-depth weighting: per-DEPTH target standardization is NOT in the retired list
  (the bans are per-scene/per-image norms, multi-candidate scoring, ranking losses, ensembles) — but it
  would change the K=1 design's "one global z" invariant, so treat as a real experiment, not a tweak.

## Verifications (pipeline contracts pinned this session)

- **`run_episode` contract** (read upstream source): `policy.step(t, state)` called exactly once per t in
  order, state = post-glimpse state of t−1. `GreedyQPolicy`'s stateful encoder (reset at t=0, prev=init at
  the t1 decision, rolling after) therefore matches the training rollout's encoder semantics exactly.
- **Readout alignment**: numerically verified that `grid_sample` at the safe-box centres under
  `align_corners=False` reads a full-image-registered 32×32 map at the exact scene coordinates of each
  candidate (linear-ramp map reproduced to 6e-8, both scales). "Aligned by construction" holds.
- **New objective live**: fresh canonical run `qrecipe_s8` (seed 8, recipe defaults, crockett @ `14a9a59`,
  run dir `runs/20260703-135931_qrecipe_s8`): step-0 eval `objective` == recomputed mean(t1–t4) val CE
  exactly, and `val_ce_t0 = 0.7649` == the band's frozen t0 anchor digit-for-digit. Untrained-net rollout
  (random-init argmax) already climbs 39.6→43.2 mIoU / 0.765→0.711 CE over t0..t4, consistent with the
  documented untrained bands.
- **Selection-rule change in practice** (`objective_rule_delta.py` over the band + 30 completed sweep
  trials): the old endpoint rule deployed a different eval than the mean rule on 24/41 runs; mean cost
  ≈ +0.0003 mean-CE (max +0.0018). Small; recorded for completeness only.

## In-flight

- `qrecipe_s8` was launched here and killed at step 1925 (see the A/B section below) — superseded by
  `qdefault_s0` as the clean-checkpoint run.

## Housekeeping observed

- crockett disk: **21 GB free** (was 33 GB at the 2026-06-24 post-mortem; sweep-checkpoint reclaim still
  pending user OK). The 5 GB gate is not imminent but the trend is down.
- A user ipykernel (`scratch-space/gpu_roofline_20260625`, idle since Jun 25) holds ~3 GB VRAM; left alone
  (not ours); training fits beside it.

## Exact-DAgger A/B (launched same day; `throwaway/dagger_ab.py`, 2×3 seeds, 80k fwd, paired by seed)

`qrecipe_s8` was killed at step 1925 [user: prioritize the A/B] — its validation purpose was already served.
Seed-0 pair (objective = mean(t1–t4) val CE; identical step-0 = same init, sanity ✓):
ctrl 0.6945@500 / 0.6914@1000; dgr (rollout-act-eval) 0.6933@500 / **0.6910@1000** — paired delta −0.0004
at 1000, −0.0012 at 500 (dgr better, both far below the step-1000 cross-seed σ 0.0024, n=8 band).

**Full 3-seed verdict (clean paired null):** ctrl 0.6907±0.0006 vs dgr 0.6906±0.0005 (best mean(t1–t4)
val CE over evals); paired deltas −0.0004 / −0.0006 / +0.0005. Exact-DAgger costs nothing and buys the
cleaner method description → `rollout_act_eval=True` stays the default [user]. Runs
`runs/*daggerab_{ctrl,dgr}_s{0,1,2}` (ctrl arms re-run at `1c03b1cc+` with the explicit
`--no-rollout-act-eval` after the default flip; seed-0 pair predates it — same effective configs).

## Reward-transform structure (no training; `throwaway/reward_transform_stats.py` @ `e48fb85`)

From `runs/valcand_seqoracle_t5_c64` (17 candidates/state, oracle-advance states, R-IID candidates only,
n=2000 scenes, decisions t2–t4; base = selected CE at t−1):

- **Scale law at c64 is α≈0.95** (log σ_s(ΔCE) vs log base; c32-era was 0.88) — dividing by base¹ (frac)
  is within noise of the fitted power law; nothing to gain from tuning α.
- **frac removes most but not all scene-scale nuisance**: per-scene σ spread p90/p10 drops from ~8–9×
  (raw ΔCE) to ~3.1–3.7× (frac), all depths.
- **~44% of global-z target variance is between-scene MEAN** — under every per-scene rescale (raw, frac,
  α-fit): rescaling c(s) cannot touch it; only a BASELINE b(s) can. This is the single biggest target-
  variance component that isn't ranking signal. (Caveat: the net sees the state and can learn the scene
  value itself, so this is an early-training gradient tax + capacity competition, not irreducible error.)
- **Noise-dominated scenes grow with depth**: σ_s(ΔCE) median 0.016→0.0096 (t2→t4) against the ~0.003
  bf16 CE floor; scenes with σ_s < 2× floor: 14% → 22% → 29%. Late-rollout per-scene supervision is
  increasingly at the measurement floor — consistent with the per-t selection-quality decline, and the
  reason FULL per-scene-z (weight 1 per scene) is the wrong ideal [user stress-test: a constant reward has
  zero nuisance and zero signal; and per-scene-z upweights exactly the noise-floor scenes].
- frac is also more t-stationary than raw (median σ ratio t2/t4: raw 1.67×, frac 1.46×) — the
  "fraction-of-remaining-gap" form partially self-normalizes across depth.

## Dueling/centering ladder (same day; `throwaway/dueling_toy.py` + `throwaway/dueling_real.py`)

Question [user]: does a dueling head (Q = V(s) + mean-zero A(s,a)) fit ranking faster under K=1? Tested as
a ladder, cheap→real:

- **MLP memorization toy** (real 16-candidate reward vectors, random scene features, K=1): centered ≈
  dueling ≈ 2.3× plain on within-scene Spearman; oracle (true mean removed) ≈ action-only ≈ 0 — an anomaly
  (mean-PREDICTION seemingly aids feature learning) noted, not explained.
- **Real ViewpointQNet** (real t0 `StateEncoder` features + TRUE 512-grid maps via `candidate_rewards`
  semantics, 96 val scenes, K=1 overfit, run CONCURRENTLY beside the live A/B with a VRAM-capped cache):
  **the MLP result did NOT transfer** — centered ≈ plain (constraint buys nothing); **dueling shows a
  modest edge** (Spearman@400: 0.437±0.020 vs plain 0.397±0.011, n=2 seeds), consistent with the V head
  acting as a useful auxiliary (oracle again slightly BELOW plain, 0.375). Fresh-grid t0 between-scene-mean
  share = 0.45 — confirms the cached-data 0.44 on the live action space. 4-seed/800-step strengthening run
  + (if it holds) a paired 1k-step training A/B are the confirmation path. Cache:
  crockett `outputs/dueling_cache_c64.pt` (rev `487cb310`).
- Lesson worth keeping: the MLP control inverted on the real arch — never promote a toy-level optimization
  finding without the real-arch rung (the ladder cost ~15 min total and prevented a wrong wiring).
- **4-seed/800-step confirmation — the n=2 estimate shrank** (Spearman, mean±std over 4 seeds):
  step 800 dueling **0.495±0.015** ≈ oracle 0.495±0.016 > plain 0.476±0.012 > centered 0.455±0.015;
  at step 200 dueling 0.325±0.032 vs plain 0.314±0.024 (overlapping). So the honest dueling edge is
  **~+0.02 at ~1.3σ**, not the n=2 run's +0.04; centered is now WORSE than plain; the oracle-below-plain
  anomaly did not replicate (likely n=2 seed noise). Dueling remained ≥ plain at every horizon and was made
  the default [user, on the n=2 read; retained after this correction — cheap, non-negative], but the
  **paired 1k-step training A/B is load-bearing** for keeping it, not a formality.

## Defaults changed + the clean-ckpt run (afternoon)

- **`rollout_act_eval=True`** (exact deploy-policy rollouts) and **`dueling=True`** (V + mean-zero A;
  `ViewpointQNet.vhead`) are the recipe defaults [user]. Dueling's evidence is the real-net ladder above
  (+0.02 at 4 seeds); the paired 1k-step TRAINING A/B is the standing confirmation item.
- **`qdefault_s0`** (`runs/20260703-160435_qdefault_s0`, git_rev `e2fd800`, 1M forwards, bare defaults,
  seed 0) auto-launched when the A/B freed the GPU — the clean checkpoint of the new recipe.
  `grad_norm_vhead` logs alive. On completion: compare against the 8-seed band
  (`docs/head_band_results.md`) — it differs from the band by BOTH new defaults, so it is a recipe
  checkpoint, not an ablation.
- **Wasted compute noted (~6 GPU-min):** the first auto-launch gate ("GPU used < 4 GB → go") fired inside
  the 30 s gap BETWEEN two A/B runs and started a premature `qdefault_s0` (rev `487cb310`, pre-dueling)
  beside `dgr_s2`; it died before its first eval (manifest-only dir, deleted) when the gate was disarmed.
  Root cause: a bare memory-threshold gate is race-prone while a multi-run driver is mid-sequence. Guard:
  gate on the DRIVER's completion (its summary line / process exit), not on instantaneous GPU memory.
  The A/B run it briefly overlapped is unaffected (sharing changes throughput, not computation).

## Ruthless cleanup + restructure (afternoon; behavior-preserving, `just` green throughout)

- Options removed [user: remove useless options]: `t0_mode`/`scale_min` (riid start states — no measured
  edge; its removal made `rollout.t0_state` == `canvas_ops.full_scene_state`, so ONE t0 builder remains),
  `augment` + the dinov3 train-transform machinery (motivating "overfit" was a biased-slice artifact,
  sweep-flat), the ladder's centered/oracle arms + the whole MLP toy (non-transferring), one-shot
  throwaways (`objective_rule_delta`, two superseded coarse-fine plot variants). Second batch, after the
  A/B null landed: `rollout_act_eval` (eval-mode selection is now unconditional), `compile` (never
  validated on), `train_split`, and the optuna rung ladder (single-study `main`; CLI renamed
  `--budget-forwards`/`--n-trials`). Third batch [user: dump_init is brittle]: `dump_init` deleted — the
  TRUE init is now saved unconditionally at launch (recovery-by-artifact; RNG-prologue replay silently
  breaks on any code change, the 2026-06-19 DANGER failure mode) — and the `probe_repo` override removed
  (the probe is determined by canvas_grid). `dueling` stays flagged until its training A/B. All
  recoverable from git.
- Restructure for one-concern-per-module (`pypatree` clarity): `q.features` (state → net input) split out
  of `q.net`; `q.train_eval` (in-training eval loops) + `q.viz` (value-map filmstrip) split out of
  `q.train` (now the trainer alone). `evals` was renamed `train_eval` (one keystroke from `q.eval`).
- DRY: `sweep_report` now imports `horizons`/`mean_ce` from `seedband_io` (the deploy rule lives in ONE
  module); `rowwise_spearman` graduated to `q.stats` (was pasted in 3 probes; also revived
  `viz_t0_channels`/`reward_corr`, whose import of it had been dead since the 2026-06-18 qcorr cleanup).
- Docstrings de-regimed: `canvas_ops`/`scoring` no longer claim "frozen" (the caller's policy, not the
  ops'); README fully rewritten in plain prose with GFM math [user: kill the LLM voice].

## qdefault_s0 mid-run mechinterp (step ~4k/12.5k; `throwaway/ckpt_trajectory_probe.py` on the cached
96-scene t0 probe set — no backbone forwards, run beside training)

Per-ckpt trajectory (t0 states; sp_true = Spearman vs TRUE 512-cand maps; regret = true frac of best −
argmax; v_corr = Pearson(V(s), true scene-mean); fine% = argmax at scale 0.25):

```
  step  sp_true  regret  v_corr  fine%  radius   (Δsp when zero-ablating a group: only ln_feat matters,
     0    0.083  0.150   -0.032   59.4   0.574    growing to ~0.05 by 3k; all scalar groups ≈ 0 — high
  1000    0.274  0.131    0.262    1.0   0.406    input redundancy at t0, ln_feat is the non-redundant
  2000    0.298  0.122    0.269    0.0   0.444    carrier. Caveat: zero-ablation passes through the BN,
  3000    0.292  0.119    0.352    0.0   0.389    so this is perturbation sensitivity, and groups are
  4000    0.294  0.116    0.342    0.0   0.472    computationally redundant — ent is derivable from feats.)
```

- **Calibration saturates by 1–2k steps (8–16% of budget)** at the familiar ~0.3 ceiling; **deploy regret
  keeps falling after Spearman flatlines** — late training buys argmax refinement, not global ranking.
- **The V head works and is load-bearing**: v_corr rises 0 → ~0.35, and the log shows `grad_norm_vhead`
  growing from 17% to a plateau of **~45% of the total gradient** by 2k — a ~131k-param head (≈2.5% of
  params) carrying ~45% of the gradient, numerically consistent with the independent measurement that
  ~45% of global-z target variance is between-scene mean. The decomposition routes scene-value learning
  away from the conv trunk as designed.
- **t0 policy geometry commits immediately**: fine-scale argmax 59% (random init) → ~0% by 1k; radius
  → ~0.4; scale-spread canary collapses to ~0.02 while spatial spread stays healthy (~0.3) — coarse+central
  landscape internalized in the first 1k steps, spatial diversity retained.
- Gates decay symmetrically 1.0 → ~0.92 (weight decay on the gate params), no differential selection yet.
- All t0-only, n=96, one seed; rerun the probe over the full ckpt ladder at completion.

## Per-depth reward standardization is the default [user 2026-07-03: "doesn't cost anything, should help"]

`q.train` now keeps ONE `RunningNorm` per rollout depth (same momentum — no new HPs; still global across
scenes, per-scene stats stay banned). Grounds: measured reward spread halves d0→d3, so the pooled z gave
depth 0 ~4× the gradient weight and left a depth-mean in the target. Deploy semantics unchanged (per-depth
affine is constant within a state; argmax unaffected). CLAUDE.md design-invariant line + README updated.
NOTE: `qdefault_s0` (in flight) PREDATES this — it carries dueling + deploy-mode rollouts only; the next
1M run is the first with per-depth z.

## Schedule default changed + flagship relaunched [user 2026-07-03]

- **Default budget 320k forwards (= 4000 steps) with 1k-step linear warmup then HOLD** (`warmup_hold`
  replaces `warmup_cosine`; cosine is in git history). Basis: qdefault_s0 reached the band's quality region
  by 4k steps (meanCE 0.6868@4k vs band-final 0.6855±0.0004) and the sweep post-mortem put convergence at
  ~2k steps. The 1M-forward HARD CAP is unchanged — only the default moved.
- **`qdefault_s0` killed at step ~6500** [user] — the recipe had moved twice under it (per-depth z,
  schedule). Its best.pt@6000 (meanCE 0.6865) is on the Hub and stays the published flagship until
  superseded. Partial run dir kept (valid record, manifest pins `e2fd800`).
- **`qflagship_s0`** launched at `c8d1d25`: the FIRST run of the complete current recipe (dueling +
  deploy-mode rollouts + per-depth reward z + 4k-step warmup-hold), seed 0, no overrides.
  **[APPEND RESULT]**

## The "ρ≈0.26 CE↔mIoU" claim was a restriction-of-range artifact [user caught it 2026-07-03]

`scripts/ce_miou_scatter.py` (graduated) over ALL 3216 logged full-val evals on disk: within-timestep, converged
(step≥2000), CE↔mIoU Pearson = −0.61/−0.53/−0.73/−0.73 at t1..t4 — STRONG agreement across real quality
differences, near-monotone globally. The canonized "ρ≈0.26" came from ~20 evals of ONE converged run
(richaux_q_20k, 2026-06-16 night) spanning ~0.004 CE, where mIoU eval noise dominates; the post-mortem's
cross-trial −0.12 is the same effect (30 trials within 0.002 CE). Correct statement: the metrics agree
strongly at mesoscale and decouple only below ~0.002 CE — which remains the reason small comparisons are
judged on CE. POOLED across t the two metrics are one axis: Pearson −0.989 / Spearman −0.967 (n=3216; converged-only
identical). README + CLAUDE.md corrected. Figure: `outputs/ce_miou_scatter.png`.

## Evening: LR/budget verdict; recipe finalized; 8-seed band launched

- `qflagship_s0` (lr 3e-4, 4k steps, hold): deploy 0.6867 mean-CE @2k, then FLAT (0.6875/0.6877) — the hot
  LR plateaus ~0.001 above the old 1M band.
- `qlr2e4_s0` (lr 2e-4, 8k steps, hold; [user]): same 0.6867 shelf by 2k, then grinds THROUGH it late —
  0.6856@6k / 0.6863 / 0.6857 final (deploy 0.6856, t4 CE 0.6653, mIoU_t4 44.98) = the 1M band's level
  (0.6855±0.0004) at 64% of its forwards. Verdict: at 4k budgets LR doesn't matter; the last ~0.001 needs
  the cool LR × longer hold. (The terminal-anneal hypothesis was NOT tested — the hold alone sufficed.)
- **Defaults finalized** (`bcb9742`): lr 2e-4, 640k forwards (8k steps), 1k-step warmup then hold — plus
  everything earlier today (dueling, deploy-mode rollouts, per-depth z, mean-CE objective, logits dedup).
- **8-seed band `qband_s0..7` launched** on bare defaults (~9.3 h). Hub repo updated to the qlr2e4_s0
  deploy ckpt (0.6856). The interrupted first band attempt (single partial seed, recipe superseded
  mid-flight) was deleted.

## E2E differentiable action selection — overfit probe [user 2026-07-03 eve; `throwaway/e2e_gumbel_overfit.py`]

Question: can the task CE train the policy END-TO-END through the differentiable glimpse sampler
(ST-Gumbel over the Q map picks the candidate; dCE/d(viewpoint) flows through the frozen backbone)?
Setup: t1-only overfit on the first 8 cached-probe scenes; readout = argmax's TRUE frac / per-scene best
(oracle mean 0.183). Ran beside the qband (B=8, ~3-4 GB). 6 arms, ~3-12 min each:

- **ST-Gumbel pathwise FAILS robustly** (5 arms: lr {1e-3, 3e-3}, tau {anneal, const 1.0}, draws {1, 4},
  entropy bonus 0.02): policy entropy collapses 6.2 -> ~0.1 within ~100 steps (ST rich-get-richer), after
  which the only gradient is a local coordinate-wiggle — noise on a landscape that is rough at sub-cell
  scale (dataset-findings entry 5). Captured%% wanders −16..+29 with argmax flips between random cells.
- **Score-function WORKS immediately** (REINFORCE over the grid, per-scene baseline, k=8 sampled cells per
  scene-step, entropy bonus 0.02): captured 67/72/80/72%% at steps 100–400, entropy healthy (~5.5). It
  does NOT use glimpse differentiability at all — it is policy-gradient with sampled CE, i.e. a k×-cost
  cousin of the deployed K=1 measured-reward regression.
- **Localized estimator 2×2 (matched budgets, same net/lr/scenes; captured%% of oracle @400 steps):**
  PG k=1 (EMA baseline) 63 | PG k=8 78 | Qreg k=1 58 | Qreg k=8 70–78. **PG ≈ Q-regression at every
  budget** (differences within single-run wobble); the samples-per-scene budget k is what matters
  (~+15pp from k=1→8 for both). Arm E's earlier "PG wins" was purely its k=8 budget.
- Verdict: the pathwise/Gumbel ingredient is the ONLY broken part. Rebuttal-ready ablation: the
  differentiable route fails for identifiable reasons (ST collapse + sub-cell reward roughness); PG and
  K=1 regression are budget-equivalent, and regression needs no baseline/entropy machinery. Caveats:
  8-scene overfit (not generalization), single runs, qreg samples uniformly vs PG on-policy.
  Logs: crockett /tmp/e2e_{A..E,pg1,pg8,qr1,qr8}.log.

## Candidate follow-ups (not run)

- **Exact-DAgger A/B**: rollout action selection under eval-mode BN (one line in `rollout_samples`) vs
  current, 1M forwards each. Tests whether the Finding-1 semantics gap costs anything.
- **Late-depth supervision reweighting** (see Finding 2 caveats) — only worth it if late-t selection
  quality is the chosen attack surface.
