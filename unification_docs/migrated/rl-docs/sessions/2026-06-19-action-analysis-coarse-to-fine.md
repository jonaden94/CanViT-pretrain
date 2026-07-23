# 2026-06-19 — action analysis: do the learned policies make sense? (coarse→fine + Q calibration)

Question [user]: "how could we analyze the learned policies vs untrained to see if they make sense."
Built tooling to characterize WHAT the deployed policy does (not just its CE), comparing trained vs its
own untrained init vs the baselines. HEAD around `1b4b69a`; local == crockett (verify `git rev-parse`).

## New / changed code
- **`tools/action_analysis.py`** — behavioral analysis from `actions.parquet` (the chosen viewpoint per
  image per t, written by every `q.eval` / `baselines.evaluate` run; schema `image,t,center_y,center_x,scale`).
  Each policy is a GROUP of per-seed dumps. Reports per t≥1: fraction of glimpses at the finest scale
  (coarse→fine signature) and center-placement entropy. `--scale-plot` draws frac-at-finest vs t, **one
  line per seed** (NOT a CI band — at small n a 95% t-CI is useless: dof=1 → t=12.7, and it dips below 0%
  on a bounded fraction; see the CI discussion below). Heatmap output existed briefly, removed [user].
- **`q.train --dump-init` + step-0 ckpt keeping** (`q/config.py`, `q/train.py`). The step-0 path never
  saved a checkpoint, so a run's random-init weights were unrecoverable — and `manual_seed(seed)+build_qnet`
  does NOT reproduce them, because `setup()` constructs the frozen backbone/probe and consumes torch RNG
  BEFORE the Q-net is built. Fix: (a) `keep_every` now also saves `step_000000.pt` at the step-0 eval so
  future runs retain their bit-exact init; (b) `--dump-init` replays only the prologue
  (seed→setup→build_qnet), saves the init, exits. For s3 (same code as its launch) this reproduced its
  logged step-0 metrics (ce_t4 0.7154 vs 0.7160, AMP noise) — but **DANGER [user 2026-06-19]: do NOT trust
  prologue-replay to recover a run's weights once the code has changed** — any RNG-consuming edit anywhere
  in the prologue silently shifts the init with no error, and verifying one seed licenses nothing for
  others (esp. seeds launched at older commits). `--dump-init` is safe only as "a random init at the
  CURRENT code" (a valid untrained CONTROL, where seed-matching isn't needed); the actual mechanism for
  exact init recovery is the saved `step_000000.pt`. [[never-rng-replay-weights-across-code]]. (This
  replaced a WRONG first attempt, `make_init_ckpt.py`, arbitrary seed-0 no-prologue, the user caught.)
- **`throwaway/q_calibration.py`** — at t1 (fixed full-scene t0), predicted Q[B,V] and the true
  `candidate_rewards`[B,V] are aligned by vp_flat order, so: Spearman(Q,true), captured-fraction vs the
  best-of-512 true-CE oracle, and regret. Micro-batched (small image batch + small candidate chunk) to
  fit in the GPU scraps the running band leaves.

## Findings (s3 deploy = best-mean(t1–t4) ckpt unless noted)
- **Coarse→fine is the COMMON mode but NOT necessary** (7 seeds: **5 zoomers, 2 coarse-stayers**). Zoomers
  s0–s4 rise monotonically ~1%→11–21% fine from t1→t4 (s0,s1 via last.pt; s2,s3,s4 via best-mean); stayers
  **s5 and s6 barely zoom (0→3% fine) yet reach IDENTICAL deploy CE** (s5 0.6649, s6 0.6657, both in-band).
  The untrained init does the OPPOSITE of either (starts ~80% fine — a readout init bias — and drifts down),
  so training reshaped a fine-biased init into coarse-first. With 2/7 winning seeds barely using fine,
  coarse→fine is clearly one of several equivalent strategies, not the mechanism of the CE gain — behavioral
  ≠ causal. (4 seeds looked unanimous; seeds 5–6 broke it — seed-band discipline paid off.) Magnitude is
  modest even in the zoomers (coarse-dominant throughout, mean scale 0.50→0.46), 2-scale action space.
  - **Zooming is image-SELECTIVE** (per-image fine-glimpse count over t1–t4, n≈2000/seed): even the zoomer
    group leaves **78% of images with 0 fine glimpses** (P(0..4)=0.78/0.16/0.05/0.02/0.00, mean 0.30);
    coarse-stayers 94% zero (mean 0.08); untrained mass at 3–4 (0.03/0.09/0.22/0.35/0.31, mean 2.81 — random
    Q is fine-biased). So the coarse→fine fraction-rise is carried by the ~22% of images a zoomer chooses to
    zoom (mostly one fine glimpse, late), not a broad shift. Viz: `throwaway/plot_coarse_fine_{strip,violin,
    violinline}.py` (per-seed dots/slopegraph and per-image violins — line+violin-per-t is the one the user
    wanted; at n=8 seeds show the points, not a KDE).
- **Coarse→fine is NOT causal — removable at deploy (mask ablation).** Forbidding the fine scale at deploy
  (`q.eval --mask-scales 0.25`, GreedyQPolicy sets those candidates' Q to −inf) costs the zoomer seed s3
  NOTHING: ce_t1–4 0.7145/0.6882/0.6737/0.6652 vs unmasked 0.7144/0.6883/0.6746/0.6650, mi_t4 44.88 vs
  44.85 — unchanged within noise. s5 (already coarse) unaffected, as expected. So the learned fine-scale
  picks are ~epiphenomenal; performance comes from WHERE coarse glimpses are placed, not the scale
  schedule. Caveat [[action-space-ablation-opt-dynamics]]: this is mask-AT-DEPLOY on a both-scales-TRAINED
  policy — it shows the deployed policy doesn't need fine, not that training without fine would match.
  **Confirmed on s4** (strongest zoomer, 21% fine): masked ce_t4 0.6667 vs 0.6663, mi_t4 44.79 vs 44.96 —
  ΔCE +0.0004, ΔmIoU −0.17, both at/below the per-batch noise floor. Robust across s3/s4/s5.
- **WHY coarse wins — the t1 reward landscape is coarse- and center-biased** (`throwaway/q_reward_landscape.py`,
  net-free, 128 val img): mean true CE-reduction is **0.045 for coarse (0.5) vs 0.018 for fine (0.25)** —
  coarse is ~2.5× more rewarding; the per-image best candidate is **coarse 90%** of the time (true top-10%
  is 78% coarse). Reward also **falls with center-radius**: 0.040 (center) → 0.017 (edge). This is the
  mechanism behind the not-causal finding: the optimal t1 move is almost always a coarse, fairly central
  glimpse, so a policy needs no fine scale to win and removing it costs ~nothing; coarse→fine is the policy
  weakly tracking the ~10% of images where fine is best. (score_res 128², t1.)
- **Per-t trajectory landscape — coarse-bias persists in the MEAN but fine becomes optimal for a growing
  minority late** (`throwaway/q_reward_landscape_traj.py`, rolls deployed greedy + measures next-candidate
  landscape at each visited state; s3, 64 img). Coarse > fine in mean reward at every t (~2×: t1 0.046/0.021
  → t4 0.0067/0.0028), so a coarse default holds throughout. BUT the share of images whose single BEST next
  candidate is FINE rises **9% → 19% → 19% → 36%** over t1→t4 — fine becomes optimal for more
  images once coarse context saturates (a real landscape shift the zoomers track, not noise). It's still
  removable at deploy (mask Δce_t4 ≤0.0004 < band σ 0.0005) because the late reward magnitudes are small:
  coarse mean reward falls 0.046 (t1) → 0.0067 (t4), so the gap between the best coarse and best fine pick is
  a fraction of that. (t1 row reproduces the standalone t1 landscape — cross-check.)
- **Selection quality DEGRADES along the rollout — replicates across seeds** (same tool, s3/s4/s5, 64 img).
  The deployed policy's Spearman(Q,true) falls monotonically t1→t4 in all three: s3 0.280/0.222/0.128/0.108,
  s4 0.300/0.215/0.156/0.132, s5 0.298/0.243/0.192/0.128 (t1 ~0.29 → t4 ~0.12). So Q is best-aligned at t1
  (largest-stakes, in-distribution) and worse deeper in — consistent with K=1 sparse supervision and the
  shrinking reward. The t1 landscape row is identical across seeds by construction (all start at full-scene
  t0) — consistency check. **Stayer s5 vs zoomers s3/s4: no clear difference** (s5 Spearman marginally higher
  mid-rollout, within seed+subset noise; similar best-is-fine share). METHOD CAVEAT: captured-fraction is
  unreliable past t1–t2 — its denominator (oracle−random) collapses as rewards shrink (0.046→0.007), so at
  64 img it's noise (e.g. s4 t3 0.005 vs s5 t3 0.126); use Spearman as the stable per-t signal.
- **Placement is image-adaptive, not a fixed schedule.** Center-entropy: trained ~6.0–6.5 bits (of 8
  max), untrained ~7.0 (near-random), EG-C2F ~1.8 (its fixed 4-quadrant tiling). Trained sits between
  rigid and random = adaptive structure — and places more centrally than EG-C2F (std_c ~0.3 vs 0.5),
  consistent with the center-biased reward landscape above.
- **Q is real but weakly calibrated, and this REPLICATES across seeds** (t1, 128 val images; oracle
  best-candidate true reward 0.184, random/mean-candidate 0.031):
    - s2: Spearman 0.337, captured 0.261, regret 0.110
    - s3: Spearman 0.324, captured 0.209, regret 0.117
    - s4: Spearman 0.340, captured 0.235, regret 0.115
    - untrained (qinit_s3): Spearman −0.039, captured −0.044 (its argmax is slightly WORSE than an
      average candidate — random weights carry no signal).
  So training creates real value signal from nothing, consistently (~0.33 Spearman, ~0.23 capture),
  but the policy reaches only ~⅓ of the way to the 1-step oracle. Consistent with deploy (t1 mIoU ~⅓ of
  the EG-C2F→oracle gap) and with the standing note that the best-of-K true-CE oracle is a LOOSE upper
  bound. [user: the rest may be impossible to capture — parked, not a deficiency to chase.]
  - Refined (s3, 256 img): chosen candidate sits at the **70th percentile** of the true-reward
    distribution (untrained 43rd — below median), and lands in the **true top-10% on 31%** of images
    (untrained 5.5% ≈ chance). So Q is a decent-but-not-sharp selector: reliably above-median,
    top-decile a third of the time. (Keep these calib runs at limit≤128 while the band trains — 256
    micro-batched noticeably slows the band.)

## Why the baselines aren't on the coarse→fine plot [user]
EG-C2F and C2F only ever emit scale 0.5 in this eval (their coarse-to-fine is a different scale
parametrization); F-IID samples its own scales. None choose from our discrete {0.5,0.25} grid, so
"% at scale 0.25" is structurally 0 / undefined for them. The scale plot is meaningful only for policies
on our action grid — the learned policy and its random-init control. → trained-vs-untrained is the proper
scope of that plot.

## Bottom line (overnight)
The deployed viewpoint-Q policy is sensible but its competence is narrower than the "coarse→fine" framing
suggested: (1) it places COARSE glimpses image-adaptively (placement entropy ~6 bits, between EG-C2F's
rigid 1.8 and random 7) and that placement is where the CE gain lives; (2) the emergent coarse→fine scale
schedule is real in 5/6 seeds but NOT causal — removable at deploy for ~0 cost (s3/s4) and absent in
a winning seed (s5); (3) the value function is really learned but weakly calibrated (Spearman ~0.33,
~70th-percentile pick, top-decile 31%, ~⅓ of the 1-step-oracle gap), replicating across s2/s3/s4.

## Live state / next (cold-start)
- Band `seedband_s*`: **ALL 8 COMPLETE** (s0–s7, 12500). Finalized: tag `result/qpolicy-8seed-band`,
  deploy band recorded in `docs/head_band_results.md` (beats EG-C2F-c64 on CE+mIoU at every t). Final
  coarse→fine tally: **6 zoomers / 2 stayers** (`coarse_to_fine_8seed.png`). Best-mean ckpt on disk for
  s2–s7 (keep_every); s0,s1 only have best.pt/last.pt → last.pt fallback.
- Supervisor exited → GPU was idle → **relaunched the perpetual HP sweep** (`throwaway/perpetual_sweep.py`,
  study `qpolicy_c64_t5_ce`) to refill the GPU (one process, GPU/disk-gated).
- Local action dumps in `throwaway/action_data/`: qact_last_s0/s1, qact_bestmean_s2/s3(=qact_trained_s3)/s4/s5,
  qact_untrained_s3, egc2f; coarse→fine plot at 6 seeds (`coarse_to_fine_6seed.png`).
  `uv run python -m canvit_pytorch_rl.tools.action_analysis 'trained=<parquets>' 'untrained=...' --scale-plot out.png`
- Eval/dump tooling: `q.eval --mask-scales 0.25` (deploy-time scale ablation); `throwaway/q_calibration.py`
  (Spearman/captured/regret/percentile/top10). All GPU dumps run `--batch-size 4` +
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to coexist with the band; keep calib limit≤128 (256
  micro-batched noticeably slows the band).
- **Mostly concluded.** Optional remaining curiosity: does coarse-glimpse PLACEMENT go to informative
  (high-probe-entropy) regions — the "is the placement smart, not just adaptive" question. Grow band to
  s6/s7 best-mean for completeness when they land (adds little — 6 seeds already show the heterogeneity).
