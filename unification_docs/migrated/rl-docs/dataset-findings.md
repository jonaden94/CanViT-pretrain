# Candidate-dataset findings (verifiable, with recipes)

Empirical facts about the cached candidate datasets, each with the exact
computation that produced it.

## 0. Metric hierarchy and known exchange rates

Deployed full-val mIoU (paper protocol) is the ultimate readout but costs
a GPU eval per arm — iteration runs on proxies. Each proxy below is
defined where used; their LINKS to mIoU are empirical and sparsely
validated, so treat exchange rates as point observations, not constants:

- within-cell z of -CE: the training target; dimensionless, per-scene.
- val Spearman (critic scores vs -CE over a cell's candidates, averaged
  over val cells): ranking proxy. Observed exchange ONCE (2026-06-12):
  +0.04 val Spearman t1 (0.26 -> 0.30) -> +0.25 pp (randomk) / +0.43 pp
  (safe-box greedy) deployed t1 mIoU. Do not extrapolate linearly.
- top1-regret (CE of critic's pick minus cell-best CE, nats): closer to
  deploy than Spearman (only the argmax matters); same caveat.
- flow NLL (action-space, nats/action): density-fit proxy for priors and
  actors; no measured link to mIoU yet.
- binned-conditioning R^2 of z (entry 4): information-content bound for
  feature subsets; bounds attainable score-vs-z correlation, NOT mIoU. Conventions: "the oracle dataset" =
`runs/traincand_seqoracle_t5_c32` on crockett (20,210 ADE20K train images,
oracle-advance T=5, 17 candidates/state: k=0 EG-C2F proposal + 16 R-IID
min-scale 0.25); z = within-(image,t) z-score of -CE over a stated candidate
subset; analyses ran on a slimmed copy
(`select("image","t","k","source","center_y","center_x","scale","ce")`,
regeneration command in docs/sessions/2026-06-12-arch-and-repro.md, PM
section). Scripts: throwaway/oracle_marginal_analysis.py and inline snippets
quoted per entry. All [CC measured 2026-06-12] unless noted.

## 1. Raw CE is 99% scene-difficulty nuisance

At t=1, R-IID candidates only: total CE variance 0.207; between-scene 0.205
(**99.1%**); within-scene 0.002 (0.9%). At t=4: 99.8% / 0.2%. The
decision-relevant (within-scene) signal is ~1% of raw variance — this is WHY
per-scene z-normalized targets train ~100x better than raw regression
[user-confirmed rationale; numbers CC].
Recipe: `group_by(image).agg(mean, std)` on `ce`; between = var of means.

## 2. Within-scene CE spread is ~proportional to the scene's CE level (CV ~ 4%)

Definitions (all at t=1, R-IID candidates only, n = 20,210 train scenes):
D(s) := mean over the 16 per-image CE values of scene s's t1 candidates
(post-glimpse CE under a random viewpoint); S(s) := sd over the same 16.

- S(s) quantiles: p10 0.009, p50 0.026, p90 0.063, p99 0.141 nats
  (p90/p10 = 6.9x). D(s): p10 0.24, p90 1.25 nats.
- Pearson(D, S) = 0.674, Spearman = 0.785, Pearson(log D, log S) = 0.823
  with OLS slope log S ~ 0.88 * log D.
- CV(s) := S/D: p10 0.025, p50 0.043, p90 0.079; Spearman(D, CV) = -0.10.

Precise reading: CE across scenes behaves close to a scale family —
candidate choice perturbs a scene's CE by ~+-4% (CV) of its level, nearly
independent of the level. The 0.674 mean-sd correlation is mostly this
proportionality, NOT an extra "harder scenes are intrinsically more
selectable" effect (residual CV-vs-level relation ~ -0.10). In ABSOLUTE
nats, selection headroom concentrates in high-D scenes (entry 7) purely
because gains scale with level; in RELATIVE terms selection is worth a
similar ~4% everywhere. Consequences unchanged in form: (a) z's
scale-division equalizes per-scene gradient mass; (b) softmax(z/tau)
actor weights carry effective temperature tau*S(s) (~7x spread); (c) for
multi-step objectives use per-state mean-subtraction (policy-invariant
baseline) + a per-SCENE scale, never per-step (see entry 8).

## 3. EG-C2F proposals are ATOMIC

All 80,840 EG-C2F proposals (k=0) in the oracle dataset have scale exactly
0.5 and only **4 distinct (y,x) values** (quadrant centers). 12.2% of
oracle-advance picks are these atoms. Any density/NLL/pick-distribution
analysis must exclude behavior-won cells or it spike-fits deltas (a
schedule-only flow scored -2.87 NLL with atoms vs -0.770 without).
Future datasets drop the behavior candidate entirely [user directive].

## 4. The image-blind prior is a (scale, radius, axis, t) schedule;
##    it explains ~5% of within-scene z-VARIANCE (definition below)

R-IID-only, z within scene. Win rate P(best-of-16) by scale bin flips with
t: at t=1 monotone UP in scale (0.033 at <0.35 vs 0.086-0.091 at 0.55-0.8);
at t=4 reversed (0.066 small vs 0.051 at >0.8). Uniform reference 1/16 =
0.0625. Within small scales (<=0.45), radius effect flips sign: t=1
central good (mean z -0.07 -> -0.265 toward |c|>0.5), t=4 eccentric good
(+0.145 -> +0.178). At t=4 the periphery preference is HORIZONTAL
(x-extreme bins +0.177/+0.225 vs center +0.137), not vertical.

The "~5%" PRECISELY: let z be the within-(image,t) z-score of -CE over the
16 R-IID candidates; population = all such candidates at fixed t (20,210
scenes x 16). Estimator: bin candidates by action (scale: 12 uniform bins
of floor(16s); position: 8x8 uniform (y,x) bins; joint: their product);
R^2 := count-weighted variance of bin means / total variance of z
(dimensionless; z-variance ~ 1 by construction). Values: scale-only
0.0435 (t=1) / 0.0512 (t=4); position-only 0.0294 / 0.0241; joint 0.0515 /
0.0535; t=2 anomalously weak (joint 0.0108). NOT a deployed-mIoU
percentage. Implication: the best possible action-only (image-blind,
within-t) predictor of z has correlation <= sqrt(0.054) ~ 0.23 at these
bin resolutions — consistent with the state-blind critic's measured 0.172
val Spearman. Pooled-over-(scale,t) views CANCEL the radial effect
(Simpson) — always condition.

## 5. The within-scene value landscape is rough and weakly multimodal

t=1, R-IID pairs within scene, scale-matched (|ds|<0.1): mean |dz| = 0.783
at center distance <0.15, rising to ~1.15-1.20 at distance >0.8. Reference:
uncorrelated z pairs give E|dz| = 2/sqrt(pi) = 1.128. So even near-identical
actions differ by ~70% of the uncorrelated level (CE is deterministic —
this is genuine landscape roughness at sub-0.15 granularity, not noise).
Scale axis similar: |dz| 0.784 at ds<0.05 -> 1.173 at ds 0.3-0.6.
Top-2 candidates' center distance: median 0.362 vs random-pair 0.514 —
good regions are broad-ish but NOT a single sharp peak (partial
multimodality; actor densities must be multimodal).

## 6. Best-of-k follows iid-normal order statistics; K=64 is worth ~+0.55 z

Exact within-cell order-statistics computation (P(max of k-subset) =
C(i-1,k-1)/C(16,k) over ascending z): E[max z] = 0.571 / 1.047 / 1.447 /
1.787 at k = 2/4/8/16. This matches iid standard-normal expectations
(E max_16 ~ 1.77) — within-cell z behaves like near-independent draws
(consistent with entry 5's weak spatial correlation). Extrapolation to
k=64 (iid normal): ~2.33, i.e. +0.55 z of oracle target headroom over
k=16 — what the K=64 regeneration buys in supervision sharpness.

## 7. Absolute selection gains concentrate in high-spread (= high-level) scenes

Raw per-scene selection headroom (D(s) - min CE over the 16, in nats):
the top-2 S(s) deciles (20% of scenes) hold **44.4%** of the summed total.
Per entry 2 this is the scale-family effect: absolute gains track the
scene's CE level; RELATIVE gains (~4% of level) are roughly uniform.
Deployed-policy per-image deltas are expected heavy-tailed; paired
per-image statistics are the right lens (consistent with the wide CI on
the +0.429 result).

## 8. Scene selectivity is a stable property across the horizon

corr(sigma_t1, sigma_t4) across scenes = 0.622 (oracle-advance states).
A per-SCENE reward scale for multi-step objectives is well-defined, and
t1-trained value structure has cross-t transfer support.

## 9. Learned image-blind prior beats R-IID by 0.082 nats (ceiling ~5%, cf. 4)

Held-out NLL of oracle-advance R-IID-won picks (atoms excluded, entry 3):
analytic R-IID -0.719; t-schedule-only flow -0.770; GRU-over-VPE-history
flow -0.801 (gap grows with t: history = coverage/anti-revisit signal).
Models: 6x(MAF h64) conditional nflows + safe-box u-transform,
throwaway/flow_prior.py; ~60 s CPU each. R-IID analytic density:
p(a) = 2L/L_max^2 * 1/(4L^2), L = 1-s, L_max = 0.75.
