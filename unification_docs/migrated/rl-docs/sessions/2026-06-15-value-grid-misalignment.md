# 2026-06-15 — Value-grid misalignment: investigation + aligned-readout arch (crossover in progress)

## Question [user]
"Review the architecture for a fundamental MISALIGNMENT issue, taking into account the precise manner
in which viewpoints/glimpses are taken and the skip connections." Then, emphatically: be skeptical of
my own prior high-context reasoning (re-derive from code, don't defend); test it cheaply with synthetic
geometric images + the step-0 reward maps. ([[skeptical-of-own-high-context-reasoning]])

## The misalignment — VERIFIED FROM CODE (not asserted)
- The value grid output `[B, n_scale, 16, 16]` is in **per-scale SAFE-BOX coords**: cell (i,j) at scale s
  scores the viewpoint centered at `grid_viewpoints = linspace(-(1-s), 1-s, 16)[i,j]` (`grid_net.py`).
- `GridValueNet` is a ConvNeXt U-Net over the **FULL-IMAGE-aligned** 32×32 canvas (`_feats` →
  `get_spatial` → row-major `grid_coords` tokens, image cell (r,c) ↔ image loc (r,c)). The skips
  (esp. the 16×16 skip = pooled input) register output cell (i,j) to **full-image** loc
  `g16[i] = (i+0.5)/16*2-1`.
- ⇒ output cell (i,j) is skip-fed the full-image feature at `g16[i]` but supervised for the safe-box
  viewpoint at `linspace(-(1-s),1-s)[i]`. **Affine offset = s − 0.0625: 0.44 @ s=0.5, 0.19 @ s=0.25**
  (zero at center, grows outward). The output corner (s=0.5) = viewpoint (0.5,0.5) = centre of the
  bottom-right *quadrant*, NOT the image corner the conv geometry pairs it with [user's framing].
- **KILLER:** one shared decoder + `Conv2d(width, n_scale, 1)` head feeds BOTH scales → a single
  spatial registration cannot serve two different safe boxes. FORCED by the arch.
- Coordinate convention cross-checked in `canvit_pytorch.{viewpoint,coords,model.base.impl}`: `[-1,1]²`,
  (y,x); `grid_coords` cell centres `(idx+0.5)/N*2-1`; glimpse WRITES via position-aware RoPE
  cross-attention at `center ± scale·retinotopic`; probe READS canvas (r,c) ↔ image (r,c). The whole
  pipeline is image-aligned; **only the grid READOUT lives in safe-box coords.**

## Synthetic probe (training-free) — `throwaway/grid_align_probe.py` @ a03382a
Real ADE object patch pasted at a controlled diagonal position sweep; step-0 TRUE reward map
(`score_grid`, no policy) + the net's input feature energy. Result:
- **INPUT geometry clean**: feature-energy peak tracks the object's full-image position p (gain ≈ 1).
- **Reward TARGET** tracks p but **compressed/smeared** (centroid gain ~0.34, noisy argmax) — a scale-0.5
  glimpse covers ±0.5, so many cells partially cover the object → the reward is a broad plateau, not a
  delta, and the safe box clamps edge objects inward. So the synthetic reward map **cannot crisply
  isolate** the offset; the architectural proof above stands on its own. Figures: `outputs/align_probe/`.

## The aligned-readout fix — `grid_net.py` `policy_arch='aligned'` @ 9e56585
[user: keep the working ConvNeXt enc-dec; no hybrid global-pool bolt-on; don't optimize for cheaper.]
Decode back to the **full 32×32 canvas-shaped informative map** → `grid_sample` at each scale's
safe-box viewpoint centres (re-register PER SCALE) → shared conv head over `[local + scale_emb]` →
value. Output cell (i,j) now reads features AT viewpoint (i,j)'s centre: **aligned by construction, no
shared-substrate conflict, location-resolved** (CLAUDE.md mandate). Contained to `GridValueNet`
(deploy/loss/eval consume `[B,n_scale,16,16]` unchanged); ckpt records `policy_arch`; `direct` path
byte-identical (tests pass). +0.16M params @ width 64 (bigger, not cheaper).

## Experiment

### c64 T=5 eval — the BAD (direct/misaligned) arch baseline [the bar to beat]
`grid_t5_fused_c64_op50_20k` (`--canvas-grid 64 --train-horizon 5 --steps 20000 --eval-every 1000
--keep-every 5000 --prime-on-policy 0.5`) was KILLED at step 8418 [user-authorized] to free the GPU
for the arch test. Snapshot `eval_snap.pt` = step 8000 (cp of last.pt). Eval:
`grid_eval --ckpt-run grid_t5_fused_c64_op50_20k --ckpt-name eval_snap.pt --canvas-grid 64
--n-timesteps 6 --batch-size 8 --run-name c64eval_step8000_t5` (results in that run dir on crockett).

| t | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| mIoU % | 39.59 | 42.58 | 43.44 | 44.39 | 44.73 | 45.00 |
| paper EG-C2F-c64 | 39.6 | 42.2 | 43.3 | 44.1 | 44.7 | — |

- **t0 = 39.59 ≈ paper Table 4 c64 t0 39.6 → PROTOCOL VERIFIED** (paper squish-512, c64 canvas, frozen probe).
- The misaligned arch beats EG-C2F-c64 at every t (+0.38/+0.14/+0.29/+0.03), at only 40% trained.
- **NUANCE [user-flagged]: the BAD arch is already this good ⇒ the misalignment is NOT crippling.**

### direct vs aligned t1 screen [isolates ONLY the readout]
`grid_train --policy-arch {direct,aligned} --run-name grid_t1_{direct,aligned}_head`, all grid_s8
defaults (c32, `train_horizon=1`, 5000 steps, lr 8.251e-5, wd 0.01192, batch 16, eval_every 250,
seed 0) @ 9e56585. Identical config; ONLY the readout differs.

val_gridcorr / spearman (direct / aligned), key steps:

| step | 250 | 1000 | 1500 | 1750 | 5000 (final) |
|------|-----|------|------|------|--------------|
| gridcorr direct  | 0.254 | 0.300 | 0.308 | 0.310 | 0.319 |
| gridcorr aligned | 0.095 | 0.282 | 0.307 | 0.315 | **0.329** |
| spearman direct  | 0.256 | 0.303 | 0.313 | 0.315 | 0.324 |
| spearman aligned | 0.099 | 0.292 | 0.317 | 0.325 | **0.337** |

**SETTLED (step 5000): aligned WINS the landscape fit** — gridcorr +0.010 (0.329 vs 0.319), spearman
+0.014 (0.337 vs 0.324) — having started slower (bigger net, `scale_emb` zero-init), crossed ~step 1500
and kept climbing while direct plateaued. **BUT `val_miou_t1_mode` is TIED: 0.4111 vs 0.4112** — the
rank-metric edge does NOT (at t1) convert to the argmax deploy metric. Mode-spread: aligned 0.26 vs
direct's COLLAPSED 0.20 (direct narrows its selections to win early gridcorr; aligned stays diverse —
why its value maps look better [user]).

**T=5 rollout of these t1 policies** (c32; OFF-DISTRIBUTION at t2+ since t1-trained):
direct 38.54/41.12/41.64/41.72/41.78/**41.89**, aligned 38.54/41.14/41.58/41.66/41.66/**41.65**. Direct
edges aligned; both plateau fast (t1 policies don't know evolved states). NOT the real T=5 test.

## Read (t1 screen SETTLED; the deploy question moves to trained-T5 / c64)
- Aligned fixes the misalignment by construction and wins the t1 LANDSCAPE FIT (gridcorr/spearman) with
  a healthier, less-collapsed mode-spread + better-looking value maps [user]. So the misalignment IS a
  real limiter on the fit — but NOT crippling (the bad arch still beat EG-C2F-c64; t1 deploy mIoU TIED).
- The DEPLOY metric (mIoU) has NOT separated at t1. Whether aligned's fit/spread edge converts to better
  ROLLOUT mIoU is THE open question — now tested at the real regime (train_horizon=5 + c64, below).
- PROCESS: I prematurely concluded a NULL at ~step 1250 mid-crossover; corrected. [[training-dynamics-patience]].

## Scale analysis → single-scale 0.5 [user decision]
c64 best-ckpt action scales (`runs/c64eval_step8000_t5/actions.parquet`): **96.6% scale 0.5** over t≥1,
**100% at t1**; 0.25 only creeps in late (t4 5%, t5 10%). EG-C2F also uses a fixed 0.5 scale for t1–t4.
⇒ [user] DROP 0.25 for the T=5 horizon — smaller action space → cleaner learning dynamics, expect
equal-or-better, re-add later if ever. (This also makes the two-scales-one-decoder conflict moot in
practice — the policy lives at one scale.)

## Decisions + the best-config c64 run [IN FLIGHT]
- **`policy_arch='aligned'` is the NEW DEFAULT** (52b92e2); `direct` byte-identical (`--policy-arch direct`).
- **`grid_t5_aligned_s05_c64_20k`** = the synthesis: aligned + single-scale 0.5 + train_horizon=5 fused
  (`prime_on_policy=0.5`) + c64, 20k steps, `keep_every 5000`, `eval_every 1000`, grid_s8 HPs. Launched
  @ 52b92e2. Startup verified: t0 **39.60** (=paper 39.6), single-scale (`scale_std 0.0`), finite
  loss/grad, 96 gf/step. Throughput **~2.75 steps/s / ~264 gf/s, SM util ~98%** (saturated; ~same as the
  direct c64 — the bigger aligned net costs ~0 wallclock, backbone-glimpse-bound). 1.92M gf @20k; the
  standard 1M lands ~step 10.4k (~1h); ETA ~2h. **Bar to beat: direct-2scale c64 45.00 @ t5 (step 8000),
  EG-C2F-c64 44.7 @ t4 / 45.6 @ t9.** JUDGE BY ROLLOUT mIoU (grid_eval `--n-timesteps 6 --canvas-grid 64
  --scales 0.5`) at each 5k ckpt — the milestone says the rollout peaks BEFORE the final step.

### c64 aligned s0.5 — ROLLOUT mIoU by checkpoint [recording as they land]
`grid_eval --ckpt-name <snap>.pt --canvas-grid 64 --scales 0.5 --n-timesteps 6 --batch-size 8` on a cp of
`last.pt` — VERIFY the snapshot step first (last.pt is overwritten each 1000-step eval). Eval run dirs:
`runs/aligneds05_step<N>_t5/`.

| ckpt | t0 | t1 | t2 | t3 | t4 | t5 |
|------|----|----|----|----|----|----|
| **step 3000 (15%)** | 39.59 | 42.47 | 43.67 | 44.05 | 44.35 | 44.55 |
| **step 5000 (25%, FINAL — killed)** | 39.59 | 42.34 | 43.67 | 44.41 | 44.69 | 44.99 |
| EG-C2F-c64 (bar) | 39.6 | 42.2 | 43.3 | 44.1 | 44.7 | 45.6@t9 |
| direct-2scale @8k | 39.59 | 42.58 | 43.44 | 44.39 | 44.73 | 45.00 |

**T=5 = t0..t4 [user 2026-06-15] — t5 is BEYOND the horizon** (an extra glimpse; my evals used
`n_timesteps=6` so the tables show a t5 column for completeness, but the T=5 comparison is t0..t4; use
`--n-timesteps 5` going forward). **KEY (softens the prediction):** from step 3000 → 5000 the LATE
timesteps CLIMBED (t3 +0.36, t4 +0.34) while t1/t2 stayed flat — so the step-3000 "trails late-t" was
largely UNDER-TRAINING, NOT (yet) the single-scale cap. At step 5000 (T=5, t0..t4), s0.5 beats EG-C2F-c64
at t1–t3 and TIES at t4 (44.69 vs 44.7). So the [user] single-scale-caps-late-t prediction is NOT
confirmed; s0.5 was doing well. Killed at 5k per plan, pivoted to 2-scale — which must beat THIS at
t3/t4 to justify 0.25. (s0.5 DONE: step 3000 + 5000 evals only; `runs/aligneds05_step{3000,5000}_t5/`.)

### c64 aligned 2-scale (0.5,0.25) — ROLLOUT mIoU by checkpoint [the 0.25 test] — LIVE `grid_t5_aligned_2scale_c64_20k`
`grid_eval --ckpt-name step_NNNNNN.pt --canvas-grid 64 --scales 0.5 0.25 --n-timesteps 5` (T=5=t0..t4); cp eval run dirs `runs/aligned2scale_step<N>_t5/`.

| ckpt | t0 | t1 | t2 | t3 | t4 | Δ vs s0.5@5k (t1..t4) |
|------|----|----|----|----|----|----|
| **step 5000 (25%)** | 39.59 | 42.75 | 43.87 | 44.47 | 44.81 | +0.41 / +0.20 / +0.06 / +0.12 |
| **step 10000 (50%)** | 39.59 | 42.94 | 44.05 | 44.55 | 44.94 | +0.60 / +0.38 / +0.14 / +0.25 |
| s0.5 @5k (ref)      | 39.59 | 42.34 | 43.67 | 44.41 | 44.69 | — |
| EG-C2F-c64 (paper)  | 39.6  | 42.2  | 43.3  | 44.1  | 44.7  | — |

**5k→10k (fixed val, deterministic — no eval noise):** STILL CLIMBING — step_10000 beats step_5000 at every
t1–t4 (+0.19/+0.18/+0.08/+0.13), consistent direction across all t (a real training gain, not a 1-ckpt
fluke). Beats EG-C2F-c64 at all t1–t4 (+0.74/+0.75/+0.45/+0.24). The LR-peak (step ≈10000, warmup_frac 0.5)
raw-loss spike (one window → 13.8, pre-clip grad → 11.8, clipped to `grad_clip=1.0`) was CONTAINED:
`train_loss_norm` flat (CV 0.07), per-depth corrs normal, policy improved THROUGH it. LR now decaying;
track 15k/20k for the peak (milestone: rollout often peaks before the final step).

**SURPRISE vs the prediction:** at MATCHED step + glimpse-budget (both 96 gf/step, 480k @5k), 2-scale beats
s0.5 at EVERY t1–t4 — but the gap is LARGEST EARLY (t1 +0.41, t2 +0.20) and SHRINKS late (t3 +0.06, t4
+0.12), the OPPOSITE of "0.25 helps late-t refinement." Also beats EG-C2F-c64 at all t1–t4 (+0.55/+0.57/
+0.37/+0.11). CAVEATS (do not over-read one ckpt): step 5000 = 25% trained; s0.5 is a DIFFERENT run (not
same-seed-comparable — different action space → different rollout trajectory). Decisive = best-vs-best at
convergence; tracking 10k/15k/20k.

**SCALE-USAGE (deploy, step_5000; `throwaway/scale_by_t.py`):** 0.25 picked only **2–9%** (t1 2.1%, t2
8.8%, t3 5.1%, t4 2.1%; t0=100% scale-1.0 = fixed full-scene init, NOT a grid choice). The biggest win
over s0.5 (**t1 +0.41**) is at the LOWEST 0.25 usage (2.1%) → the t1 edge is **NOT from deploy-time 0.25
selection**. **BUT low deploy-usage ≠ 0.25 irrelevant [user correction]:** having 0.25 in the TRAINING
action space affects optimization dynamics (extra value targets, different gradient landscape) INDEPENDENT
of deploy-selection frequency. So the edge = opt-dynamics-of-training-with-0.25 OR run variance — one run
each can't separate them (my earlier "win is NOT from 0.25 → confound" binary was WRONG). CLEAN SEPARATOR
(if ever needed): deploy the 2-scale net with the 0.25 channel MASKED (same weights, 0.5-only argmax) —
still beating s0.5 ⇒ training-dynamics not deploy-0.25 (needs a deploy mask; eval asserts
ckpt.scales==cfg.scales). NB aligned's 0.25 profile peaks MID-traj (t2), unlike direct-misaligned which
grew 0.25 LATE (t4/t5). **At step_10000, 0.25 usage DROPPED to 0.3–2.4%** (t1 0.3 / t2 1.2 / t3 1.3 / t4
2.4%; vs 2–9% @5k) — the converged policy is ~all-0.5 (97.7–99.7%); with training it prefers 0.5 MORE while
rollout mIoU climbs ⇒ stronger evidence the 2-scale edge is NOT deploy-time 0.25 selection (opt-dynamics or
run-variance; matched evidence stays 2scale@5k > s0.5@5k). **DECISION [user]: close enough — DEFAULT to both scales** (already the GridConfig
default `(0.5,0.25)`); it works well. Let the run reach 20k, eval periodically, record + keep probing.

**FUTURE [user, 2026-06-15]:** after the 2-scale run converges, consider RL robustness tricks — e.g.
**clipped double-Q** (two value heads, take the min) to counter the maximization / optimizer's-curse
positive bias in the deploy-time argmax over 512 candidates (16×16×2). NB our value net is REGRESSED to the
fractional-CE target (not TD-bootstrapped), so the overestimation here is from argmax-SELECTION among many
noisy candidates, not from bootstrap — double-Q still applies via that pathway.

**PREDICTION [user 2026-06-15]:** if, after sufficient training, aligned-s0.5 beats EG-C2F cleanly at
EARLY timesteps but FAILS at LATER ones, that gap is the **single-scale restriction** — later timesteps
benefit more from the smaller 0.25 scale (fine refinement; the direct run's 0.25 usage grew t4 5% → t5
10%), which s0.5 can't provide. It would be structural (won't close with more steps). CONTINGENCY: if
the late-t gap persists at convergence, re-add 0.25 → aligned **2-scale** (architecturally clean now);
that's the test of whether late-t needs the fine scale. (Track the early-vs-late pattern as it converges.)

## Provenance / repro
- HEAD **52b92e2** (aligned = default). Arch 9e56585; synthetic probe 52699f5 + a03382a. Cached
  `throwaway/grid_align_ab.py` (a4a0c06) was a throwaway un-mlflow'd screen — superseded by the real arch.
- t1 screen runs (DONE, mlflow :5500): `grid_t1_direct_head`, `grid_t1_aligned_head` (c32, 5000 steps);
  T=5 rollouts of each in `runs/t5eval_{direct,aligned}_latest/`.
- c64 direct baseline: `runs/grid_t5_fused_c64_op50_20k/eval_snap.pt` (step 8000) + `runs/c64eval_step8000_t5/`.

## Waste ledger
Killed `grid_t5_fused_c64_op50_20k` (direct, 2-scale) at step 8418 (~40%) to prioritise the arch test
[user-authorized]. Result PRESERVED: `eval_snap.pt` (step 8000) + `c64eval_step8000_t5` (the T=5 numbers
+ the scale-pick `actions.parquet`). Re-runnable (`grid_train --canvas-grid 64 --train-horizon 5 --steps
20000 --prime-on-policy 0.5 --policy-arch direct --scales 0.5 0.25`).

## HANDOFF (cold-start-resumable) — as of step ~12.5k
**LIVE: `grid_t5_aligned_2scale_c64_20k`** (crockett, mlflow :5500) — the best-config c64 policy: aligned +
2-scale (0.5,0.25) + train_horizon=5 fused (prime 0.5), 20k steps, keep_every 5k, **96 glimpse-fwd/step**
(=batch16×(1+H)=16×6; 960k by step 10k, ~1M by step 10.4k). At ~step 12.5k/20k, healthy, LR decaying from
its step-10k peak (warmup_frac 0.5). The s0.5 single-scale run was killed at 5k (pivot DONE); 2-scale is the
keeper [user: default to both scales — already GridConfig default `(0.5,0.25)`].

**RESULTS so far (T=5=t0..t4, full val n=2000, deterministic).** 2-scale beats EG-C2F-c64 AND s0.5 at every
t1–t4 and is STILL CLIMBING: step_5000 39.59/42.75/43.87/44.47/44.81 → step_10000 39.59/42.94/44.05/44.55/
44.94 (+0.19/+0.18/+0.08/+0.13). The matched-5k 2-scale>s0.5 edge (+0.41/+0.20/+0.06/+0.12) is NOT from
deploy-time 0.25 (usage 2–9%@5k → 0.3–2.4%@10k; opt-dynamics or run-variance — [[action-space-ablation-opt-dynamics]]).

**NEXT:** at each remaining 5k ckpt — `python -m canvit_pytorch_rl.grid_eval --ckpt-run
grid_t5_aligned_2scale_c64_20k --ckpt-name step_NNNNNN.pt --canvas-grid 64 --scales 0.5 0.25 --n-timesteps 5
--batch-size 8 --run-name aligned2scale_stepNNNNN_t5` (T=5=t0..t4); then `throwaway/scale_by_t.py <run>`
(scale usage) + regenerate `throwaway/plot_c64_miou.py` (c64 result plot). Judge ROLLOUT mIoU vs step_10000:
still rising or peaked? Training health: `python -m canvit_pytorch_rl.metric_stats --run-dir runs/<run>` —
read CV as a NOISE FLOOR, never eyeball a single window ([[per-batch-metrics-are-noisy-samples]], CLAUDE.md
RULE ONE). FUTURE [user]: clipped double-Q for the 512-way argmax maximization bias.

**BARS (c64, T=5=t0..t4, our apples re-evals):** EG-C2F-c64 39.60/42.22/43.30/44.04/44.65 (`egc2f_t5_apples`;
paper Table 4 rounds to 39.6/42.2/43.3/44.1/44.7); s0.5@5k 39.59/42.34/43.67/44.41/44.69
(`aligneds05_step5000_t5`); direct/misaligned 2-scale@8k 39.59/42.58/43.44/44.39/44.73 (`c64eval_step8000_t5`);
oracle ceiling (val seq, the headroom) 39.60/45.41/47.89/49.24/50.24 (`valcand_seqoracle_t5_c64`).

- Keep `policy_arch` (aligned=default); do NOT delete the `direct` path (byte-identical; tagged results).
- git_rev captured at launch (`harness.TrainLogger.git_rev`); crockett checkout synced to origin.
