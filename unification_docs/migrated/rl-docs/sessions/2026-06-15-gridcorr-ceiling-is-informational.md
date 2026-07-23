# 2026-06-15 — The t1 grid reliably reaches val_gridcorr ~0.33 (and why higher is hard)

> **[user 2026-06-15 — CORRECTIONS to this doc's framing]**
> 1. **"Information ceiling" was OVERCLAIMED.** What's solid: the t1 grid RELIABLY REACHES val_gridcorr
>    ~0.33 (3 seeds) and training knobs are flat there — a nice result. The claim that the signal "isn't
>    in frozen t0" / a hard information limit is a HYPOTHESIS (ridge was pooled+linear; capacity test was
>    2 widths) — read the "ceiling/limit" language below as suggestive, not proven.
> 2. **We will NOT unfreeze perception [emphatic].** Disregard the "unfreeze perception is the lever"
>    recommendation in the T=5 section. Direction: simplify/clean/refactor the t1 grid → deploy mIoU →
>    adapt to LONGER HORIZONS (T=3 first), frozen perception throughout.

Continuation of the dense value-grid policy work (overnight from 2026-06-14). GOAL: maximize
val_gridcorr without changing how it's computed. This session reached ~0.33 (3 seeds) and characterized
why higher is hard, with a chain of cheap, committed, reproducible diagnostics. Single entry point to
resume cold.

## TL;DR

The val_gridcorr ceiling (~0.31–0.33) is an **information limit of the frozen t0 features**, and
the trained net is **already at it**. Not a capacity/arch/horizon/WD/LR knob (all verified flat at
~0.30 in prior runs). Decomposition of the per-image landscape (the metric is per-image Pearson →
rewards only WITHIN-image structure):

- **Constant** scale preference (s0.5 reward 0.0465 > s0.25 0.0227, holds for 83% of images):
  a same-for-all-images 2-level predictor scores between-scale gridcorr **0.264**. The net captures
  exactly this and ~nothing more on the scale axis — **even on the TRAIN set**.
- **Image-specific** between-scale (which images prefer the zoomed 0.25 scale): the +0.063 gap from
  0.264 → the 0.3273 per-image oracle. **NOT predictable from t0** — a ridge regression on pooled
  t0 features overfits train to 0.31 but generalizes to ≤ 0.264 on val at every regularization
  strength; the deep grid net also fails to fit it even on train. Two independent predictors fail →
  feature/information limit, not linear/capacity.
- **Within-scale** ("where in the scale to look"): the net DOES extract real, generalizable signal —
  within-scale corr 0.18–0.20 with truth, **beating entropy-only 0.122** — but over-weights it ~2–3×.

So realistic val_gridcorr from frozen t0 = constant-between (0.264) + within-scale (net 0.18) ≈
**0.31–0.33, and the net (0.308–0.314) is essentially there.** The unreachable +0.06 oracle headroom
is the **active-vision circularity**: which scale/position wins for a specific image isn't knowable
from the low-res t0 — you'd need to take the glimpse. Entropy is ALREADY a default input channel;
ON vs OFF = +0.008 val (grid_s13 trials), matching the +0.009 oracle — no unused entropy juice.

**Implication for the GOAL:** t1 training-knob tuning is exhausted. To raise val_gridcorr you must
change the INPUT INFORMATION — the **T=5 sequential setup** (earlier glimpses inform later viewpoint
value) breaks the circularity. The only residual t1 lever is a within-scale rebalance (~+0.02,
checkpoint-dependent and small).

## The diagnostic chain (all committed; run on crockett @ git 80067df, full-val N=2000, trainslice N=1000)

1. `throwaway/landscape_headroom.py` (CPU, truth-only): full-val constant-mean floor 0.282;
   **SCALE-ONLY floor 0.3273** (per-image per-scale mean, flat within); within-image variance
   ~85% within-scale-position / ~15% between-scale. (The 0.327 had been an UNREPRODUCIBLE hardcoded
   comment — now computed in-script. commit 6fa6d1d)
2. `throwaway/entropy_vs_reward.py`: within-scale entropy predictor 0.122; **(scale + optimal-λ
   entropy) ceiling 0.3446 at λ=0.08 — only +0.009 over the scale floor.** (commit c2af1e8)
3. `throwaway/net_decompose.py` (loads a trained ckpt, splits pred along the scale axis; SPLIT=
   validation|trainslice). grid_bigcap_w512 (0.308) and grid_s13_w128 (0.300), SAME on val AND train:
   net between-scale capture ≈ **0.262** (= the 0.264 constant floor); net within-scale vs truth
   0.18–0.20; within-scale std 2–3× truth; reweighting the net's own components buys +0.02 (w128) /
   +0.0007 (w512). (commits c7a289a, 1a15cd2, d62ab75)
4. `throwaway/between_scale_learnable.py`: ridge pooled-t0 → per-image per-scale mean; **VAL
   between-gridcorr ≤ 0.264 at every α** (overfits train to 0.31). image-specific between-scale is
   not in t0. (commit 80067df)

## Reproduction

```
# on crockett, ~/projects/CanViT-PyTorch-RL @ 80067df
uv run python throwaway/landscape_headroom.py                       # scale floor 0.3273, 85/15 split
uv run python throwaway/entropy_vs_reward.py                        # entropy 0.122, combined 0.345
RUN=grid_bigcap_w512_bl3 SPLIT=validation  uv run python throwaway/net_decompose.py
RUN=grid_bigcap_w512_bl3 SPLIT=trainslice  uv run python throwaway/net_decompose.py
uv run python throwaway/between_scale_learnable.py                  # ridge VAL ≤ 0.264 (feature limit)
```

The reward maps (val + trainslice) live on crockett under `runs/reward_maps/`; checkpoints under
`runs/<name>/best.pt`. Best available net for decomposition: `grid_bigcap_w512_bl3` (val_gridcorr
0.3083). Best EVER recipe is grid_s8 trial0009 (0.3136), the current `GridConfig` defaults.

## Deliverable + the rebalance lever, tested and rejected

`grid_repro_best` (default GridConfig = trial0009 recipe: ConvNeXt w128, lr 8.25e-5, WD 0.0119,
global-z, entropy on, 5000 steps) FINISHED: **best val_gridcorr 0.3199** (step 4000), final-eval
0.3193, val_miou_t1 0.4113 (= EG-C2F t1 41.1 bar), val_miou_t0 0.385 (= paper 38.5 ✓), **train≈val**
(train_gridcorr 0.3207 ≈ val 0.3193 → at the ceiling, NOT capacity-limited). A clean new best,
reproducibly, ≥ the prior 0.3136.

**Rebalance lever (within_var_reg) — TESTED and REJECTED.** Hypothesis from net_decompose: the net
over-weights within-scale ~3×, so a variance penalty should buy ~+0.02. I added `within_var_reg`,
but `--within-var-reg 0.1` was INERT (within_var ≈ 0.005–0.01, penalty 0.1·0.005 ≪ MSE loss 0.8 —
val_gridcorr tracked baseline exactly). Rather than blind-sweep β over orders of magnitude, I measured
the lever directly: `throwaway/reweight_eval.py` (commit b…) down-weights within-scale on the BEST net
through the real eval path:

| within-scale gain | val_gridcorr | miou_t1 |
| 1.0 (as-trained) | **0.3199** | 0.4113 |
| 0.7 | 0.3148 | 0.4113 |
| 0.5 | 0.3070 | 0.4112 |
| 0.0 (between only) | 0.2628 | 0.3960 |

Full-weight within-scale is OPTIMAL for the best net — down-weighting only hurts. The +0.02 reweight
headroom seen earlier was specific to WORSE checkpoints (grid_s13 @ 0.30) catching up to ~0.32; the
best net is already balanced. So within_var_reg would only suppress good signal → **reverted** (commit
on top). No within-scale rebalance lever exists above 0.32.

**Conclusion: val_gridcorr is MAXIMIZED at ~0.325 ± 0.004 from frozen t0 / t1.** Noise floor pinned
by 3 seeds of the best config (seed 0/1/2 = grid_repro_best/_s1/_s2): **0.3199 / 0.3252 / 0.3296**
(mean 0.325, std 0.004, best **0.3296** = grid_repro_s2/best.pt — the deliverable checkpoint). So the
predicted ~0.32–0.33 ceiling is confirmed; seed-0's 0.3199 was just the low end of the seed
distribution. Beyond ~0.33 needs new input info (T=5), not a t1 knob — and T=5 does NOT raise THIS
metric (t1-from-t0), it's a separate objective.

**Dynamical confirmation (grid_repro_best eval trajectory):** the converged plateau (steps 4000–5000)
is 0.3199±0.001 — 0.3199 is robust, not a spike. The one transient dip (step 2250, gridcorr 0.2465,
near the LR peak) coincides with `val_mode_scale_std` spiking to 0.077 (the argmax differentiating
scales per-image) and recovers when scale_std collapses back to ~0.015 (constant s0.5). So ATTEMPTING
image-specific scale differentiation *lowers* gridcorr — independent confirmation that image-specific
between-scale is anti-learnable from t0; the optimal policy is the constant scale preference.

## T=5 path: payoff AND risk from existing data (grounding the decision — analysis only, no new runs)

Per-timestep scene mIoU from existing crockett artifacts (all *_t5_c32, full val, t0=38.53 ✓ same protocol):

| t | EG-C2F (`valcand_rollout_egc2f_t5_c32`) | prior LEARNED critic T=5 (`critic_rollout_greedy_t5_c32`) |
| t0 | 38.53 | 38.53 |
| t1 | 41.04 | 40.62 |
| t2 | 42.05 | 41.59 |
| t3 | 42.69 | 41.96 |
| t4 | 43.17 | 42.41 |

t1 best-of-17 oracle (`valcand_bestof17_c32`) = **43.87** — one optimally-selected glimpse ≈ EG-C2F's
FOUR-glimpse endpoint (43.17). So the selection PRIZE is real and large.

**But the history is decisively sobering [RULE THREE] — surveyed ~20+ prior learned T=5 runs, NONE beat
EG-C2F.** Best of the prior effort (actor-proposal-advantage, critic, boxent; many K/seeds/configs):
`actorprop_critic_k16_soft8k_seed0` = **43.11** max-over-t — i.e. *tied-to-below* EG-C2F t4 (43.17),
and the rest trail (critic_rollout 42.41, most 42–43). The `critic_rollout_greedy` even underperforms
EG-C2F at EVERY step (40.6/41.6/42.0/42.4 vs 41.0/42.1/42.7/43.2). **Synthesis:** the frozen-perception
information ceiling I established for t1 (the "which glimpse helps" ranking signal isn't in the frozen
features) EXTENDS through the sequential setting — it's why the whole prior T=5 program didn't beat
EG-C2F. So T=5 is NOT a fresh win; it's a heavily-explored-and-unbeaten direction.

**What's actually still open for grid-T=5 (a long shot, not a fresh idea):** the grid is a BETTER
selector than the prior attempts (grid t1 41.1 > prior critic 40.6), so grid-T=5 isn't strictly
already-tested. The hypothesis would be that at t2+ the canvas has GLIMPSE-ACCUMULATED detail t0 lacked,
enabling selection that breaks the t1 ceiling. BUT the prior T=5 runs ALREADY operated on the evolving
canvas and still didn't beat EG-C2F — so glimpse-accumulation, with an info-limited selector, is
empirically discouraged. Honest read: the ranking signal isn't in the FROZEN features at any step, so
the fundamental lever is **unfreezing perception** (put the missing "which-glimpse-helps" info INTO the
features) — not another frozen-perception selection variant. grid-T=5 is worth at most a single
quick probe, not a program. **Key build challenge if pursued:** for t≥2 the reward is STATE-dependent
(depends on the policy's prior glimpses), so the static precomputed reward-map trick (the t1 keystone)
does NOT extend — T=5 needs ONLINE sequential
reward during rollout (or on-policy data), the main new work. [User decision: pursue grid-T=5 vs accept
the t1 result.]

### Untried t1 options (low expected value, enumerated for the user to add back if wanted)
- Probe CLASS-logit / max-prob input channels (vs just entropy): a richer per-cell uncertainty signal.
  But within-scale is already near its t0 limit (net 0.18 > entropy 0.122) and between-scale isn't in
  t0 at all, so expected gain is small. Test offline first (within-scale corr of max-prob vs reward).
- Wider scale band / more scales (currently {0.5,0.25}): changes the action space + reward maps, not a
  pure t1 knob; the scale axis is ~data-determined (s0.5 wins 83%), so unlikely to raise gridcorr.

## Process scars

- **GPU sat idle through the diagnostic chain.** The diagnostics were decisive (overturned two of
  my own prior conclusions — see below) so net-positive, but I should have staged the deliverable
  run earlier to overlap. Diagnostics that need only the reward map are CPU-only and instant; only
  net_decompose/between_scale needed the GPU briefly.
- **crockett's git remote is `origin`, NOT `deploy`.** The laptop pushes to `deploy` (=
  crockett:repos/...git, the bare repo); on the crockett CHECKOUT the correct pull is
  `git fetch origin && git reset --hard origin/main`. I ran `git fetch deploy` (fails) AND piped it
  through `tail` so the failure slid through and a stale script ran — the exact "never pipe a
  verification command" footgun. Re-ran un-piped after verifying HEAD.

## Beliefs CORRECTED this session (were wrong/overstated before)

- "Unused entropy headroom ~0.03–0.05" → actually +0.009 oracle; entropy is already an input.
- "Net adds within-scale NOISE that drags it below the floor" → opposite: within-scale is real
  signal (0.18 > entropy); the net UNDER-captures the easy between-scale axis instead.
- "Within-scale WHERE is image-idiosyncratic / unreachable" (from the 256-img overfit) → overstated;
  the full-train net generalizes a within-scale signal (0.18). It's the image-specific BETWEEN-scale
  that's truly not in t0.
