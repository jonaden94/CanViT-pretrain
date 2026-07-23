# What the Codex era tried (synthesis, 2026-06-11)

Source: archived ledgers (`archive/2026-05-06-agentic-experiments/RESULTS.md`,
`docs/RESULTS_COMPACT.md`, `HANDOFF_20260611.md`), read 2026-06-11.
ALL numbers are Codex-computed and UNTRUSTED until reproduced; treat as leads.
Its eval protocol is the "scene" protocol (squish-512 masks) in our evaluator.

## The setting every number lives in

ADE20K val (2000 images), frozen `canvitb16-add-vpe-...-2026-02-02` backbone
+ `probe-ade20k-40k-s512-c64-in21k`, scene 512, canvas 64, glimpse 128px,
**T=2**: fixed full-scene t0, choose ONE t1 viewpoint `(y, x, scale)`.
Headline = t1 mIoU. Budget unit = CanViT glimpse-forwards (t0 + candidates
both counted during training; deploy cost is 1 extra forward for all
methods below — the critic scores candidates WITHOUT forwarding them).

Reference points:

| Policy | t1 mIoU | Δ vs EG-C2F |
|---|---:|---:|
| C2F (fixed pyramid order) | 41.20 | −1.03 |
| **EG-C2F** (entropy-ordered pyramid) | **42.23** | 0 |
| Constant train-mean action (central crop, scale 0.55) | 42.15 | −0.08 |
| Best learned (critic argmax-grid) | 42.67 | +0.44, CI [+0.09, +0.76] |
| Best-of-17 oracle (val labels) | 45.58 | +3.35, CI [+2.85, +3.68] |

Two context numbers that calibrate everything: entropy guidance alone is
worth **+1.03 pp** over plain C2F for free, and a CONSTANT central crop is
only −0.08 pp below EG-C2F — so every learned method is fighting for a few
tenths of a pp above an almost-trivial prior, against +3.35 headroom that
nobody captured more than ~13% of.

## Method families, with budgets

1. **Reward critic, argmax over a fixed grid** — the winner. Train a critic
   to predict per-scene-z-normalized ΔCE(t0→t1) of candidate viewpoints
   (16 fresh random candidates/scene/step, all forwarded); deploy by scoring
   a 16×16-center × 4-scale grid (1024 candidates, no CanViT forwards) and
   taking argmax. Global-pooled scene state + VPE-encoded candidate.
   **42.672 (+0.441, paired CI [+0.089, +0.763])** at a HUGE budget:
   964k optimizer steps ≈ **16.4M glimpse-forwards**. An rbf_pool variant
   hit 42.658 at 294k steps. Mean selected scale 0.56; scales {0.5, 0.7}
   dominate.

2. **Strict 1M-glimpse budget arm** (the only efficiency-framed experiment):
   joint flow+critic co-trained on shared rollouts (136 forwards/step, 7352
   steps ≈ 1M). Result: critic-selected best-of-16 (flow or random
   candidates, indistinguishable) ≈ **42.44 ± 0.05** over 6 training seeds
   (+0.21). Single-sample flow stays BELOW EG-C2F. So at 1M forwards the
   recipe is: 16 random candidates + critic argmax = +0.2 pp.

3. **Decomposition findings** (spatialcond runs, 13.6M-forward variant):
   - Critic SELECTION is the dominant lever: critic-over-16-random beats
     one-random by **+0.8–1.0 pp**; everything else is second-order.
   - Learned flow PROPOSALS beat a single random draw (+0.5–0.74 pp) but
     under critic selection flow ≈ random candidates (slight +0.21 edge
     only late in training).
   - Scene-swap controls: the critic's scene-conditioning eventually does
     real work (+0.41 pp matched−swapped at step 20k); the flow's own
     scene-conditioning stays ≈ null throughout.

4. **Oracle distillation** (from saved best-of-17 datasets; train split has
   20210 images, oracle 50.86 vs EG-C2F 47.32 there). Cheap (160k–640k
   forwards, amortized to zero extra since targets are precomputed):
   - Soft spatial-router actor, expected-MSE: 42.509 (+0.278, CI crosses 0).
   - Point head (cross-attn + linear, action MSE): 42.483 (+0.252);
     overfits by 20k steps; imitation MSE does NOT track mIoU.
   - Conditional flow (NLL, deterministic base-mean deploy): 42.529
     (+0.298) but seed/checkpoint-sensitive (replications +0.14/+0.23);
     same-head MSE objective ties it (42.535) → NLL not required.
   - Flow analysis: the learned policy is a narrow continuous crop prior
     (x_std 0.009 vs oracle 0.32) with image-dependent vertical placement —
     NOT an oracle-distribution matcher.

5. **Direct-CE pathwise actor** (differentiable glimpse sampling through
   frozen CanViT, full-res CE loss): best snapshot 42.389 (+0.158, CI
   crosses 0) as a continuation from an EG-C2F-distill init; later
   snapshots worse. Fragile; mostly abandoned.

6. **EG-C2F flow distillation** (clean): never reached EG-C2F (best −0.03).
   NLL on a deterministic target drives volume collapse (their log-det
   lesson). Distilling a deterministic policy into a flow is a dead pattern.

7. **Negative/diagnostic**: two-image critic overfit gates don't transfer;
   mean-pooled conditioning structurally destroys WHERE (their biggest
   architecture lesson); MNIST-on-canvas CE gate failed from a setup bug
   (fixed-scene memorization), corrected version partially run — first
   full-scene glimpse has WHERE (~0.95 probe acc) but not WHAT (~chance);
   train-holdout checkpoint-selector control NEVER completed (all small
   deltas are validation-selected, i.e. mildly validation-tuned).

## What seems robust (still untrusted, but consistent across their runs)

- (EG-C2F > C2F > random orderings is PUBLISHED fact — paper Figure 4B,
  arXiv:2603.22570 — not a Codex finding; the +1.03 pp at T=2 just
  quantifies it at that budget.)
- Critic-based SELECTION of candidates is the one learned lever that
  repeatedly beats EG-C2F (+0.2 at 1M, +0.44 at 16M forwards).
- Learned PROPOSAL distributions add ≈ nothing once a critic selects from
  random candidates.
- Deterministic one-action heads distilled from oracle data land at
  +0.25–0.30 cheaply but plateau there; their imitation losses are
  uninformative about mIoU.
- Stochastic deploys need multi-draw CIs (their 42.77 "best" was the max of
  10 draws of a 42.59±0.13 distribution).

## Interestingly-different directions they never touched

- T > 2 for anything learned (all learned policies are one-shot t1 choosers;
  no sequential credit assignment, no recurrent policy state).
- Efficiency CURVES (mIoU vs budget) as the headline — only the strict-1M
  arm thought in budget terms at all, and only at T=2.
- Deploy-time candidate forwards (their critic never re-forwards candidates;
  best-of-K with REAL forwards at deploy would trade inference budget for
  quality — a different point on the efficiency curve, unexplored).
- Training the critic ON the probe's own uncertainty/entropy (EG-C2F's
  signal) rather than ΔCE regression.
- Any non-ADE20K environment (their MNIST gate was just starting).
