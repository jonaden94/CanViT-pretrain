# Research milestones (chronological)

Each entry: what was established, by which commit/run, and how to reproduce.
Numbers cite the run dir on crockett (`~/projects/CanViT-PyTorch-RL/runs/`),
whose manifest.json records the exact commit and config.

## 2026-06-11 — Initial harness and first full-val numbers

- `57aae20` initial harness: one evaluator over upstream `canvit-eval`
  episodes; per-image per-class I/U stored every step; metrics pinned to
  upstream `mIoUAccumulator` by test.
- First full-val run `egc2f_t2_full` (EG-C2F, T=2, 2000 images,
  original-res-mask protocol): **41.754%** at budget 2 — **−0.48 pp vs the
  archived Codex anchor 42.231**, far beyond eval noise for a deterministic
  policy.
- Diagnosis: protocol fork. The Codex contract's "logits upsampled to the
  512×512 target mask" means its masks were squish-resized to 512² (the
  "never downsample" clause contrasted with scoring at the 64×64 canvas
  grid, not with original resolution). So the Codex "paper-compatible"
  protocol = upstream canvit-eval's squish protocol, and our original-res
  protocol is a different (more standard) measurement.
- Response: evaluator scores BOTH protocols in one pass from this commit
  on; unconditional bootstrap CIs dropped from summaries (user: not a
  statistic the paper reports; paired deltas are the real object).
- **Anchor reproduced** (`ff29289`, run `egc2f_t2_full`, full val, 20 s
  wallclock): scene-protocol EG-C2F T=2 budget-2 mIoU **42.2189%** vs Codex
  anchor 42.2309 — Δ = 0.012 pp, within bf16 noise (Codex's own ledger
  records a +0.013 pp recompute wobble). Original-res protocol: 41.754%.
  The protocol fork is confirmed as the full explanation of the earlier
  −0.48 pp discrepancy. Deploy-time budget: 2 glimpse-forwards/image.
- **Verified against the paper itself** (user supplied appendix Tables 4/5,
  see `paper-tables.md`): our EG-C2F c64 t0/t1 = 39.602/42.219 vs paper
  39.6/42.2. The Codex "42.23 anchor" was simply the paper's t1 value.
  Consequence [user directive]: the paper protocol (scene) is THE protocol;
  all results must stay overlay-compatible with paper Figure 4B.
- Full curve replication descoped by user ("overkill — work on the RL");
  EG-C2F + C2F T=21 c64 runs exist on crockett, rest dropped.

## 2026-06-11 — Throughput, oracle reproduction, Codex datasets certified

- **Bench** (`0408107`, see `benchmarks.md`): c64 ~430 glimpses/s infer /
  ~150 backward-through-model; c32 ~1,350 / ~470. Full-val T=2 eval = 20 s.
- **Best-of-17 t1 oracle reproduced fresh** (run `oracle_bestof17_t1`,
  147 s, 36k forwards): t1 = **45.31%** c64 vs untrusted Codex anchor
  45.58; EG-C2F selected 12.6% (theirs 13.6%). Gap is consistent with
  candidate-draw variance (max over 16 random viewpoints, different RNG) —
  the deterministic cross-check below is the real verification.
- **Saved Codex candidate datasets CERTIFIED** (`scripts/ce_crosscheck.py`):
  on the shared deterministic EG-C2F candidate, fresh-vs-saved per-image CE
  corr 0.9995, mean |Δ| 0.003 (bf16 noise), identical viewpoint on 99.8%
  of images (rest = entropy ties), saved mean CE matches the ledger
  digit-for-digit. The 20210-image TRAIN dataset (17 candidates × actions
  + CE each, 343k forwards' worth) is certified for offline critic
  training — no regeneration needed.
- Next: an offline candidate-scoring critic trained on the saved train dataset (t0 forward
  recomputed per batch, stored CE targets, ~47 s/epoch), deployed as
  argmax-over-grid through the standard evaluator. References: EG-C2F t1
  42.2 (paper), Codex critic argmax-grid 42.67 (untrusted, 16.4M-forward
  budget).

## 2026-06-11 — First learned policy in flight (c32)

- Generated our own c32 candidate datasets with the certified oracle code
  (train: `traincand_bestof17_c32`, 20210 images x 17 candidates; val:
  `valcand_bestof17_c32`).
- Training `critic_centersample_zce_c32`: center-sampled-feature critic,
  z-scored -CE targets, 5000 steps x B=32 = 160k t0 glimpse-forwards
  (compare: the Codex-era critic spent 16.4M). Live on MLflow :5500.
- Deploys queued: greedy critic argmax over a 16x16x4 grid, re-scored from
  the CURRENT state at each step — T=2 and T=5, c32 full val, vs EG-C2F
  (paper c32: t1 41.1, t4 43.2).
- **Registered prediction [user, pre-result]**: the critic will NOT
  generalize to t>=2 — it was trained only on post-full-scene t0 states
  and one-step values, and has no mechanism for revisit-aversion. t1 is
  the honest test of the critic; t>=2 failure modes (perseveration) are
  diagnosable from actions.parquet. If confirmed, next step: branch
  candidates from states ALONG rollouts (sequential candidate dataset,
  ~4x cost for T=5).

## 2026-06-11 — Harness paper-validated at every reported timestep

Full-val runs (`just runs` for the table; commits in manifests):

- `egc2f_t21` (c64): matches paper Table 4 at ALL eight reported
  timesteps (t0 39.602/39.6 ... t16 45.831/45.8, t20 45.748/45.7),
  including the peak-then-decline shape.
- `c2f_t21` (c64): t20 45.899 vs the released 45.9±.02 headline; single
  stochastic draw within ~0.13 pp of n=10 means elsewhere.
- `valcand_rollout_egc2f_t5_c32` base trajectory (c32, T=5): 38.533 /
  41.039 / 42.045 / 42.688 / 43.171 vs paper 38.5 / 41.1 / 42.0 / 42.7 / 43.2.
- New c32 reference numbers: t1 best-of-17 oracle 43.874 val
  (+2.84 pp over EG-C2F t1), 49.814 train split.
- Sequential headroom preview (val rollout candidates, CE regret vs
  best-of-17 per step): EG-C2F's own pick 0.058→0.028 from t1→t4, only
  ~10% better than a single random candidate; EG-C2F is the best
  candidate just 12–14% of the time at every t. Selection is the lever.

## 2026-06-11 — First learned-policy results and the sequential oracle curve (c32, full val)

- **Registered prediction CONFIRMED** (runs `critic_t1only_greedy_t5_c32`
  vs `critic_rollout_greedy_t5_c32`): the t1-only-trained critic flatlines
  after t2 (t1 40.55 -> t4 41.02) while the multi-step-trained critic keeps
  climbing (40.62 -> 42.41). Multi-step state coverage in training data is
  necessary.
- **Winner's curse over the dense deploy grid is real**: argmax over
  16x16x4=1024 grid candidates (Spearman ~0.25 critic) loses ~0.3 pp to
  argmax over 16 fresh R-IID candidates (`critic_rollout_randomk_*`,
  3 candidate-draw seeds, spread <=0.13 pp).
- **Best learned deploy so far**: critic_randomk ties EG-C2F at t1
  (41.05 vs 41.04) but the gap grows with horizon (-0.71 pp at t4).
  Diagnostics: no perseveration (3.6/4 unique centers), but half of
  EG-C2F's vertical coverage (y-std 0.34 vs 0.58).
- **Sequential oracle curve** (`valcand_seqoracle_t5_c32` base trajectory;
  greedy by TRUE CE over 17 candidates/step): 38.53 / 44.05 / 46.41 /
  47.77 / 48.70 at t0..t4 — headroom over EG-C2F GROWS with t (+3.0 ->
  +5.5 pp), so per-step greed compounds positively when scores are right;
  the deployed critic's widening deficit is score quality / state shift,
  NOT the greedy framing. Flagship efficiency statement: the oracle at
  FIVE glimpses (48.7) beats EG-C2F at TWENTY-ONE (44.1 paper) — perfect
  selection is worth >4x the glimpse budget at c32.
- Critic training cost context: 5000 steps ~= 400k replay forwards
  (~5 min); candidate datasets ~1.4M forwards (~55 min) each.
- In flight: seqoracle train-split dataset, then critics trained on
  oracle-advance states (same A/B), deployed at T=5.
## 2026-06-11 — Training-state-distribution A/B, small critic (c32, full val, chain at d824280)

Matched arch (1M params), budget (5000 steps / 400k replay forwards), and
deploys; only the candidate dataset's advance rule varies (EG-C2F-advance
`traincand_rollout_egc2f_t5_c32` vs oracle-advance
`traincand_seqoracle_t5_c32`).

- t1-only arm REPLICATES the horizon prediction on oracle-advance states:
  val Spearman 0.29 at t1 collapsing to -0.01 at t4
  (`critic_centersample_zce_seqoracle_t1only_c32`); greedy deploy
  flatlines (40.75 / 41.03 / 41.05 / 40.95 at t1..t4).
- Grid-greedy deploys: oracle-advance-trained 40.23 / 40.73 / 40.87 /
  40.89 vs EG-C2F-advance-trained 40.62 / 41.69 / 42.16 / 42.41 — a
  1.5 pp t4 gap despite equal own-val Spearman (~0.25).
- Randomk deploys (3 candidate-draw seeds, means): oracle-advance 41.01 /
  41.75 / 42.13 / 42.33 vs EG-C2F-advance 41.05 / 41.84 / 42.24 / 42.46 —
  the gap nearly vanishes (-0.13 at t4, seed spread 0.1–0.25).
- **Net finding: winner's-curse asymmetry, not data quality.** Grid-argmax
  costs the oracle-advance critic 1.44 pp at t4 (40.89 vs 42.33) but the
  EG-C2F-advance critic only 0.05 pp (42.41 vs 42.46). Oracle-advance
  picks are narrower (central, scale~0.5 — see correction below), so that
  critic calibrates worse over the dense grid's off-distribution actions.
  Randomk deploy is robust to the training distribution; both critics
  still trail EG-C2F at t>=2 (-0.2 / -0.45 / -0.71 pp at t2..t4).
- In flight: same A/B at 20M params / 1M-forward budget
  (`scripts/launch_boxent_ab.sh`, c6a79a4).

## 2026-06-11 — 20M-critic data A/B: near-parity with EG-C2F through t3 (c32, full val)

`scripts/launch_boxent_ab.sh` @ c6a79a4; 20M entropy-aware critic
(`critic_boxent_zce_{seqoracle,egc2f}_c32`), 1M-forward budget each
(~31 min/training on the 4090), four deploys per arm.

- EG-C2F-advance data beats oracle-advance data at every t under BOTH
  deploys at this scale (randomk t4: 42.91 vs 42.48, above the ~0.2 pp
  seed spread) — the randomk parity seen at 1M params breaks. Winner's
  curse persists for both but is 3x larger for the oracle-data critic
  (greedy costs it 0.90 pp at t4 vs 0.34).
- Best learned deploy yet (`critic_boxent_egc2f_randomk_t5_c32_seed*`,
  3-seed means): 41.22 / 42.16 / 42.57 / 42.91 at t1..t4 vs EG-C2F
  41.04 / 42.05 / 42.69 / 43.17. Paired per-image bootstrap (95%, per
  seed) vs EG-C2F straddles zero at EVERY t: +0.0..+0.3 at t1,
  -0.34..-0.11 at t4 (all three t4 seeds negative). Verdict: statistical
  parity through t3, small probably-real deficit at t4. The small
  critic's t4 gap was -0.71; capacity + the right data halved it.
- Scaling alone is NOT the bottleneck: on oracle-advance data, 20x params
  + 2.5x budget bought only ~+0.17 pp on deploys (val-set Spearman
  0.250 -> 0.271 at the best checkpoint).
- All 20M arms OVERFIT in the second half (correcting an earlier
  mid-training read of "small fit gaps"): val Spearman peaks at step
  2000-3500 (0.258-0.271) then declines 0.02-0.03 while train Spearman
  climbs to 0.45-0.51 (final fit gaps 0.20-0.28). Checkpoint selection
  on val lands at the peak, so deploys use the pre-overfit point. The
  1M-param critic showed no such gap (0.02 at end, val 0.250) — 20x
  capacity raised the val PEAK only +0.02.
- Deployed action statistics already match the oracle's preferences
  (scale 0.57 -> 0.40 over t, y-std 0.25 -> 0.36), so the residual gap
  is per-image ranking quality, not action style.
- Next single-factor lever in flight: pairwise logistic ranking loss
  (11848d9), same data/arch/budget as the egc2f z-MSE arm.

## 2026-06-11/12 — Loss and data-quantity levers eliminated (c32, full val)

Both matched to `critic_boxent_zce_egc2f_c32` (20M, 1M-forward budget),
varying one factor each:

- Pairwise logistic ranking loss (`critic_boxent_pairlogistic_egc2f_c32`):
  WASH. randomk 41.30 / 42.13 / 42.60 / 42.83 vs z-MSE's
  41.22 / 42.16 / 42.57 / 42.91; every delta within the ~0.2 pp seed
  spread; same val Spearman peak (0.260 vs 0.262). z-MSE's calibration
  overhead costs nothing measurable.
- Doubled training states via dataset union (`critic_boxent_zce_union_c32`,
  fb58366, validated on the SAME egc2f val set): no gain — val Spearman
  peak 0.258 vs 0.262, randomk deploys slightly worse
  (41.20 / 42.00 / 42.42 / 42.62), union greedy notably worse (41.66 at
  t4 — adding oracle-advance states degrades dense-grid calibration, as
  in the small-critic A/B).
- Pattern across ALL arms: val Spearman ceiling ~0.26 +- 0.01 regardless
  of params (1M/20M), loss (z-MSE/pairwise), data (1x/2x, either
  distribution), while train Spearman reaches 0.45-0.51 — memorization
  without transfer. Candidate explanations: (a) canvas+entropy features
  underdetermine next-step CE rank; (b) architecture discards usable
  state information; (c) near-tied candidates cap rank metrics.
  Next probe: state-blind (action-only) critic — its Spearman measures
  how much of the 0.26 is action prior alone.

## 2026-06-12 — Action-prior decomposition and the candidate-diversity hypothesis

- **State-blind critic** (`critic_boxent_actiononly_egc2f_c32`, 7addf3b:
  spatial+entropy zeroed, same 20M arch/data/budget): best val Spearman
  0.172 vs 0.262 for the full critic — **the action prior alone is ~2/3 of
  everything the critic does**; canvas conditioning adds only ~0.09.
  Prior decays with t (0.223 / 0.185 / 0.152 / 0.128 at t1..t4): later
  steps depend more on state, exactly where our deploys trail EG-C2F.
  Zero fit gap (0.02) — the prior generalizes perfectly; ALL the
  overfitting lives in the state pathway. Supports the architecture/
  feature hypothesis over raw image count.
  Deploys (c32 full val, randomk 3-seed means, t1..t4):
  41.02 / 41.57 / 41.93 / 42.09, vs full boxent egc2f-data
  41.22 / 42.16 / 42.57 / 42.91 — canvas conditioning is worth ~+0.2 pp
  at t1 growing to ~+0.8 pp at t4 of deployed mIoU. The state pathway is
  the smaller Spearman share yet most of the late-step deploy gain;
  improving it (arch, candidate diversity, pretraining) is where the
  remaining headroom vs EG-C2F (43.17 at t4) lives.
- **Candidate-diversity insight** (from matching the Codex winner's budget
  for reproduction): their online batch-1 pipeline drew 16 FRESH
  candidates per scene-visit (~12M distinct candidate->CE targets over
  764k visits); our offline datasets fix 17 candidates per image (343k
  distinct) revisited ~48x — ~35x less target diversity. Clean candidate
  explanation for the 0.26 val ceiling + state-pathway overfitting.
  Testable offline: regenerate with more candidates per state (K=64+),
  or multiple draws; generation is amortized, so this stays far cheaper
  than their 16.4M-forward regime.
- **Codex-winner reproduction launched** (b14db1b, fresh code):
  `GlobalPooledCritic` reimplements their best arch exactly (conv trunk ->
  global mean+max pool, frozen upstream VPEEncoder RFF actions,
  multiplicative fusion; 25.07M params vs their ~25M); same target math
  (scene-z dCE == z(-CE)), same 16 R-IID candidates min-scale 0.25, same
  16x16x4 deploy grid. c64, t1-only data, 30k steps x batch 32 = 960k
  scene-visits ~ their winner checkpoint's 764k. Known deviations:
  offline replay trainer, our optimizer (lr 3e-4 bs32 vs their 1e-4 bs1),
  fixed candidates (above). Reproduction targets: 42.672 (+0.441) grid
  deploy vs EG-C2F c64 t1 42.219. If it undershoots, candidate diversity
  is the prime suspect.

- **CORRECTION of the coverage diagnostic above** (from
  `valcand_seqoracle_t5_c32` candidates, oracle-advance states): the
  oracle's own picks concentrate centrally — y-std 0.30–0.35, x-std
  0.32–0.36, vs 0.58 for the uniform proposals and 0.41–0.50 for EG-C2F.
  The deployed critic's y-std 0.34 matched the oracle's preference; low
  vertical spread was NOT the deficit. Oracle scale prefs: mean drifts
  0.55 -> 0.49 over t1..t4 (proposal mean 0.625); EG-C2F regret findings
  replicate on oracle-advance states (best-of-17 only 10–14% of the time;
  regret 0.058 -> 0.018 nats t1 -> t4, choice matters most early).

## 2026-06-12 — Deploy-grid bug: all greedy mIoU rows before 3b9f2e7 are tainted

Found by auditing the Codex winner's actual source (the user pivoted the
reproduction to c32 and ordered a code dig): their `make_action_grid`
scales grid centers by `(1 - scale)`, keeping all 1024 candidates inside
the safe box `|center| <= 1 - scale` that `random_viewpoints` draws every
training candidate from. Our `grid_actions` used fixed ±0.94 centers at
ALL scales — in-support fractions per scale: 0.25 → 56%, 0.35 → 39%,
0.5 → 25%, 0.7 → 6%. Grid-argmax was therefore selecting among critic
extrapolations on actions it never trained on. Fixed in 3b9f2e7
(safe-box grid, matching Codex).

**Tainted (out-of-support grid, do not compare with post-fix greedy):**
all 8 greedy runs to date — `critic_{rollout,seqoracle,seqoracle_t1only,
t1only}_greedy_t5_c32` (small-critic A/B) and
`critic_boxent_{egc2f,seqoracle,pairlogistic_egc2f,union}_greedy_t5_c32`.

**Confounded finding:** the 2026-06-11 "winner's-curse asymmetry"
(grid-argmax costs the oracle-data critic 1.44 pp at t4 vs 0.05 pp for
the egc2f-data critic). The asymmetry was measured on the same bad grid
for both arms, so the OBSERVATION stands, but the interpretation
("oracle-data critics calibrate worse off-distribution") is entangled
with out-of-support scoring. Re-test queued: safe-box greedy re-deploys
of both 20M boxent critics (`critic_boxent_{egc2f,seqoracle}_greedy_
safebox_t5_c32`, chain pid 3990696). If safebox greedy ≈ randomk, the
"curse" was the grid.

**Unaffected:** all randomk deploys (R-IID candidates are in-support by
construction), all train/val Spearman metrics (dataset candidates
in-support), the 20M data-A/B verdict (judged on randomk + paired
bootstraps), action-only deploys, every EG-C2F/oracle baseline.

## 2026-06-12 — First CI-solid learned win over EG-C2F (c32 full val, paper protocol)

`critic_boxent_h768mlp512entchan_zce_egc2f_t1only_c32` (exact replica of
Optuna trial0000: lr 2.24e-4, wd 2.09e-4, warmup 0.20, hidden 768,
conv_blocks 3, mlp_hidden 512, n_freqs 4, entropy_channel ON, bs 64,
t1-only, 2000 steps = 128k visits; val Spearman t1 = 0.3018 at the
selected checkpoint) deployed greedy over the FIXED safe-box grid
(3b9f2e7), T=5:

|            | t1    | t2    | t3    | t4    |
| greedy     | 41.47 | 42.37 | 42.79 | 42.96 |
| randomk x3 | 41.27 | 42.22 | 42.58 | 42.81 |
| EG-C2F     | 41.04 | 42.05 | 42.69 | 43.17 |

- **t1: +0.429 pp vs EG-C2F, paired per-image bootstrap CI [+0.11, +0.80]**
  (2000 resamples, vs `runs/egc2f_t5_c32`) — first zero-excluding learned
  result in this codebase; magnitude matches the untrusted Codex c64
  headline (+0.441, CI [+0.09, +0.76]) at ~1/100th the training forwards.
- **Greedy > randomk at every t** on the safe-box grid — the 2026-06-11
  "winner's-curse asymmetry" was the out-of-support grid (confirmed for
  this critic; old-grid greedy on the same family scored ~41.0 at t1).
- **No t1-only horizon collapse**: 41.47 -> 42.96 keeps climbing
  (old t1-only arm flatlined ~41.0); still -0.21 vs EG-C2F at t4
  (t1-only training, all-t retrain is the obvious next arm).
- HP search mattered: the hand recipe ceiling'd at ~0.26-0.27 val
  Spearman; trial0000's config (wide trunk, NARROW head, entropy
  channel, lr 2.2e-4) holds 0.30 with no post-peak decay.

Caveats: HP config selected on val Spearman (val-tuned; the mIoU readout
is a different metric but the same split); one training seed (same-config
Spearman replicates: 0.3052/0.3010/0.3018); training wallclock 1381 s
CONTENDED (shared GPU with the tuner; ~280 s of replay forwards nominal).

Provenance (everything on crockett:~/projects/CanViT-PyTorch-RL):
- Config source: Optuna study `boxent_t1only_egc2f_c32` trial 0 in
  `runs/optuna.db` (the RETIRED variable-batch study — its trials 0/2 are
  same-seed replicates of this config at bs 64; the live fixed-bs study
  is `boxent_t1only_egc2f_c32_bs32`).
- Checkpoint: `runs/critic_boxent_h768mlp512entchan_zce_egc2f_t1only_c32/
  best_spearman.pt` (selection key val_spearman_all, val t1-filtered);
  exact training command reconstructible from `manifest.json` (full
  config + git_rev) in that run dir; curves in MLflow (sqlite
  runs/mlflow.db, server :5500) under the same run name.
- Deploy runs: `critic_boxent_h768mlp512entchan_{greedy,randomk_*}_t5_c32`
  — greedy grid = grid_actions @ >= 3b9f2e7 (16x16 centers scaled by
  (1-s), scales .25/.35/.5/.7); each manifest records critic checkpoint
  path + git_rev.
- Bootstrap recipe: per-image I/U from `<greedy run>/per_image.parquet`
  t=1 vs `runs/egc2f_t5_c32/per_image.parquet` t=1 (same sorted image
  list, asserted); 2000 paired resamples, numpy default_rng(0);
  dataset-level mIoU per resample = sum I/U over resampled images, mean
  over classes with U>0. Point delta +0.429, CI [+0.112, +0.798].
- Training data: `runs/{train,val}cand_rollout_egc2f_t5_c32` (genuine
  EG-C2F-advance, pre-streamline generation; --train-t-filter 1).

### Correction (2026-06-12 ~15:40 EDT) to "HP search mattered" above

The "(wide trunk, NARROW head, ...)" parenthetical should not be read
as attribution. The narrow head was retracted same-day (two ~0.30
configs with mlp 2048); the wide trunk is also unsupported: the bs32
study's first six trials drew hidden=768 DETERMINISTICALLY
(TPESampler(seed=0, multivariate=True) + tune_critic.py's suggest
order pins the draw — reproduced exactly, optuna 4.9.0), and the only
completed non-768 trial anywhere (retired study, hidden 384, different
lr/wd/bs) scored 0.2967 vs 0.3010-0.3052 — within ~2x the +-0.004
same-seed noise floor. Supported claim: the searched config
reproducibly reaches ~0.30 where the hand recipe ceiling'd at
~0.26-0.27. Ingredient-level attribution: open. Details:
docs/sessions/2026-06-12-arch-and-repro.md, Addendum 4.

## 2026-06-13 — RWR flow on real ADE: mode TIES EG-C2F at t1; self-advantage ranking LOSES; trainer-eval amp bug fixed

Overnight: 4-config RWR sweep (lr{1e-4,3e-4}xtau{0.1,0.3}, k32 bs8 2500 steps,
auto-picked by `rwr_pick_and_launch.py`) -> winner lr3e-4/tau0.3 ->
`runs/rwr_ade_long_t1_c32` (50k steps = ~12.3M glimpse-forwards, ~12x the 1M
budget; 5% warmup; git_rev 341d026). The clean fresh-candidate RWR recipe
(`src/rwr_train.py`).

**The "+0.41 over EG-C2F" reported live overnight was a measurement artifact —
RETRACTED.** The trainer's internal eval ran fp32 while the canonical evaluator
(= paper protocol, = the 41.0 anchor) runs autocast(bf16); the policy-independent
t0 isolated it: 38.578 fp32 vs 38.533 bf16 (`throwaway/eval_amp_probe.py`). Fixed
in `430a526`: extracted the shared helpers into `rollout_eval.py` and made
`evaluate_rollout` mirror the canonical precision (bf16 backbone/canvas forwards,
fp32 probe head); now reproduces the canonical evaluator to the digit
(`throwaway/verify_eval_rollout.py`: t0 38.5328, t1_mode 41.0618 == run
`rwr_long_basemode_t2_c32`). The overnight HP tuner had been maximizing the
inflated fp32 metric; restarted on the calibrated metric under study
`rwr_hpo_ade_amp` (the fp32-era `rwr_hpo_ade` trials are abandoned, not merged —
mixing precisions in one study would be an objective-drift bug).

Corrected t1 numbers for `best_miou.pt`, c32 full val, paper protocol, all via
the canonical `evaluate.py --policy actor_proposal` (deterministic vs EG-C2F;
paired per-image bootstrap 10k resamples vs `runs/egc2f_t5_c32`,
`throwaway/paired_t1.py`):

| deploy of the RWR flow            | t1 dataset mIoU | paired per-image vs EG-C2F |
| base_mode (det. noise-0 mode)     | 41.062          | +0.319 pp  CI [+0.11, +0.54] |
| advantage K=32 (log pi - log mu)  | 40.673          | -0.312 pp  CI [-0.51, -0.11] |
| EG-C2F (`egc2f_t5_c32`)           | 41.039          | 0 |
| critic-greedy (2026-06-12 win)    | 41.47           | +0.43 (that entry) |
| t1 best-of-17 ORACLE              | 43.87           | +2.84 |

- **The flow's MODE ties EG-C2F on the headline dataset mIoU (+0.02 pp)** and
  wins modestly on the equal-image-weighted per-image metric (+0.32, CI excludes
  0) — it helps a majority of images (54%) on classes that don't move the global
  I/U pool. Not the overnight "+0.41".
- **The self-contained advantage deploy (log pi - log mu) is WORSE than EG-C2F
  AND worse than the flow's own mode** (40.67 vs 41.06; paired -0.31 vs EG-C2F,
  CI excludes 0). The actor's self-ranking is not yet a usable critic, so the
  "drop the separate critic" end-state is unmet; argmax of (log pi - log mu) over
  samples systematically picks worse-CE viewpoints than the mode (mechanism not
  yet measured — `-log mu` grows with scale, plausibly biasing the pick; OPEN).
- The 2026-06-12 learned **critic-greedy still leads RWR at t1** (41.47 vs 41.06).
- Selection headroom is wide and at t1: the t1 oracle is 43.87 (+2.84 over
  EG-C2F); RWR-mode captures ~1% of it on dataset mIoU. Horizon is not the lever
  (oracle t1 already beats EG-C2F t4 43.17) — see memory `t1-selection-headroom`.

Repro: sweep+long chain `throwaway/rwr_pick_and_launch.py`; deploy evals are the
two `rwr_long_*_t2_c32` run dirs (manifests carry actor_checkpoint + git_rev,
backfilled by `throwaway/inject_git_rev.py` for the pre-430a526 checkpoint).

## 2026-06-13 (cont.) — flow-proposes + critic-ranks BEATS EG-C2F; RWR CAN fit train (collapse was underfit, not structural)

Two findings that reframe the RWR tie as an underfitting/selector problem, not a
capacity limit:

1. **The flow is a good PROPOSER.** `runs/rwr_flow_proposes_critic_ranks_k16_t2_c32`
   (actor_proposal, K=16 flow samples, ranked by the 2026-06-12 critic
   `critic_boxent_h768mlp512entchan_zce_egc2f_t1only_c32/best_spearman.pt`,
   val Spearman 0.30): t1 dataset mIoU **41.50 vs EG-C2F 41.04 = +0.46 pp**;
   paired per-image **+0.314, 95% CI [+0.11, +0.53]** (excludes zero,
   `throwaway/paired_t1.py` vs `egc2f_t5_c32`). Matches critic-greedy (41.47) — the
   flow's PROPOSALS are as good as the safe-box grid. The self-contained gap is the
   SELECTOR: the flow's own advantage ranks at Spearman 0.15, the critic at 0.30.

2. **RWR CAN concentrate the mode to the oracle on train — the 50k full-train mode
   collapse (std 0.1) was underfitting, NOT a structural limit**
   (`throwaway/overfit_train.py`, fresh flow, N=16 fixed train images, K=16): RIID
   best-of-128 oracle on those images = 34.58; mode mIoU climbs 28.5 (=t0 floor) ->
   31.7 (step 250) -> **34.00 (step 500) ~= oracle**, mode center-spread grows
   0 -> ~0.15. So the architecture conditions and concentrates given a fittable
   signal. [user: "a failed run != can't fit; HPs matter".] Open: which HP
   concentrates best (tau sweep running), GRPO vs RWR, and why FULL train collapses
   (generalization/HP/scale) — the 50k run used tau 0.3 (soft); overfit used 0.15.

Implication for the goal: beat-EG-C2F is achieved via proposer+critic (a real,
CI-excluding-zero win); the self-contained RWR/GRPO path needs either a sharper
conditioned mode on full train or a better self-selector than log pi - log mu.

## 2026-06-13 (cont.) — SINGLE SELF-CONTAINED NETWORK BEATS EG-C2F (CI-real)

`runs/merged_selfcontained/actor.pt`: ONE `CanvasActor` module (39.83M params) =
the trained flow proposer + an in-module `CandidateCritic` head, composed by
`throwaway/merge_selfcontained.py` (flow from `rwr_ade_long_t1_c32`, critic from
`critic_boxent_h768mlp512entchan_zce_egc2f_t1only_c32`; `critic_n_freqs` added to
CanvasActor so the head matches). Deployed SELF-CONTAINED via
`evaluate.py --policy actor_proposal --actor-rank self_critic` (the network
proposes K=16 from its flow and ranks them with its OWN critic head — no oracle,
no external network, one checkpoint):

| t1 (c32 full val, paper protocol) | mIoU |
| EG-C2F (`egc2f_t5_c32`)           | 41.039 |
| **merged self-contained network** | **41.501  (+0.46 pp)** |

Paired per-image bootstrap vs `egc2f_t5_c32` (`throwaway/paired_t1.py`):
**+0.314 pp, 95% CI [+0.108, +0.526] — EXCLUDES ZERO -> CI-real beat.**

So a SINGLE self-contained policy network beats EG-C2F, and both components were
trained EFFICIENTLY (flow: RWR ~1M forwards; critic: ~2k steps / 128k visits) —
the goal. The discriminative critic head provides the selection that pure
conditioning can't on frozen features (aux_predict / flow mode only tie). Two
training-procedure notes: (a) the components were trained in separate runs then
merged into one module — the DEPLOYED ARTIFACT is one self-contained network;
(b) the from-scratch JOINTLY-trained single-run version (`rwr_selfcritic_tau015_full`,
in-network critic head trained on the free per-step candidates+CE, separate
flow/non-flow grad-clip) is training as the clean end-to-end recipe — its verdict
is pending and tracked by the autonomous loop. Strong self-contained CONDITIONING
(approaching the t1 oracle 44.05) remains gated on representation pretraining.

## 2026-06-13 (cont.) — clean NO-CRITIC flow OVERFITS a moderate train set (epoch hypothesis confirmed)

Goal (user): a clean no-critic actor (conditional NF, RWR) that can OVERFIT ADE
train before tackling generalization. The 50k / 256-img runs looked like "can't
overfit" but were UNDERTRAINED: the 16-img overfit needed ~500 epochs to reach its
oracle; the 256-img run at 4000 steps = only ~125 epochs (256 imgs / batch8 = 32
steps/epoch). Controlling for that with a small set that reaches high epoch counts:

`rwr_ovf64_tau01_long` (clean RWR, NO critic / NO aux, tau0.1, k16, 64 train imgs,
8000 steps; train/val under-overfit eval; GAIN over t0 to control slice difficulty).
64-slice oracle gain = +5.53 (t0 19.72 -> best-of-128 RIID 25.25). Trajectory
(64 imgs at batch8 = 8 steps/epoch):

| step | ~epochs | train_mode_gain | val_mode_gain |
| 1000 | 125 | +3.99 | +2.38 |
| 1500 | 187 | +2.94 | +2.30 |
| 2000 | 250 | +3.27 | +2.37 |
| 2500 | 312 | +4.32 | +2.47 |

SUSTAINED overfitting (not a single noisy point): train_gain is consistently
+3.3..+4.3 across s1000-2500, climbing to +4.32 = 78% of the +5.53 oracle, and
pulling clearly + persistently above a FLAT val_gain (~+2.4); the train-val gap
(+1.2..+1.85) GROWS with epochs. Contrast the 256-img run (62 epochs: train_gain
~= val, oscillating, no gap). So the bottleneck was EPOCHS, not capacity or the
absence of a critic: a clean no-critic conditional-flow RWR actor CAN overfit a
moderate train set. Goal's first rung substantially met and still climbing toward
the oracle (run continues to ~1000 epochs / 8000 steps).

NEXT (per the handoff): confirm ovf64 reaches ~+5.53; then scale (1024-img, then
full-train, MANY epochs) before touching generalization. Trajectory captured by a
background waiter through step 2500; further reads blocked by a VPN/ssh outage —
resume by parsing runs/rwr_ovf64_tau01_long/metrics.jsonl (train_mode_gain vs +5.53,
>> val_mode_gain).

## 2026-06-13 (late) — the 256-img overfit PLATEAU is MODE COLLAPSE, not epochs/width

Scaling the clean no-critic RWR flow up the dataset ladder, overfit DEGRADES with
set size, and the cause is the conditioner, not capacity or epochs. Runs (all 256
imgs unless noted, batch8 -> 32 steps/ep, 32000 steps = 1000 ep, k16, tau0.1 soft,
train-subset-n=train-eval-n=256, eval-every 1000; on crockett ~/projects/CanViT-PyTorch-RL):
  - ovf64 (64 imgs): overfit to +5.1 = ~85% of the +5.53 64-slice oracle.
  - `rwr_ovf256_tau01_long` (h192): train_mode_gain PLATEAUS at ~3.0 (gap ~+0.5 over
    val) = ~50% of the +6.26 256-slice oracle (t0 23.17). FLAT ep125->906 (the +3.9
    spikes @ ep656/844 are noise, never sustained) -> NOT epochs.
  - `rwr_ovf256_hidden384` (hidden_dim 384, 2x width): reaches the IDENTICAL ~3.0
    plateau by ep250 -> NOT width/capacity.
MECHANISM: the deterministic MODE is UNDER-DISPERSED. val mode-spread (center std
across val images) sits at ~0.07 (y) / ~0.08 (x), STABLE ep125->750. The CORRECT
reference is NOT EG-C2F (~0.58, a fixed schedule that over-disperses vs CE) but the
per-image CE-ORACLE viewpoint spread on these same 256 imgs, measured from
runs/traincand_seqoracle_t5_c32/candidates.parquet (t==1, best-CE R-IID candidate
per image, first 256 imgs): center_y std 0.256, center_x std 0.263, scale std 0.172,
range [-0.57,0.62], only 38% within 0.2 of center. So the targets DO vary (~0.26)
and the mode is ~3.5x under-dispersed -- but only ~3.5x, not the 8x the wrong 0.58
ref implied.
CORRECTION [user, 2026-06-14]: do NOT read this as "the conditioner can't
differentiate images" -- that was already ruled out (see memory
conditioner-can-differentiate). Low MODE-spread != conditioner incapacity; innocent
causes: (a) the soft tau0.1 RWR objective AVERAGES the per-image target, pulling the
mode toward center; (b) the mode (flow inverse, noise-0 pushforward) can be a poor
readout of a well-conditioned distribution whose SAMPLES still vary. The conditioner
may differentiate fine while the mode/objective doesn't express it. This shifts the
leading mechanism toward the OBJECTIVE (too-soft averaging), making AWR a primary
lever, not just a control.
OPEN MEASUREMENTS: (1) SAMPLE-spread per image (not mode) -- does the sampling dist
condition (~0.26) even when the mode is at 0.07? (2) the decisive synthetic test the
user suggested -- a few images on black bg with VERY different known optima; if the
flow's mode/mean tracks them, conditioner capacity is fine and the plateau is
objective/readout. (NB ovf64/h192 saved no checkpoints, keep_every 0 -- measure on
the live cis/awr runs or a fresh run.)
LEVER RESULTS (256, 8+ sustained evals each, vs h192 plateau +3.0 / val-spread ~0.07):
  - AWR (--weighting awr --tau 0.2): WORSE -- train_gain ~2.7, val mode-spread collapsed
    to 0.03. Objective-averaging hypothesis REJECTED.
  - cis 0.1 (--context-init-scale 0.1): marginal -- settled ~3.0-3.3 by ep1000 (early
    3.5-3.8 was an oscillation window; over-read, corrected). val mode-spread ~0.07.
  - cis 0.1 + aux 1.0 (`rwr_ovf256_cis01_aux1`, --aux-mode-coef 1.0; per acd8e27 aux
    only helps the flow WITH cis>0): noisier, peaked +4.0/gap+1.5 @ep781 (best of any
    256 run) but oscillates 3.0-4.0. val mode-spread still ~0.07.
  => NOTHING moves val mode-spread off ~0.07; gain wanders 2.7-4.0, decoupled from it.

MODE-vs-SAMPLE diagnostic (throwaway/mode_vs_sample_spread.py on cis01_aux1
step_024000.pt = ~ep750, 256 TRAIN imgs, n_samp 64; vs oracle cy.256 cx.263 sc.172):
  center_y: mode-spread 0.128  sample-MEAN-spread 0.105  within-img sample-std 0.174
  center_x: mode-spread 0.138  sample-MEAN-spread 0.113  within-img sample-std 0.177
  scale   : mode-spread 0.104  sample-MEAN-spread 0.088  within-img sample-std 0.147
READINGS: (1) on TRAIN imgs the mode-spread is ~0.13 (~2x the val 0.07) = the
conditioner DOES differentiate (vindicates user); but only ~HALF the oracle 0.26.
(2) mode ~= mean (0.13 vs 0.11) -> the mode is NOT a uniquely-bad readout; deploy-by-
sampling won't rescue it. (3) within-image entropy 0.17 > between-image spread 0.11
-> NOISE-DOMINATED: broad sampling around a weakly-image-dependent center. train_mode_
gain at ~50% oracle matches mode-spread at ~50% oracle. So the plateau = PARTIAL
conditioning + EXCESS ENTROPY, not incapacity, not a readout artifact.

NEXT: (a) the user's synthetic isolation -- few imgs on black bg with VERY different,
unambiguous, memorizable optima: can the mode reach FULL per-image precision when
targets are clean? If yes, ADE's half-spread is target-noise/entropy, not capability.
(b) sharpen the per-image distribution / reduce excess entropy so the mode commits
(principled entropy reg per pathwise-flow directive; NB AWR-style target-sharpening
already failed, so attack the flow's OWN entropy floor, not the candidate weights). git_rev in each run manifest.

## 2026-06-14 — Dense value-grid policy: best result to date (TENTATIVE, 2000-step horizon)

The grid value-policy direction (NOT the flow): a U-Net (`grid_net.GridValueNet`) over t0
canvas features predicts a value for every viewpoint in a `{0.5,0.25} × 16×16` overlap
grid; **argmax = the t1 policy** (self-contained, no separate critic/generator). All numbers
below are SNAPSHOTS at a 2000-step (~1.6-epoch) horizon and WILL change — the horizon ladder
may ×100 the steps and many knobs are still moving. Recorded precisely so it's reproducible.

**BEST CONFIG TO DATE — run `grid_s6_h2000__trial0003` (crockett `~/projects/CanViT-PyTorch-RL/runs/`):**
- **Code: commit `ea6c3b7c9471e08f0cf8e14692d69c85dfc5ec82`** (recorded in the ckpt's `git_rev`).
- **Result (val, full 2000-img, paper squish protocol):** `val_gridcorr` (Pearson) **0.3074**,
  Spearman 0.3118, `val_ce_t1_mode` **0.7423**, **`val_miou_t1_mode` 41.18%** — first config to
  clear the central-crop bar (41.10, measured through our own eval at scale 0.625). `train_gridcorr`
  0.4557 (overfit gap 0.148). `val_mode_scale_std` 0.0 (scale still collapses to one value).
- **Checkpoint: `runs/grid_s6_h2000__trial0003/best.pt`** (== `last.pt`; eval only at end). Holds
  `net_state`, `config`, `git_rev`, `step`, `metrics`. Net = 3.68M params (width 128).
- **Hyperparameters (the searched ones):** lr **5.169e-5** (≈ the search floor — low LR clearly wins),
  weight_decay **1.444e-3**, adam_beta2 **0.95**, target_momentum **0.98**.
- **Fixed config (`GridConfig`):** canvas_grid 32, scales (0.5, 0.25), grid 16, width 128,
  entropy_channel True, **t0_mode `riid`** (random t0 = start-state augmentation; eval always
  full_scene), scale_min 0.25, **score_res 128** (reward CE scored at 128², not 512²), grad_clip 1.0,
  adam_beta1 0.9, steps 2000, warmup_frac 0.5, batch_size 16, **K=1** (one viewpoint/scene, fixed).

**The recipe pieces that matter (the little details):**
- Files: `grid_net.py` (U-Net, **replicate conv padding** — zero-pad caused spurious corner
  artifacts in the predicted grid, fixed in `ea6c3b7`), `grid_train.py` (online sparse K=1
  trainer), `reward_maps.py` (precomputed objective), `grid_optuna.py` (horizon ladder).
- Target = fractional CE `(t0_ce - vp_ce)/t0_ce` + online global-z via **bias-corrected EMA**
  (`grid_train.RunningNorm`) — no per-scene stats, unlocks K=1 (memory `fractional-ce-target-unlocks-k1`).
- **Objective = `val_gridcorr`** (per-image Pearson of pred grid vs the PRECOMPUTED true reward
  landscape), NOT mIoU (argmax-noisy, ~1pp jitter, can't distinguish a 30× LR range). Maps:
  `runs/reward_maps/grid_{validation,trainslice}_s0.5-0.25_g16_r128_c32.pt` (precompute via
  `python -m canvit_pytorch_rl.reward_maps --canvas-grid 32`, ~24 min, ~1.5M forwards; reusable
  while the action space is unchanged).
- Frozen perception: backbone `canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02`,
  probe `canvit/probe-ade20k-40k-s512-c32-in21k`. Data: `Ade20kSquish`, **no augmentation**
  (val transforms), full ADE20K train (20210) / val (2000).

**What's working / why (TENTATIVE, n=6 trials @ 2000 steps, single-eval):**
- The gridcorr objective WORKS: `corr(gridcorr, CE) = -0.81`, `corr(gridcorr, mIoU) = +0.49` across
  the 6 trials — fit and deploy move together (the whole reason for the objective).
- **LOW LR wins** (best at the 5e-5 floor; high-LR 1.7e-3 baseline = worst) — OPPOSITE of the
  noisy-mIoU grid_s1 era; gridcorr could finally tell LRs apart. Floor probably wants to drop below
  5e-5, esp. at longer horizons.
- WD secondary at 2000 steps (high WD 6e-2 tolerable, smaller overfit gap); expected to matter at
  longer rungs as the gap (~0.13–0.15) grows. Seed now defaults to WD 5e-2 [user].
- Scale collapse persists across ALL configs (one scale, position varies) — not config-dependent.
- Padding fix (replicate) lifted gridcorr/CE/mIoU together vs the artifact-laden runs.

## 2026-06-15 — The val_gridcorr ceiling is INFORMATIONAL (frozen-t0 limit), not a knob

Settled why val_gridcorr plateaus at ~0.30–0.31. Full reproduction + numbers in
`docs/sessions/2026-06-15-gridcorr-ceiling-is-informational.md`; diagnostics committed
`@80067df`, all re-run on crockett (full-val N=2000).

- **Realistic ceiling ~0.31–0.33; the net is already at it.** The per-image-Pearson metric rewards
  only WITHIN-image structure (per-image level is centered out), which splits ~85% within-scale
  position / ~15% between-scale. The SCALE-ONLY oracle floor (per-image per-scale mean) is **0.3273**
  (`landscape_headroom.py`).
- **The "scale collapse" is the DATA, not a bug:** s0.5 reward (0.0465) > s0.25 (0.0227), and s0.5
  wins for 83% of images. The net correctly learns this CONSTANT preference (between-scale gridcorr
  0.264 = the same-for-all-images floor) and the argmax picks s0.5.
- **Image-specific between-scale (+0.063 to oracle) is NOT predictable from t0** (`between_scale_learnable.py`:
  ridge on pooled t0 → val ≤ 0.264 at every α; `net_decompose.py`: the deep net fails to fit it even
  on TRAIN). Two independent predictors fail → a feature/information limit = the active-vision
  circularity (can't know a glimpse's value without taking it).
- The net DOES learn a real within-scale signal (0.18 corr > entropy 0.122) but over-weights it ~2–3×.
  Entropy is already a default input; ON vs OFF = +0.008 (= the +0.009 oracle).
- **Deliverable** `grid_repro_best` (default `GridConfig` = trial0009 recipe, ConvNeXt w128) reproduces:
  val mIoU t1 ~0.41 (at the EG-C2F t1 41.1 bar), t0 38.5 ✓. **Noise floor (3 seeds):** val_gridcorr
  0.3199 / 0.3252 / 0.3296 (mean 0.325, std 0.004, best `grid_repro_s2` 0.3296) → MAXIMIZED at
  ~0.325 ± 0.004, confirming the ~0.32–0.33 ceiling.
- **Implication:** t1 training-knob tuning is exhausted (arch/cap/horizon/WD/LR all flat). Raising
  val_gridcorr needs new INPUT information → the **T=5 sequential setup** (earlier glimpses inform
  later viewpoint value). Only residual t1 lever: within-scale rebalance (`within_var_reg`, ~+0.02).

## 2026-06-15 — Longer-horizon grid policy reaches EG-C2F PARITY across T=5 (clean recipe)

Refactored the t1 trainer (removed per_position + overfit probe; `advance_state` seam) and made the
grid net a `canvit_eval` Policy (`grid_policy.GridPolicy` + `grid_eval` → Fig-4B-comparable rollout at
any horizon). Full arc: `docs/sessions/2026-06-15-refactor-and-longer-horizon.md`. Commits 789a938
(refactor, verified 0.3199 identical), ec3a818 (policy+eval), 8472a82 (train_horizon).

- **The lever [user "the canvas is a state"]:** the value net is `V(canvas_state)`; training reward is
  already online, so longer horizons = ENRICH THE START-STATE DISTRIBUTION. `grid_train.train_horizon`:
  each step prepends k~U{0..train_horizon-1} random priming glimpses (chained via `advance_state`), one
  SHARED net learns the value grid at every rollout state. K=1 + global-z keystone intact.
- **Zero-shot baseline** (t1-only net `grid_repro_s2` rolled out): mIoU front-loads then PLATEAUS —
  41.34 / 41.80 / 42.00 / 42.10 (t1–t4), below EG-C2F (41.04/42.05/42.69/43.17): picks redundant glimpses.
- **`grid_t3` (train_horizon=3, seed 0):** rollout 41.24 / **42.18 / 42.75 / 43.24** (t1–t4) — fixed the
  plateau (+0.4/+0.8/+1.1 at t2–t4), **EG-C2F PARITY across T=5** (within mIoU noise), and EXTRAPOLATES
  to t4 (trained ≤t3). Bonus: t1-from-t0 gridcorr improved same-seed 0.3199→0.3359. Notable because the
  ~20 prior elaborate T=5 attempts only matched/underperformed EG-C2F; this is a clean, simple recipe.
  Repro: `grid_train --train-horizon 3` then `grid_eval --ckpt-run grid_t3 --n-timesteps 5`.
- **RECALIBRATED [user 2026-06-15] — it's a REAL BEAT at every horizon, not "parity".** The 5k sweep was
  UNDERTRAINED per depth (steps/H per depth); training H× longer fixes it. `grid_t5_25k` (train_horizon=5,
  25k=5×) mid-run hit val_gridcorr **0.347** (exceeds the supposed "0.33 ceiling" → it was undertraining)
  and rollout **41.32/42.21/42.87/43.21** (t1–t4) vs EG-C2F-c32 41.1/42.0/42.7/43.2 → **+0.2 at t1–t3,
  ties t4**. EG-C2F is DETERMINISTIC (no eval noise) so a consistent +0.2 across timesteps AND runs
  (grid_t3, grid_t5op) is SIGNAL, not noise — earlier "parity" framing was defeatist. t4 convergence is
  expected (glimpses saturate). **val_gridcorr stays the PRIMARY HP metric; push it higher (5k→25k gave
  0.33→0.347; 100k next) — big gridcorr gains translate to mIoU (~0.4 target).** Tooling `grid_policy`+
  `grid_eval`; the [[grid-where-underfitting]] frozen-feature "ceiling" is SUPERSEDED (undertraining).
  **[SUPERSEDED same-day → see the next bullet: the comparison was glimpse-budget-confused and judged on
  t1-gridcorr; corrected, trajectory-wide supervision BEATS EG-C2F at EVERY horizon incl t4.]**
- **TRAJECTORY-WIDE SUPERVISION beats EG-C2F-c32 at EVERY horizon [2026-06-15, tag `result/grid-t5-trajsup-10k`].**
  Two corrections to the bullet above: (a) **judge by ROLLOUT mIoU, not t1 `val_gridcorr`** (t1-only — mis-ranks
  schemes) [user]; (b) **match TRAINING `glimpse_forwards`, not scorings** [user] — the scoring-match had silently
  given the old run 2× the glimpses. At equal glimpse budget (1.6M), `grid_t5_trajsup_10k` (train_horizon=5, 10k
  steps, supervise a K=1 sample at every rollout state via `_rollout_samples`) rolls out **41.43 / 42.40 / 42.97 /
  43.37** (t1–t4) — beats EG-C2F-c32 (41.04/42.05/42.69/43.17; our `evaluate.py --policy entropy_coarse_to_fine
  --canvas-grid 32` reproduces paper Table 4) at EVERY horizon **+0.39/+0.35/+0.28/+0.20** (deterministic = real),
  and beats old `grid_t5_25k` too. Then [user "do it WELL"]: FUSED the rollout so the advancing glimpse IS the
  supervised action (ε-greedy Q, ~2× cheaper) — revalidation in flight. Details:
  docs/sessions/2026-06-15-refactor-and-longer-horizon.md. Repro: `grid_train --train-horizon 5 --steps 10000`,
  `grid_eval --ckpt-run <run> --ckpt-name last.pt --n-timesteps 5`.
- **FUSED ε-greedy Q rollout — same win, ~2× cheaper, plateaus ~1M glimpses [2026-06-15].** `_rollout_samples`
  (27c797f) now takes ONE ε-greedy glimpse/step that BOTH advances the rollout AND is the supervised action (no
  discarded probe; argmax w.p. `prime_on_policy`, else random) → cost `b·(1+H)` vs the separate scheme's `b·2H`,
  ~504 glimpse-forwards/s. `grid_t5_fused_op50_20k` (prime_on_policy=0.5) beats EG-C2F-c32 at EVERY horizon (peak
  ~step 11.7k, t1–t5: **41.33/42.30/43.17/43.48/43.70** vs 41.04/42.05/42.69/43.17/43.53) AND matches/beats
  trajsup_10k at 30% fewer glimpses; per-depth t1 corr 0.272→0.327 (fusing fixes the cross-depth interference).
  **SCALING: rollout PLATEAUS by ~10k steps (~1M glimpses)** — 15k/20k don't lift it, t3 slightly regresses at 20k
  under full LR decay (deployable peak is ~step 11–12k, NOT the final). `grid_t5_fused_op50_100k` (warmup_frac=0.1,
  longer decay) in flight to test data-saturation vs schedule-artifact. Repro: `grid_train --train-horizon 5
  --steps 20000 --prime-on-policy 0.5`; `grid_eval --ckpt-name step_NNNNNN.pt --n-timesteps 6`.

## 2026-06-15 (eve) — Value-grid misalignment, aligned readout (new default), 2-scale (0.5,0.25)

- **MISALIGNMENT found + fixed; aligned readout is the new DEFAULT [`52b92e2`].** The value grid
  `[B,n_scale,16,16]` is in per-scale SAFE-BOX coords (cell i,scale s → viewpoint `linspace(-(1-s),1-s)[i]`),
  but the ConvNeXt U-Net registers its output to FULL-IMAGE coords (skips from the image-aligned canvas);
  one shared decoder + 1×1 head can't serve two safe boxes (affine offset 0.44 @ s=0.5). Verified from code
  incl. the `canvit_pytorch` coord convention. FIX `policy_arch='aligned'` (`9e56585`): decode to the 32×32
  canvas-shaped map → `grid_sample` at per-scale viewpoint centres → per-scale conv head over `[local+scale_emb]`
  (aligned by construction; `direct` byte-identical). t1 c32 A/B (`grid_t1_{direct,aligned}_head`, identical
  grid_s8 cfg, only readout differs): **aligned wins the landscape fit** — gridcorr 0.329 vs 0.319, spearman
  0.337 vs 0.324 @ step 5000 — with a healthier, less-collapsed mode-spread (0.26 vs 0.20). **BUT t1 deploy
  `val_miou_t1_mode` is TIED (0.4111 vs 0.4112)** — the rank-metric edge doesn't (at t1) convert to argmax mIoU.
  So the misalignment is a real fit-limiter but NOT crippling (the BAD/direct c64 still beat EG-C2F-c64: T=5
  eval of `grid_t5_fused_c64_op50_20k` step 8000, t0 39.59=paper 39.6, **39.59/42.58/43.44/44.39/44.73/45.00**).
- **Scales: explored single-scale 0.5, then REVERSED to 2-scale (0.5,0.25) [user].** The misaligned c64 best
  ckpt picked 0.5 **96.6%** (100% @t1; `c64eval_step8000_t5/actions.parquet`; EG-C2F also fixes 0.5 t1–t4), so
  single-scale 0.5 (`grid_t5_aligned_s05_c64_20k`) was tried first — killed at 5k. BUT at MATCHED 5k the aligned
  **2-scale beats single-scale s0.5** at every t1–t4 (+0.41/+0.20/+0.06/+0.12), so [user] DEFAULT to both scales
  (already GridConfig default `(0.5,0.25)`). The 2-scale edge is NOT from deploy-time 0.25 (usage 2–9%@5k →
  0.3–2.4%@10k, converging to ~all-0.5) — it's opt-dynamics-of-training-with-0.25 or run-variance
  (`action-space-ablation-opt-dynamics` memory / session doc).
- **BEST T=5 c64 policy: `grid_t5_aligned_2scale_c64_20k`** (aligned + 2-scale + train_horizon=5 fused prime0.5
  + c64, 20k, grid_s8 HPs; **96 gf/step**, 960k by step 10k). STILL TRAINING (~step 12.5k+); beats EG-C2F-c64
  AND single-scale s0.5 at every t1–t4 and CLIMBING: step_5000 39.59/42.75/43.87/44.47/44.81 → step_10000
  39.59/42.94/44.05/44.55/44.94 (vs EG-C2F-c64 `egc2f_t5_apples` 39.60/42.22/43.30/44.04/44.65). Judge ROLLOUT
  mIoU at each 5k ckpt (still rising or peaked?). Details + HANDOFF:
  `docs/sessions/2026-06-15-value-grid-misalignment.md`. Repro: `grid_train --canvas-grid 64 --scales 0.5 0.25
  --train-horizon 5 --steps 20000 --prime-on-policy 0.5` (aligned is default).

## 2026-06-15 (night) — `grid_t5_aligned_2scale_c64_20k` finished; clean-break refactor to viewpoint-Q; budget-in-forwards; CE-judged perpetual sweep

Full narrative: `docs/sessions/2026-06-15-refactor-to-q-and-ce-sweep.md`. Tag `pre-refactor-2026-06-15`
preserves the old tree; refactor commits f67460a / a5cecee / 1fc8bc8 (branch `refactor/codebase-cleanup`,
merged `--no-ff`).

- **The 20k run finished and PLATEAUED past ~1M glimpse-forwards.** `grid_t5_aligned_2scale_c64_20k` full-val
  T=5 (paper protocol): step_20000 **39.59 / 42.73 / 43.97 / 44.57 / 44.91** (t0–t4) vs EG-C2F-c64
  39.60/42.22/43.30/44.04/44.65 → beats at every t1–t4. But the within-budget checkpoint (step_10000 ≈ 1M
  forwards) is as good or better: 39.59/42.94/44.05/44.55/44.94; step_15000 = 39.59/42.75/43.87/44.48/44.86.
  The 2× budget bought nothing on val (t1 even slipped). Judged by VAL CE ALONE — the train-vs-val gap is a
  CONFOUND (the frozen probe is already overfit to ADE train, so `train_ce` < `val_ce` is the PROBE, not the
  policy [user 2026-06-16]) — val CE bottoms ~step 10k (≈1M forwards) and the BEST is INTERMEDIATE, not the
  last → no val benefit past ~1M → **the 1M-forward cap**. Lever = HPs/regularization (the sweep), not more
  steps. [memory: probe-overfit-confounds-train-val.]
- **Clean-break refactor `grid` → viewpoint-Q (`canvit_pytorch_rl.q`).** `ViewpointQNet` predicts
  Q(state, viewpoint); `GreedyQPolicy` argmaxes; `candidate_viewpoints`/`centers_per_axis`/`QConfig`. Grouped
  layout `q/` + `baselines/` + `tools/` + shared root substrate. Dropped the misaligned `direct` readout
  (always aligned now), the unused `nflows` dep, dead `scripts/`, the stale `plot_training`. Metric
  `gridcorr`→`qcorr`. **VERIFIED behavior-preserving:** the new `q.eval` reproduces the old `grid_eval` to 4
  decimals on step_15000 AND step_20000 (after `throwaway/migrate_q_ckpts.py --apply` rewrites pre-refactor
  ckpt metadata: `grid`→`centers_per_axis`, drop `policy_arch`). No in-code back-compat shims.
- **Budgets are in GLIMPSE-FORWARDS now** [user]: `QConfig.budget_forwards` (1M) replaces `steps`; the trainer
  derives step count = `budget_forwards // (batch_size × (1+train_horizon))`. q.optuna sweeps `--base-forwards`.
- **Judge by val CE** [user]: the trainer's best.pt AND the optuna objective both MINIMIZE `val_ce_t1_mode`
  (less noisy than mIoU; we optimize CE reduction). mIoU + qcorr still logged as diagnostics.
- **Perpetual CE sweep live** (study `t5_c64_ce`): `throwaway/perpetual_sweep.py` runs `q.optuna` forever —
  disk-gated (stops at <5 GB free, in-study callback + supervisor relaunch), crash-resilient, detached
  (`nohup setsid`). Broad `--search` (lr, weight_decay, betas, target_momentum, warmup_frac, width,
  block_layers, t0_mode, prime_on_policy, entropy_channel, frontend_mlp); minimize val t1 CE; c64/T=5/1M.
- Repro the canonical run: `python -m canvit_pytorch_rl.q.train --run-name <name>` (defaults ARE c64/T=5/1M).
  Eval: `python -m canvit_pytorch_rl.q.eval --ckpt-run <name> --ckpt-name last.pt --n-timesteps 5`.

## 2026-06-16 — first c64/T=5 val-CE sweep (FLAT); defaults re-tuned by synthesis; sweep redesigned (pin design, tune optimizer)

Session: `docs/sessions/2026-06-15-refactor-to-q-and-ce-sweep.md` (cont.). First 1M-forward CE sweep
(study `t5_c64_ce_f1000000`, 7 trials, broad 12-dim `--search`), then a defaults update + sweep redesign.

- **The landscape is FLAT and single-seed — do NOT read the ranking as truth.** `rollout_eval` iterates the
  WHOLE val loader, so `val_ce_t4` is full-val; even so the top-5 span only **0.6656–0.6706** (top 4 within
  0.0022 — inside single-seed noise). CE and mIoU DISAGREE on the winner: trial4 is lowest CE, but **trial3
  has the best val mIoU t4 (44.99) and is still improving at the budget end (best-at-last)**. trial3 is
  essentially the user-preferred profile (riid). [memory: per-batch-metrics-are-noisy-samples,
  understand-history-question-metrics.]

  | trial | val_ce_t4 | best@ | val mIoU t1/t2/t3/t4 | warmup | wd | t0 | prime | lr | blk | ec | fm |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | 4 | 0.6656 | 6000 (mid) | 42.33/43.77/44.51/44.86 | 0.05 | 6e-4 | full | 0.0 | 6.8e-5 | 3 | F | T |
  | 3 | 0.6662 | 10416 (last) | 42.48/43.75/44.51/**44.99** | 0.5 | 6e-4 | riid | 1.0 | 1.0e-4 | 3 | F | T |
  | 1 | 0.6675 | 10000 | 42.46/43.78/44.27/44.83 | 0.1 | 1.4e-2 | full | 1.0 | 6.3e-5 | 2 | F | F |
  | 0 (seed) | 0.6678 | last | 42.77/43.81/44.55/44.86 | 0.5 | 1.2e-2 | full | 0.0 | 8.3e-5 | 2 | T | F |
  | 2 | 0.6706 | last | 42.59/43.75/44.53/44.94 | 0.5 | 1.8e-3 | riid | 0.5 | 5.5e-5 | 3 | T | T |
  | 5 | 0.686 | (pruned @2000) | underfit — lr 5.5e-6 |
  | 6 | 0.6708 | (killed @~2k) | incomplete (killed when sweep retired); high lr 5e-4 + wd 8.6e-2; at 2k 0.6708/44.69, inconclusive (NOT divergence — 0.7258 was its step-0 baseline) |

- **Defaults re-tuned by SYNTHESIS, not by copying the CE-winner** [user 2026-06-16: "evaluate the whole
  trajectory, all HPs … choose sensible defaults"; priors — less warmup, more WD, riid t0, 0.5 prime, and
  "good results without the entropy channel is all the better"]. On a flat single-seed landscape the
  strongest signal is the user's priors, and the data is consistent with them. New QConfig defaults:
  **t0_mode=riid, prime_on_policy=0.5, warmup_frac=0.1, weight_decay=1e-2** (kept substantial — NOT
  trial4's 6e-4), **lr=7e-5, block_layers=3, entropy_channel=False**. Reversible (tagged); the old recipe
  survives in tags. Copying trial4 verbatim would have set full_scene / prime 0 / wd 6e-4 — the opposite of
  the priors, on a within-noise 0.0022 CE edge.
- **Sweep redesigned** [user 2026-06-16: "fix warmup, entropy, riid start, on-policy priming, and sweep the
  rest … also tune beta1 and beta2"]. New study **`t5_c64_ce_optdyn`**: the 4 DESIGN dims pinned via QConfig
  defaults (+ target_momentum=0.997, pinned [user] — target-norm EMA, an infra knob, not tuned); `--search`
  = lr, weight_decay, adam_beta1, adam_beta2, width, block_layers, frontend_mlp (optimizer + capacity). Narrower space → TPE converges faster on the dims that matter; its
  trial0 seed = the new defaults = the confirming 1M run. Old 12-dim study `t5_c64_ce_f1000000` retired.

## 2026-06-16 (cont.) — twin-Q (clipped-double-Q) is the DEFAULT; deploy-ensemble probe NULL

Session: `docs/sessions/2026-06-16-twinq-default.md`.

- **Deploy-ensembling the 5 overnight nets — NULL.** `throwaway/ensemble_eval.py` aggregates their `best.pt`
  Q-maps before argmax (min/mean/median, raw + z-scored), full-val T=5. Best (min-raw) TIES the best single
  net (45.02 vs 44.99 mIoU@t4; 0.6659 vs 0.6662 CE) — within noise; the nets agree on the good viewpoints
  (flat landscape) so there's little per-net overestimation to cancel. Don't re-try deploy-ensembling.
- **Twin-Q is the DEFAULT** [user 2026-06-16]. `QConfig.n_critics=2`: 2 distinct-init `ViewpointQNet` critics
  (`EnsembleQNet`), `forward`=per-state **min** = the rollout+deploy policy; each critic regresses the SAME
  measured fractional-CE target (no bootstrap → the min is ONLY the policy). `critic_qspread` diagnostic
  ≈0.043 ⇒ the critics differ (min is not a no-op). 200k-forward validation (`twin_min_nd_200k` last.pt):
  t0–t4 = **39.60 / 42.35 / 43.84 / 44.45 / 44.64**, CE@t4 0.6685 — ≈ EG-C2F-c64 @t4, ahead @t1–t3, at 1/5 the
  budget. `q.eval` reconstructs the ensemble for twin ckpts; perpetual sweep → fresh twin study
  `t5_c64_ce_twinq` (n_critics not searched ⇒ every trial twin); single-critic `t5_c64_ce_optdyn` retired.
- **Run dirs datetime-prefixed** (`RunConfig.__post_init__`, UTC `YYYYMMDD-HHMMSS_`, idempotent) ⇒
  chronological + collision-free; `tools/sweep_report` glob `*{study}__trial*`.
- Throughput: ~198 glimpse-forwards/s, ~84 min per 1M trial, ~91% GPU util (twin ≈ single — the 2nd critic is
  a tiny U-Net vs the frozen CanViT-B that dominates every glimpse).
- **Horizon off-by-one fixed** [user 2026-06-16]: `train_horizon` now = the deploy horizon (4 decisions,
  d0..d3 → t4); `eval_horizon=train_horizon+1`; objective `val_ce_t4` unchanged. Was: train_horizon=5 took 5
  actions (reached t5 for the d4 reward) while deploy ended at t4 → 1/5 of supervision never deployed. Same 1M
  budget ⇒ ~20% more supervision on the deployed decisions; `corr_dX` now d0..d3. Fresh study
  `t5_c64_ce_twinq_hfix`; future train_horizon=4 runs differ from the (5-action) documented anchors.

## 2026-06-16 (eve) — RICH-AUX policy graduated into `q/`: beats EG-C2F everywhere, ≈ anchor at ~1/6 forwards

Full arc + recipe: `docs/sessions/2026-06-16-rich-aux.md`. A viewpoint-Q policy fed curated scale-equalized
aux features (probe entropy, entropy-Δ, cos-to-prev, cos-to-init, LN feats, LN-feat-Δ) through a
**per-channel BatchNorm2d + per-group gate + 1×1 proj** frontend into the SHARED `ViewpointQNet` body. The
delta groups use the **INIT canvas as the t0 reference** (deviation-from-template, not dead-zeros).

- **Result (single seed, full-val 512² paper protocol).** Beats EG-C2F-c64 at every t1–t4 by step 1000
  (~80k fwd) and ≈ the anchor: `richaux_q_20k` @step1000 `42.46 / 43.71 / 44.44 / 44.82` (t1–t4), CE@t4 0.672
  vs EG-C2F `42.22/43.30/44.04/44.65` and anchor `42.94/44.05/44.55/44.94`. The pre-graduation throwaway hit
  ~44.9 t4 by step 1500 (~120k fwd) — **anchor quality at ~1/6 its ~960k forwards.** OPEN (the 20k run tests
  it): does more training push **t1** past the anchor's 42.94 (still −0.12 behind at 80k fwd)? Single seed.
- **Gates** (per-group favor/suppress): entropy ↑ (~1.02), LN-feat-Δ ↓ (~0.97).
- **Graduated into `q/` behind `cfg.rich_aux`** (`bc37fad`/`503d3cd`): `RichAuxNet` + rich state-encoder in
  `net.py`; prev/init threaded through `rollout_samples`/`GreedyQPolicy`/`evaluate_q`/`q_map_figure`
  (gated; **`rich_aux=False` byte-identical to base, pinned by `test_q`**). Single-net (`n_critics=1`;
  ensembling TODO). Run: `q.train --rich-aux --n-critics 1 --lr 3e-4`. Throwaway retired (`03978f8`).
- Context: the probe-free-proxy hunt (`reward_corr`, `viz_t0_channels`) found NO cheap probe-free substitute
  for the entropy channel (best eig-spectral spearman ~0.37); it's ~free (probe in loop) so it stays.

## 2026-06-16 (eve) — `t0_mode` default flipped to `full_scene` for next runs [user]

`QConfig.t0_mode` default `riid → full_scene` (config.py). Rationale [user 2026-06-16]: judged at the **t1**
selection point, the one sweep that varied t0_mode (`t5_c64_ce_f1000000`) gives **no riid advantage** — the
best t1 there was a full-scene trial (0000, 42.77) and the two riid trials sat mid-pack (42.48/42.59); riid's
only edge was at **t4**, and that edge **flipped sign** vs t1, on a flat single-seed landscape where t0_mode
was confounded with prime/width/warmup in every trial (`sweep_sets.md`). With `train_horizon=4` +
`prime_on_policy`, mid-rollout states t1..t3 are already diverse, so riid's start-state augmentation is
largely redundant. Deploy/eval was always full_scene regardless, so this aligns train with deploy. riid stays
available (optuna option / explicit flag). NOT an isolated ablation — no clean riid-vs-full run exists,
none under rich-aux; the live `richaux_q_20k` was launched riid and is unaffected (default change only hits
future runs). Reversible (one-line default).

## 2026-06-16 (night) — the anchor's 42.94 is a single-seed PEAK, not a stable bar; seed band queued

While judging the live `richaux_q_20k`, caught a comparison error worth recording. We had been measuring
rich-aux's noisy t1 band against the anchor's **single** number 42.94 (`grid_t5_aligned_2scale_c64_20k`,
the step-10000 = 960k-forward budget checkpoint). Pulling the anchor's OWN per-eval t1 trajectory
(`val_miou_t1_mode`, the argmax deploy-t1 metric; @step10000 = 42.91, confirming it's the same number):

```
anchor t1 over its 20 evals: mean 42.63, std 0.20, min 42.00, max 42.91 — and 42.91 @step10000 is the MAX.
```

So **42.94 is the high point of one seed's path**, and it coincided with the budget-matched checkpoint we
used to define the bar (legitimate selection rule, but n=1). The anchor's actual t1 *level* is ~42.63.

- **Method note (corrects a mistake I made):** a within-run eval trajectory is NOT a reliability estimate.
  A full-val eval at a fixed checkpoint is deterministic (fixed val set + argmax → zero measurement noise);
  eval-to-eval scatter is the weights wandering along ONE optimization path. Its mean/std/range say nothing
  about cross-seed reliability and must NOT be compared across runs as such. Reliability = the spread, across
  independent SEEDS, of the metric at a fixed checkpoint-selection rule. We have one seed each → no estimate.
- **Implication:** rich-aux (seed 0, ~42.5 at t1) is NOT clearly below the anchor; the apparent ~0.4 deficit
  was band-vs-peak. Cannot decide rich-vs-base without seeds.
- **Overnight seed band (queued, `throwaway/seed_band.py`):** single-critic, 1M budget, full_scene t0, c64/
  2-scale. RICH (`--rich-aux --n-critics 1 --lr 3e-4`) seeds 1,2 (seed 0 = the live run); BASE (`--n-critics 1`,
  current defaults) seeds 0,1,2. Interleaved; launches when the GPU frees. Decides rich-vs-base on mean AND
  spread. Confound to keep in mind (documented, not hidden): the two arms differ in lr (3e-4 vs 7e-5) and in
  entropy availability (rich has entropy as a feature group; base has entropy_channel off) — this is a
  method-vs-method comparison at each method's own recipe, not a one-variable ablation.

## 2026-06-16 (night) — rich-aux promoted to the DEFAULT recipe [user]; val-CE view; best-by-metric tables

[user 2026-06-16: "i want rich aux by default and the 20k run's stuff by default"]. QConfig defaults flipped
to the single-critic rich-aux recipe: **`rich_aux=True`, `n_critics=1`, `lr=3e-4`** (the richaux_q_20k recipe).
Kept `t0_mode=full_scene` (prior user call, not the 20k run's riid) and `budget_forwards=1M` (the hard cap, NOT
the 20k run's 1.6M override). Base/twin is the retained alternative (`--no-rich-aux --n-critics 2 --lr 7e-5`).
Updated config docstring + fields, README, CLAUDE.md (target + rich-aux + a new conventions bullet). `uv run
just` green (10 tests; `rich_aux=False` still byte-identical to base, pinned by test_q). **NOT deployed to
crockett's checkout tonight** — the parked seed band's base arm passes no `--rich-aux`, so a `rich_aux=True`
default on disk would silently turn base runs into rich runs and corrupt the confirming experiment; deploy
after the band finishes. The SEED BAND (throwaway/seed_band.py) confirms/reverts this default.

**Val CE view (the optimized, less-noisy metric — judge by it) [user: "always look at both, remember the
nuances"].** richaux_q_20k learns MONOTONICALLY on CE across all horizons where t1 *mIoU* looked like a noisy
plateau: ce_t1 0.7200→0.7158, ce_t2 0.6939→0.6887, ce_t3 0.6805→0.6764, ce_t4 (objective) 0.6724→**0.6678**
(min @step13000). At t1 the **anchor is marginally better on CE** (mean 0.7159 / min 0.7126 vs richaux mean
0.7187 / min 0.7158) even though t1 *mIoU* was ~level — CE and mIoU disagree (ρ≈0.26).

**Best-by-metric × horizon (new standing convention [user]: keep BOTH, they land at different steps):**
richaux_q_20k through step 15000 — t1: CE 0.7158@15k / mIoU 42.64@14k; t2: 0.6887@15k / 43.99@8k;
t3: 0.6764@13k / 44.57@14k; t4: 0.6678@13k / 44.86@14k; overall objective val_ce_t4 0.6678@13k.

**Pivot [user, night]:** dropped the rich-vs-base seed band (user confident in richaux, doesn't want a base
comparison); instead the perpetual HP sweep (`throwaway/perpetual_sweep.py`) now tunes richaux overnight —
new study `richaux_c64_t5_ce`, GPU-gated to wait for the live richaux_q_20k run to finish first. Requires the
richaux config defaults deployed on crockett (q.optuna reads rich_aux/n_critics from on-disk QConfig defaults).
`throwaway/seed_band.py` retired.

## 2026-06-17 — richaux HP sweep: confirms the default

Overnight perpetual sweep `richaux_c64_t5_ce` (tuning lr/wd/betas/width/blk/fmlp around the richaux default,
1M/trial, minimize val_ce_t4). 9 trials, 0 errors. **It confirms the default rather than beating it:** trial0000
= the default recipe (full_scene, w128, lr3e-4, wd1e-2) is #1 by BALANCED mean-CE(t1..t4) (0.6852), the only
top trial still improving at 1M (best-at-last), cheapest, no overfit; #1 (tied) on balanced mIoU (44.05) too.
Width-256 trials are noise-tied on ce_t4 but plateau/overfit early — w256 buys nothing. **full_scene validated**
(trial0000 vs the riid richaux_q_20k baseline, same recipe: CE 0.6875→0.6852, mIoU4 44.76→44.95). mIoU
corroborates CE but is noisier (judge by CE). Both metrics + trajectories + balanced rankings, leaderboard,
preserved ckpts (`preserved_ckpts/richaux_sweep/`). Regenerate the leaderboard/trajectories from the optuna db:
`tools.sweep_report --study richaux_c64_t5_ce [--trajectories]`.

## 2026-06-17 — tooling consolidation + repo cleanup

Consolidated sweep analysis into one tool `tools.sweep_report` (renamed from `sweep_miou`; CE-primary,
balanced-mean-CE default sort, `--trajectories` + shape) and graduated `tools.ckpt_meta` (ckpt metadata /
step verification). Deleted dead throwaways (`build_ckpt_table`, `enqueue_seeds`, and the earlier
`sweep_summary`/`sweep_trajectories` dupes) + the gitignored `revisit/` scratch. `q.optuna` now enqueues a
curated init seed set (`SEED_TRIALS`: width=256 at the winning optimizer + capacity corners) and fills missing
seeds into existing studies (not just fresh) — 5 injected into the live sweep. README de-duplicated (no HP
values/dates; points to `QConfig` + a defaults-dump). crockett hard-reset clean to origin/main.

## 2026-06-17 — remove ensembling; unify to ONE ViewpointQNet; drop "richaux" terminology

Big code-clarity refactor [user]; NO behavior change (verified bit-identical).

- **Removed twin-Q / clipped-double-Q ensembling** (`973f5be`): `EnsembleQNet`, `n_critics`, `ensemble_agg`,
  the per-critic loss branch, the `grad_norms` critic-stripping, `throwaway/ensemble_eval`. Single net always;
  the ensemble lineage lives in git history.
- **Unified RichAuxNet + ViewpointQNet into ONE `ViewpointQNet`** (`973f5be`): a pluggable `frontend`
  (`CanvasFrontend` | `CuratedFrontend`, both subclass `Frontend`) maps the input to `[B,width,32,32]`; the
  ConvNeXt body + readout are shared. `build_qnet(cfg, ...)` is the one constructor; gate logging is
  polymorphic `frontend.log()` (no isinstance). `state_dict` keys move under `frontend.*`.
- **Dropped "richaux" terminology** (`b348520`) [user: "all there is is a network with various ways to feed
  data into it"]: `rich_aux: bool` → `input_mode: Literal["canvas","curated"]="curated"`; `rich_dim`→
  `curated_dim`; `RichFrontend`→`CuratedFrontend`; `rich_*` helpers → `curated_*`. ckpt schema
  `rich_aux`/`rich_dim` → `input_mode`/`curated_dim`. (`input_mode` chosen over `frontend` to avoid clashing
  with the swept `frontend_mlp`. The live study name `richaux_c64_t5_ce` is kept — a persistent sqlite ID.)
- **Migration** `throwaway/migrate_qnet_keys.py`: pure state_dict key-rename + metadata translation;
  strict-load into the new arch + finite forward is the built-in equivalence gate. Run on preserved/old ckpts
  to eval under new code (writes `<stem>_unified.pt`).
- **Verification (loading old ckpts in new code) [user-required]:**
  - **Q-net is BIT-IDENTICAL** old↔new (amp/backbone/probe-free fixed-input forward): curated sum
    −71.74588112 == −71.74588112; canvas −468.53779167 == −468.53779167. So the unification is provably
    lossless for the policy net.
  - End-to-end `q.eval` of the migrated curated anchor/trial0000 reproduces documented per-t mIoU/CE to the
    bf16-backbone run-variance floor (trial0000 t4 mIoU 44.78 vs documented 44.95; full-fp32 `--no-amp` gives
    44.73 — *lower*, so bf16 is NOT a consistent quality loss). Base anchor reproduced near-exactly
    (t1 42.9121, t4 44.9252 vs 44.94). The **probe is already fp32** (`scoring.head_logits` forces
    `autocast(enabled=False)`+`.float()`; `candidate_ce`/`probe_entropy` route through it) — only the backbone
    canvas integration is bf16.
- `just` green throughout; live sweep undisturbed (ran on its launch-time code; crockett checkout NOT reset).

## 2026-06-18 — curated 8-seed band: viewpoint-Q beats EG-C2F on CE, seeded (+ frontend redesign WIP)

The reliability band the goal needed. **Full record + repro: docs/old_frontend_band_results.md.** Tags:
`result/curated-8seed-band` (`55fda96`, the band arch+ckpts) and `result/egc2f-c64-baseline` (`e805277`).

- **Result:** curated viewpoint-Q (8-seed band, 1M/seed) beats EG-C2F-c64 on **CE at every t1–t4 by ~6–13σ**
  of the band (endpoint CE 0.7151/0.6887/0.6751/0.6657 ±≤0.0012 vs EG-C2F 0.7258/0.7004/0.6828/0.6707; margin
  largest at t1). mIoU margin smaller (+0.11 endpoint / +0.26 deploy at t4) — CE↔mIoU misalignment, judge by CE.
  The single-seed "44.94 peak" was best-of-trajectory optimism; honest seeded endpoint mIoU_t4 = 44.76±0.11.
- **Tooling:** `tools.seed_report` (per-t mean±std, matched-endpoint + deploy bands), `throwaway/run_traj.py`
  (single-run trajectory). EG-C2F CE confirmed present in `baselines.evaluate` (the old "evaluate.py lacks CE"
  note was stale).
- **Frontend redesign [user, WIP]:** the curated frontend was reworked — concat+shared-proj (unequal per-group
  roles) → per-group [proj→LayerNorm(affine)→gate] → sum → pre-norm MLP, `curated_dim` 32→128 (no narrow
  bottleneck). NOTE: a plain per-group-proj-then-SUM is algebraically identical to the shared proj (verified
  1.9e-6); the LayerNorm is the non-trivial part. Validation run `curated128_s0` in flight vs the band.

## 2026-06-18 (eve) — collapse to ONE frontend; method moved into the README

Committed the single-frontend regime [user: "commit to it, delete the fork"]. HEAD `a1233d3`.

- **Deleted the `input_mode` fork:** removed `CanvasFrontend`, the `Frontend` base class, `qnet_input`, and the
  canvas-only knobs (dropout, input_norm, frontend_mlp, entropy_channel). ONE `Frontend` (per-group
  proj→LN(affine)→gate→sum → token-MLP) + ONE `StateEncoder` (the single place a canvas state → net input;
  built once, `reset()` at t0; init reference cached, not recomputed per step). `curated_dim` eliminated (never
  swept, == width). Renames: GROUP_NAMES→FEATURE_GROUPS, curated_*→feature_*/`init_reference`/`Frontend`.
  Net −152 LOC; `just` green (12 tests). Old canvas/curated ckpts load ONLY at their git tags.
- **Flat-index made structural:** the random sampler draws a flat `randint(n_candidates)` (was
  `scale*cells+cell`); sampling/gather/argmax now index ONE layout (`vp_flat`). Dropped `vp_cells`.
- **`curated128_s0`** (git_rev `adaa231`, the validation run) is FUNCTIONALLY EQUIVALENT to HEAD — the refactor
  is behavior-preserving except the random-exploration RNG stream (same distribution) [user: doesn't care about
  seeds]. At step ~7k/12500 it tracks at/above the band (ce_t4 0.6652, mIoU_t4 45.10 @ step 6000 vs band
  endpoint 0.6657 / 44.76). Let it finish; then seed-band the recipe under HEAD (`throwaway/seed_band.py`,
  prefix `seedband_s`).
- **Docs:** the method (action space, γ=0 reward, features, net, training, deploy) now lives in the README;
  CLAUDE.md points to it. README "Known warts": training reward scored at 128² px (masks downsampled) — move to
  512² if throughput allows. Throwaway fixed for the HEAD CLI: `seed_band` (no input_mode/lr), `perpetual_sweep`
  (fresh study `qpolicy_c64_t5_ce`, GPU-memory gate replacing the dead richaux_q_20k wait), `ablation_scan`
  (`--feature-groups`). STALE/broken under HEAD (not re-run, not yet removed): `overfit_batch`, `synth_fixation`,
  `migrate_q_ckpts`, `migrate_qnet_keys`.

## 2026-06-18 (eve) — remove the dead qcorr / reward-map precompute path

`qcorr` (predicted-Q vs the precomputed true-reward-landscape correlation) is dead in the c64/val-CE regime:
no current run logs it (the only reward-map files on disk are c32; `load_map` finds nothing at c64 → None →
skipped), and the HP objective is `val_ce_t4`, not qcorr. Removed `reward_maps.{load_map,map_path,
PrecomputeConfig,main}`, the `reward_map`/qcorr plumbing in `train.evaluate_q`, `stats.{rowwise_pearson,
rowwise_spearman}` (qcorr-only), and the qcorr panel in `training_curves` (now plots val CE per horizon).
Kept `reward_maps.{candidate_rewards,expand_state}` — still feeds the value-map filmstrip. Behavior-preserving
for training and the band's headline metrics (val_ce/mIoU come from the rollout); ~120 LOC removed. just green.

## 2026-06-18 (night) — curated128 confirms the frontend (1 seed); 8-seed band under HEAD running

`curated128_s0` (git_rev `adaa231`) finished the full 1M: endpoint ce_t4 **0.6665**, ce_t1 0.7154, mIoU_t4
44.86 — inside the old 8-seed band's per-seed spread (0.6644–0.6667). So the single-`Frontend` arch reproduces
the band (the mid-run "t1 lead" was noise that settled to the mean). One seed isn't a band → an **8-seed band
under HEAD** (`seedband_s0..7`, `throwaway/seed_band.py`) is now running on a free GPU (seed 0 smoke clean,
t0-floor reproduced). When it lands: aggregate (`tools.seed_report --prefix seedband_s`), and if ≥ the old band
tag `result/qpolicy-8seed-band` + update `docs/old_frontend_band_results.md` (the canonical band record for HEAD,
since the old band's ckpts load only at their tag). Liveness audit this session removed the dead qcorr subsystem
and 4 dead throwaway scripts; no other dead subsystem found.

## 2026-06-19 — HEAD 8-seed band landed (beats EG-C2F); action analysis; sweep relaunched

The 8-seed HEAD band (`seedband_s0..7`) completed. Deploy = best-mean(t1–t4)-CE per seed: val_ce
**0.7141/0.6881/0.6743/0.6654 (±0.0005)**, mIoU 42.74/43.85/44.51/44.87. Beats measured EG-C2F-c64 on CE at
every t (t4 margin 0.0053 ≈ 10× band σ) and on mIoU at every t. Tagged `result/qpolicy-8seed-band`; recorded in
**`docs/head_band_results.md`** — a NEW doc, NOT `old_frontend_band_results.md` (that stays the old-frontend
record; the 2026-06-18 plan to reuse it was superseded). When it landed the supervisor freed the GPU → relaunched
the perpetual sweep (study `qpolicy_c64_t5_ce`).

**Action analysis** (`docs/sessions/2026-06-19-action-analysis-coarse-to-fine.md`; new tools `tools/action_analysis.py`,
`q.eval --mask-scales`, `throwaway/q_calibration|q_reward_landscape|q_reward_landscape_traj.py`, `q.train --dump-init`):
the deployed policy places coarse, fairly-central glimpses image-adaptively; **coarse→fine emerges in 6/8 seeds but
is not causal** — 2/8 win staying coarse and masking the fine scale at deploy costs ≤0.0004 ce_t4 (< band σ). The
t1 reward landscape is coarse+center-biased (best candidate coarse ~90%); along the rollout coarse keeps the mean
but the fine-is-best share rises 9%→36% t1→t4 over a shrinking-reward regime. Q is real but weakly calibrated
(t1 Spearman ~0.29, ~⅓ of the best-of-512 oracle, replicates s2/s3/s4) and its selection quality declines along
the rollout (Spearman t1 ~0.29 → t4 ~0.12, replicates s3/s4/s5; stayer≈zoomer).

## 2026-06-24 — perpetual sweep post-mortem: recipe is on an HP plateau (sweep stopped)

Stopped the perpetual sweep (study `qpolicy_c64_t5_ce_f1000000`, 245 trials: 30 COMPLETE / 202 PRUNED / 12 FAIL).
**Confirmatory-negative:** `val_ce_t4` across the 30 completed trials = 0.6640 (best #31) / 0.6653 / 0.6659; the
honest noise check is **cross-trial std (0.00050) == 8-seed band σ (0.0005), ratio 1.0** — 30 different configs
vary by the same amount as re-seeding one, so the HP effect is indistinguishable from seed noise. Best #31 beats
the median by 0.0013, within winner's-curse for the luckiest of 30 noise draws (~0.0010). Capacity irrelevant
(w64≈w256, bl2≈bl3); CE↔mIoU cross-trial Pearson −0.12 (decoupling persists). **No reliably-bad/unstable region:**
prune rate ~80–91% uniform across all HP values (189/202 pruned at first eval — pruner aggressive on a flat
landscape, not a bad-region detector); 12 fails are sporadic infra (0 at width 256). **Actionable positive:** this
recipe converges within ~160k forwards (16% of the 1M budget reaches within 0.0005 of final) → future HP search
should run at ~250k fwd/trial, reserve 1M for confirmation only. Full record + verifications:
`docs/sessions/2026-06-24-perpetual-sweep-postmortem.md`. Flagged: crockett disk 99% (38 GB of sweep checkpoints;
pruned-trial ckpts the reclaim target — not yet deleted). Seed-replication of #31 declined by user.

## 2026-07-03 — objective = mean(t1–t4) val CE lands in code (audit session)

Repo audit (session `docs/sessions/2026-07-03-audit-mean-ce-objective.md`). The 2026-06-24 user directive
("mean(t1–t4) val CE is the only thing I care about") had been recorded in the post-mortem + memory but never
applied: `q.train` still set `objective = val_ce_t4`, so best.pt, optuna ranking, and pruning selected on the
endpoint-only metric. **Fixed:** objective = mean val CE over t1..t{train_horizon} — best.pt now coincides with
the deploy-ckpt rule (`seedband_io.best_mean_ckpt`); `tools.sweep_report` re-anchored on per-t `val_ce_t*` keys
so old and new runs rank under one rule; pre-change optuna studies must never be resumed (objective drift).
Secondary: `keep_every % eval_every == 0` now asserted (was a silent no-step-ckpt failure), wasted full-val t0
scoring in `evaluate_q(selection=False)` removed, several wrong comments + stale doc pointers fixed (CLAUDE.md
sweep status, README band pointer). No training-math change; `uv run just` green.

## 2026-07-03 (cont.) — deep study: BN train/deploy argmax gap; per-depth supervision quantified

Session: `docs/sessions/2026-07-03-deep-study-bn-probe-depth-diag.md`. Probes at `14a9a59`.

- **The DAgger rollout doesn't follow the deployed policy:** rollout action selection runs the net in
  train() (Frontend BatchNorm batch stats), deploy runs eval() (running stats) — the two modes' argmax
  agrees on only **~25–33% of val images** (3 band seeds, t0 and t1) even though their Q maps correlate at
  Spearman ~0.9. On disagreements the chosen actions' TRUE rewards differ by |Δ|≈0.05 with signed mean ≈0:
  the Q top is a broad near-tie, so the argmax identity is fragile while its reward level is not. Open
  cheap A/B: eval-mode action selection in `rollout_samples` (exact DAgger).
- **Per-depth supervision** (band runs' logged `fracstd_d/corr_d/predstd_d`): within-depth reward spread
  halves d0→d3 (0.085→0.044) → the pooled global-z weights depth-0 discrimination ~4× depth-3
  (variance shares ~49/23/15/13%); per-depth fit declines only mildly (corr 0.44→0.36);
  `predstd ≈ corr×σ` at every depth (no output compression); global-z calibrated (target_std ≈0.97).
- **Contracts pinned:** `run_episode` step-order/state semantics match the stateful encoder exactly;
  `grid_sample` safe-box readout alignment verified numerically to 6e-8; fresh run `qrecipe_s8` confirms
  `objective` == mean(t1–t4) live and reproduces the frozen t0 anchor (0.7649) digit-for-digit.

## 2026-07-03 (evening) — recipe v-next lands: 0.6856 mean-CE at 64% of the old budget; band overnight

Full arc in the session doc. The recipe is now: dueling head + deploy-mode ε-greedy rollouts + per-depth
reward z + mean(t1–t4)-CE objective + lr 2e-4, 640k forwards (8k steps), 1k warmup then hold (`bcb9742`).
Single-seed evidence: `qlr2e4_s0` deploy 0.6856 mean-CE / 44.98 mIoU_t4 = the old 1M-forward 8-seed band's
level at 64% compute; lr 3e-4 plateaus ~0.001 higher regardless of the 4k budget (`qflagship_s0`). CE↔mIoU
"ρ≈0.26" corrected (restriction-of-range artifact; pooled ρ=−0.99, `scripts/ce_miou_scatter.py`). Hub:
`canvit/qpolicy-ade20k-c64-t5-2026-07-03` (public) serves the deploy ckpt via
`ViewpointQNet.from_pretrained` (release-stack hub mixins). 8-seed band `qband_s*` running on bare
defaults; report via `tools.seed_report` (now also THE comparison-figure tool, `--overlay`). Dead-era
ckpts pruned (~34 GB; every metrics.jsonl kept — 488 runs).

## 2026-07-03 (afternoon) — exact-DAgger + dueling become the recipe; reward-transform structure; big clean

Session (all numbers + provenance): `docs/sessions/2026-07-03-deep-study-bn-probe-depth-diag.md`.

- **`rollout_act_eval=True` default** (rollout argmax under deploy-mode BN): 3-seed paired 80k-forward A/B
  vs the historical train-mode selection is a clean null (0.6906±0.0005 vs 0.6907±0.0006 mean(t1–t4) val
  CE; paired deltas ≤0.0006 both signs) — costs nothing, buys the honest method description.
- **`dueling=True` default** (`Q = V(s) + mean-zero A`; V = MLP on mean-pooled input features; deploy
  argmax unchanged, pinned by test): real-net overfit ladder on TRUE 512-grid maps, 4 seeds — Spearman@800
  0.495±0.015 vs plain 0.476±0.012 (the n=2 read of +0.04 shrank to +0.02; an MLP-toy rung did NOT
  transfer). The paired 1k-step training A/B is the standing confirmation. Ckpts record `dueling`.
- **Reward-transform structure** (no training; cached 2026-06-12 candidates + fresh 512-grid maps): the
  scene-scale law is σ ∝ base^0.95 — frac (÷base) is already the right rescale (8×→3× nonuniformity);
  the between-scene MEAN is ~0.44–0.45 of global-z target variance (rescaling can't touch it — the dueling
  V is the K=1-compatible attack); scenes at the CE measurement floor grow 14%→29% over t2→t4. Plot:
  `outputs/reward_transform_c64.png`. The frac form generalizes label-free as fraction-of-remaining-gap.
- **`qdefault_s0`** launched: the clean 1M checkpoint of the new recipe (both defaults on), seed 0.
- **Cleanup/restructure** [user]: riid t0 + augment removed (with the duplicate t0 builder and the dinov3
  transform machinery); `q` split one-concern-per-module (features / net / rollout / train / train_eval /
  viz); deploy rule + rowwise_spearman deduped into single sources; README rewritten in plain prose with
  GFM math; module docstrings no longer claim the frozen regime as their own property.

## 2026-07-04 — qband lands: the 2026-07-03 recipe is 8-seed-confirmed at 64% of the old compute

Full numbers + provenance: `docs/qband_results.md`. `qband_s0..7` (640k forwards each, bare `QConfig`
defaults, one training-code version across the band): deploy mean(t1–t4) val CE **0.6853±0.0007**, per-t
CE 0.7143/0.6878/0.6741/0.6652, mIoU_t4 44.97±0.10 — statistically identical to the 1M-forward HEAD band
(0.6855±0.0004 / 44.87±0.12) at 64% of its training compute (~80 min/seed incl. nine full evals), and
beats EG-C2F-c64 on CE and mIoU at every t (t4 CE gap ≈7σ). Deploy selection (best-mean steps 4k–8k)
buys ~0.001 CE over last-step. Tag `result/qband-8seed-640k`. CLAUDE.md validated-result paragraph and
README reference-result line now point here; `head_band_results.md` stays the 1M reference.

## 2026-07-05/06 — the actor era: PG objective, entropy floor, 3-seed band parity; bicubic reopens pathwise

Full record: `docs/sessions/2026-07-05-e2e-diff-single-image-probe.md` (one doc, both days). Landmarks,
with commits: objective sum type QReg|PG + one trainer (merge `cd70632`); SAC-style entropy floor
`entropy_target` (`8d34cdf`, windup fix `4a24a3d`); **pgfloor_s0/s1/s2 (floor 1.0, 640k) =
0.6857/0.6859/0.6853, band 0.6856±0.0003 vs qband 0.6853±0.0007 — the score-function actor is
band-equivalent to Q-regression, per-t CE within qband sigma at every t.** Floor-0.5 sweep winner
(trial0012, 0.6862@400k) FAILED 640k confirmation (0.6877) — short-budget artifact. Published:
pgpolicy-...-2026-07-05-s0, -floor-2026-07-05-s0, -2026-07-06-sweep-trial0012 (Hub). Estimator
program: bicubic grid_sample removes bilinear's derivative ripple (8x pointwise-grad cut, CE values
unchanged) -> single-scene reparam works (0.479->0.390), race estimator parity, sub-cell refinement
below the grid ceiling (train-time; needs mask); amortized pathwise on real data fails at both
apertures with a Gaussian head while matched-jitter REINFORCE ties it at 0.75 — landscape roughness +
policy class bind, not plumbing (sanity ladder passed). AVA archeology cross-check: their pathwise
successes were smooth-loss/trained-frontend/large-aperture; their small-patch runs failed like ours.
