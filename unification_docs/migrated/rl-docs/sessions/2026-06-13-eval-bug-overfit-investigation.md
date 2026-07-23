# 2026-06-13 — RWR eval bug, retraction, and the overfit-train investigation

Session goal (user): efficiently train policies that ACTUALLY learn on ADE20K,
stably, quickly, and beat EG-C2F. Two horses: RWR/AWR and GRPO.

## 1. The overnight "+0.41 over EG-C2F" was a measurement artifact — RETRACTED

The RWR trainer's internal eval ran **fp32**; the canonical evaluator
(`evaluate.py`, = paper protocol = the 41.0 anchor) runs **autocast(bf16)**.
The policy-independent t0 isolated it (`throwaway/eval_amp_probe.py`): t0 mIoU
**38.578 fp32 vs 38.533 bf16**. So the trainer read ~0.4 pp high at t1; the
overnight 41.45 was inflated.

Fix `430a526`: extracted shared helpers into `src/canvit_pytorch_rl/rollout_eval.py`
(`candidate_ce`, `full_scene_state`, `mle_weights`, `evaluate_rollout` — renamed
from the misnomer `evaluate_pathwise`); `evaluate_rollout` now mirrors the
canonical precision (bf16 backbone/canvas forwards, fp32 probe head). Verified
to the digit (`throwaway/verify_eval_rollout.py`): t0 38.5328, t1_mode 41.0618
== canonical run `rwr_long_basemode_t2_c32`. Also `fb7bdb5`: RWR checkpoints now
embed `git_rev` (were unloadable by evaluate.py + violated provenance); backfilled
pre-fix checkpoints from the run manifest (`throwaway/inject_git_rev.py`).

## 2. Corrected RWR result (run `rwr_ade_long_t1_c32`, 50k steps, lr3e-4/tau0.3/k32)

c32 full val, paper protocol, canonical `evaluate.py --policy actor_proposal`;
paired per-image bootstrap (10k) vs `runs/egc2f_t5_c32` (`throwaway/paired_t1.py`):

| RWR-flow deploy | t1 dataset mIoU | paired vs EG-C2F |
| base_mode (det. mode)    | 41.06 | +0.32 per-image [+0.11,+0.54]; +0.02 dataset = TIE |
| advantage K=32 (logpi-logmu) | 40.67 | -0.31 [-0.51,-0.11] = LOSS |
| EG-C2F | 41.04 | 0 |
| critic-greedy (2026-06-12) | 41.47 | leads |

The flow's mode only TIES EG-C2F; its self-advantage ranker LOSES (worse than its
own mode). See milestones.md 2026-06-13.

## 3. WHY it barely learns (diagnostics on best_miou.pt)

- `throwaway/flow_ranking_diag.py` (val, K=32): **Spearman(advantage, -CE) = 0.15**
  (dedicated critic ceiling ~0.30); top-1 CE regret of adv-argmax = 0.079 nats
  (worse than EG-C2F's own 0.058) → explains the advantage-deploy loss.
- **Mode collapsed to a near-constant central glimpse**: across 2000 val images,
  base_mode center_y std **0.116**, center_x std **0.082**, scale 0.586±0.039
  (EG-C2F y-std ~0.58). It ties EG-C2F by reproducing a generic fixed glimpse, NOT
  by conditioning on the image.
- `throwaway/reward_vs_density.py` (corr of -CE landscape vs flow logpi over a
  viewpoint grid): inconsistent — train[0]@0.3 +0.557 but most images/scales
  ~0.05-0.2, one negative. Density only weakly/unreliably tracks the reward.

## 4. THE TARGET: t1 selection margin is HUGE (verified, was being defeatist)

best-of-K true-CE oracle, c32 (`{val,train}cand_seqoracle_t5_c32/summary.json`):
val t1 **44.05** (+3.0 pp over EG-C2F 41.04), train t1 **49.80**. A smart policy
has +3 pp (val) / huge (train) headroom AT t1 — selection, not horizon (oracle t1
44.05 > EG-C2F t4 43.17). Burned into CLAUDE.md top (`30e73f5`), with the
OVERFIT-TRAIN-FIRST sanity: a policy that can't overfit train toward the oracle
is broken training.

## 5. Investigation in progress

`throwaway/overfit_train.py` (--method rwr|grpo): fresh flow, overfit a fixed
N-image train batch, HP-swept (lr × tau / G), report mode/best-of-K train mIoU vs
the RIID-best-of-128 oracle ceiling on those images + mode spread (conditioning).
Question: can ANY HP config push the mode toward the oracle and grow the spread? A
failed run != can't fit [user] — hence the sweep. [RESULTS PENDING — fill in.]

## 6. Overfit results (the gate): RWR FITS, GRPO DIVERGES

`overfit_train.py`, fresh flow, N=16 fixed train images, K=16, paper-protocol bf16
eval (`eval_set`, same primitives as the verified `evaluate_rollout`).

- **RWR FITS — conditioning VERIFIED real** (config lr3e-4/tau0.15, N=16 seed0):
  t0 floor **28.09**, best-CONSTANT-glimpse (one viewpoint for all 16) **31.98**,
  per-image best-of-128 oracle **34.58**; RWR-trained **mode 28.5 -> 35.2 (step1250)**,
  *above* the discrete oracle and **+3.2 pp over the best constant** -> the gain is
  NOT achievable by any single fixed glimpse, so the flow genuinely conditions
  per-image (mode center-spread grows 0 -> ~0.15-0.19). [control in
  `overfit_train.py`; mIoU via the canonical-verified `eval_set` primitives.]
- **GRPO DIVERGES as implemented** (lr3e-4, G=32, on-policy `-(adv*logpi).mean()`,
  z-scored within-group advantage): loss explodes (-1e8 -> +3e12), mode DROPS to
  ~27, **mode-spread stays 0.000** (never conditions). The score-function term is
  unbounded — on-policy logpi of the flow's own samples blows up as the policy
  moves; grad_clip=1.0 bounds the step but not the divergence. FIX candidates
  (untested): much lower lr, normalize/standardize logpi, entropy floor, or
  trust-region. GRPO needs stabilization before it's a contender. [the 2nd horse]

INSIGHT for future-me: the architecture is NOT the bottleneck (RWR overfits,
proposer+critic beats EG-C2F). The two real problems are (a) FULL-TRAIN
conditioning (the 50k run used tau0.3 = too soft -> mode collapsed to a constant;
overfit at tau0.15 conditions -> try sharper tau + more capacity/steps on full
train, watch val_mode_*spread the new canary), and (b) a SELF-SELECTOR as good as
the critic (flow advantage Spearman 0.15 vs critic 0.30). Fastest beat-EG-C2F path
already in hand: flow-proposes + critic-ranks (41.50, CI-real).

## 7. WHY full-train conditioning fails — grad-norm + trajectory diagnosis [user: check curves/grad-norms]

From the full-train HP sweep (`fulltrain_hp_sweep.py`), reading per-eval grad
norms + spread/mIoU trajectories (not endpoints):

- **The conditioner is GRADIENT-STARVED.** Post-clip per-module grad shares,
  full-train: `grad_norm_flow ~= 1.00` (the flow head absorbs ~99% of the
  gradient), while the conditioner that produces the per-image context gets ~1%:
  `head ~0.015, down ~0.001, trunk ~1e-4, proj ~1e-4, ent_fuse ~1e-4`. So the flow
  tunes its UNCONDITIONAL density; the conditioner barely updates -> context ~=
  constant -> mode ~= constant -> no conditioning. THE likely root cause of the
  full-train mode collapse.
  - Hypothesized mechanism: the MAF context pathway is zero-initialized
    (`flow_head.py`: `nn.init.zeros_(...final_layer.weight)` for identity init).
    Zero final-layer weight => context has zero forward effect AND ~zero gradient
    back to the conditioner at init; on 16 overfit images the consistent gradient
    still turns it on, on diverse full-train it's averaged out and stays off.
  - FIX TO TEST: small NON-ZERO context-pathway init (keep near-identity but give
    the conditioner gradient from step 0) and/or a higher conditioner LR.
  - ROOT CAUSE CONFIRMED locally (instantiate CanvasActor, feed 8 random states,
    measure logpi std of a fixed action ACROSS states): cis=0.0 -> std **0.0000**
    (flow EXACTLY context-blind at init => zero gradient to the conditioner);
    cis=0.01 -> 0.0028; cis=0.1 -> **0.092**. So zero-init makes context have zero
    forward effect AND zero backward gradient; `context_init_scale>0` (added,
    flow_head/actor/rwr_train, default 0.0) restores context-sensitivity + gradient.
    GPU `fulltrain_hp_sweep.py --vary cis` tests whether this turns on conditioning
    during full-train (watch grad_norm_trunk rise from ~3e-4 + yspread grow).
- **tau has a sweet spot** (~0.1-0.15): tau 0.3 (50k run) too soft -> collapse to
  constant; tau 0.05 too sharp -> the flow over-sharpens and `base_mode` (computed
  via the flow INVERSE path) blows up (a few inf/nan modes; val_mode_yspread
  artifact 5.58 while sample-path u_sat stays ~0; val mIoU dips 41.12->40.97).
- **`base_mode` is a fragile deploy readout** for sharp flows (inverse-path clamp-
  edge explosion). Prefer a forward-path / sample-based deterministic readout. The
  val_mode_*spread canary should be made inf-robust.

## 8. Full-train conditioning is SLOW & SEQUENTIAL — not collapse [user insight, confirmed]

Reading WITHIN-run trajectories (not endpoints), full-train cis0 control, tau0.1,
1000 steps: `grad_norm_trunk` 3e-5 (s200) -> 5.6e-3 (s500) -> 7.7e-3 (s1000)
[~250x growth]; `val_mode_yspread` 0 -> 0.022 -> 0.046; `grad_norm_flow`
0.9998 -> 0.9970. So the conditioner gradient is RISING and conditioning is just
beginning at 1000 steps. [user]: "expect initial learning to reap UNCONDITIONED
behavior first (easier), then become CONDITIONED (harder) — easier then harder."
The flow learns its easy unconditional density first (flow dominates the gradient
early); as that saturates the conditioner warms up and conditioning emerges. So
1000-step full-train runs are FAR too short to judge conditioning — do NOT read
the early mode-plateau as a structural collapse. The 50k tau0.3 run reached
yspread ~0.1 (still low, but it WAS growing). The `context_init_scale` fix is
expected to ACCELERATE the conditioner warm-up (start its gradient at step 0
instead of waiting for the flow's final-layer weight to grow).

TEST RUNNING: controlled A/B at tau0.15, lr3e-4, 8000 steps, eval_every 500 —
`rwr_cond_cis0_tau015` (control) vs `rwr_cond_cis01_tau015` (fix, cis0.1). Watch
whether cis0.1 reaches healthy yspread (~0.3-0.6) + mode>41.5 FASTER than cis0.

FIX MECHANISM CONFIRMED on full train (early grad-norm A/B, both tau0.15 lr3e-4):
cis0 control `grad_norm_trunk` ~6e-5 -> noisy ~1e-4..0.04, `grad_norm_flow` ~0.999
(conditioner ~0.01% gradient share early); cis0.1 fix `grad_norm_trunk` STEADY
**0.04-0.10 from step 25**, `grad_norm_flow` 0.74-0.83 (conditioner ~5-10% share
from step 0). So context_init_scale=0.1 un-starves the conditioner ~100-1000x as
intended. Cost: cis0.1 starts at mode 39.93 (context-init perturbs the good
center-glimpse init); OPEN whether conditioning recovers + exceeds 41.12 by 8000
steps. Killed the control once the contrast was clear; cis0.1 now runs full-GPU
to its answer faster. If un-starvation still doesn't condition, the next lever is
per-param-group LR (conditioner >> flow) or the issue is deeper (conditioner can't
extract a GENERALIZING signal from t0 canvas features -> large-scale pretraining).

UPDATE (cis0.1, step 1000/8000, full-GPU): mode recovered from the init
perturbation (39.93 -> 41.18, marginally over the 41.12 baseline) but yspread
stays low/fluctuating (~0.04-0.10, not climbing to ~0.3-0.6) and `grad_norm_trunk`
dropped 0.05-0.10 (early) -> ~0.005 (step1025): the cis-init boost looks TRANSIENT
(flow re-dominates as it learns its unconditional density). => built+testing the
SUSTAINED lever `conditioner_lr_mult` (param-group LR, default 1.0). Running A/B:
`rwr_cond_cis01_tau015` (cis only) vs `rwr_cond_cis01_lrm10_tau015` (cis0.1 +
conditioner_lr_mult 10), tau0.15 lr3e-4 8000 steps, concurrent. NOT concluding
cis0.1 (7000 steps left; conditioning is slow). Watching mode>41.5 + healthy
yspread + sustained gn_trunk.

## 9. THE deeper diagnosis: the conditioning GRADIENT vanishes (not an init/lr problem)

Reading `grad_norm_trunk` (the conditioner's GRADIENT norm, computed pre-optimizer
-> lr-mult-independent) on full train: even with conditioner_lr_mult=10 + cis0.1,
it decays 0.097 (s25) -> 0.014 (s125) -> 5e-4 (s175) -> **1e-5 (s375)**. The
cis-init only DELAYED the decay. So neither lever sustains conditioning because the
underlying gradient VANISHES — lr_mult amplifies zero.

MECHANISM (hypothesis, well-supported): RWR fits the flow to the per-state
softmax-weighted candidate distribution. Once the flow's UNCONDITIONAL density
matches the AVERAGE good-viewpoint distribution, the objective is nearly satisfied
and the marginal benefit of conditioning (hence its gradient) is tiny. On 16
overfit images the per-image optima differ enough to keep that gradient alive
(conditioning emerged); on full train (CE is ~99% inter-scene variance) the average
absorbs it -> conditioning gradient -> 0. So full-train RWR conditioning is NOT an
init/lr bug — the conditioning SIGNAL is weak in the objective at scale.

IMPLICATION / redirected plan (away from the now-futile init/lr levers):
- DATA-SIZE / CURRICULUM: smaller data preserves per-image signal. `subset256` run
  (256 imgs, same levers) is the key test — does it condition + GENERALIZE to full
  val? If conditioning survives at 256 but not full, curriculum (grow data while the
  conditioner stays engaged) is the route.
- OBJECTIVE that REQUIRES conditioning: e.g. penalize the unconditional solution /
  reward context-action mutual information, or sharpen tau (more per-state target
  spread — but base_mode saturates, needs the forward-path readout fix first).
- Or the per-image signal in frozen t0 features is genuinely weak -> the user's
  large-scale policy-pretraining direction.

## 10. Data-size route is a DEAD END; the fix is a non-vanishing conditioning signal (aux head)

Data-size sweep of the conditioning gradient (gn_trunk, cis0.1+lrm10, full val):
16 imgs -> grew to 0.05+ (conditions, mode->oracle); 64 imgs -> ~0.0015 (decaying);
256 -> ~0.0007; full -> ~1e-5. So the conditioning-gradient threshold is BETWEEN
16 and 64 — far too low for a curriculum to generalize (you can't generalize from
<64 imgs). Curriculum/data-size route ABANDONED.

THE FIX (built + launched): an auxiliary context->action head (`aux_mode_coef`,
actor.aux_predict) regressed by MSE to the per-state best-CE candidate (already
computed each RWR step, free). Its SUPERVISED gradient is non-vanishing (unlike
RWR's marginal-matching), flows into the conditioner -> un-starves it durably at
ANY scale and forces the context to encode per-image "where to look". Bonus:
aux_predict is itself a direct conditioned-viewpoint regressor — a candidate
self-contained deterministic deploy (behavior-clones the oracle viewpoint).
RUNNING: `rwr_aux1_tau015_full` (full train, tau0.15, aux_mode_coef 1.0, 8000
steps). Watch: gn_trunk STAYS high (not decaying), aux_loss drops, and val
mode/sample rise above the 41.12 tie (conditioning that GENERALIZES). This is the
most direct shot at full-train conditioning; decisive read ~step 1000-1500.

## 11. aux head conditions the CONTEXT, but a cis=0 flow ignores it (need aux + cis>0)

`rwr_aux1_tau015_full` (aux_mode_coef 1, cis0.0): aux WORKS on its own terms —
aux_loss 0.49->0.055, gn_trunk sustained ~30x the vanishing baseline. BUT the
flow's base_mode stays CONSTANT (val yspread 0.0001, mode ~41.0) because cis=0
leaves the flow's MAF context weights zero-init = context-blind: the aux makes the
context informative, but the flow doesn't READ it. So conditioning lives in
`aux_predict` (the direct context->action head, deployable via actor_rank=aux),
NOT the flow's mode. For the FLOW to condition self-contained: aux + cis>0 (aux
informs the context, cis>0 lets the flow use it). LAUNCHED `rwr_auxcis_tau015_full`
(aux_mode_coef 1 + context_init_scale 0.1). Caveat: aux_loss plateaus ~0.055
(RMS ~0.23/dim) — the per-state best-of-32-random target is noisy AND frozen-t0
features may carry only weak per-image "where to look" signal (the fundamental
limit -> the user's large-scale-pretraining direction). Evaluate BOTH the flow
mode and aux_predict deploys vs EG-C2F once trained.

## 12. VERDICT: self-contained RWR TIES EG-C2F on frozen features (a real result, not a tuning miss)

Across `rwr_auxcis_tau015_full` (aux_mode_coef1 + cis0.1) steps 500/1000 (val=full):
flow yspread 0.053->0.043 (does NOT rise to healthy ~0.3-0.6), mode ~41.0 (~=
EG-C2F 41.04 baseline), gn_trunk decaying 0.05->0.01, aux_loss plateaued 0.053.
Step-500 deploy-eval (canonical, paired vs egc2f_t5_c32): aux_predict t1 41.20,
paired +0.038 CI [-0.17,+0.26] = TIE. The flow conditioning does not develop
strongly; the aux head modestly helps (aux_predict 41.2 vs flow mode 41.0) but
still ties.

CONCLUSION (well-supported by the full chain): with FROZEN perception, a
self-contained RWR policy TIES EG-C2F at t1 and does not strongly beat it. The
binding constraint is that the per-image "where to look" signal in frozen t0
features does not GENERALIZE (the same conditioner memorizes 16 imgs to the oracle
but plateaus at aux_loss 0.053 / yspread ~0.04 on full train). A strong
self-contained beat requires either (a) the SELECTION route — flow proposer + a
discriminative critic, which DOES beat EG-C2F CI-real (41.50, +0.31 [+0.11,+0.53])
— or (b) representation pretraining so the features carry generalizable per-image
signal (the user's hypothesis). This is a negative result for self-contained
conditioning-on-frozen-features, paired with the positive proposer+critic result.

ANSWER to "how to train policies that beat EG-C2F efficiently": use the SELECTION
recipe (flow proposes K, discriminative critic ranks) — demonstrated, CI-real,
~1M-forward budget. Self-contained single-network conditioning is the open frontier
gated on representation pretraining.

## Waste ledger / lessons

- This 24GB GPU is ONE-HEAVY-JOB-AT-A-TIME: co-running tuner + overfit + an eval
  OOM'd twice (512px canvas forwards at batch 48 alloc ~7 GB). Keep <=2 light jobs;
  a K=32 candidate-CE training step at batch N=32 is ~4x the 50k run's batch-8 step.
- `pgrep -af <pat>` self-matches the ssh shell when the shell's own command string
  contains <pat> (e.g. the pattern appears in a piped `grep`); verify a kill with
  `ps -p PID`, never trust pgrep's listing for the kill decision.
- Tuner restarted on the calibrated metric (study `rwr_hpo_ade_amp`), then KILLED
  to give the overfit investigation the full GPU; relaunch a TARGETED tuner once
  the right objective is known. Monitoring cron `3a1af83b` deleted (active driving).

## COMPACTION HANDOFF (2026-06-13, late) — read this first on resume

GIT: local HEAD `08b87eb` (AWR weighting), tree clean, origin+deploy synced through it.
All code changes below are COMMITTED. crockett checkout was synced to HEAD before
the VPN cut ssh (a few late commits may need a `git pull --ff-only` on crockett).

### Current goal (active /goal stop-hook)
"Get a clean, NO-CRITIC actor (conditional normalizing flow, RWR/AWR) to train
cleanly/reliably/well. Full ADE20K train + val. OVERFIT before tackling
generalization." => first rung is: demonstrate the clean flow can OVERFIT a train
set (train-mode-gain >> val-mode-gain, climbing toward the slice oracle).

### Honest status: NOT yet demonstrated. The decisive run is training.
- DECISIVE RUN (should be alive on crockett, nohup, SURVIVES compaction):
  `rwr_ovf64_tau01_long` = clean RWR, no critic/aux, tau0.1, k16, --train-subset-n 64
  --train-eval-n 64 --steps 8000 (~1000 epochs; 64 imgs/batch8 = 8 steps/epoch),
  eval_every 500. Log /tmp/ovf64_long.log. Last seen ~step 675 (~84 ep),
  train_mode_gain +2.78 (baseline-level, PRE-TREND). 
- VERDICT TO READ at step ~2000-3000 (~250-375 ep): does `train_mode_gain` climb
  toward the **64-slice oracle gain +5.53** AND pull clearly above `val_mode_gain`
  (~+2.5), SUSTAINED? Yes => clean flow overfits a moderate set => scale to full
  ADE (many epochs). Plateaus ~+2.5 => capacity/optimization limit => fast 64-img
  sweep: `--weighting awr --tau 0.2`, or `--tau 0.05`, or `--hidden-dim 384`, or
  `--conditioner-lr-mult 5` (one change at a time).

### SESSION-ONLY state that DIES on compaction (recreate if continuing autonomously)
- cron `5a1bd45e` (overfit monitor, fires :18/:48) — GONE after compaction.
- background waiter `b38zgcl5u` (waits ovf64 step 2500, prints trajectory) — GONE.
- After compaction + VPN/ssh restored: reconnect, parse runs/rwr_ovf64_tau01_long/
  metrics.jsonl (fields: train_mode_gain, val_mode_gain, train_val_miou_t1_mode,
  val_miou_t1_mode), decide per the verdict branches above, and recreate a monitor
  cron if running unattended. crockett nohup runs persist; only the cron/waiters die.

### Oracle / reference targets (c32, paper protocol, all measured this session)
- EG-C2F t1 = 41.04 (run `egc2f_t5_c32`). val t1 oracle (best-of-K) = 44.05; train
  t1 oracle = 49.80. 64-slice (first 64 train) oracle GAIN +5.53 (t0 19.72->25.25).
  256-slice oracle GAIN +6.26 (t0 23.17->29.43). Slice t0s are LOW because the first
  train images are hard — ALWAYS use GAIN over t0, never raw mIoU, across slices.

### Earlier goal ACHIEVED (committed, milestones.md): beat EG-C2F
- Proposer+critic: flow proposes K, separate critic ranks. t1 41.50 vs 41.04,
  paired per-image +0.31 [+0.11,+0.53] (CI excludes 0).
- SINGLE self-contained network also beats it: `runs/merged_selfcontained/actor.pt`
  (one CanvasActor = flow + in-module critic head; built by
  `throwaway/merge_selfcontained.py`; deploy `evaluate.py --policy actor_proposal
  --actor-rank self_critic`). Same 41.50, paired +0.31 [+0.11,+0.53].
- Pure self-contained CONDITIONING (flow mode / aux_predict) only TIES EG-C2F on
  frozen features (~41.0-41.2, CI straddles 0). User then steered to: drop the
  critic, make the actor-only flow OVERFIT first (the current goal).

### Code changes this session (all committed, all default-OFF / backward-compatible)
- `rollout_eval.py` NEW: extracted candidate_ce/full_scene_state/mle_weights/
  evaluate_rollout from train_actor (rwr_train no longer imports a sibling trainer).
  evaluate_rollout now matches the canonical evaluator's precision (bf16 backbone,
  fp32 head) — the eval-bug fix. NaN-robust mode-spread canary + val_mode_nan_frac.
  mle_weights gained weighting="awr" (exp(z/tau) clamped, unnormalized).
- `rwr_train.py`: git_rev in ckpts; train-slice under/overfit eval (train_eval_n,
  reports train_*/val_* + train_mode_gain/val_mode_gain); SEPARATE grad-clip for
  flow vs non-flow params (the flow was starving the conditioner/critic budget);
  levers context_init_scale, conditioner_lr_mult, aux_mode_coef, self_critic_coef
  (+critic_hidden/critic_mlp_hidden), train_subset_n.
- `actor.py`: aux head (aux_predict) + in-network critic head (critic_score,
  CandidateCritic, critic_n_freqs) — both optional, default off.
- `flow_head.py`: context_init_scale (small non-zero MAF context-init).
- `critic_policy.py` + `evaluate.py`: actor_rank in {critic,advantage,base_mode,aux,
  self_critic}; aux & self_critic clamp to the safe box.
- throwaways: paired_t1, eval_amp_probe, verify_eval_rollout, flow_ranking_diag,
  reward_vs_density, overfit_train, fulltrain_hp_sweep, train_vs_val_mode,
  merge_selfcontained, inject_git_rev, rwr_optuna_ade, rwr_watch_and_chain.

### LESSONS / corrections (do NOT repeat — these cost time this session)
- Judge overfit by MATCHED EPOCHS, not steps. 16-img overfit needed ~500 epochs to
  reach its oracle. A full-train 30k-step or 256-img-4k-step run is only ~12-125
  epochs = UNDERTRAINED, not "can't overfit". I wrongly concluded non-overfit from
  undertrained/short runs several times.
- NEVER read a trend off a single noisy eval point. I called "overfit onset" from one
  +3.56 that regressed to +2.05 next eval. Require a SUSTAINED climb across evals.
- Control slice difficulty via gain-over-t0 (train slices much harder than val).
- "the conditioning gradient cancels across images" was BOGUS (retracted) — that's
  not how conditional learning works; don't reach for hand-wavy mechanisms.
- Concurrent compute-bound runs HALVE each other (no throughput gain); decide
  deliberately, don't flip-flop. <=2 GPU jobs (3 OOMs at 24GB). 512px canvas
  forwards are memory-heavy; one heavy job ~5-12GB.
- Kill ONLY by explicit PID (ps -p verify); never pkill -f a pattern matching the
  ssh shell (exit 255); bracket-grep `[c]anvit`. crockett clock ~13.5min behind.
- The /goal stop-hook fired every turn for ~hours and I kept responding on
  wall-clock-gated experiments — burned a lot of compute re-stating "pending".
  When gated on a long run, set a background waiter / cron and YIELD; don't poll.

### SCALE-UP PHASE (2026-06-13 ~17:00 crockett, post-ovf64) — full-ADE-train overfit
ovf64 (64 imgs, cis=0, condlr=1, tau0.1, k16, soft) OVERFIT cleanly: final s8000
train_mode_gain +4.46 / val +2.44, peak +5.12 @ s7000 (~880 ep), sustained climb
above flat val from ~s5500. First rung MET on a moderate set. ovf64 finished, freed
a slot. Now two concurrent full-ADE-train runs (20210 imgs, batch8 -> ~2526 steps/ep;
60000 steps ~= 24 ep total -- NOTE: only 24 ep, vs the hundreds ovf64 needed, so a
weak/null full-train overfit in this budget is NOT proof of incapacity):
  - `rwr_ovf_full_tau01_k16` -- BASELINE, default recipe (cis=0, condlr=1).
  - `rwr_ovf_full_condlr10` -- `--conditioner-lr-mult 10`, sole change; tests the
    conditioner-starvation hypothesis (trunk grad ~1e-3..1e-2 vs flow clipped 1.0;
    Adam normalizes per-param so the 10x LR still moves the trunk ~10x/step).
Step-0 eval both: train~=val~=+2.6, t0=38.54 (= canonical anchor, eval precision OK),
val_mode_nan_frac 0. Baseline s2000 (~0.8 ep): train +2.76 / val +2.51 -- faint
correct-signed nudge, NOT yet overfit (single sub-epoch point; RULE ONE).
OVERFIT VERDICT = train_mode_gain pulling SUSTAINED above val across matched epochs
(as ovf64). Watcher armed at baseline step 8000 to dump both trajectories.
Other un-starve levers if condlr10 null: `--context-init-scale >0` (gives the flow
context-gradient from step 0 -- the SIGNAL lever, vs condlr's STEP lever),
`--aux-mode-coef`, sharper `--weighting awr`. One change at a time.
