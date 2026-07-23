# 2026-06-12 — Lever elimination, arch diagnosis, Codex-winner reproduction

Continues `2026-06-11-bootstrap.md`. All numbers c32 full val paper protocol
unless marked c64. Commits this session: 3ffb6b5 .. b14db1b (+ docs).

## Results (details in milestones.md)

1. **Seqoracle small-critic A/B** (chain at d824280): t1-only horizon
   collapse REPLICATES on oracle-advance states; the apparent 1.5 pp
   data-distribution deficit resolved to a **winner's-curse asymmetry**
   (grid-argmax costs the oracle-data critic 1.44 pp at t4, the
   EG-C2F-data critic 0.05 pp; randomk deploys nearly tie).
2. **20M boxent critic, data A/B** (`scripts/launch_boxent_ab.sh`,
   c6a79a4): best learned deploy yet —
   `critic_boxent_egc2f_randomk_t5_c32` 3-seed means
   41.22 / 42.16 / 42.57 / 42.91 vs EG-C2F 41.04 / 42.05 / 42.69 / 43.17.
   Paired per-image bootstraps straddle zero at every t (parity through
   t3; t4 deficit -0.11..-0.34, all seeds negative). EG-C2F-advance data
   beats oracle-advance at 20M under BOTH deploys.
3. **Levers eliminated at matched budget**: pairwise logistic loss = wash
   (11848d9); 2x data via dataset union = wash (fb58366, multi-run
   RolloutCandidateSet). 20x params bought val Spearman 0.250 -> 0.271
   only. ALL arms ceiling at val ~0.26 while train hits 0.45-0.51
   (overfit after step ~2000-3500; checkpoint selection lands at the
   peak — earlier "small fit gaps" read was mid-training, corrected).
4. **Action-only probe** (7addf3b): val Spearman 0.172 of the full 0.262 —
   the critic is ~2/3 action prior; state pathway adds ~0.09, decays with
   t, and carries all the overfitting. Deploys (c32 full val, randomk
   3-seed means): 41.02 / 41.57 / 41.93 / 42.09 at t1..t4 vs full boxent
   egc2f-data 41.22 / 42.16 / 42.57 / 42.91 — the state input is worth
   ~+0.2 pp at t1 growing to ~+0.8 pp at t4 in deployed mIoU, so the
   state pathway DOES matter at deploy time even though it's the smaller
   Spearman share.
5. **Train-eval slice bug** (6061b88): old strided slice aliased the
   t-cycle — pre-fix `train_*_all` metrics are t1/t3 mixes (t1-only for
   the union run); compare per-t keys only. Train-side comparison plot:
   `outputs/train_side_comparison.png` (local).

## User directives added this session (also in CLAUDE.md)

- Method comparisons read TRAIN-side metrics while the val ceiling stands
  (pre-overfit segment, per-t keys, t1 = only fully cross-dataset-fair).
  Plan: fix generalization later via large-scale pretraining (IN21K,
  cossim-style label-free targets). Suggested prep: record per-candidate
  cossim-to-reference at the next dataset generation to validate
  cossim-rank vs CE-rank cheaply.
- "Try to reproduce" + "prioritize repro": Codex winner reproduction is
  the priority job.

## In flight on crockett (queue order, single GPU)

1. `critic_boxent_actiononly_egc2f_c32` training + 3 randomk deploys —
   DONE 09:15 EDT (numbers above), log /tmp/rl_actiononly.log.
2. ~~Repro chain pid 3936946 (c64)~~ **KILLED 11:00 EDT on user pivot**
   ("don't do the training with c64 / make the codex arch work at c32"):
   both c64 dataset generations COMPLETED first (val 572 s, train 5764 s;
   data kept, renamed to `{val,train}cand_seqoracle_t5_c64` by waiter pid
   3983637) — only the c64 training/deploys were cancelled.
3. **Globalpool c32 chain pid 3983582** — log /tmp/rl_globalpool_c32.log:
   `critic_globalpool_zce_egc2f_t1only_c32` (--arch globalpool 25.07M,
   --train-t-filter 1, bs 32 x 30k steps, eval-every 2000, on the c32
   egc2f-advance datasets) -> `critic_globalpool_greedy_t5_c32` (uses
   the FIXED safe-box grid) + 3-seed randomk T=5. The t1 column is the
   Codex-recipe readout; t2..t4 diagnose t1-only horizon behavior.
   Mid-training read (steps 2k/6k/10k/14k): val_spearman_t1
   0.265/0.258/0.218/0.194, train_spearman_t1 0.270/0.307/0.437/0.584 —
   same ~0.26 val ceiling as every boxent arm, overfitting past ~2k;
   train-side learnability at 2k (0.270) is BELOW boxent-on-same-data
   (0.319 t1@2k). So far the winner arch shows no learnability edge on
   our offline data; checkpoint selection will land near step 2000.
4. **Entchan chain pid 3983711** — log /tmp/rl_entchan.log: re-queued
   behind globalpool c32; entropy-map-as-trunk-channel arm (712ac41) +
   4 deploys.
5. **Safe-box greedy re-deploy chain pid 3990696** — log
   /tmp/rl_safebox_redeploy.log, queued behind entchan:
   `critic_boxent_{egc2f,seqoracle}_greedy_safebox_t5_c32` re-run the two
   20M boxent critics' greedy deploys with the fixed (1-s)-scaled grid
   (3b9f2e7). Readout: if seqoracle-data greedy jumps from 41.58 toward
   its randomk 42.48 at t4, the winner's-curse asymmetry was the
   out-of-support grid, not the data. (Old-grid greedy numbers:
   egc2f 40.98/41.79/42.26/42.57, seqoracle 40.88/41.42/41.48/41.58.)
6. **Matched-K 2x2 completion chain pid 3996294** — log
   /tmp/rl_matchedk.log, queued behind 5. [user called out that
   grid-vs-randomk confounded candidate DISTRIBUTION with K=1024 vs 16
   — both rules are argmax; "greedy" naming is wrong.] evaluate.py now
   exposes --k-random/--grid-n (ac588a6). New cells for the boxent
   egc2f critic: `critic_boxent_egc2f_random1024_t5_c32_seed0`
   (R-IID K=1024, 1 seed — max-of-1024 has low draw variance) and
   `critic_boxent_egc2f_grid16_safebox_t5_c32` (2x2 centers x 4 scales).
   Existing cells: random-16 = randomk 3-seed runs; grid-1024-safebox =
   chain 5. Readout: K-effect (curse exposure vs coverage) separated
   from distribution-effect. TODO after queue drains: rename policies
   critic_greedy -> critic_grid, since both deploys are greedy argmax.

## Readout recipe for the globalpool c32 arm (when chain finishes)

USER PIVOT 11:10 EDT: no c64 training — run the Codex-winner arch at c32
("make your stuff close to the codex stuff that worked but for c32").
The c64 readout recipe is superseded; c32 equivalents:

- Compare `critic_globalpool_greedy_t5_c32` t1 vs EG-C2F c32 t1 = 41.04
  and vs our best boxent t1 (randomk 3-seed mean 41.22, greedy 40.98).
  No direct Codex anchor at c32 (their winner was c64-only); the question
  is whether the globalpool arch beats boxent on OUR protocol — train-side
  per-t metrics first (user directive), deploys second.
- t2..t4 from the same runs diagnose t1-only horizon collapse (expected
  from the d824280 A/B; if it collapses, retrain all-t).
- If globalpool > boxent at t1: arch (global pooling + multiplicative VPE
  fusion) is the ingredient — fold into the main line, consider all-t
  training next.
- If ~tie: arch eliminated at c32; candidate diversity (343k fixed targets
  vs their ~12M fresh) becomes prime suspect — regenerate c32 t1
  candidates with K=64+ (~25 min) before touching optimizer knobs.

## Reference numbers (c64, paper Table 4 / our runs)

EG-C2F c64: t0 39.602, t1 42.219 (our `egc2f_t21` matches paper exactly).
Codex c64 t1 (UNTRUSTED archive ledger, orig-mask protocol, their EG-C2F
42.231): winner = global-pool scene-z VPE critic over the 16x16x4 grid,
42.672 (+0.441 CI [+0.09,+0.76]) — online, 16 fresh R-IID cands/scene,
LR 5e-6 wd 1e-4 batch 1 scene, last.pt @ 964k total steps = 16.39M
forwards (NOT lr 1e-4 as previously mis-recalled — that was the
spatialcond flow run). rbf_pool variant 42.658 @ 294k (~6.3M forwards).
The "strict-1M" 42.44-42.46 rows are a DIFFERENT pipeline (spatialcond
flow + co-trained critic, critic_flow16 stochastic deploy, sd ~0.11);
the winner recipe was never run at a 1M budget. critic_random16 ~42.49
at 1M: critic selection dominates, flow proposals add nothing. Their
only c32 row: CE-finetuned actor 41.378 vs their c32 EG-C2F 41.064
(+0.314). Oracle best-of-17 c64 t1: 45.583.

## Archive code audit (2026-06-12, train_ade20k_t2_scene_z_vpe_critic.py
## + eval; verifying our globalpool transplant against the real winner)

Verified IDENTICAL (their code -> ours):
- Arch module-for-module: conv1x1+GN+GELU -> 4x ResidualConvBlock
  (conv3-GN-GELU-conv3-GN, post-residual GELU) -> global mean+max pool ->
  LN MLP state_proj; frozen VPEEncoder(rff 256, seed 42) -> LN MLP;
  fusion cat[state, vpe, sv*vpe, |sv-vpe|] -> LN head + 4x ResidualMLP ->
  LN+Linear. Param count matches theirs exactly: 25,074,945.
- Critic input: get_spatial(state.canvas).float() in both eras (their
  critic_input="features"); no LayerNorm before the conv (GN(1) inside).
- Target math: their scene_z(CE_t0 - CE_t1) == our z_targets(-CE)
  (constant shift per scene); loss = plain MSE on z. Training CE uses
  squish-512 masks in both eras.
- Candidate sampler: same upstream random_viewpoints, min_scale 0.25,
  no full-scene; centers always inside the safe box |c| <= 1 - s.
- Deploy: argmax over 16x16 centers x scales (.25,.35,.5,.7) on the
  current state's get_spatial.

DEVIATION FOUND AND FIXED (3b9f2e7): their deploy grid scales centers by
(1 - s) (make_action_grid) so all 1024 candidates are in-support; our
grid_actions used fixed +-0.94 centers at all scales — at s=0.7 ~90% of
candidates were OUTSIDE the safe box the critic ever trained on.
Grid-argmax was selecting extrapolations: prime suspect for the
winner's-curse asymmetry (greedy << randomk for our critics, while the
Codex winner's greedy worked). All greedy deploys before 3b9f2e7 used
the out-of-support grid; re-deploy before comparing greedy numbers.

Known remaining deviations (documented, accepted for this arm):
- Offline fixed candidates (17/image incl. EG-C2F cand, z over 17,
  revisited ~48x) vs online 16 fresh R-IID (z over 16, never repeated).
- Optimizer: ours lr 3e-4, bs 32 samples x 17 targets, 30k steps,
  warmup 10%; winner lr 5e-6, 1 scene x 16 targets/step, 964k steps,
  warmup 50%, wd 1e-4 (code default lr 1e-4 was overridden).
- Readout init: winner run used readout_weight_std 1e-3; their code
  default and our GlobalPooledCritic use 1e-4 (init-scale only).
- Protocol: their table is original-res-mask mIoU (our secondary "orig"
  column); our headline is paper squish-512.

## ALL THE BAD SHIT — complete error ledger (2026-06-11..12 sessions)

Every mistake found, its blast radius, and status. [user demanded this
list 2026-06-12 after the grid bug.]

1. **Out-of-support greedy deploy grid** (worst; mine, in the original
   CriticGreedyPolicy). Fixed grid centers ±0.94 at all scales vs
   training support |c| <= 1-s; in-support fraction 56/39/25/6 % at
   scales .25/.35/.5/.7. Wrote the deploy rule from prose, not from the
   reference source; never checked support. TAINTS all 8 pre-3b9f2e7
   greedy runs (listed in milestones) and CONFOUNDS the winner's-curse
   asymmetry interpretation. Status: fixed 3b9f2e7; safebox re-deploy
   test queued (pid 3990696); milestone correction written; lesson in
   memory (transplant-fidelity-discipline).
2. **c64 datasets silently misnamed egc2f** — the rollout_candidates
   streamline hardcoded oracle advance, deleting the behavior-advance
   mode the d824280 A/B later showed trains better-deploying critics,
   and kept writing "egc2f" names. Caught only by the base-trajectory
   sanity check (t1=45.41=oracle). Status: dirs renamed
   *_seqoracle_t5_c64; --advance restored + manifest records it
   (a05f3b9). Lesson: never bake a distribution-determining choice out
   of the config; names must derive from config.
3. **Winner-recipe optimizer mis-recalled to the user** ("lr 1e-4 wd
   1e-5") — actual winner run: lr 5e-6, wd 1e-4 (code DEFAULT was 1e-4).
   Status: corrected in chat + 01170cb.
4. **"Strict-1M arm" misattributed to the winner recipe** — the 42.44
   rows are the spatialcond flow+critic pipeline; the global-pool winner
   was NEVER budget-matched. Consequence: no evidence their recipe works
   at 1M; our 960k-forward arm has no like-for-like Codex anchor.
   Status: corrected in chat + 01170cb.
5. **Train-eval slice aliased the t-cycle** (pre-compaction; fixed
   6061b88): strided slicing of (image,t)-sorted items sampled only
   t1/t3 (or t1) — pre-fix `train_*_all` metrics are t-subset mixes;
   compare per-t keys only.
6. **Mid-training fit-gap misread as "underfit/data-limited"** — final
   gaps were 0.20–0.28 with val declining. Status: corrected; rule:
   read full curves, never mid-run snapshots.
7. **"Coverage pathology" misdiagnosis** — the deployed critic's narrow
   y-spread matched the oracle's own preference; low spread was not the
   deficit. Status: corrected in milestones.
8. **Unqualified numbers + overclaiming in chat** (train vs val
   unstated; "beating" when CIs straddle zero). Status: procedural;
   every number now carries metric+split+config.
9. **readout init deviation in the transplant**: winner run used
   readout_weight_std 1e-3; we use 1e-4 (their own code default).
   Accepted, init-scale only; documented in the audit section.
10. **VPEEncoder output-dim assumed 2x rff_dim** — actually rff_dim;
    caught by LayerNorm shape error before any run.

Still-open consequences: (a) winner's-curse milestone interpretation
pending the safebox re-deploys; (b) data-A/B greedy columns tainted —
randomk columns and the paired-bootstrap verdict stand; (c) t1-only
horizon collapse expectation means the in-flight globalpool T=5 deploys
are read at t1 (t2-4 are a diagnostic, not the headline).

## Gotchas added this session

- **c64 dataset misnamed** (caught via base-trajectory sanity check:
  t1 = 45.41 = oracle curve, not EG-C2F's 42.2): the streamlined
  rollout_candidates always advanced by argmin-CE, so
  `{val,train}cand_rollout_egc2f_t5_c64` was really SEQORACLE-advance
  data. FIXED 2026-06-12 (a05f3b9 + rename waiter): dirs renamed to
  `{val,train}cand_seqoracle_t5_c64`; `--advance {oracle,behavior}`
  restored (manifest records the mode). The c32 dirs named egc2f are
  genuine egc2f-advance (generated pre-streamline) — only c64 was
  affected.

- **crockett suspended overnight (~22:40 -> 09:10 EDT)**: chains stall
  but survive; CLOCK_MONOTONIC freezes, so `done in N s` / `wall_time_s`
  spanning a suspend report GPU-active time, not elapsed wallclock.
  Repro chain therefore delayed ~10.5 h — results expected midday
  2026-06-12, not overnight.

- Old checkpoints (pre-registry) load as arch "boxent" via
  load_critic's get() default; d824280-era centersample checkpoints are
  NOT loadable by current code (arch changed) — their deploys are all
  recorded in runs/.
- Don't `uv run` inside the archive — it syncs/builds the untrusted
  archive package into its own venv (harmless but noisy).
- MLflow on :5500 (sqlite); tunnel: `ssh -f -N -L 5500:localhost:5500 crockett`.

## PM session (post-pivot, same day) — state for the next session

User directives added (all encoded in CLAUDE.md): batch size chosen by
saturation probe, never searched, prefer low bs + low lr + more steps
(probe result lives next to `TuneConfig.batch_size`); no "tN" trial
names; pooled-only conditioning BANNED for location-resolved outputs;
sub-native zoom helps, widen min_scale to ~0.05 with STRATIFIED scale
sampling at next generation; future datasets oracle- or random-advance,
never egc2f-advance; method goal reframed: replace the hand-coded R-IID
prior with a LEARNED image-independent but history-conditioned proposal
(GRU over VPE + conditional flow); benchmarks only on idle hardware;
notes must be reproduction-grade.

Live on crockett (single 4090, ~/projects/CanViT-PyTorch-RL):
- Optuna study `boxent_t1only_egc2f_c32_bs32` (storage runs/optuna.db,
  launcher: `uv run python -m canvit_pytorch_rl.tune_critic`, log
  /tmp/rl_tune.log, trial curves in MLflow as
  tune_boxent_t1only_egc2f_c32_bs32/trialNNNN). The earlier studies are
  RETIRED: `boxent_t1only_egc2f_c32` (searched batch size — curves
  incomparable; its trial0/trial2 = the 0.3052/0.3010 same-seed pair)
  and the unnamed pre-restart trials.
- MAIN RESULT of the day: see milestones.md "First CI-solid learned win"
  (+0.429 t1, CI [+0.11,+0.80], full provenance there). Next arm when
  GPU frees: SAME config retrained all-t (drop --train-t-filter) +
  greedy T=5 — targets the remaining -0.21 at t4.

Local-laptop exploration (throwaway/, CPU; figures in outputs/):
- Ephemeral inputs in /tmp regenerate via:
  `ssh crockett 'cd ~/projects/CanViT-PyTorch-RL && uv run python -c
  "import polars as pl; pl.read_parquet(\"runs/traincand_seqoracle_t5_c32/candidates.parquet\").select(\"image\",\"t\",\"k\",\"source\",\"center_y\",\"center_x\",\"scale\",\"ce\").write_parquet(\"/tmp/oracle_cands_slim.parquet\")"'`
  then rsync; same pattern for valcand_slim + {train,val}cand actions
  parquets (see throwaway/flow_prior.py header).
- throwaway/oracle_marginal_analysis.py + the R-IID-only variant:
  image-blind structure = (scale, radius, x-vs-y, t) schedule worth ~5%
  of z-variance; EG-C2F proposals are ATOMIC (scale 0.5, 4 positions) —
  always exclude behavior-won cells when fitting pick distributions.
- throwaway/flow_prior.py: GRU-over-VPE + nflows MAF on R-IID-won
  oracle-advance picks: val NLL -0.801 vs R-IID analytic -0.719
  (schedule-only -0.770); history's edge grows with t. 6x(MAF h64) +
  ctx 64, lr 1e-3 cos, 2k steps bs 1024, ~60 s CPU.
- throwaway/flow_fixate_toy.py: conditioning gate — cnn conditioner
  fixates (med center err 0.035), mean-pool fails position (0.437 ~
  blind 0.507) while nailing scale (0.017). Settled; led to the
  pooled-conditioning ban.
- throwaway/flow_fixate_seq.py: 3-blob any-order-no-revisit sequential
  toy (CNN+GRU vs no-history), coverage metric; figure renders EXACT
  per-step conditional density heatmaps. Run in flight at session end.

Errors added to the ledger this PM: searched batch size in the first
study (curves incomparable — retired); claimed "kernel-launch-bound"
GPU behavior and CPU scaling from CONTENDED benchmarks without clean
measurement (user: confounded results are worthless); repeated the
documented pkill-over-ssh self-kill once more (use explicit pids,
always); trial0000's +0.04 Spearman initially mis-referenced against
an all-t metric (correct t1-only references: globalpool 0.265, old
small t1-only arm ~0.29).

## Research direction logged (from another CC instance, via user, 2026-06-12)

MaxEnt-RL unified flow actor-critic: a conditional normalizing flow's
exact log-density is a soft Q up to a state value, Q = alpha*log pi + V;
one flow + one scalar head trained by a single soft-Bellman / path-
consistency (PCL, Nachum 2017) residual replaces separate actor & critic.
Fit to our setting:
- Our softmax(z/tau)-weighted-MLE actor IS the offline corner:
  pi ∝ mu*exp(A/tau) with mu = R-IID, whose density we have IN CLOSED
  FORM — so log pi_flow - log mu_analytic is an exact advantage critic,
  one model, no second network.
- TRAP: our z targets divide by per-state sigma(s) -> state-dependent
  effective temperature; harmless for the bandit actor, but the full
  Bellman/PCL version needs RAW dCE rewards (cached in candidates.parquet
  — no regeneration needed).
- Where it pays: the t4 gap = myopia of per-step-greedy critics; PCL on
  cached T=5 rollouts is offline at replay economics.
- Validation ladder: weighted-MLE actor toy (throwaway/
  flow_actor_weighted_toy.py) -> advantage-flow on real t1 cells ->
  flow-PCL on the sequential 3-blob world (known dynamics/rewards,
  exact V to check against) -> PCL on real rollouts. Use NSF (spline)
  couplings for conditional multimodality; Trust-PCL KL-to-lagged-flow
  if the coupled parametrization is unstable.

## COMPACTION HANDOFF (2026-06-12 ~14:10 EDT laptop / ~13:55 crockett)

Read first: CLAUDE.md (rulebook, grew a lot today), docs/milestones.md
(the +0.429 result + provenance), docs/dataset-findings.md (verifiable
data facts + metric hierarchy). This section = what is RUNNING, what is
UNRESOLVED, and chat-only results not recorded elsewhere.

### Chat-only results now recorded here

- Strongest HP attribution: THREE configs reach val Spearman t1
  0.295-0.305 sharing ONLY hidden_dim 768 — retired-study trial0
  (lr 2.2e-4, wd 2e-4, mlp 512, entchan, bs64), bs32-study trial0000
  (lr 2.3e-4, wd 8.3e-3, mlp 2048, no entchan), bs32-study trial0001
  (lr 1.6e-5, wd 1e-6, conv 5, mlp 2048, no entchan). lr spans a DECADE,
  wd spans 4 DECADES. Trunk width is the load-bearing ingredient; the
  earlier "20x capacity is a wash" elimination scaled the WRONG dims
  (head/embedding, not trunk). Sampler gotcha: a restarted tuner process
  re-draws the same first config (TPESampler(seed=0)) — the retired
  study's trial0/trial2 same-seed pair measured GPU nondeterminism
  noise: +-0.004 Spearman.
- Cost accounting of the headline result: training 128k replay forwards
  (~5 min uncontended at measured 410-480 visits/s), t1 data slice
  attributable 364k forwards (~14 min), deploys 10k forwards each.
  Total ~0.5M forwards vs Codex winner 16.4M training forwards alone.
- Toy gates (scripts in throwaway/, committed; figures in local
  outputs/, regenerable):
  - flow_fixate_toy.py (single blob, 2.5k steps): cnn conditioner
    NLL -7.31 / med center err 0.035; mean-POOL conditioner -2.95 /
    0.437 (~blind 0.507) while scale err 0.017 -> pooling destroys
    location, recovers global stats. Led to the pooled-conditioning BAN.
  - flow_fixate_seq.py (3 blobs, any order, no revisit; 4k steps):
    GRU-history arm coverage 75.9% / all-3 rate 42.9% vs no-history
    65.8% / 18.2%; analytic memoryless baseline = 63.3% / 22.2%, so
    no-history == "sees blobs, can't remember visits" exactly, and the
    GRU learns anti-revisit. Density-heatmap figure code is IN the
    script but the completed run predates it (figures are scatter
    version; rerun for heatmaps).
- Actor-vs-critic framing (analysis, user-vetted): the critic gets 17
  labeled examples per state, an imitation actor gets 1 noisy argmax;
  actors cannot be evaluated off-policy (critics can); the value
  landscape is multimodal (findings entry 5) so actors must be density
  models; critic+grid-argmax IS already an amortized actor, so a
  parametric actor's value is (a) learned proposal for K-sample
  selection, (b) behavior policy for on-policy data generation, (c) the
  IN21K pretraining target. CRITICAL correction [user]: an actor
  conditioner may contain NO action-dependent reads (no box sampling) —
  state only, location-preserving (CNN+flatten or learned-query
  cross-attn); the action is the OUTPUT.
- Planned actor experiment (designed, not built): conditional flow over
  safe-box u, conditioner = LN->proj->conv trunk over canvas tokens +
  STANDARDIZED entropy channel (scale caveat [user]), weighted MLE
  w = softmax(z/tau) over the 17 cached t1 candidates. Note
  pi ∝ mu*exp(A/tau) with ANALYTIC mu (R-IID) -> log pi - log mu is an
  exact advantage critic for free (see MaxEnt direction note above).

### Running at handoff

- crockett: Optuna study boxent_t1only_egc2f_c32_bs32, pid 4025394,
  log /tmp/rl_tune.log, 200 trials, ~2 scored at last check (best
  0.2999). Read: `uv run python -c "import optuna; ..."` on
  runs/optuna.db, or MLflow :5500. GPU otherwise idle.
- laptop: throwaway/flow_actor_weighted_toy.py ladder running
  (log /tmp/actor_toy.log): 9 arms — exact-MLE upper bound; soft tau
  0.1/0.3/1.0; hard argmax-of-17; K=64; noise=0; 3-blob soft+hard.
  Read the printed table: regret@1 = 1 - true_value(one sampled action),
  regret@8 = best-of-8 samples; compare vs rand@1 and best-of-17-RIID
  columns. Purpose: which (tau, K, noise, modality) regimes weighted-MLE
  actors work in, BEFORE building the real one.

### Unresolved ambiguities / open questions (do not silently drop)

1. Why does the tuned t1-only critic NOT collapse at t2-4 deployed
   (41.47->42.96) when the old t1-only arm flatlined? Untested suspects:
   safe-box grid, trunk width, entropy channel.
2. t4 still -0.21 vs EG-C2F: all-t retrain of the winning config is
   designed and WAITING FOR USER GO (one command + 4 deploys).
3. greedy>randomk confirmed for ONE critic only; the killed safebox
   re-deploys of the two 20M boxent critics would settle generality.
4. Matched-K 2x2 (distribution vs K confound) — killed, unresolved.
5. Entropy-channel attribution: entchan arm was killed; sweep data is
   the only evidence (mixed: winner has it ON, two 0.30 configs have it
   OFF). Per-module grad norms (grad_norm_ent_fuse... note: module name
   is ent_fuse) now logged for future reads.
6. t=2 blind-structure anomaly (findings entry 4) unexplained.
7. Sub-native scales (min 0.05) + K=64 + no-behavior-candidate + 
   oracle/random-advance regeneration: SPEC'D (CLAUDE.md) not run
   (~2.5 h generation for K=64 train split — scale linearly: 69 fwd/img
   at K=16 -> ~261 fwd/img at K=64).
8. Learned-prior (GRU flow) deploy test (sample-K proposals + critic)
   not run; flow code is throwaway-grade.
9. globalpool checkpoint (runs/critic_globalpool_zce_egc2f_t1only_c32,
   killed at 79%, best@~2k saved) never deployed — only Spearman read.
10. c64 datasets ({train,val}cand_seqoracle_t5_c64) generated, unused.
11. val-tuned caveat on the +0.429 stands; a fresh-seed replication of
    train+deploy (~20 min) would harden it cheaply.
12. PCL/MaxEnt direction: validated path is toys-first (3-blob world has
    exact V*); needs RAW dCE rewards (cached) not z (see findings #2).

### User decision queue (explicitly waiting on user)

all-t retrain arm | K=64+wide-scale regeneration | real actor build |
safebox re-deploys of old critics | PCL toy rung | anything GPU-queued
(standing instruction: do NOT queue without user word).

NOTE on throwaway/ scripts: they are LAPTOP-LOCAL and GITIGNORED
(exploration regime). The durable reproduction recipes are the estimator
descriptions in docs/dataset-findings.md + the /tmp regeneration commands
above; the toy/flow scripts' RESULTS are recorded in this file. When the
actor is built for real, the flow machinery graduates into src/ with full
harness treatment (committed, typed, MLflow) — do not copy throwaway code
verbatim.

### Compaction addendum — nuances that would otherwise be lost

- **NO monitors/ticks are armed post-compaction.** The crockett tuner
  runs unattended (nothing will notify); the laptop actor-toy ladder
  writes to /tmp/actor_toy.log and a task notification may fire, but if
  lost, read that file. Standing user goal of <=45-min wakeups needs
  re-arming after compaction.
- **crockett SUSPENDS ~22:40 EDT**: the 200-trial study (~4-6 min/trial
  uncontended -> 13-20 h) will stall overnight and resume on wake. Do
  not promise morning completion; wall_time spanning suspend
  undercounts (monotonic clock freezes).
- **Reading the actor-toy ladder correctly**: weighted-MLE at temperature
  tau fits the BOLTZMANN-TILTED candidate distribution, not the argmax —
  soft arms SHOULD show higher regret@1 than the exact-MLE arm even when
  working perfectly; judge soft arms against each other and against the
  intended tilted-distribution regret, not against exact-MLE. Also: the
  exact arm is single-blob only, so 3-blob arms compare only within
  themselves; regret@8 confounds actor sharpness with best-of-8
  selection; 3-blob scenes have 3x the pixel noise (summed channels).
- **Leaderboard discipline for the ongoing sweep**: as trials accumulate,
  the top value creeps up by selection (winner's curse on the
  leaderboard itself). Do NOT deploy every new "best"; pre-commit a
  rule (e.g. top config must replicate within +-0.004 across a fresh
  seed before a deploy is spent).
- **The +0.43-greedy vs +0.25-randomk gap is K-confounded** (1024 vs 16
  candidates), not pure distribution-effect — the user's own matched-K
  point applies to the GOOD result too; grid16/random1024 cells remain
  unrun (open Q4).
- **Milestone narrative partially stale**: "wide trunk, NARROW head" —
  later bs32-study evidence (two ~0.30 configs with mlp 2048) shows the
  narrow head is NOT load-bearing; hidden 768 is. The milestone's
  what-mattered sentence overstates the head's role.
- **entchan retrofit warning**: the winning config's ent_fuse consumes
  the RAW [0,1] entropy channel (no standardization). Standardizing it
  is a sensible improvement but is a NEW ARM — do not silently "fix" it
  inside the winning recipe and keep comparing.
- **Next-gen dataset changes FIVE things at once** vs the data the winner
  trained on (advance rule, no behavior candidate, K 16->64, min_scale
  0.25->0.05, stratified scale sampling). When it lands, run a bridge
  arm (winning config, new data, t1-only) before attributing any change.
- **Cross-study config comparisons**: the retired study searched batch
  size (winner = bs64); the live bs32 study fixes bs32 — a "new best"
  from the live study trains at bs32 and is NOT the same regime as the
  deployed bs64 winner; lr comparisons across studies are bs-relative.
- **flow_prior detail**: histories include behavior-won PREFIX steps
  (only targets were filtered to R-IID-won); and val NLL at t1 for the
  GRU arm equals the t-only arm (-0.901 both) as it must (no history
  exists at t1) — a passed consistency check, useful regression probe.

### Addendum 2 (~14:30 EDT) — tuner killed, toys moved to GPU

- **bs32 tuner KILLED ~13:55 EDT by user decision** (toys outrank it for
  GPU time; its marginal value had dropped to plateau-mapping +
  parameter importances). Final state: 5 scored, 0 pruned, best
  trial0004 = 0.3049 (lr 2.5e-5, wd 4e-6, h768, conv5, mlp2048,
  entchan ON, bs32). Per-trial wallclocks 8.2-25.6 min (early ones
  CONTENDED by the deploy chain; patience-3 saved ~25% on 3 of 5).
  RESUMABLE: state in runs/optuna.db, same launcher command
  (tune_critic module); remember the restart re-draws the first prior
  samples (seeded sampler).
- **Sampling oddity, unresolved**: ALL FIVE scored bs32 trials drew
  hidden_dim=768 (p ~ 0.4% if uniform over {192,384,768}). Either fluke
  or degenerate categorical sampling — VERIFY before leaning on this
  study for the hidden-768 attribution (the retired study did sample
  384/192, so the attribution stands on cross-study evidence).
- **Local CPU actor-ladder killed mid-run, zero results** (its output
  was block-buffered — nothing ever flushed; the buffering flaw is
  fixed in all toys now: per-arm flush prints + wallclocks).
- **throwaway/*.py are now GIT-TRACKED** (.gitignore exception; outputs
  still ignored) and GPU-ready (device field defaults to cuda when
  available; Agg matplotlib). Git is the transport to crockett.
- **IN FLIGHT on crockett GPU, pid 4059610, log /tmp/rl_toys_gpu.log**
  (streams; watcher armed on ===ALLDONE | Error | Traceback | CUDA):
  seq-fixation tuning grid — A: 4k steps GPU baseline; B: 16k steps;
  C: 16k + lr 3e-3; D: 16k + ctx 128 + flow_hidden 128 + 8 layers —
  each runs history=False AND True; THEN the 9-arm weighted-MLE ladder
  at 4k steps. Figures land in crockett:outputs/ (NOT synced to laptop
  automatically).
- **Pre-registered prediction [CC]**: if sequential fixation is as
  trivial as the user expects, arm D pushes all-3 coverage well past
  80% (history arm) while no-history stays near the 22% memoryless
  bound. If it plateaus ~50% regardless of tuning, audit the rollout
  history path (k=0 uses a (0,0,1) t0 placeholder; k=0 and k=1 share
  GRU length 1 and differ only by content — train/rollout are
  consistent by construction but this is the most delicate spot).
- User directives encoded this round [chat-level, now here]: toys run
  on the GPU when it is free — kill lower-value GPU work for iteration
  speed; TUNE toy lr/capacity before treating toy numbers as findings;
  "sequential fixation should be trivial" — persistent mediocrity
  after tuning means bugs, not task difficulty.

## Addendum 3 (~15:15 EDT): GPU toy chain results — both gates passed

Chain (crockett pid 4059610, log crockett:/tmp/rl_toys_gpu.log, RTX 4090
otherwise idle, finished ===ALLDONE ~15:05 EDT). Scripts as committed in
throwaway/ at this commit. Two parts: seq-fixation HP grid, then the
9-arm weighted-MLE ladder.

### Sequential-fixation HP grid (throwaway/flow_fixate_seq.py)

Task: 3 colored blobs per 16x16x3 grid, flow must fixate all 3 in one
3-step rollout, no revisits. Metrics on 1,024 val scenes, one greedy
3-step rollout each, blob matched if sampled center within 0.18 of blob
center (coverage = matched blobs / 3,072; all-3 = scenes with all three
matched / 1,024). hist=True conditions on grid + GRU-over-VPE of taken
actions; hist=False conditions on GRID PIXELS ONLY — verified at
flow_fixate_seq.py SceneConditioner.forward: the no-history branch
returns the grid embedding alone, t_idx and hist are IGNORED. So
no-history cannot key on timestep; its ceiling is i.i.d. sampling from
the learned 3-mode mixture (expected coverage 1-(2/3)^3 = 70.4%,
all-3 = 6/27 = 22.2% if uniform over modes).
CORRECTION: the earlier "analytic memoryless 63.3%" in chat/handoff was
a different (stricter) accounting; use 70.4%/22.2% as the memoryless
i.i.d.-uniform reference.

| arm (16k unless noted)          | hist=False cov / all-3 | hist=True cov / all-3 |
|---------------------------------|------------------------|-----------------------|
| A: 4k steps, lr 1e-3            | 66.11% / 18.95%        | 74.97% / 41.31%       |
| B: 16k steps, lr 1e-3           | 71.71% / 28.52%        | 93.10% / 80.86%       |
| C: 16k steps, lr 3e-3           | 72.46% / 27.83%        | 95.90% / 88.48%       |
| D: 16k, lr 1e-3, ctx128/fh128/fl8 | 72.14% / 26.56%      | 96.26% / 89.26%       |

Pre-registered prediction (Addendum 2) CONFIRMED: history arm pushed
well past the 80% all-3 bar (89.3% at D, 88.5% at C with default
capacity), no-history pinned at its memoryless ceiling (~72% ~ 70.4%
bound). The earlier mediocre 41% all-3 was UNDERTRAINING at 4k steps,
not an architecture or rollout bug — the user's "trivial to learn" call
was right. lr 3e-3 at default capacity captures nearly all of arm D's
gain; capacity is not the binding constraint here. Train NLL still
falling at step 16k in C/D (not converged). Wallclock: ~150-190 s per
16k-step arm at 84-107 it/s (batch 256).
Figures: crockett:outputs/flow_fixate_seq_hist{False,True}.png — NOTE
each run() overwrites the same filename, so the saved figures are from
the LAST arm only (D).

### Weighted-MLE ladder (throwaway/flow_actor_weighted_toy.py, 4k steps/arm)

Known true value v(a) = exp(-|c-blob|^2/(2*0.15^2)) * exp(-(s-s*)^2/(2*0.10^2)),
max over blobs; regret = 1 - v. Eval on 2,048 fresh scenes. regret@1 =
single flow sample; regret@8 = best-of-8 flow samples by TRUE value
(emulates flow-as-proposal + perfect critic ranking); rand@1 / best17 =
single R-IID draw / best-of-17 R-IID by true value (yardsticks, same
for all arms of a given n_blobs). Default arms: K=17 candidates,
noise sd 0.2 on v, 1 blob. ~36-43 s/arm on the 4090.

| arm                      | regret@1 | regret@8 | rand@1 | best17 |
|--------------------------|----------|----------|--------|--------|
| exact-MLE (upper bound)  | 0.125    | 0.040    | 0.953  | 0.611  |
| soft tau=0.1             | 0.748    | 0.297    | 0.953  | 0.611  |
| soft tau=0.3             | 0.762    | 0.306    | 0.953  | 0.611  |
| soft tau=1.0             | 0.872    | 0.471    | 0.953  | 0.611  |
| hard argmax-of-17        | 0.748    | 0.299    | 0.953  | 0.611  |
| soft tau=0.3, K=64       | 0.561    | 0.154    | 0.953  | 0.611  |
| soft tau=0.3, noise=0    | 0.633    | 0.243    | 0.953  | 0.611  |
| soft tau=0.3, 3 blobs    | 0.697    | 0.275    | 0.881  | 0.391  |
| hard, 3 blobs            | 0.693    | 0.278    | 0.881  | 0.391  |

Readings (all single-seed, seed 0; no CIs — treat as directional):
1. Architecture is not the bottleneck: exact targets give 0.125 regret@1.
2. The candidate set bounds the actor. Hard argmax-of-17 imitates a
   target whose own quality is best17 = 0.611; it reaches 0.748 (fitting
   gap ~0.14, consistent with the tilted-distribution caveat from
   Addendum 2 — weighted MLE fits the TILTED distribution, so regret@1
   of soft/hard arms is NOT expected to approach exact-MLE).
3. K=64 is the single biggest lever among supervision arms: 0.762 ->
   0.561 regret@1, 0.306 -> 0.154 regret@8. Directly supports the K=64
   regeneration decision (user decision queue).
4. DEPLOY-RELEVANT headline: 8 samples from the K=64-trained flow,
   ranked by value, give regret 0.154 vs 0.611 for best-of-17 R-IID —
   better quality at half the forwards. The actor's first practical role
   is PROPOSAL DISTRIBUTION for the existing critic, not standalone policy.
5. Reward noise (sd 0.2 on a [0,1] value) costs ~0.13 regret@1 at K=17
   (0.762 noisy vs 0.633 clean).
6. tau in {0.1, 0.3, hard} barely matters; tau=1.0 clearly worse
   (over-flat tilt). Consistent with Codex's hard-selection choice not
   being load-bearing.
7. 3 blobs is not harder for the flow (0.697 vs 0.762 at K=17 — more
   targets make the tilted target easier to hit).

[CC assessment — single seed, synthetic value function; the K=64 and
proposal-distribution readings are the ones worth carrying to real data.]

## Addendum 4 (~15:40 EDT): sampling oddity RESOLVED — and the
## hidden-768 attribution is downgraded to UNSUPPORTED

### Mechanism (verified by exact reproduction)

The "all five bs32 trials drew hidden_dim=768" oddity (Addendum 2) is a
deterministic property of the seeded sampler, not a bug and not a fluke
to 0.4%: TPESampler(seed=0, multivariate=True) on optuna 4.9.0 (same
version local and crockett), with tune_critic.py's exact suggest order
(lr, weight_decay, warmup_frac, hidden_dim, conv_blocks, mlp_hidden,
n_freqs, entropy_channel), draws hidden_dim index 2 (=768) for trials
0 THROUGH 5, then 192/384/192/768/384/192 for trials 6-11. Reproduced
locally with a 12-trial dummy objective; the reproduction's lr draws
match runs/optuna.db to all stored digits (trial 0: 2.28811e-4,
trial 1: 1.64373e-5), so it is the same stream, not a lookalike.
tune_critic.py's suggest call and the DB-stored CategoricalDistribution
both carry all three choices {192,384,768} — code is correct. The
killed study simply died before trial 6, the first non-768 draw.
mlp_hidden has the same flavor: seed pins index 1 (2048) in 5 of the
first 6 draws (matches DB).

Consequence if the bs32 study is RESUMED: trials 6+ explore 192/384
automatically; nothing to fix in code. But any "what mattered" reading
of the first 6 trials must treat hidden_dim and (mostly) mlp_hidden as
CONSTANTS, not sampled variables.

### Attribution downgrade

This guts most of the evidence behind "Trunk width is the load-bearing
ingredient" (COMPACTION HANDOFF, "Chat-only results", the three-configs-
sharing-only-hidden-768 paragraph) and behind Addendum 2's "the
attribution stands on cross-study evidence":

- The two bs32-study configs in that paragraph COULD NOT have drawn
  anything but 768. They are not evidence about hidden_dim.
- Remaining evidence, all from the retired study
  (boxent_t1only_egc2f_c32; draws by trial: 768,384,768,384,192; only
  trials 0,2,3 completed): 768 -> 0.3052, 768 -> 0.3010 vs a SINGLE
  384 completion -> 0.2967. Gap ~0.004-0.008 against the +-0.004
  same-seed GPU-nondeterminism noise floor, with lr/wd/bs all
  differing between those trials. The 384 and 192 arms that would have
  settled it (retired trials 1 and 4) were orphaned RUNNING at the
  study kill, never scored.

Verdict [CC assessment, from runs/optuna.db read 2026-06-12]:
"hidden 768 is load-bearing" is UNSUPPORTED — one confounded trial
within ~2 noise floors. Trunk width moves from "finding" back to "open
question". This also further stales the milestone's what-mattered
sentence ("wide trunk, NARROW head"): the narrow head was already
retracted in the handoff; the wide trunk now lacks support too. What
remains supported about the winning config: it reproducibly hits
~0.30 val Spearman t1 (0.3052/0.3010/0.3018 replicates) where the
hand recipe ceiling'd at ~0.26-0.27 — the HP search mattered; WHICH
ingredient mattered is unresolved.

Cheap decisive probe if wanted (NOT queued — user decision): rerun the
winning config with hidden 384 and 192, everything else identical,
3 seeds each at ~7 min/run uncontended; compare against the
0.3018-0.3052 replicate band.

## Addendum 5 (~17:00 EDT): actor stack built; init analysis; both actor
## gates passed; probe killed by user decision

### hidden-dim probe (killed by user after 4/9 arms — decisive enough)

throwaway/probe_hidden_seed.py, winning critic config, only hidden/seed
varied, val Spearman t1: h384/s0 0.3021 (306s), h192/s0 0.3064 (289s),
h768/s0 0.3024 (622s), h384/s1 0.3007 (308s). All within +-0.003 across
a 4x trunk-width range -> trunk width NOT load-bearing at this data
scale (closes the Addendum-4 open question); the 0.30 plateau is
data-limited. Operational bonus: h192 halves training wallclock vs h768
(critic compute is NOT negligible vs replay — 622s vs 289s).
Pre-kill partial runs live under runs/probe_hidden_winnercfg/ (no
checkpoints; metrics in MLflow). h192_seed1 killed mid-run.

### Actor stack (committed d7ce3a0 + init fixes)

src/canvit_pytorch_rl/{flow_head,actor,train_actor}.py + test_flow_head
contracts (bijection round-trip, analytic log-det vs autograd at 1e-6,
MC normalization, box containment). Trains on the 16 R-IID candidates
per cell (k=0 behavior candidate EXCLUDED). Val readout: advantage
log pi - log mu vs -CE -> spearman/top1_regret keys, same convention as
the critic trainer. TrainLogger extracted to harness (manifest +
metrics.jsonl + MLflow, both trainers).

### Init analysis (throwaway/actor_init_analysis.py — keep as template)

Measured at init on synthetic cells (GPU): (1) conditioner activation
collapse — trunk std 0.34 -> post-down 0.010 -> ctx 0.036 (~30x), with
conditioner grad norms ~1000x below the flow's; (2) random-init MAF MADE
heads compound softplus scales over 6 layers on atanh-tailed safe-box
inputs (|u| absmax ~7.5) -> the first gate attempt went NaN within 200
steps. FIXES (committed): LayerNorm at the context head output (ctx std
0.996 after); zero-init every MADE final layer (each MAF starts at
scale softplus(0)=0.693, shift 0 — bounded log_prob at any depth).
nflows' own conditional_moons example (cloned, read) never faces this:
hidden_features=4 on O(1) data; it ALSO conditions the BASE
(ConditionalDiagonalNormal(context_encoder)) — a context-gradient path
that bypasses the transforms.

### Zero-init conditioner-gradient question [user] — settled empirically

Moons reproduction (throwaway/nflows_moons_gradflow.py, their exact
config, 2000 steps, 1 seed/arm, CPU): zero-init ctx-pathway grad is 0 at
step 1, alive by step 5 (0.008), 0.24 by step 100; final NLL identical
to random init (0.725 vs 0.730 at scale x1; 3.96 vs 3.92 at x6). The
conditional-base variant gives instant ctx gradient (1.6 at step 1) but
no final-quality edge here (0.92 at x1 — slightly worse, single seed).
Heavy-tail x6 data did NOT NaN their tiny config — tails alone are not
sufficient; our blowup needed tails x depth x width. Verdict: zero-init
transient is harmless; conditional base stays a config idea for real
data if the conditioner engages slowly (do NOT bundle it into the first
real runs).

### Synthetic gate (throwaway/actor_synthetic_gate.py) — PASSED

Real CanvasActor + mle_weights + log_prob loss, planted-target scenes in
the real tensor shapes, 1200 steps, batch 32, ~25 s/arm on the 4090
(after moving data gen on-device; the CPU-gen version ran 3.2 it/s and
its randn-per-step was the bottleneck). 1024 fresh eval cells:

  conditioned: spearman 0.723, top1-of-16 regret 0.748,
               sampled regret@1 0.885 / @8 0.561 (random: 0.957 / 0.765)
  blind:       spearman 0.201 (action-prior floor, cf. blind z R^2 ~0.05
               on real data), sampled regret == random (sanity).

The contract learns image-conditioned structure end-to-end. Modest
sampled-regret gain is expected (weighted MLE fits the K=16-tilted
distribution; 1200 steps).

### In flight

Real actor run actor_flow_soft_tau03_egc2f_t1only_c32 (chain2 pid
4089181, /tmp/actor_gate_chain2.log on crockett, MLflow run of the same
name; defaults: soft tau 0.3, bs 32, lr 1e-3, 2000 steps, t1-only).
First read: val_spearman_all vs the critic's ~0.30 and the gate's
blind-floor analogy.

### New CLAUDE.md rules this round (user)

GPU-first for minute-scale checks; experiments ALWAYS in background
(never sleep-and-tail in the conversation loop); GPU usefully busy AT
ALL TIMES incl. while thinking/editing; non-finite loss = instant exit
everywhere; init-time analysis (activations, logp components, grad
norms) before training a new arch.

## Addendum 6 (~18:00 EDT): codebase review + refactor; value-map logging

### Review snapshot (post-refactor; `tokei src/` = 2,452 code lines, 26 files)

Where LOC is spent: trainers 450 (critic 226 + actor 224, was 609 before
dedup), shared infra 374 (harness 216 + candidate_data 158), policies +
eval 325, data generation 327 (rollout_candidates 184 + oracle 143),
models 277 (critic 106 + globalpool 100 + actor 71), flow_head 67,
value_maps ~90, the rest plotting/listing/reference utilities.
throwaway/: 1,222 code lines, 10 scripts, all reproduction recipes for
documented findings (kept deliberately).

### Duplication removed this pass (commit "Refactor: shared candidate_data
### + value_maps modules", net -148 lines while ADDING image logging)

- candidate_data.py: RolloutCandidateSet, SameTBatchSampler, replay_state,
  z_targets, make_loaders (incl. the train-eval-slice construction and its
  stride-aliasing comment), CandidateTrainConfig base. Both trainers were
  carrying full copies; train_actor imported critic-trainer internals.
- harness: warmup_cosine (was 3 inline copies incl. toys), aggregate_per_t
  (was 2), TrainLogger.log_figure. metrics: spearman (was in train_critic).
- value_maps.value_map_figure: one dense-map renderer for ANY scorer;
  throwaway/actor_reward_maps.py reduced to a thin CLI over it.
- tqdm disabled on non-tty (CR spam in every redirected chain log).

### Known remaining debt (deliberate, not forgotten)

- evaluate_critic vs evaluate_actor: parallel ~25-line shapes; metric
  definitions genuinely differ — merging needs a callback abstraction
  that obscures more than it saves. Revisit if a third evaluator appears.
- save()/best/patience/loop skeleton still duplicated (~35 lines/trainer);
  a Trainer driver class is the fix IF a third trainer appears.
- critic_policy.py name is stale (it now also holds ActorProposalPolicy
  and load_actor) — rename to policies.py at next quiet moment.
- rollout_candidates.py + oracle.py (generation side) not audited
  line-by-line this pass.
- bench/figure4b/plot_training/list_runs/paper_reference: utility tier,
  unaudited this pass.

### Good patterns to PRESERVE (the load-bearing conventions)

One evaluator/one artifact bundle + manifests with git_rev; frozen tyro
dataclass configs (code is the spec); registry-dict dispatch (CRITIC_ARCHS,
loss_fn) over isinstance branching; contract tests pinning math
(test_flow_head: bijection/log-det-vs-autograd/normalization;
test_metrics: upstream mIoU equivalence); action-space densities ONLY
(flow_head owns the bijection + Jacobian; nobody else touches u-space);
analytic mu kept analytic; MetricWindow on-device accumulation; per-module
grad norms everywhere; instant NaN exit; zero-init MAF heads + LN'd
context (measured, documented in Addendum 5).

### Bad patterns CAUGHT this session (already fixed, listed for pattern
### recognition)

ImageNet-normalized images displayed without denorm (saturated figures);
CPU-generator randn in a GPU toy loop (16x slowdown); block-buffered
prints in long scripts; sleep-and-tail blocking the conversation loop;
stale MLflow store path in CLAUDE.md vs code.

### Trainers now log images

Both trainers render a fixed 4-cell val panel (value_map_figure: critic
score / actor advantage maps, scales 0.25/0.35/0.5/0.7, cached candidates
overlaid) to MLflow at EVERY eval interval incl. step 0 — artifact path
value_maps/step_NNNNNN.png per run; log_value_maps=False for HP search.

## Addendum 7 (~16:25 EDT): first actor deploys — matched-K direction
## positive, actor-16+critic matches greedy-1024

Full val, T=5, c32, squish protocol. Actor = actor_flow_soft_tau03_8k
(best_spearman.pt, val spearman 0.2568); critic = the +0.429 winner.
Per-t mIoU (t0 fixed 38.53 for all):

  actorprop_critic_k16 seed0:  41.39 / 42.40 / 42.90 / 43.11
  actorprop_critic_k16 seed1:  41.44 / 42.34 / 42.63 / 42.84
  actorprop_adv_k16   seed0:   41.05 / 41.96 / 42.42 / 42.62
  randomk16+critic seeds 0-2:  41.22-41.32 / 42.20-42.24 / 42.43-42.69 / 42.69-42.97
  greedy1024+critic:           41.47 / 42.37 / 42.79 / 42.96
  EG-C2F:                      41.04 / 42.05 / 42.69 / 43.17

Paired bootstrap t1 (2000 resamples, default_rng(0), n=2000 images;
recipe = throwaway/deploy_paired_bootstrap.py; per_image.parquet inputs
rsynced to /tmp/deploy_cmp/<run>/ — regenerate with the rsync loop in
that script's docstring):

  actor-16+critic vs randomk-16+critic (MATCHED-K, same critic):
    seed0 pair +0.167 pp CI [-0.171, +0.454]
    seed1 pair +0.186 pp CI [-0.185, +0.504]
    -> direction positive in both seed pairs (and every actor seed's t1
       beats every randomk seed's), but NOT CI-solid; single-seed pairs
       of stochastic policies are underpowered for a ~0.18 pp effect.
       2-3 more seeds/side (~30 s GPU each) would settle it.
  actor-16+critic vs EG-C2F: seed0 +0.352 [-0.017, +0.723];
    seed1 +0.402 [+0.063, +0.772] (excludes zero).
  advantage-only (NO critic at deploy) vs EG-C2F: +0.013
    [-0.436, +0.373] — the standalone 8k actor EQUALS EG-C2F at t1,
    neither better nor worse; its own ranking (spearman 0.257) adds
    nothing over its sampling yet.

Headline readings: (1) 16 actor proposals + critic = 41.39/41.44 at t1
vs greedy-1024's 41.47 — the same pick quality at 1/64th the candidate
scorings (and greedy's +9 s/episode critic cost avoided). (2) The
proposal-distribution effect on real data is ~+0.18 pp at t1 at K=16
(direction consistent, CI pending more seeds). (3) t4: EG-C2F (43.17)
still edges everything — the t1-only horizon issue, all-t training is
the known fix. (4) All numbers from a 14-minute actor training run.

## COMPACTION HANDOFF 2 (2026-06-12 ~17:30 EDT) — actor-line state

Read order for a fresh session: CLAUDE.md (grew: self-contained-policy
goal, ML-questions checklist verbatim, optimization-skepticism rule,
background/GPU-first/NaN rules), then Addenda 3-7 above, then this.

### Results in chat but not yet recorded above

- **8k actor arms (final)**: soft tau0.3 0.2568 / hard 0.2589 / soft
  tau0.1 0.2557 best val spearman t1 — weighting choice flat (spread
  ~= noise), matching the toy-ladder prediction. Runs
  actor_flow_{soft_tau03,hard,soft_tau01}_8k_egc2f_t1only_c32.
- **BLIND FLOOR (big)**: actor_flow_blind_8k_egc2f_t1only_c32
  (state_blind=True, 8k steps) = **0.2238** best val spearman t1. The
  conditioned actor (0.251-0.257) is only ~+0.03 above the
  image-independent floor, and 0.224 saturates the dataset-findings
  blind bound (z R^2 ~0.05 -> max corr ~0.23). Image conditioning
  currently contributes LITTLE to the actor's ranking. (Critic
  comparison: its state_blind analogue exists as the old
  critic_boxent_actiononly_* family — pull its val spearman before
  quoting a critic-side conditioning margin.)
- **Fit analysis (decisive for direction)**: actor 8k train spearman
  STALLS at 0.263 (val 0.251, fit_gap <= 0.012 throughout, 12.7 epochs
  over the 20,210 t1 cells). Critic winner on the SAME cells: train
  0.289 at 64k visits, 0.338 and climbing at 128k (gap 0.036). Targets
  are deterministic (cached CE), so the actor is fit-limited: it cannot
  express/optimize what the critic memorizes in a quarter of the visits.
- **FLOPs audit** (throwaway/flops_audit.py; FlopCounterMode, matmul/conv
  only): backbone one c32 glimpse fwd 13.5 GFLOPs (95.3M params);
  critic h768 winner score-call 68.1 (K=16) / 72.4 (K=1024) GFLOPs
  (35.3M) — FIVE backbone-forwards per scoring call, trunk-dominated,
  K nearly free; critic h192 4.6/6.5 GFLOPs (3.2M); actor
  context+log_prob(16) 4.8 GFLOPs (4.5M). => same weight class as h192
  critic which scores 0.306: the actor's gap is NOT size/FLOPs.
  Also measured from deploy manifests: greedy-1024 evals 27 s vs
  randomk-16 18-19 s vs no-critic 18 s (uncontended ~20M-class critics)
  — earlier "critic scoring is cheap" claim WRONG at K=1024.
- **Budget audit** (manifest-derived): critic winner training 127,916
  replay forwards; its HP search 1,183,202 (retired 479,678 + bs32
  703,524); hidden-probe 527,664. Actor deployed ckpt 255,818 forwards,
  ZERO search; sibling arms 575,594. Dataset gen 69 fwd/image x 22,210
  images = 1.53M (shared, amortized).

### In flight on crockett RIGHT NOW

- 32k-steps actor run (actor_flow_soft_tau03_32k_egc2f_t1only_c32),
  part of batch2 (pid 4113611, /tmp/actor_batch2.log) — interpretive
  frame: 51 epochs; if train spearman stays ~0.26, the steps lever is
  EXHAUSTED and arch/objective is confirmed as the binding constraint.
- Waiter chain batch3 (pid 4124914, /tmp/actor_batch3.log) starts when
  batch2's pid exits: (1) throwaway/actor_arch_suite.py — 9 arms, same
  prod archs on synthetic planted-target cells, early stopping,
  late-training gradient-share diagnostic, critic-h192 yardstick;
  expectation: if ctx bottleneck is binding, ctxgrid8/ctxdim1024 arms
  learn the toy task markedly faster/higher than base, while lr arms
  separate optimization-vs-arch [pre-registered]; (2) base_mode
  deterministic deploy eval (actorprop_basemode_soft8k_t5_c32);
  (3) matched-K CI tightening: actor seeds 2,3 + randomk seed 3
  (-> 4v4 seeds; rerun throwaway/deploy_paired_bootstrap.py with the
  new pairs added).
- Watchers in this CC session: b38qlhswj (batch3 ALLDONE/error),
  bhfrk32v5 (batch2 — will fire when 32k ends; benign overlap).
- crockett SUSPENDS ~22:40 EDT. After batch3 (~18:15) the GPU is idle
  unless the next thing is queued — next planned GPU work is tune_actor
  (NOT YET WRITTEN; write only after reading suite results, which set
  the search space; remember the TPESampler(seed)-pins-early-categoricals
  lesson from Addendum 4 when configuring it).

### Standing user directives from this stretch (all in CLAUDE.md except *)

Fully self-contained policy is the end-state (critic = scaffolding);
ML-questions checklist consulted before any run design; apparent
capacity limits = optimization pathologies until proven otherwise (lr
spread + late grad shares + synthetic race); experiments ALWAYS detached
in background; GPU usefully busy at all times; NaN instant exit;
*density maps are a STANDING DELIVERABLE — keep rendering and showing
them (trainers auto-log to MLflow value_maps/; rsync+open notable ones,
e.g. blind-floor maps should be scene-INVARIANT — a pending visual check
the user should make), and base-mode/noise-0 deploy was an explicit
user curiosity — report its mIoU when batch3 lands.

### Open questions (new since handoff 1)

1. Why does the actor's train spearman stall at 0.26 while an equal-FLOPs
   critic reaches 0.34 on identical cells? (Suite + tuner attack this;
   candidates: ctx bottleneck, affine-MAF expressiveness, weighted-MLE
   objective smoothing at tau=0.3, lr.)
2. Is the matched-K proposal effect (+0.17/+0.19 pp t1) CI-solid at 4v4
   seeds? (Bootstrap after batch3.)
3. base_mode deterministic deploy: where does it land between
   advantage-ranked (41.05) and critic-ranked (41.4)?
4. Critic's own blind floor (actiononly family) — pull for symmetry.
5. The t1-trained actor's maps at t2-t4 states (value_maps now make this
   visible) — does conditioning degrade off-distribution?
6. All of handoff 1's open questions that remain (all-t retrain, K=64
   regen with bridge arm, etc. — user decision queue unchanged).

## Addendum 8 — the 32k run breaks the fit-limit story (2026-06-12 ~17:00-17:30 EDT laptop clock; crockett clock runs ~13.5 min behind, see CLAUDE.md)

OPEN QUESTION 1 ABOVE IS ANSWERED: the stall was schedule truncation, not
a fit limit. `actor_flow_soft_tau03_32k_egc2f_t1only_c32` (same config as
the 8k run, steps=32000 so cosine decays 4x slower):

- step ~9.75k: train rho 0.295, val rho 0.2717 (new val best; 8k peak was 0.257)
- step 18k: train rho 0.358 — PAST the h768 critic's 0.338 train-side
  benchmark on identical cells — train_loss -1.23 and still accelerating
  per-step; val rho rolled over to 0.235 (fit gap ~0.12 and widening).
- Per-module grad dynamics over the same window [user observation]:
  conditioner modules' (proj/trunk/ent_fuse) grad share rising steadily,
  flow share falling. Norms are logged POST-clip (shares of the clipped
  ball); raw total fell 4.5->2.8, so pre-clip conditioner growth is even
  steeper. NOT a clipping artifact. (Also corrected in-chat: global-norm
  clipping preserves gradient direction, and AdamW is approximately
  invariant to constant gradient rescale — persistent clipping does NOT
  mean effective-LR reduction. grad clip does not confound lr [user].)
- Reading: unconditional bulk fits first; conditioning signal wakes late
  (val-vs-blind-floor gap grew from ~0 early to ~+0.05 at the val peak).
  Past ~step 10k (~16 epochs of 20210 t1 cells) the extra train fit no
  longer transfers — val ceiling (~0.27) consistent with the dataset
  blind-bound analysis. User stance (now in CLAUDE.md): fit train first,
  regularize later; this is the desired problem.

Interim deploy of the step-9500 ckpt (run names accidentally lack the
step — a snapshot-script var was empty; provenance is in each manifest's
actor_step=9500; contended GPU so wall_time_s tainted, mIoU fine):
- actorprop_critic_k16_32kinterim_step_t5_c32  t1 41.27
- actorprop_adv_k16_32kinterim_step_t5_c32     t1 41.30  <- SELF-RANKED
  vs 41.05 for the 8k ckpt (same seed 0, K=16, T=5, c32, full val): more
  MLE training closed the self-ranking gap to the critic ranker at this
  checkpoint (41.30 vs 41.27, within seed noise). Direct progress toward
  the self-contained end-state.

In flight / queued (crockett):
- 3 parallel LR arms [user: "1e-3 was never swept; run sweeps in parallel"]:
  actor_lrswp/lr{3e-5,1e-4,3e-4}, 3000 steps, t1-only, bs32, nw4
  (pids 4134295-7, logs /tmp/actor_lrswp_*.log). Stability readout
  requested: val-rho oscillation, loss/grad spikes, from metrics.jsonl.
- batch3 unchanged (arch suite, base_mode 8k deploy, matched-K seeds).
- batch4 (waiter 4134847, /tmp/actor_batch4.log): 32k-final deploys
  (critic/advantage/base_mode) then throwaway/actor_lever_battery.py
  --arms base hard ctxgrid8 ctxdim1024 hidden384 flow12x128 (lr arms
  dropped — covered by the parallel sweep; battery = warm-started
  single-change 3k-step arms, b77865b).
- CPU probe (pid 4137537, /tmp/actor_ckpt_probe.log):
  throwaway/actor_ckpt_probe.py on best(9750)+last(18k) snapshots —
  init-vs-trained activation scales, ctx cosine collapse, logp variance
  split (action- vs state-discrimination), flow context/final weight
  growth [user: probe what it learned to speed it up].

Value-map sanity check [user request]: maps rendered correctly
(denormalized imgs; blind rows scene-invariant as constructed). Both
blind and conditioned maps are LARGELY FLAT — visual counterpart of the
small conditioning gap at those checkpoints. Colormap semantics:
surface viridis = model advantage, brighter better; dots coolwarm =
true z(-CE), red better; panels autoscale independently (only colorbar
gives absolute range). Renderer improvement candidate: shared color
scale per row.

Deprioritized by user this stretch: fresh-seed +0.429 replication ("not
sure we care yet"), safebox re-deploys of old critics ("eh"), PCL toy
(re-scope if it comes up).

## Addendum 9 — explosion mechanism SOLVED; map anisotropy; LR settled (2026-06-12 evening, laptop clock; commits b7ef1fc..ff2e2ce)

### The 10^8 trunk activation explosion: GELU-escape (hard gating)

Chain of evidence, all on real data/checkpoints (toy repro RETIRED — at
matchable scales a toy grew 50x slower than the phenomenon; commit 5981459
records the negative result):

1. ckpt probe (Addendum 8): trunk act std 0.37 -> 6e7 over 18k steps; head
   LN makes the scale loss-invariant; wd=1e-5 gives ~1e-8/step restoring
   force (nil). [throwaway/actor_ckpt_probe.py]
2. alpha-sweep on /tmp/actor32k_last_snap.pt (step 18k), 16 TRAIN t1 cells
   [throwaway/norm_explosion_repro.py part B]: wnll(alpha) =
   3.4 / 0.20 / -0.32 / -0.74 / -1.19 / -1.21 / -1.21 at alpha
   1e-7 / 1e-4 / 5.6e-3 / 5.6e-2 / 0.25 / 1 / 4 — gentle slope up,
   saturated at trained scale.
3. DECISIVE — relu-swap probe [throwaway/relu_swap_probe.py, commit
   6618343]: trunk GELU->ReLU at eval on the same ckpt: relu@1 -1.213 ==
   gelu@1 -1.211; relu FLAT across two decades (alpha 5.6e-3: relu -1.156
   vs gelu -0.322). All three pre-registered predictions hit.

CONCLUSION: the optimizer grew activations to escape GELU's soft-gating
regime — at the grown scale every GELU IS a ReLU. Fix options now in the
arch [commits b7ef1fc, 3817d1b]: ResidualConvBlock(pre_norm=, act=) —
battery arms trunknorm (GN pre-norm), relutrunk (pre-norm + relu, the
mechanistically-indicated combo), wd1e-2 (user's restoring-force fix).
At 3k steps lr1e-3: trunknorm 0.2411, wd1e-2 0.2448 val spearman;
lr-matched plain base + relutrunk in flight (actor_bat1/*).

### Value-map artifacts -> renderer fixes; the maps were never flat

[throwaway/map_artifact_stats.py, commit 370b7e2] on the step-9500 best
ckpt: interior p2-p98 spans 2-4 nats (REAL structure); full range is
stretched by safe-box bijection tails at the rendered box edge, so the
old autoscale rendered only 20-69% of span on tails — the earlier "maps
are flat" read was substantially a rendering artifact. Fixes in
value_maps.py: percentile clim p2/p98 [743a072] + cell-centered grid that
never evaluates the singular boundary [32bf7ed].

### Learned-density anisotropy: y-dominance is MODEL-side, not reward-side

- Conditioned ckpt maps: row-mean std (y) 2-8x col-mean std (x), tiny xy
  residual. Blind actor maps: scene-invariant to numerical identity
  (sanity PASS) and mildly y-dominant (~1.6-2x) — explainable by the
  data marginal (center bias stronger in y: rho(-ce,|y|) -0.148 vs |x|
  -0.101).
- Dataset per-cell reward is nearly ISOTROPIC: mean |spearman(-ce, .)|
  over 2000 val t1 cells = y 0.333, x 0.304, scale 0.353 (signed scale
  +0.21) [polars on runs/valcand_rollout_egc2f_t5_c32/candidates.parquet].
- => the conditioned model under-uses x. Leading hypothesis [user]:
  y-conditioning is EASIER (sky/ground universal regularity), x is
  scene-specific placement — learned later and memorization-flavored;
  links to the val rollover. Pre-registered test: row/col map stats on
  the ckpt series (9500 / 18k / 32k-final), train + val cells: easier-
  first predicts col-std grows late and more on train.
- Visual-neuroscience tie-in [user]: horizon-bias / elevation priors
  (Foulsham & Kingstone; Torralba contextual guidance) — the elevation
  prior is real in the marginal, the model exaggerates it.

### LR + defaults

Sweep (3k matched steps, same data order, actor_lrswp/*): 3e-5 0.180 /
1e-4 0.199 / 3e-4 0.225; oscillation rises with lr, no spikes
[throwaway/run_stability.py, d9f4a8b]. ActorTrainConfig.lr default now
3e-4 [743a072; user: "1e-3 clearly too high" from 32k trajectories].
In-flight wave arms explicitly pin --lr 1e-3 for internal comparability.

### Ops lessons (encoded in CLAUDE.md)

Thread-cap + nice ALL CPU torch jobs on crockett (uncapped diag took 11.7
cores, GPU dipped to 46%) [5413906]; never pipe verification commands
(two pushes with lint/format failures via `| tail`) [ff2e2ce]; zsh does
not word-split unquoted vars — ctxgrid8/ctxdim1024 arm launches silently
died on a one-token flag string, caught by post-launch log check.

## Addendum 10 — battery complete; proposal-null; easier-first confirmed; evening queue (2026-06-12 ~18:45 laptop clock)

### Full single-change battery (3k steps, bs32, matched data order, best val spearman; lr 1e-3 unless noted; runs actor_bat1/*, actor_lrswp/*, actor_fix/*)

  hidden384   0.2512      flow8x64    0.2500      base        0.2463
  relutrunk   0.2463      wd1e-2      0.2448      ctxgrid8    0.2444
  trunknorm   0.2411      hard        0.2368      lr3e-4      0.2250
  flow6x128   0.2140      ctxdim1024  0.1953      lr1e-4      0.1993
  lr3e-5      0.1801

Reads: nothing decisively beats base (hidden384/flow8x64 within eval
oscillation ~0.01-0.05); WIDENING consistently hurts at this lr/horizon
(ctxdim1024 -0.05, flow6x128 -0.03); explosion fixes are FREE (relutrunk
== base) -> adopted for hygiene in tonight's long runs; hard weighting
slightly behind soft. Conclusion: at 3k steps the fit is not arch-bound
— consistent with the data-limited story below. NOTE: lr ladder at 3k
says 1e-3 > 3e-4 at MATCHED SHORT steps; default stays 3e-4 on the
user's long-horizon read [user decision; no lr race].

### Matched-K proposal effect: NULL at 4v4 (pre-pivot thread CLOSED)

Pooled paired bootstrap (seed-averaged, 2000 resamples, default_rng(0),
throwaway/deploy_paired_bootstrap.py + /tmp/deploy_cmp parquets):
actorprop vs randomk (both K=16, h768 critic ranker): +0.080 pp t1,
CI [-0.087, +0.253]; per-seed +0.167/+0.186/+0.124/-0.158, all CIs
straddle 0. randomk_seed3 = 41.55 (above all four actor seeds
41.39/41.44/41.45/41.39). WHY a good actor shows no proposal gain:
proposer and judge encode the SAME learned value surface (rho 0.26 vs
0.30) — knowledge used twice adds nothing; selection quality is the
binding constraint (oracle best-of-17 = 45.3 vs critic-picked ~41.4).
Actor proposals matter at small K (K=1: actor ~41.0 vs random-draw
~39.6 baseline). The big critic's remaining roles: matched-K instrument
+ train-side fit yardstick. All deploy numbers that matter for the
end-state are advantage-ranked (self-contained line).

### Easier-first conditioning CONFIRMED (ckpt series, /tmp/ckpt_series_maps.log)

Mean col-std (x-structure) on 4 fixed val cells across the 32k run's
ckpts: 0.19 (step 9500) -> 0.34 (18k) -> 0.83 (32k final); y/x ratio
4.6 -> 1.9 -> 1.2; xy-residual 0.23 -> 0.23 -> 0.82. Order learned:
scale -> elevation -> azimuth/placement. Val rho rolled over during
exactly the placement phase => placement conditioning is DATA-limited
(hallucinated on unseen scenes), elevation+scale generalize. Supports
the large-scale-pretraining route over arch work.

### base_mode verdict [user question CLOSED]

Deterministic noise-0 deploy: 8k ckpt t1 40.98, 32k-best ckpt 40.94 —
EG-C2F level, and t2-t4 plateau ~41.3 (vs 42.9 ranked). Sampling +
self-ranking is load-bearing; mode-only deploy is not viable.

### Ops

batch4's "32k-final" critic/advantage deploys silently re-evaluated the
step-9500 best ckpt (best never updated after 9500) — keep_every
retained ckpts added to train_actor to prevent this class of waste.
Synthetic arch suite: OOM'd next to parallel arms, then RETIRED
(superseded by the real-data battery). zsh launch bug: unquoted $FLAGS
is one token — always verify launches with a post-launch log grep.

### Evening queue (all watcher-armed, runs THROUGH the ~22:40 suspend)

longrun actor_flow_soft_relutrunk_32k_egc2f_t1only_c32 (lr 3e-4 default,
keep_every 4000) -> ckpt-trace: advantage-ranked K=16 deploy per retained
ckpt (actorprop_adv_k16_relutrunk32k_step*_t5_c32) — decides whether
ckpt selection follows train fit or val rho -> 128k t1-only run
(truncates at suspend; ckpts survive). All-t option fully specced in
chat (egc2f-advance state bias is the main caveat; data already cached;
2.5x cost/step) — USER-GATED, not queued.

## COMPACTION HANDOFF 3 (2026-06-12 ~19:10 laptop clock; crockett ~13.5 min behind)

READ ADDENDA 8-10 FIRST — they hold the day's evidence chain. This block:
everything chat-only since Addendum 10, the live queue, and tomorrow's
pre-registered reads.

### New results since Addendum 10

- Overfitting onset, cached 32k run (metrics.jsonl, exact): fit gap >0.01
  sustained from step 7500 (11.9 ep), >0.02 from 9750, >0.05 from 13750;
  VAL PEAK step 9500 = 15.0 epochs (rho 0.2718). step/val/train:
  4k .234/.234 | 8k .255/.268 | 12k .267/.296 | 16k .240/.319 |
  20k .223/.374 | 24k .190/.398 | 28k .163/.393 | 32k .155/.389.
- relutrunk ckpt probe (runs/actor_bat1/relutrunk/best_spearman.pt,
  /tmp/relutrunk_probe.log): GN pre-norm does NOT bound the stream —
  post-trunk std x335 in 2k steps (0.47 -> 158), post-down 7.8e4. With
  relu this is a function-preserving GAUGE (positive homogeneity), so
  outcomes match base; numerical hazard only. critic.py comment
  corrected. Containment options if wanted: wd (user's pick, arm was
  cost-free) or LayerScale — now implemented as trunk_layer_scale
  (CaiT, default off) [user suggestion].
- 4v4 proposal-null details, base_mode verdict, easier-first ckpt
  series, full battery table: all in Addendum 10.

### ON-THE-FLY TRAINING MODE — built, smoked, queued (the evening's main event)

User direction: inline on-the-fly preferred over periodic-regen hybrid
("cleaner"); fixes BOTH cached-data defects (egc2f-advance state bias,
old scale band) plus candidate memorization. Implementation in
train_actor.py (committed): --on-the-fly replaces the cached loader;
per visit: t0 full-scene fwd, t-1 advance steps under --otf-advance
(actor = ON-POLICY states | random), then --otf-k fresh R-IID candidates
scored by frozen-probe CE (rollout_candidates inner loop). t per batch =
train_t_filter or uniform [1, otf_t_max]. Cost (t-1)+1+otf_k fwd/visit
vs 1 cached. Val/train-eval stay cached as BRIDGE ANCHOR ONLY; deploys
are the verdict metric. Smoke passed (30 steps, 17 fwd/visit exact,
~0.45 steps/s contended; solo unknown ~1-1.5 est). GPU does ~430-470
backbone fwd/s in ALL regimes — wall time = forwards/rate.

### Live queue (crockett; all logs /tmp, all watcher-armed locally)

1. CACHED TWIN actor_flow_soft_relutrunk_32k_egc2f_t1only_c32 (lr 3e-4
   default, trunk-norm+relu, keep_every 4000) — running,
   /tmp/actor_longrun.log, done ~19:45 crockett.
2. CKPT-TRACE (waiter 4170985, /tmp/ckpt_trace.log): advantage-ranked
   K=16 T=5 deploy per retained ckpt -> runs
   actorprop_adv_k16_relutrunk32k_step*_t5_c32. DECIDES: does deploy
   mIoU track train fit or val rho (checkpoint-selection metric).
3. OTF TWIN actor_flow_otf_relutrunk_32k_t1only_c32 (waiter 4185149,
   /tmp/post_trace.log -> /tmp/actor_otf_twin.log): IDENTICAL config to
   cached twin except --on-the-fly, keep_every 2000. Launches ~20:15,
   truncated by ~22:40 suspend around step 8-10k (fine — ckpts+metrics
   survive). PRE-REGISTERED READ vs cached twin at matched steps:
   (a) OTF fit gap ~0 and val keeps climbing past cached's step-9.5k
   peak => rollover was CANDIDATE-SET memorization (freshness fixes it);
   (b) OTF rolls over at same place => IMAGE memorization => more
   images needed (IN21K pretraining route). Cached val anchor: peak
   0.2718 @9.5k.

### Decisions/retirements this stretch (user)

lr default 3e-4 by fiat (no 1e-3 race; 1e-3 won at 3k matched but long
horizon is the regime); 128k t1-only run REPLACED by OTF twin; synthetic
arch suite RETIRED; self-ranked K-sweep CUT (changes no decision);
proposal thread CLOSED (4v4 null); big h768 critic = instrument +
yardstick only, all end-state deploys advantage-ranked; queue through
the suspend, never idle GPU, ~100x budgets/data when out of hypotheses
(in CLAUDE.md).

### Tomorrow's menu (priority order)

1. Read OTF-vs-cached twin A/B (the pre-registered read above).
2. Read ckpt-trace -> pick checkpoint-selection metric.
3. If freshness wins: OTF all-t run with --otf-advance actor (on-policy
   states, no t filter) — supersedes the cached all-t debate entirely;
   spec discussion in chat ~18:30 laptop.
4. Gauge-pin decision: wd 1e-2 vs trunk_layer_scale arm (both cost-free
   at 3k; pick one for the recipe).
5. Critic h768 activation probe (same block class, never measured).
6. Visited-mask channel arm (history toy won big: coverage 66->76%,
   all-3 18->43% [chat-grade]; entropy partially covers it).
7. Open user gates: K=64/min_scale 0.05 dataset regen (only matters for
   CACHED training now — OTF makes it moot if freshness wins), all-t.

### Stale-fact warnings for the next session

- best_spearman.pt of the OLD 32k gelu run = step 9500 — batch4's
  "32kfinal" critic/advantage deploys are EXACT DUPLICATES of the
  interim ones (same ckpt, same seed). Don't double-count.
- actor_bat1/base ran at lr 1e-3 (pre-default-change) — bat1 arms are
  internally lr-matched at 1e-3, NOT at the new 3e-4 default.
- Smoke run throwaway_otf_smoke evaluated 8000 val cells (no
  val-t-filter) — ignore its metrics.
- /tmp snapshots: actor32k_best_snap.pt = step 9500, actor32k_last_snap
  = step 18000 of the gelu 32k run.

## Addendum 11 — PATHWISE PIVOT (2026-06-12 ~21:00 laptop; commits through pathwise toy)

USER DIRECTION: stop spending 16/17 of compute scoring R-IID candidates
that are predictably bad. R-IID weighted-MLE = BURN-IN ONLY; afterwards,
directly maximize reward of the flow's own reparameterized samples
(end-to-end differentiable: action -> grid_sample -> frozen backbone ->
probe CE). throwaway/pathwise_toy.py, same actor arch, two equal planted
reward bumps per scene (collapse canary):
- MLE burn-in plateaus at sample reward 0.033 (16 R-IID rarely hit
  narrow bumps); pathwise: 0.033 -> 0.94 in 600 steps (ent=0), full
  collapse std 0.25 -> 0.001, ONE mode kept; naive entropy 0.01 bonus
  destabilizes (0.90->0.76) without preserving coverage.
- User thesis: collapse-to-the-RIGHT-answer is fine; harmful collapse ==
  OVERFITTING (sharpness beyond the model's conditional accuracy on new
  images); anti-collapse should come from IRREDUCIBLE state ambiguity.
- Key refinement: plain mean-reward keeps point-mass optima even under
  ambiguity; the BEST-OF-K objective (== deploy: sample 16, self-rank)
  makes calibrated spread STRICTLY optimal under ambiguity. Density then
  maps the policy's own conditional uncertainty. In flight:
  /tmp/pathwise_toy2.log — finite 2k-scene pool (overfitting enabled) x
  {plain, best-of-16}; pre-registered: plain val decays as std under-
  shoots conditional error; best-of-16 val holds.
- NEXT RUNG: pathwise fine-tune the real actor from tonight's bs8
  burn-in ckpt with best-of-K objective, judge by deploys.
Also this evening: GRU history (fourier-encoded taken viewpoints ->
GRUCell -> context head; history_dim config; cached path feeds replay,
OTF feeds own picks; ActorProposalPolicy tracks taken; maps asserted
off) committed; OTF advance-policy mismatch identified (training advance
= argmax over R-IID != deploy actor-proposals; fix = +1 fwd/step, NOT
yet implemented); marginal calibration probe + datamined lifts (scale
band 1.4x, sky 0.51x, x flat) in throwaway/marginal_calibration.py;
bs8 flagship (t1 OTF, compiled, 6.9 steps/s) ends ~00:50 then all-t
auto-starts (/tmp/post_bs8.log) — RECONSIDER the all-t run given the
pathwise pivot before letting it burn the night? (it's burn-in-grade
data either way). Suspend = PAUSE not deadline; beacon
/tmp/clock_beacon.log.

Addendum 11 results (finite 2k-scene pool, held-out eval,
/tmp/pathwise_toy2.log): plain pathwise collapsed to the CORRECT
conditional argmax out-of-sample (val mean 0.956 stable, std 0.001) —
overfitting decay did NOT trigger (toy conditional map too learnable;
on real data, where placement-conditioning is known not to generalize
at 20k images, decay remains likely — deploys are the detector).
Best-of-16 objective: val best-of-16 0.986 > plain 0.962, with 20x the
spread (std 0.018, coverage 0.11-0.15) — deploy-matched hedging works
as theorized. Irreducible-ambiguity-preserves-spread remains weakly
tested (toy states encode modes cleanly). RECIPE: bs8 burn-in ckpt ->
best-of-16 pathwise fine-tune on the real actor -> deploys + maps.

## Addendum 12 — REAL pathwise stage launched (2026-06-12 ~21:40 laptop)

Run: actor_pathwise_bok8_from_bs8_t1only_c32 (/tmp/actor_pathwise.log).
Init: /tmp/bs8_burnin.pt = last.pt of the killed bs8 OTF-MLE run (TMP —
copy into a runs/ dir for durability). bs4, K=8, lr 3e-5, 20k steps,
keep_every 2000. The queued all-t MLE run was CANCELLED (waiter 18011)
— superseded by the pathwise pivot.

Logic and subtleties (READ BEFORE TOUCHING):
- Loss = mean over scenes of MIN-over-K per-image CE of the flow's OWN
  reparameterized samples == maximize best-of-K true reward. Best-of-K
  is the deploy objective (sample 16, self-rank): hedging across
  plausible argmaxes is part of the LOSS where the state is ambiguous,
  sharpness where it is not -> calibrated density (pathwise_toy:
  val best-of-16 0.986 vs plain 0.962, 20x sample spread).
- Gradients: t0 state built under no_grad (detached); the K sampled
  viewpoints enter sample_at_viewpoint (grid_sample — differentiable in
  the viewpoint), then frozen backbone + probe CE. Weights frozen;
  input-grads flow back to the flow params via reparameterized samples.
  K separate backbone graphs are held for backward => memory is the
  constraint => bs4/K8 v1 (no compile — untested under grad).
- train_loss in this run = best-of-8 CE (LOWER better, positive) — NOT
  comparable to the MLE runs' negative wnll. Expect ~2.0-2.5 region
  (per-image CE scale) falling if it works.
- val_spearman on cached cells still computed: under reward ascent the
  density sharpens, so log pi - log mu ranking may IMPROVE top-1 while
  bulk spearman over 16 R-IID drifts — top1_regret is the better key
  here. Deploys + density maps are the verdict. Maps auto-log to MLflow
  (value_maps/), percentile-clim renderer.
- Gradient flow only through the per-scene WINNING sample (min-CE) —
  sparse but standard; losers still shape the density via the flow's
  coupling across samples (shared context/params).
- Failure modes to watch: NaN via box-edge samples under atanh tails
  (EPS clamp protects log_prob; sampling path bounded by tanh — safe);
  reward hacking impossible (frozen probe); collapse detectable as
  sample-std -> 0 in maps + marginal_calibration probe.

Addendum 12b — BoK hedging, formal grounding (external-Claude exchange):
BoK hedges iff the K samples are scored under a SHARED draw of an
uncertain reward. OUR TRAINING SATISFIES THIS BY CONSTRUCTION: psi = the
unobserved image details given the lossy c32 canvas; each scene scores
all K samples under the same image's true CE. Hedging optimum = covering
p(argmax | canvas); self-anneals as conditioning improves (collapse-to-
right-answer = correct). Naive-BoK-collapses critique applies to FULLY
OBSERVED toys — explains pathwise_toy coverage decay (states encoded
modes cleanly). K ~ reparameterized temperature (~log K nats of
selection sharpening; BoN-KL bound). TODO: tau-softened CEM-style
credit over K samples as a flag (winner-starvation guard); synthetic
toys must HIDE reward structure from the state or they test the wrong
regime.

## COMPACTION HANDOFF 4 (2026-06-12 ~22:00 laptop; crockett -13.5 min)

READ ADDENDA 11-12b FIRST (pathwise pivot, BoK theory, launch subtleties).

### IN-FLIGHT (the only GPU job)

actor_pathwise_bok8_from_bs8_t1only_c32 — best-of-8 pathwise reward
ascent, bs4, lr 3e-5, 20k steps, eval_every 500, keep_every 2000,
init /tmp/bs8_burnin.pt (= step-13000 last.pt of the killed bs8 OTF-MLE
run; ALSO still in runs/actor_flow_otf_bs8_relutrunk_t1only_c32/ —
copy the /tmp snapshot somewhere durable). Log /tmp/actor_pathwise.log.
NO long-term watcher armed (the first-steps one fired) — ARM ONE.
train_loss = windowed best-of-8 per-image CE (positive, lower=better,
~0.65 at step 50). CE PATH VALIDATED: candidate_ce reproduces cached
dataset CE to 3 decimals (k=1/2/3 of val cell 0: .2791/.3096/.2849 vs
.2782/.3068/.2827). My "CE should be ~2.2" alarm was FALSE MEMORY —
per-scene CE is broad (99% inter-scene variance), 0.65 batch values
are normal. Crash #1 was the GRU-history files being uncommitted while
train_actor referenced history_dim — fixed, all committed.

### BASELINE for the pathwise curve (just measured)

actorprop_adv_k16_otfbs8burnin_t5_c32 (self-ranked, no critic, full
val): t1 41.17 / t2 41.98 / t3 42.45 / t4 42.58. References: EG-C2F
41.04 (t1) 43.17 (t4); cached-MLE self-ranked band t1 40.9-41.3, best
t4 42.68. NOTE: 5 epochs of FRESH data, fit gap 0.0022 — already
matches actors trained 3-10x longer. NEXT: same 50s deploy per retained
pathwise ckpt (step_002000.pt, ...) -> mIoU-vs-pathwise-steps curve vs
this row; step-500 value maps rsync+open (sharp-vs-spread per scene);
throwaway/marginal_calibration.py on pathwise ckpts (collapse watch:
sample std, scale/sky lifts vs data targets 1.4x band / 0.51x sky).

### PAUSED / AMBIGUOUS / SUBTLE

- bs8 freshness experiment: paused at step 13000 (5.1 ep), val 0.2337
  RISING, gap 0.0022. Verdict needs ~30-38k steps (cached rollover at
  12-15 ep = bs8 step ~30-38k). Warm-resume via --init-from (weights
  only — optimizer moments + LR schedule RESET; at lr 3e-5-7.5e-5 a
  minor tax, but note when comparing curves). User: "not full restart
  if we reuse weights".
- GRU history (actor.py fourier->Linear->GRUCell->context head;
  history_dim): committed but NEVER SMOKED. Default off. Smoke before
  any history arm. Value maps asserted off with history.
- OTF MLE advance mismatch: training advance = argmax-self-advantage
  over R-IID candidates != deploy (actor proposals). Fix designed
  (+1 fwd/step: sample 16 from actor for the advance only) but NOT
  implemented — only matters if all-t OTF MLE is revived; the pathwise
  line samples from the actor by construction (no mismatch when it is
  extended to t>=2).
- All-t OTF MLE run: CANCELLED pre-launch (waiter killed) — superseded
  by pathwise. The all-t harvest smoke PASSED (520 fwd/traj exact).
- Pathwise TODOs: tau-softened CEM credit over the K samples (flag;
  winner-starvation guard); t>=2 extension (rollout under own samples —
  naturally on-policy); compile under grad untested (run is eager).
- Toy caveat (12b): synthetic reward toys MUST hide reward structure
  from the state, else they test the deterministic regime where BoK
  collapses (explains pathwise_toy coverage decay).
- Suspend = pause-not-deadline; beacon /tmp/clock_beacon.log decides
  overnight behavior; crockett clock -13.5 min vs laptop.
- bs32 OTF twin run dir (actor_flow_otf_relutrunk_32k_t1only_c32,
  killed @ step ~13k) and cached relutrunk twin (killed @ ~16k, ckpts
  4k-16k + trace deploys done) both retain curves/ckpts for analysis.

Addendum 12c (external-Claude refinements, verified and adopted):
(1) The BoK-rendered uncertainty is ALEATORIC-GIVEN-STATE (p(image|
canvas) ambiguity; contracts with better glimpses, not more data).
Model ignorance does NOT widen the density — asymptotically confidently
sharp even where perception is wrong. Do not read self-doubt into maps;
epistemic widening would need an explicit ensemble source.
(2) Optimal BoK allocation: 1-p_i prop. w_i^(-1/(K-1)) — a concave
K-tilt of the argmax posterior: flatter than proportional, modes below
~O(1/K) dropped (K=16). Maps = good SUPPORT estimator, biased weight
estimator; uniform-over-plausible look is expected, not miscalibration.
(3) Sharpening across ckpts confounds (a) fitting the fixed argmax
posterior and (b) better conditioning via better glimpses. Our fixed-
cell value maps ARE the frozen-canvas control isolating (a); residual
on own-rollout states = (b). KEY: t>=2 pathwise with terminal reward
credits disambiguating glimpses AUTOMATICALLY (backprop through canvas
updates into earlier glimpse choices) — the strongest argument for the
t>=2 pathwise extension.

### PRESERVATION RECORD: actor_flow_otf_bs8_relutrunk_t1only_c32 [user: good run, may resume]

Host: crockett ~/projects/CanViT-PyTorch-RL/runs/actor_flow_otf_bs8_relutrunk_t1only_c32/
Code: git_rev 6a210b379db8ba9e360e948f532b131d86e18234 (full config in manifest.json)
Checkpoints: best_spearman.pt + last.pt (= step 13000) + step_008000.pt
  + step_013000_pathwise_init.pt (DURABLE copy of the /tmp pathwise-init snapshot).
HPs (manifest, verbatim): on_the_fly, otf_k 16, otf_t_max 1 (t1-only via
  the NEW knob, val_t_filter 1), bs 8, lr 7.5e-5 (linear-scaled from
  3e-4@bs32), wd 1e-5, warmup_frac 0.1, steps 128000 (killed at 13000),
  eval_every 1000, keep_every 8000, relutrunk (trunk_norm True, relu),
  h192/ctx4/256/flow6x64, soft tau 0.3, scale_min 0.25, seed 0, nw 4,
  compile_model True (TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor-canvit).
State at kill: step 13000 = 5.1 epochs, val_spearman_t1 0.2337 (rising,
  = peak), train 0.2359, fit_gap 0.0022; throughput 6.9 steps/s solo.
RESUME (weights-only; optimizer/LR phase reset): same command + 
  --init-from runs/actor_flow_otf_bs8_relutrunk_t1only_c32/step_013000_pathwise_init.pt
  --run-name <new name>. Freshness verdict needs ~30-38k total steps.

### UNADDRESSED USER REQUESTS / OPEN TODOs LEDGER (as of compaction, 2026-06-12 ~22:15)

USER ASKS NOT YET DELIVERED:
1. Pathwise density maps on screen (standing deliverable) — run had not
   reached step 500 at compaction; NO long-term watcher armed on
   /tmp/actor_pathwise.log. Arm watcher, rsync+open maps, run
   marginal_calibration on pathwise ckpts (collapse watch).
2. Per-ckpt deploys of the pathwise run vs baseline 41.17/42.58 — none
   run yet (step_002000.pt etc. as they appear).
3. Recurrence arm: GRU history implemented per user design but never
   smoked, never trained — the user's original "synthetic test with
   recurrence" intent has no REAL-model arm yet.
4. OTF freshness verdict (user's central question this evening) —
   paused at 5.1 ep; needs warm-resume to ~30-38k steps.
5. Vectorized/batched candidate scoring (user: "16 sequential B=8
   forwards?????") — designed (chunked CE + batched K), NOT built;
   superseded in priority by pathwise but still wanted for throughput.
6. OTF-regime batch-size probe (bs probed only on cached regime; bs8
   chosen by user fiat, never probed) — relevant if OTF MLE revived.
7. tau-softened CEM credit over K samples (12b) — flag not implemented.
8. t>=2 pathwise with terminal reward (12c: BPTT credits disambiguation)
   — not implemented; strongest next move on the pathwise line.
9. Critic h768 activation-explosion probe — never run (same block class).
10. Full resume support (optimizer+sched state) — only weights-level
    init_from exists; user cared ("not full restart if we reuse weights").

OPEN QUESTIONS *TO* THE USER (asked, no answer):
- Wire marginal-calibration into the trainer eval loop, or keep as
  standalone probe?
- Value maps: shared color scale per row ("say the word")?
- 4e-4 vs 3e-4 peak lr — 3e-4 set by default; 4e-4 never tested.

PARKED BY USER (do not start unprompted): all-t cached MLE (superseded),
K=64/min_scale-0.05 dataset regen (moot if OTF/pathwise wins), PCL toy
(re-scope first), tune_actor HP search (superseded by pivot), fresh-seed
+0.429 replication and safebox re-deploys (dropped).

### PRE-COMPACTION USER ANSWERS (2026-06-12 ~22:20) — NEXT-SESSION MARCHING ORDERS

1. Do NOT jump to t>=2 pathwise. Stay at t1: tune HPs, understand,
   debug, activation analysis, training dynamics, FULL-VAL mIoU evals,
   reflect, heatmaps. MAY prepare the arch so the same ckpt extends to
   t>=2 later (no behavior change at t1).
2. Metrics overhaul for pathwise [user]: drop legacy metrics that stop
   making sense under reward maximization (wnll/spearman keys are
   MLE-era); compute per-image/batch mIoU AS YOU GO for train and val,
   accumulate properly (NEVER throw away metrics between logging calls
   — MetricWindow discipline), log continuously. "Ultimately mIoU is
   the truth."
3. Map color scales: my call, verified empirically.
4. Peak lr: 3e-4 stands ("whatever").

## ADDENDUM 13 — pathwise boundary collapse: autopsy, mechanism, fixes (laptop ~21:20 EDT; crockett clock ~13.5 min behind)

RUN: actor_pathwise_bok8_from_bs8_t1only_c32 (bs4, K=8 best-of-8, lr 3e-5
warmup 2000, init /tmp/bs8_burnin.pt = bs8-run step 13000). KILLED at step
~2050 (pid 32297, explicit kill). Artifacts kept: step_002000.pt, last.pt,
/tmp/pathwise_snap.pt (= step-1500 last.pt copy, crockett), metrics.jsonl.

COLLAPSE (onset ~step 1300, exactly as warmup lr crossed ~2e-5):
- val_wnll_t1 -0.87 -> +1.77, val_spearman 0.23 -> 0.04 at the 1500 eval.
- marginal calibration (throwaway/marginal_calibration.py, 64 t1 states x64):
  top-scale-bin lift 1.17 -> 10.46; y/x extreme-band lifts ~3x (sky band has
  DATA lift 0.51 — mass moved into provably bad regions).
- value maps: image-INDEPENDENT bright ring at the safe-box boundary, every
  scene, every scale slice (/tmp/pathwise_maps_step1500.png on laptop).
- objective (windowed best-of-8 CE) FLAT 0.654 -> 0.654; on the 16 autopsy
  states winner CE WORSENED 0.614 -> 0.638. Distortion without ascent.

AUTOPSY (throwaway/pathwise_autopsy.py @ 16549c2, ran on crockett,
log /tmp/pathwise_autopsy2.log):
- u-saturation: burn-in frac|u_c|>3 = 0.000 -> step1500 0.539 (HALF the
  distribution tanh-saturated); u_s |.|>4: ~0.01 -> 0.49. Step 2000 same
  (absorbed, stable).
- WHERE PARAMS MOVED (|delta| vs init norm): flow final layers 0.006-0.013
  on 14.8; flow hidden 0.13-0.2 on 115 — vs head 0.94/21, down 1.24/14.3,
  trunk 0.74/31, proj 0.52/8.3. THE COLLAPSE WENT THROUGH THE CONDITIONING
  STACK, not the flow weights: one ctx vector feeds all 6 MAF conditioners
  (max leverage); Adam integrates small-but-consistent conditioning grads
  into large drift while the flow's huge-but-oscillating grads (0.998 of
  norm) random-walk. The logged "grad reallocation to trunk/proj" at 1300
  was the flow's gradient DYING of saturation, not conditioning learning.
- dCE/d(y,x,s) at planted mid-box actions (16 states x 9 actions, s=0.55):
  per-state |grad| ~11.8, across-state mean (+0.85,-0.71,+0.67), SNR
  0.05-0.075 — gradient is ~93% state-specific (fine: it SHOULD be image-
  conditioned), and the mean dCE/ds is POSITIVE (pushes AWAY from s=1):
  boundary drift was NOT reward-driven.
- MECHANISM: instability kick (clipped direction-only steps, 99.8%% into
  the flow, lr ramping) + ABSORBING SET (tanh/sigmoid saturation = zero
  gradient; pathwise never evaluates log_prob, so unlike MLE there is no
  restoring force on u-space). Mass that saturates never returns. The
  action-space density ring is partly the atanh Jacobian diverging at the
  box edge.
- NUMERICS EXONERATED: pathwise per-K glimpse forwards run OUTSIDE autocast
  (fp32); flow zero-init was fine at burn-in (u well-calibrated, s mean
  0.556 = winner E[s]).

TOY FINDINGS (throwaway/pathwise_toy.py; all logs /tmp/toy_*.log crockett):
- SIGN BUG (fixed): burn-in passed +reward where CE expected ->
  mle_weights anti-fitted. All previous toy burn-in rows were adversarial
  starts. Correct burn-in: coverage 0.84-0.89, bok16 0.73.
- Mean-reward pathwise from CORRECT burn-in: stable (zero u-saturation,
  flow-grad-frac 0.6-0.9 vs real run's 0.998), reward 0.92+, but per-scene
  coverage collapses 0.84 -> 0.09 in 200 steps.
- SCORE-IDENTITY BUG (fixed @ 16549c2): entropy bonus on DETACHED samples
  has zero expected gradient (E_pi[grad log pi] = 0) — the entire first
  entropy sweep (0.01-1.0, all collapsed, alpha=1 wrecked reward 0.24) was
  a null experiment measuring added gradient noise. Correct max-ent keeps
  the sample attached (SAC-style); corrected sweep queued (trueent_*).
- Forward-KL anchor to frozen burn-in (--anchor-coef): beta=0.1 keeps
  coverage 0.33, best stoch-bok16 so far 0.97 (collapsed 1.0 / hedged opt
  1.5); beta>=0.3 pins to burn-in. Works as a dial; user verdict: ugly,
  not understanding — superseded by correct entropy if it works.
- STOCHASTIC SHARED-REALIZATION REWARDS (--stochastic): two modes'
  amplitudes a1~U[0,2], a2=2-a1 redrawn per scoring call, one draw shared
  across K. Hedged best-of-16 optimum 1.5, any collapse 1.0; mean
  objective indifferent (linearity) — measures the collapse PRICE.
- EXPRESSIVENESS (throwaway/flow_expressiveness.py: pure conditional MLE
  on samples from known two-bump GT, no RL): affine MAF fits fine —
  coverage 1.000, val NLL -3.35 @2k steps lr3e-4, -3.55 @lr1e-3, still
  descending at 8k probe (GT bound ~ -4.5). rq-spline (now a SafeBoxFlow
  option, transform='rq-spline') is NOT better at equal budget (-3.16).
  ARCH IS NOT THE BOTTLENECK; lr matters more than transform.

TRAINER (committed @ 16549c2): pathwise_objective mean|min (mean default,
user directive), pathwise_ent_coef (attached), windowed canaries
u_sat_center/u_sat_scale/sample_logp. Per-step isfinite assert REMOVED
everywhere (ANY sync is a sync — user); guard moved into emit_checked() at
the log boundary. CLAUDE.md rule zero added.

USER DIRECTIVES THIS BLOCK: mission = stable+fast end-to-end flow training
through CE; easy/synthetic data is the acceleration vehicle; mean reward +
PRINCIPLED entropy (measure, don't guess), not best-of-K; anchor coef ugly;
always have GPU work queued; plot GT/image context next to heatmaps; mind
wallclock (date regularly); memory: pathwise-flow-training-directive.md.

## ADDENDUM 14 — toy verdicts: entropy is not the answer, hybrid is (laptop ~21:45 EDT)

All arms: throwaway/pathwise_toy.py --stochastic (shared-realization amps,
hedged best-of-16 optimum 1.5, any collapse 1.0), 800 burnin + 800 pathwise
unless noted, logs /tmp/toy_*.log (crockett), heatmaps outputs/pathwise_toy_*.

CORRECTED-ENTROPY SWEEP (attached samples): alpha=0.03 collapses (cov 0.13);
0.1 fragile middle (cov 0.19); 0.3 holds cov 0.86 but NO ascent (reward 0.05)
and eventually NaNs (entropy pushes samples into the EPS-clamped atanh edge;
action-space log-det diverges — boundary numerics again, caught by the
report-boundary NaN guard). STRUCTURAL: on-policy pathwise feels reward only
where it samples; entropy adds zero reward information about dark modes.
Max-ent pi* prop exp(r/alpha) is bimodal but unreachable by local
sample-based ascent once a mode goes dark.

WHY WEIGHTED MLE WORKED ALL ALONG: scoring R-IID candidates = OFF-POLICY
reward evaluation — information from regions the policy does not sample.

HYBRID (--mle-coef: pathwise mean ascent + continued R-IID weighted MLE),
matched 1600-step budget, stoch-bok16 / coverage:
  mle-only 1600:   0.766 / 0.83
  hybrid 1.0:      0.794 / 0.81
  hybrid 0.3:      0.865 / 0.74   <- pathwise ACCELERATES the recipe (+0.10)
First measured win for end-to-end differentiability on this problem.

OBJECTIVE TENSION (flag to user, measured): under the MEAN objective with
shared-realization noise, collapse-to-either-mode and perfect hedging have
IDENTICAL expected reward (linearity) — the mean objective does not care
about the multimodality the irreducible-uncertainty thesis wants. NO arm of
any kind exceeded stoch-bok16 = 1.0 (the collapse ceiling); the 1.0 -> 1.5
hedging premium is only optimized by best-of-K-style credit under shared
draws — i.e. the deploy protocol itself. hybridbok arms (mle 0.3 + BoK
credit) in flight = the only config that could beat collapsed.

PRINCIPLED-ALPHA NOTE (for any future real entropy term): the validated MLE
tilt exp(z/0.3) corresponds to alpha = 0.3 * sigma_within-scene ~ 0.008 nats
of CE — data-derived anchor, brackets {0.003, 0.01, 0.03}.

REAL-RUN IMPLICATION: hybrid real = OTF freshness loop (already built: fresh
scored R-IID cells per visit) + pathwise term on the same batch. Trainer
already has pathwise_objective/pathwise_ent_coef + u_sat canaries @ 16549c2.

## ADDENDUM 15 — real alpha sweep launched; hybrid+BoK toy breaks the collapse ceiling (laptop ~22:00 EDT)

USER DECISION (overrides my BoK push): real runs use MEAN reward + entropy
regularization, single reward maximization, swept — "you need to sweep your
shit and give me mlflow curves".

LIVE on crockett (chain pid 57387, sequential): 4 arms x 5000 steps,
actor_pathwise_meanK8_ent{0,0003,001,003}_t1only_c32 — pathwise mean K=8,
ent_coef in {0, 0.003, 0.01, 0.03} (0.01 ~ data-derived alpha: validated MLE
tilt = 0.3 * sigma_within ~ 0.008 nats), init /tmp/bs8_burnin.pt, bs4 lr 3e-5,
eval_every 500, keep_every 2000, logs /tmp/sweep_ent*.log, ~45-55 min/arm.
Canaries in MLflow per run: u_sat_center, u_sat_scale, sample_logp +
grad_norm_* shares. Watch: train_loss (windowed mean own-sample CE) DOWN,
sample_logp rising = sharpening rate (alpha should set its plateau),
u_sat_* near zero = no boundary collapse. MLflow http://localhost:5500
(tunnel verified 200 at 22:00 laptop). Monitor armed on all 4 logs
(milestones + failure signatures).

TOY RECORD — hybrid+BoK (mle_coef 0.3, best-of-K credit, stochastic), 8k
steps: stoch-bok16 0.84 -> 1.19 (ONLY arm ever past the 1.0 collapse
ceiling; hedged opt 1.5), coverage -> 1.000, u-sat ~1%, still rising at 8k.
Measured alternative if mean+entropy sweeps disappoint; log
/tmp/toy_hybridbok030_8k.log, heatmaps outputs/pathwise_toy_hybridbok_030*.

## ADDENDUM 16 — real sweep KILLED (wrong sample allocation); toy battery, auto-alpha (laptop ~22:20 EDT)

USER (emphatic): real sweep was wrongly designed — bs4 x K=8 samples/scene.
THE DESIGN IS MANY SCENES, ONE SAMPLE PER SCENE (batch over scenes, not K
per scene): 99%% of CE variance is inter-scene, so K>1/scene wastes forwards
on correlated rewards under a mean objective. K>1 only pays for BoK credit.
Killed chain 57387 + trainers 57388/57392 (explicit pids). Partial run dirs
runs/actor_pathwise_meanK8_ent0_t1only_c32 remain on crockett (record).
ANY FUTURE REAL PATHWISE RUN: --batch-size >=16 --pathwise-k 1 (t0 forward
per scene dominates; consider t0-state reuse later — full-canvas caching is
infeasible at ~4MB/image).

ALSO USER: the "entropy can't preserve modes" claim was OVERREACHED — 3
alpha points, 800 steps, old allocation; alpha=0.3 died of clamp-edge
gradient explosion (atanh derivative ~5e3 at the 1e-4 clamp; fp32 sq-norm
inf -> NaN), a numerics artifact not a verdict. Retest queued; user recalls
POCs showing entropy preserves modes.

TOY BATTERY (serialized chains on crockett, logs /tmp/toy_1samp*.log,
/tmp/toy_autoalpha*.log, /tmp/toy_lr1e*.log; monitor armed):
1. corrected allocation (512 scenes x 1 sample), ent {0.003,0.01,0.03,0.1};
   early: reward mean 0.96 (vs 0.92 old allocation), low-alpha still
   collapses coverage (0.11).
2. long-budget ent {0.15,0.2,0.3,0.5} @ 3200 steps — the unprobed middle.
3. AUTO-ALPHA (SAC dual; --auto-alpha-drop): H* = H(burn-in) - drop nats,
   alpha_loss = log_alpha*(H - H*); drops {0.5,1.5,3.0} @ 3200. Plus lr
   probe {1e-4,1e-3} at ent 0.01 (pathwise-phase lr was never swept — user
   called the gap).

## ADDENDUM 17 — WASTE LEDGER (rule added to CLAUDE.md; entries hereafter logged at discovery time)

1. REAL ALPHA SWEEP, WRONG ALLOCATION (~20 GPU-min + a misdesigned 3h plan).
   Launched 4x5000-step runs at bs4 x K=8 samples/scene under a MEAN
   objective despite knowing 99%% of CE variance is inter-scene — K>1/scene
   buys correlated rewards; only BoK credit justifies it. User caught it.
   Guard: addendum 16 directive — any real pathwise run is batch-over-scenes,
   K=1; allocation is part of run design review before launch.
2. DETACHED-ENTROPY NULL EXPERIMENT (4 toy arms + analysis + a WRONG
   STRUCTURAL CLAIM delivered to the user). E_pi[grad log pi(detached)] = 0
   (score identity): the entire first entropy sweep measured gradient noise.
   Cost compounded: I told the user "entropy can't preserve modes" off a null
   experiment; user pushed back; claim retracted. Guard: entropy terms keep
   samples attached (now in toy + trainer); structural claims need the
   mechanism verified, not just the sweep run.
3. TOY BURN-IN ANTI-FITTING SIGN BUG (every pre-fix toy burn-in row).
   mle_weights received +reward where CE expected — burn-in actively avoided
   modes; all "pathwise recovers from burn-in" baselines were adversarial
   starts. Guard: riid_mle_loss factored to one place; sign documented.
4. INVERSE-PATH LOG-PROB IN ENTROPY TERMS (3 arms NaN-killed: fixed
   alpha=0.15, one long arm, auto-alpha d0.5; ~10 GPU-min + the alpha>=0.15
   region blocked all evening). atanh EPS-clamp derivative ~1/EPS explodes
   once the flow concentrates. Guard: SafeBoxFlow.sample_with_log_prob
   (forward u-space, exact softplus closed forms) + contract tests; all
   own-sample log-pi terms use it.
5. BOK8 REAL RUN TRAINED ~700 STEPS PAST COLLAPSE (~25 GPU-min). Collapse at
   ~1300 was visible in grad-share reallocation but no canary metric
   existed; killed at ~2050. Guard: u_sat_center/u_sat_scale/sample_logp
   windowed canaries in every pathwise run.
6. REDUNDANT DELIVERABLES (user attention): collapse-curves figure
   duplicated what MLflow already showed ("stuff i could see 30 minutes
   ago"); deploy eval launched on a checkpoint the user could already call
   dead (also crashed on a shape assert). Guard: plots add information not
   already on a dashboard; ask what decision the artifact changes.
7. PER-STEP GPU SYNCS WRITTEN THREE TIMES (.item() EMA, isfinite asserts;
   user escalation x3). Guard: CLAUDE.md rule zero, no exceptions; sync
   audit of any loop before running.

## ADDENDUM 18 — ROOT CAUSE of the near-singular flow events (laptop ~21:55 EDT)

The d3.0 auto-alpha "freeze" decomposed into THREE findings (diag arm
/tmp/toy_autoalpha_d3_diag.log, H-train instrumentation):
1. THE DUAL WORKS: H-train tracked H* = -5.105 within ~0.05 nats for 1600
   steps, eval-H agreeing — SAC-style target entropy is validated in the toy.
2. ONE batch then produced log pi ~ +1e4 on own samples. Root cause READ FROM
   NFLOWS SOURCE (transforms/autoregressive.py): sampling runs
   _elementwise_inverse u = (z - shift)/scale with scale = softplus(raw)+1e-3.
   The 1e-3 floor protects the FORWARD direction but makes the INVERSE
   explosive: entropy pressure pushes sum(log scale) DOWN (that is how a flow
   broadens), raw goes very negative for some contexts, scale pins at the
   floor, the inverse multiplies by ~1000 PER LAYER -> |u| ~ 1e3-1e4 -> tanh
   fully saturated -> |log det da/du| ~ 4|u| -> log pi_a ~ +1e4. The entropy
   force ITSELF drives the parameterization into the explosive regime.
3. The unclipped dual gradient (-11700) poisoned alpha-Adam's second moment
   (beta2=0.999) -> effective alpha lr ~4e-6 for thousands of steps ("alpha
   frozen"). Actor side survived because actor grads are clipped.

INTERIM (user: "arbitrary bullshit clamps... temporarily"): winsorize log pi
+-30, clamp dual error +-10 — inert in healthy regimes, removed once the
structural fix lands.
STRUCTURAL FIX IN RACE: rq-spline transform (already a SafeBoxFlow option) is
inverse-stable BY CONSTRUCTION — linear tails = identity outside |u|=6 (no
amplification), bounded bins inside. Queued head-to-head on the failing
config: autoalpha_d3_rqspline + d15_rqspline vs the winsorized-affine guard
arm. Backup design if splines lose on sharpness: bounded affine
parameterization (shift = 8 tanh(raw/8), log scale = 1.5 tanh(raw/1.5)),
justified by data support |u| <= atanh(1-1e-4) ~ 5.

AUTO-ALPHA SCOREBOARD (smooth estimator, 512 scenes x 1 sample, stochastic):
  d0.5: reward 0.31@3200 ascending, cov 0.74, dual converging — healthy
  d1.5: reward 0.50@3200 ascending, cov 0.59, alpha eq ~0.13-0.15 — BEST
        (agrees with fixed alpha=0.15 arm: cov 0.74, reward 0.34@1800)
  d3.0: spike event (above); guarded rerun in queue
  12.8k d1.5 PROOF RUN queued (heatmaps = the gate artifact for bigger runs).

## ADDENDUM 19 — ENTROPY IS MODALITY-BLIND (the decisive toy result, laptop ~22:00 EDT)

NLL-FIT CEILING (throwaway/flow_expressiveness.py, affine flow, plain NLL on
GT samples, 3000 steps, ~30s): val NLL -3.67, coverage 1.000. Scatter
(outputs/flow_expressiveness_scatter_nllfit_affine.png, NEW sample-scatter
viz: flow samples in (y,x) colored by scale, modes marked) shows TWO clean
tight clusters per scene, each on its star, cluster color matching that
star's scale. THE AFFINE FLOW FITS THE BIMODAL TARGET PERFECTLY — position
AND scale. Architecture is NOT the bottleneck; the earlier "ugly ramp"
heatmaps were a fixed-scale-slice + top-25-nat-clip viz artifact (replaced).

THE RESULT: this perfect bimodal fit has H = -3.67. The d1.5 mean+auto-alpha
run has H ~ -3.5 at coverage 0.5. SAME ENTROPY, HALF THE COVERAGE. Entropy is
a scalar that constrains total spread, not modality: a single broad blob and
two tight scale-matched blobs can have identical differential entropy, and
the MEAN objective (provably indifferent to hedging under symmetric
shared-realization noise) accepts either. So NO target-entropy setting forces
bimodality — at any H*, a collapsed-correct-spread solution satisfies both
terms. Confirms the d3.0->0.25 / d1.5->0.5 ordering: more entropy buys more
SPREAD (incidentally more scenes keep both modes), never targets coverage.
Answers user's "maybe target too high": NO, target is calibrated to the data
entropy (-3.67); the level is right, entropy is just the wrong lever.

IMPLICATION: mean+entropy is stable and understood but CANNOT deliver the
clean per-scene bimodality the irreducible-uncertainty thesis wants, because
the toy's uncertainty (symmetric amplitude noise) is invisible to a mean
objective by construction (linearity). The two levers that broke the
stoch-bok16=1.0 collapse ceiling earlier: best-of-K credit (rewards hedging
directly, hit 1.19) and the off-policy MLE term (teaches both modes from
scored candidates, hybrid 0.865 vs 0.766). TOY-DESIGN CAVEAT: the real
per-image CE reward is NOT symmetric-equal-modes; whether real aleatoric
structure pays hedging under a mean objective is a SEPARATE empirical
question this toy cannot answer.

OPEN DECISION for user: (a) accept mean+entropy's stable ~0.5 coverage and
test on real data (where modes are unequal), or (b) adopt BoK/hybrid credit
which provably preserves modes in the toy. Scatter reruns d15_scatter/
d30_scatter queued for the same-viz contrast.

## ADDENDUM 20 — REAL PATHWISE RUN LAUNCHED, clean metrics, step-0 validated (laptop ~22:17 EDT)

After the toy phase derisked the machinery (addenda 13-19), launched the real
maxent pathwise run [user: "lets do real experiment indeed... good metrics, no
warts"].

RUN: actor_pathwise_mean_entdrop15_t1only_c32 (crockett pid 82267, log
/tmp/pathwise_mean.log, code @ 29c1c15). Command:
  uv run python -m canvit_pytorch_rl.train_actor
    --run-name actor_pathwise_mean_entdrop15_t1only_c32
    --pathwise --pathwise-k 1 --pathwise-objective mean
    --pathwise-target-entropy-drop 1.5
    --init-from /tmp/bs8_burnin.pt --batch-size 32 --lr 3e-5
    --steps 8000 --eval-every 500 --keep-every 2000 --val-t-filter 1 --num-workers 4
Design: MEAN reward + SAC auto-alpha entropy (H* = H(init) - 1.5 nats), BATCH
OVER SCENES (bs32) / K=1 [user: not best-of-K, not hybrid; mean over scenes].
init = bs8 burn-in step 13000 (/tmp/bs8_burnin.pt, durable
runs/actor_flow_otf_bs8_relutrunk_t1only_c32/step_013000_pathwise_init.pt).
~512k glimpse forwards (8000 x 32 x 2). MLflow: canvit-pytorch-rl /
actor_pathwise_mean_entdrop15_t1only_c32 (localhost:5500).

METRICS OVERHAUL (committed @ 29c1c15, evaluate_pathwise): MLE-era
wnll/spearman/regret DROPPED from the pathwise path (meaningless when the actor
samples its own candidates). New honest set, full val (2000 img, paper
protocol), per eval:
  val_miou_t1_sample / _mode  -- mIoU after one sampled glimpse / the flow mode
  val_ce_t1_sample / _mode    -- the reward (per-image CE) being optimized
  val_miou_t0                 -- full-scene floor
  val_entropy, val_u_sat_center/scale  -- spread + boundary-collapse canaries
Checkpoint selection: val_miou_t1_sample (best_miou.pt). "mIoU is the truth".

STEP-0 (burn-in, pre-training) VALIDATES THE PIPELINE:
  val_miou_t0 0.3858  == paper EG-C2F t0 38.5  [plumbing correct]
  val_miou_t1_mode 0.4112  == EG-C2F t1 41.1 == earlier deploy 41.17  [baseline honest]
  val_miou_t1_sample 0.4081, val_ce 0.744/0.750, val_entropy -0.95 nats,
  u_sat_center 0.000 u_sat_scale 0.009  [no boundary saturation]
THE BAR TO BEAT: val_miou_t1 = 41.1. Watch: does it rise while u_sat stays ~0
(no collapse) and mode stays near sample (multimodal where ambiguous).

## ADDENDUM 21 — WHY PATHWISE FAILS: competing zoom-out attractor (mechanism found, laptop ~23:15 EDT)

After the real pathwise run died (boundary saturation as it sharpened), built a
known-answer fixation toy (small ADE crop pasted on a featureless field, mask
IGNORE off-paste so CE scores only the paste; optimal glimpse = the paste). Same
frozen backbone+probe, CanvasActor via actor_kwargs_from_cfg, pathwise CE
gradient. A CLEAN CONTROLLED CHAIN nailed the mechanism (all CPU/fast, canvit-free
where noted):

1. ARCH CAPABLE. fixation_toy.py --supervised (MSE sample->oracle viewpoint,
   clean gradient, no backbone): center_err 0.561 -> 0.000, mode_scale -> 0.300
   (exact paste scale), in ~1000 steps, both lr 1e-4 and 3e-4, u_sat 0. The
   conditioner+flow (4.48M params: trunk 2.0M, down 1.0M, head 0.85M, flow only
   0.40M) fit the scene->viewpoint mapping perfectly. NOT capacity/arch.

2. PATHWISE FINDS BASINS (user's intuition CONFIRMED). throwaway/pathwise_basin_toy.py
   (production SafeBoxFlow, identity init -> broad; reward = Gaussian basin at a
   target): concentrates to the basin even at width w=0.05 and even with a HARD
   cutoff (exactly zero gradient outside R=0.15, like CE's flat black-bg plateau),
   from broad init with only 0.5% of init samples in the basin. 'Flat landscape'
   is NOT the killer; rare in-basin samples get pulled in and suffice.

3. THE KILLER = a COMPETING WIDE-SHALLOW ATTRACTOR. Add a decoy: reward 0.6 at
   scale~1 (zoom out) for ANY center, vs the deep narrow basin (1.0) at target.
   The flow ABANDONS the target and collapses to zoom-out: mean_scale 0.62->1.00,
   mode_dist 0.42->0.70, reward pinned at 0.6. EXACT reproduction of every real
   run's signature (mode_scale drifts UP, center never converges, settles for
   mediocre CE). The real CE has this structure: a big glimpse CONTAINS an
   off-center paste (moderate CE, wide easy basin) while a small off-center
   glimpse MISSES it (bad CE) -> the narrow fixation basin is only reachable once
   center is already right. Local pathwise ascent climbs the easy wide one.

4. ENOUGH ENTROPY ESCAPES IT. decoy 0.6 + ent_coef: 0.0 collapses (mode_dist 0.70,
   scale 1.0); 0.02 stuck; 0.10 ESCAPES (mode_dist -> 0.01, mean_scale -> 0.39 ~
   target, fixates). Maintaining breadth keeps sampling the deep basin until it
   (1.0 > 0.6) wins.

WHY THE REAL RUN FAILED, PRECISELY: auto-alpha targeted H* = H(init) - 1.5 nats =
SHARPEN. Driving entropy DOWN accelerated collapse into the zoom-out attractor --
exactly backwards. WHY WEIGHTED-MLE WORKS: it scores many R-IID candidates and
selects the best GLOBALLY, so it finds the deep basin and never falls into the
local attractor; pathwise climbs locally.

FIX (being confirmed on real CE now, /tmp/fix_entrescue_e*.log, ent_coef sweep
0.05/0.2/0.5 from broad fixed init): maintain HIGH entropy / anneal from very
broad (NOT sharpen from init), or supply off-policy candidate coverage (the
hybrid / burn-in). Also fixed this session: flow identity init (was std-9
saturated sampling from scratch -> blew up); fixation_toy has warmup+cosine,
grad_clip, density maps. ce_gradient_field.py probe still WIP (state-tiling bug).

## COMPACTION HANDOFF 5 (laptop ~23:18 EDT, code @ 50aa862, tree clean, both remotes synced)

THE HEADLINE (this session's result — addendum 21, established by a clean
controlled chain, NOT asserted): pathwise CE training fails to learn fixation
because of a COMPETING WIDE ZOOM-OUT ATTRACTOR, not capacity/arch/flat-landscape.
Evidence chain: (1) supervised MSE->oracle fits perfectly (center_err 0.000) =>
arch capable; (2) pathwise concentrates a broad flow even to a narrow/hard-cutoff
basin with 0.5% init overlap => flat landscape is NOT the killer, user's
stochastic-sampling intuition CORRECT; (3) adding a decoy (reward 0.6 at scale~1
any center vs deep 1.0 basin) makes the flow collapse to zoom-out (mean_scale->1,
mode_dist 0.42->0.70) = EXACT real-run signature; (4) entropy 0.1 escapes it
(decoy toy). => real run's auto-alpha SHARPENED (target H(init)-1.5), which is
backwards; weighted-MLE works because it selects GLOBALLY over candidates.

IN FLIGHT ON CROCKETT (sequential bash chains; this session's monitors/tasks do
NOT survive compaction — re-check by ssh+grep):
- entropy-rescue sweep (betas 0.9/0.999, the OLD pre-correction): logs
  /tmp/fix_entrescue_e{0.05,0.2,0.5}.log. e0.05 STILL RUNNING ~23:18 (too weak,
  center_err 0.59 — expected null); e0.2 and e0.5 QUEUED, NOT YET RUN. These are
  the first (weaker-optimizer) test of the fix.
- THEN ce_gradient_field.py (the confirmatory CE-landscape probe): logs
  /tmp/ce_field_scene{0,2}.log, figs outputs/ce_field_scene{0,2}.png — CE vs
  scale at true vs off center (zoom-out attractor = CE drops toward scale~1 at
  off center) + CE(y,x) at paste scale vs 0.85. Gate: /tmp/ce_field.done.
- THEN entropy-rescue at CORRECTED betas (0.8/0.9 [user], 6000 steps, coef
  0.5/1.0): logs /tmp/fix_entlowbeta_e{0.5,1.0}.log. Gate: /tmp/fix_entlowbeta.done.
  THIS is the proper end-to-end confirmation; reads success as center_err->0,
  mode_scale->0.30, CE_mode->~0.95 (oracle). All log to MLflow canvit-toys
  (fixation_* runs) with density-map + scatter + scene-panel artifacts.

HOW TO READ THE FIXATION TOY (throwaway/fixation_toy.py): n_scenes=8 ADE crops
pasted (paste_scale 0.3) on a featureless field, mask IGNORE off-paste so CE
scores only the paste. Deterministic-mode metrics (center_err, scale_err,
mode_scale, ce_mode) are NOISE-FREE (fixed set); samp/H/u_sat use 8 samples
(noisier). oracle CE ~0.946 (glimpse@paste), t0 ~2.125 (full scene). Flags:
--supervised (MSE->oracle, the arch-capability control, FITS), --ent-coef
(broaden), --init-from (burn-in /tmp/bs8_burnin.pt), warmup+cosine built in.

KEY EARLIER RESULTS THIS SESSION (full detail addenda 13-21):
- Real pathwise run actor_pathwise_mean_entdrop15_t1only_c32 (bs32 K=1 mean+auto
  alpha drop1.5 from burn-in) DIED: sharpened (H -0.95->-2.27) -> boundary
  saturation (u_sat 0->0.09) -> mIoU 41.1->40.1 by step 1000. Killed. Metrics
  overhaul shipped (evaluate_pathwise: val_miou/ce at t1 sample+mode, t0 floor,
  no MLE-era wnll/spearman). Step-0 validated pipeline (t0 38.58=paper, t1
  mode 41.1=EG-C2F).
- Flow identity-init fix (@flow_head.py): zero-init gave std-9 SATURATED sampling
  from scratch; now scale-channel bias = softplus^-1(1) => u~N(0,1). Burn-in
  ckpts unaffected.
- actor_kwargs_from_cfg factored (one source of truth, toy reuses prod arch).
- Actor = 4.48M params (trunk 2.0M, down 1.0M, head 0.85M, flow 0.40M) — huge
  for 8-scene overfit, so failure is optimization not capacity.

NEXT STEPS (priority): (1) read entlowbeta e0.5/e1.0 verdict — does maintained
breadth + reactive optimizer make real CE fixate? (2) if yes, the recipe is
"anneal from VERY broad / high entropy, never sharpen-from-init" — take to the
real T=5 run (replace auto-alpha-drop with a broaden/anneal schedule, or use the
hybrid off-policy term). (3) if no, run the ce_field figure to see how the real
landscape differs; consider reward shaping to kill the zoom-out attractor
(penalize large scale, or score CE at glimpse resolution). (4) the gradient-field
/ competing-attractor understanding should inform whether to keep pure-pathwise
at all vs the hybrid (burn-in does off-policy global search, pathwise refines).

NEW RULES THIS SESSION (repo CLAUDE.md): rule zero GPU syncs (no exceptions);
rule one "a number is not a trend" (like-for-like + bound noise); waste-ledger
discipline. New memory: training-dynamics-patience (transients + low betas).

## COMPACTION HANDOFF 6 (laptop ~00:54 EDT 2026-06-13, code @ f351649, tree clean, both remotes synced)

THE ARC THIS SESSION: pathwise CE failed on real ADE -> deep mechanistic
investigation via a fixation toy -> conclusion: VALUE-BASED (RWR/GRPO) wins,
pathwise gradient is unusable -> built a clean RWR trainer -> launched an
autonomous overnight RWR sweep on real ADE20k.

=== THE MECHANISM (why pathwise fails, measured not asserted; toys in throwaway/)
- fixation_toy.py: known-answer task — a small ADE crop pasted on a featureless
  field, mask IGNORE off-paste so per-image CE scores ONLY the paste; optimal
  glimpse = the paste. Same frozen backbone+probe, CanvasActor, c32.
- SUPERVISED (MSE sample->oracle viewpoint) fits PERFECTLY (center_err 0.000):
  arch/flow fully capable, 4.48M params (conditioner ~4.07M, flow only 0.40M),
  NOT capacity.
- PATHWISE CE: stuck (low lr) or unstable/boundary-collapse (high lr). The
  pathwise CE gradient w.r.t. viewpoint is APPROXIMATELY UNBIASED but
  texture-noise dominated: cos-to-paste ~0 at ALL distances (measured,
  ce_gradient_field.py), |grad| large near the paste but directionally random,
  ~0 on the flat background. WHY low SNR: DIFFERENTIATION AMPLIFIES HIGH
  FREQUENCIES — glimpse texture is high-freq in viewpoint space, its gradient
  amplitude ~ frequency, swamping the smooth basin's small gradient. SNR ~0.04
  => ~1/SNR^2 ~ 600 samples/step needed to average it out, vs ~16 for the VALUE.
  RETRACTED overclaims this session (user caught each): "informative inside the
  basin" (FALSE — cos~0 even near), "zoom-out attractor / CE drops at scale 1"
  (UNSUPPORTED — off-center CE-vs-scale is noisy, no clean trend; weak
  scene-dependent boundary outward pull frac_outward 0.55-0.65), "averaging
  can't recover it" (the averaging test pathwise_avg_gradient.py was CONFOUNDED
  by safe-box clamping near edge pastes — inconclusive).
- THE VALUE is clean while the GRADIENT is noise because differentiation
  amplifies the high-freq texture; value-based methods use the reward VALUE and
  skip the expensive useless backbone backward (~3x cheaper/sample AND it works).

=== THE METHODS (all on the flow; verified on the toy)
- RWR (reward-weighted regression = our "weighted-MLE"): score K fresh R-IID
  candidates by CE, fit flow log-prob to mle_weights = softmax(z_targets(-CE)/tau).
  WORKS. The flow is BOTH actor and critic: log pi - log mu ~ -CE/tau. Toy 8-img
  best (Optuna): center_err 0.231 at lr 3.08e-4, batch 4, mle_k 32, tau 0.113,
  betas 0.8/0.99. Density (saved) = SMOOTH, scene-conditioned, paste-covering,
  NON-COLLAPSING; scatter = broad clouds biased to each paste. Robust (R-IID
  always covers), soft (mode ~0.23 off, breadth from mu-anchor + tau).
  RWR vs AWR [user asked]: it is RWR — the z_targets mean-subtraction (the AWR
  baseline) is a NO-OP under softmax (shift-invariance); the real twist is the
  std-division = per-state std-normalized temperature (for heteroscedastic CE).
  True AWR would drop the softmax (w ~ exp(A/beta) unnormalized). GRPO is the
  genuinely advantage-based one.
- GRPO (on-policy group-baseline PG, group = 1 scene): sample G from current
  policy, z-score advantage WITHIN the group, score-function loss
  -(adv * log pi(samp)).mean(). Works DIRECTLY on the flow (needs only sample +
  exact log-prob; score-function => no backbone backward, value-based). FAST on
  1 image (center_err 0.56->0.09 in 100 steps, lr ~1e-4), but collapse-prone
  (near-delta) and coverage-dependent (on-policy: if it stops covering the
  target, no signal). No PPO clipping (single-step on-policy, user: "useless").
- RWR vs GRPO trade: RWR robust/soft/non-collapsing (off-policy R-IID uniform
  coverage); GRPO fast/sharp/collapse-prone (on-policy, init flow = centered
  mid-scale BELL, under-covers extremes — init_vs_riid.py).
- FAIRNESS [user caught]: per step, glimpse-forwards = supervised batch /
  pathwise 2*batch / rwr (1+K)*batch / grpo (1+G)*batch. RWR's apparent win
  used ~8-16x more compute; fixation_optuna.py now logs glimpse_forwards (fair
  axis) + grad norms + entropy + u-sat.

=== MEow (arxiv 2405.13629) [user asked] — applicable for the T>1 extension:
flow as actor+critic via energy-based flow, EXACT soft value from the partition
function, soft-Bellman TD on reward VALUES (sidesteps the pathwise gradient).
Principled multi-step generalization of what RWR does single-step. Needs the
EBFlow arch (not our MAF) + RL machinery; for T=1 RWR is simpler. Flow matching
= a different (continuous, regression-trained, sample-first) flow paradigm; not
what we use (MAF's cheap exact density is what the critic needs).

=== THE CLEAN RWR TRAINER (src/canvit_pytorch_rl/rwr_train.py) — NEW, the deliverable
Validated recipe productionized: one clean loop, fresh R-IID candidates scored
by CE, weighted-MLE loss, mIoU+CE POLICY-ROLLOUT eval (evaluate_pathwise reused;
its cfg type broadened to RunConfig), warmup+cosine, grad norms, MLflow
(canvit-pytorch-rl), checkpoint on best val_miou_t1_sample. RWRConfig defaults:
k=32, tau=0.15, lr=3e-4, betas 0.8/0.9, bs8, c32, t1, flow_transform affine,
relutrunk. Smoke OK: step-0 val_miou_t0 38.58 (paper floor), t1_mode 41.13
(EG-C2F), u_sat ~0. NOT the train_actor mode-soup.

=== AUTONOMOUS OVERNIGHT RUN (live, crockett pid 150601; this session's monitors
do NOT survive compaction — re-check by ssh+grep /tmp/rwr_sw_*.log):
1. RWR sweep: 4 configs lr{1e-4,3e-4} x tau{0.1,0.3}, k32 bs8 2500 steps (~30min
   each), real ADE t1/c32 from scratch. logs /tmp/rwr_sw_lr{LR}_tau{TAU}.log,
   run dirs runs/rwr_sw_*, MLflow canvit-pytorch-rl.
2. throwaway/rwr_pick_and_launch.py auto-picks best by val_miou_t1_sample, reads
   its manifest HPs, launches:
3. rwr_ade_long_t1_c32, 50000 steps (20x), SAME 5% warmup, eval_every 1000,
   keep_every 5000. log /tmp/rwr_pick.log -> the long run's own log.
   Gate flag /tmp/rwr_sweep.done. BASELINE TO BEAT: t1 mIoU 41.1.
   ETA: sweep ~2h, long run ~9h.

=== EARLIER (full detail addenda 13-21 + handoffs 3-5): pathwise bok8 collapse
autopsy; flow identity-init fix (was std-9 saturated sampling from scratch);
sample_with_log_prob (forward u-space, no clamp blowup); actor_kwargs_from_cfg
factored; the metrics overhaul (evaluate_pathwise: mIoU+CE, dropped MLE-era
wnll/spearman); bs8 burn-in preserved (runs/actor_flow_otf_bs8_relutrunk_t1only_c32
step_013000, git_rev f5551493).

=== NEW RULES (repo CLAUDE.md): rule zero (GPU syncs, no exceptions), rule one
(a number is not a trend), waste-ledger discipline. NEW MEMORY:
pathwise-flow-training-directive (updated w/ mechanism), training-dynamics-patience.

=== KEY THROWAWAY TOOLS: fixation_toy.py (supervised/pathwise/pathwise_ent/rwr/grpo
modes, density_map, scatter, warmup, betas), fixation_optuna.py (Optuna comparison,
glimpse_forwards/grad-norm/entropy/u-sat logging, enqueue, ckpt+heatmap save,
{study}__{algo}_t{n}_{datetime} naming), pathwise_basin_toy.py, ce_gradient_field.py
(CE landscape + bias/boundary tests), pathwise_avg_gradient.py (confounded),
init_vs_riid.py, ckpt_scatter.py, rwr_pick_and_launch.py.

=== NEXT (after the overnight results): read sweep + long-run val_miou (beat 41.1?);
if RWR works on real ADE, extend to T>1 (consider MEow framework) and/or compare
GRPO at matched glimpse-budget on real data; the fixation_optuna comparison sweep
(algo_cmp8) was killed mid-way — resume with seeded HPs if a fair toy comparison
is still wanted.

## COMPACTION HANDOFF 6b — final state + POST-COMPACTION ACTION (laptop ~01:05 EDT 2026-06-13, tree clean)

RWR TRAINER NOW LOGS PROPERLY (was too sparse): log_every=25 (windowed
train_loss/mean_ce/grad-norms/lr every 25 steps) + density filmstrip every
eval (logger.log_figure "density": 4 fixed val scenes, denorm image + log pi at
scales 0.3/0.55). Chain RELAUNCHED with these (new pid 154617, old killed).

MEASURED THROUGHPUT (rwr_train, bs8, k32, real ADE): ~1.85 opt steps/s, 264
glimpse-forwards/step (bs8*(1+32)), ~490 glimpse-forwards/s (all no_grad
candidate scoring; flow backward trivial). GPU 100% util but only 4.9 GB —
COMPUTE-bound, not memory-bound [user: "memory headroom is not compute
headroom" — parallel runs just time-share the saturated GPU, do NOT parallelize].
ETAs (crockett ~01:01): per sweep config ~25 min; 4-config sweep done ~02:40;
50k long run done ~10:30; total chain ~9.5h.

OVERNIGHT CHAIN LIVE (crockett pid 154617; survives compaction; re-check
/tmp/rwr_sw_*.log + MLflow canvit-pytorch-rl): 4-config RWR sweep (lr{1e-4,3e-4}
x tau{0.1,0.3}, k32 bs8 2500 steps) -> rwr_pick_and_launch.py auto-picks best
val_miou_t1_sample -> rwr_ade_long_t1_c32 50k steps (20x, same 5% warmup).
Baseline to beat: t1 mIoU 41.1. Step-0 evals confirm val_miou_t0 38.58.

>>> POST-COMPACTION ACTION [user, explicit]: queue INDEFINITE ~1h RWR HP-tuning
jobs on real ADE so the GPU never idles. Tool BUILT and committed:
throwaway/rwr_optuna_ade.py — Optuna, each trial a full ~1h rwr_train (lr/tau/k/
batch/beta1 sampled), MAXIMIZE val_miou_t1_sample, runs forever, logs to MLflow
(canvit-pytorch-rl) with mIoU+CE+grad-norms+density. Launch (ONE worker, GPU
compute-bound):
  ssh crockett 'cd ~/projects/CanViT-PyTorch-RL && nohup env OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 nice -n 5 uv run python throwaway/rwr_optuna_ade.py \
    --steps 6000 --study-name rwr_hpo_ade > /tmp/rwr_hpo.log 2>&1 & '
Start it AFTER the current chain frees the GPU (gate on the long run finishing,
or run it as the perpetual successor). Keep ONE worker (parallel = time-share).
