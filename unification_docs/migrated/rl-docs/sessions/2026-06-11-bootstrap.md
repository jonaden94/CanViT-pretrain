# 2026-06-11 — repo bootstrap (Claude Code)

Context: same-day full workspace reset (Codex era archived; see
`../../../SESSION_LOG_2026-06-11.md`). User directives: archive is untrusted
("tons of stupid reimplementation, duplication, subtly incorrect stuff");
run fresh code on crockett, proceed cleanly; document understanding once,
in one place.

## What was learned (the expensive part)

- The m2b3 release stack already provides almost everything the Codex repo
  reimplemented: `canvit_eval.episode.run_episode` + `Policy` protocol,
  `make_policy` with `entropy_coarse_to_fine` (EG-C2F: 3 C2F levels =
  1 full + 4 half + 16 quarter tiles = 21 viewpoints; within a level,
  descending mean probe entropy), `canvit_specialize.datasets.ade20k`
  (squish transforms, -1/ignore mask convention), `mIoUAccumulator`
  (dataset-level: sum I/U across images, mean over classes with union > 0,
  ignore filtered on target only).
- **Eval protocol fork**: upstream CanViT-eval squish-resizes MASKS to 512
  (its released 45.9% T=21 headline); the Codex anchors (EG-C2F T=2 42.23,
  oracle 45.58) used original-resolution masks. We chose original-res as
  primary. This single detail decides whether numbers are comparable.
- ADE20K on crockett: `/datasets/ADE20k/ADEChallengeData2016` (2000 val);
  both frozen checkpoints already in crockett's HF cache; GPU idle.

## What was built (commit 57aae20 + doc restructure)

- `pyproject.toml`: deps on the three m2b3 repos via git + cu128 wheels on
  Linux; CPU wheels on macOS for laptop lint/test.
- `data.py`: thin wrapper over upstream `ADE20kDataset` — original-res
  masks, image names, stride/limit subsetting. (First version duplicated
  upstream globbing/conventions; user caught it, rewritten to delegate.)
- `metrics.py`: per-image per-class I/U + dataset mIoU + bootstrap CI;
  `test_metrics.py` pins exact agreement with upstream accumulator
  (4 tests pass on laptop CPU).
- `evaluate.py`: the one evaluator → `runs/<name>/` artifact bundle.
- `plots.py` (efficiency curve, action distributions), `list_runs.py`
  (`just runs`), justfile.
- Remotes: origin = m2b3/CanViT-PyTorch-RL (private GitHub, primary),
  deploy = bare repo on crockett, checkout `~/projects/CanViT-PyTorch-RL`.
- Name history: glimpse → CanViT-RL → CanViT-PyTorch-RL (user).

## State at session end

- `uv sync` on crockett: in progress at write time.
- Next: pytest on crockett, smoke eval (`--limit 16`), then EG-C2F T=2 on
  full val vs the ~42.23 anchor; then the best-of-K oracle path.
- Not yet built: compare CLI (paired run-vs-run deltas), oracle evaluator,
  MNIST-on-canvas env, learned policies.

## Part 2 — paper grounding and milestone 1 (same session, later)

User interventions reshaped the harness mid-build; chronologically:

- Anchor test v1: EG-C2F T=2 full val (orig-resolution masks) gave 41.754%
  vs the 42.23 anchor → diagnosed the protocol fork (Codex's
  "paper-compatible" = squish-512 masks, NOT original resolution).
  Dual-protocol scoring confirmed: scene 42.219 ≈ anchor 42.231 (`ff29289`).
- User supplied the actual paper (arXiv:2603.22570) + appendix Tables 4/5:
  our t0/t1 (39.602/42.219) match the paper's 39.6/42.2 EXACTLY. The Codex
  anchor was just the paper's t1. Directives encoded in CLAUDE.md: paper
  protocol always; Figure-4B overlay compatibility; wallclock alongside
  budgets; no unconditional mIoU CIs (image-resampling bootstrap is
  rare-class-dominated and not a paper statistic).
- Dual-protocol scoring then DELETED (user: "idk what the fuck this means")
  — diagnostic scaffolding, question answered, single paper protocol
  remains (`004b859`). Side win: squish masks batch → vectorized
  offset-bincount per-image I/U, no custom collate.
- Refactor to `harness.py` (RunConfig/setup/write_run shared by all entry
  points); `oracle.py` added (best-of-17 t1, CE selection, persists EVERY
  candidate's viewpoint+CE+I/U per image — user: never waste forwards).
- `paper_reference.py` + `figure4b.py`: Table 4 pinned in code; overlay +
  per-timestep ours-vs-paper verification with the paper's own n=10 CIs.
- Name history note: GitHub repo is m2b3/CanViT-PyTorch-RL (private),
  description deliberately broad ("RL on CanViT").
- Verification queue at session end: six paper policies T=21 c64 + oracle
  (queue4, `004b859`); codex exec reviewing oracle correctness in parallel.

Key lesson encoded: `pkill -f` over ssh kills its own session before any
chained commands run — kill and relaunch in SEPARATE ssh invocations
(archive's AGENTS.md documented this; it bit anyway).

## Part 3 — RL work begins (same day, evening)

- User redirected from paper-curve replication to RL proper; scope set:
  **c32 only until told otherwise; T=5 horizon; full val always (10 s at
  c32 T=2); LOUDLY flag partial evals** (mechanized in write_run).
- Throughput benched (docs/benchmarks.md): c32 ~1,350 glimpses/s infer /
  ~470 backward; c64 ~430 / ~150. 4090, bf16, frozen weights.
- Codex-saved oracle datasets found and CERTIFIED (per-image CE corr
  0.9995 on the shared deterministic candidate), then superseded: we
  generate our own c32 candidate data (oracle.py, then the more general
  rollout_candidates.py which branches 17 candidates from EVERY state
  along a behavior rollout — behavior pick's forward doubles as the next
  base step).
- Critic stack landed: CandidateCritic (center-sampled features + global
  pool + Fourier action enc), train_critic.py (offline from candidate
  parquet, replay-based state rebuild, same-t batching, per-t eval
  metrics, MLflow sqlite + metrics.jsonl), CriticGreedyPolicy (re-scores
  a 16x16x4 grid from the CURRENT state at each t; --policy critic_greedy).
- Registered prediction [user]: t1-only-trained critic won't generalize
  to t>=2. A/B queued: --train-t-filter 1 vs all-t training on the same
  rollout dataset, both deployed greedily at T=2/T=5 vs EG-C2F (c32 refs:
  t1 41.1, t4 43.2).
- In flight at session end (self-sequencing nohup chains on crockett):
  valcand_rollout_egc2f_t5_c32 -> traincand_rollout_egc2f_t5_c32 (~55 min)
  -> critic_centersample_zce_rollout_c32 -> greedy evals; then the t1only
  ablation chain. Logs: /tmp/rl_overnight.log, /tmp/rl_ablation.log.
  MLflow server: crockett:5500, sqlite store runs/mlflow.db (file store is
  maintenance-mode in mlflow 3.6 — crashed the first training attempt).
- Gotchas hit twice: pkill -f over ssh kills its own session (kill and
  relaunch in SEPARATE ssh calls); zsh eats === literals.

## Part 4 — first results, scaling response, compaction handoff (evening)

Results (all c32 full val; milestones.md has details):

- Prediction CONFIRMED: t1-only critic flatlines (t4 41.02) vs multi-step
  critic climbing (t4 42.41); both LOSE to EG-C2F (43.17 at t4) at every t.
- Winner's curse measured: grid-argmax (1024 cands) ~0.3 pp worse than
  argmax over 16 fresh R-IID; critic_randomk TIES EG-C2F at t1 (41.05 vs
  41.04) but gap grows to -0.71 at t4.
- Sequential oracle (greedy true-CE best-of-17/step): 44.05/46.41/47.77/
  48.70 at t1..t4 — headroom GROWS (+3.0 -> +5.5 pp); oracle at 5 glimpses
  beats EG-C2F at 21 (44.1). Greedy framing is fine; critic quality is the
  bottleneck.
- Codex comparison answered precisely (see chat + archive-synthesis):
  their +0.44 t1 winner used 16.4M forwards and 25M params at c64; ours
  used 0.4M/1M-params and ties at t1 — on their scaling curve.

Scaling response (committed edd526c, NOT yet pulled on crockett):

- CandidateCritic now 20M params: residual conv trunk, 3x3 box-spanning
  feature sampling, probe ENTROPY MAP as explicit input (EG-C2F's signal),
  wider MLP. All call sites take (spatial, entropy, actions) via
  score_candidates().
- Standards set [user]: 1M glimpse-forwards = standard training budget
  (defaults batch 64 x 6250 steps); always assess under/overfitting
  (train-slice eval + fit_gap_spearman every interval, checkpoint selects
  on val_spearman_all); plot_training module + `just curves <run>`;
  T=k evals subsume T<k; goal set incl. <=45 min wakeups (memory:
  canvit-operating-goals).

In flight + NEXT ACTIONS for the successor session:

1. seqoracle chain (crockett pid 3556237, watcher set): traincand
   generation -> SMALL-arch critic A/B on oracle-advance states
   (critic_centersample_zce_seqoracle_c32 + _t1only) -> greedy T=5
   deploys. These use checkout a6e6fba (small critic) ON PURPOSE —
   matched-arch state-distribution comparison.
2. When chain completes: `git pull` on crockett (brings edd526c 20M
   critic), then launch 1M-budget training on traincand_seqoracle_t5_c32
   + critic_randomk T=5 deploys (3 seeds) + grid deploy for the curse
   re-check. Compare per-t vs EG-C2F refs and oracle ceiling above.
3. Old-format runs traincand/valcand_bestof17_c32 are superseded (rollout
   datasets cover t=1); delete when convenient.
4. MLflow sqlite server on crockett:5500 (laptop tunnel open;
   `ssh -f -N -L 5500:localhost:5500 crockett` to reopen).

Logs: /tmp/rl_seqoracle.log (chain), /tmp/mlflow_server.log. Stale t1-only
datasets and run dirs from the file-store mlflow crash were cleaned.

Post-compaction corrections (same evening):

- The chain's checkout is `d824280` (per valcand manifest git_rev), not
  a6e6fba — still pre-edd526c, so the small-arch A/B remains matched.
- `valcand_seqoracle_t5_c32` done (371 s, full val): base trajectory
  reproduces the sequential-oracle curve exactly. Candidate analysis
  yielded a diagnostic CORRECTION (oracle picks are centrally
  concentrated; low y-std was never the deploy deficit) — see
  milestones.md.
- Committed 3ffb6b5: CriticTrainConfig defaults = the 20M arch (384/2048),
  so step 2's training launch is the bare default command:
  `uv run python -m canvit_pytorch_rl.train_critic --run-name critic_boxent_zce_seqoracle_c32`
  then `evaluate --policy critic_randomk --n-timesteps 5 --seed {0,1,2}`
  and one `--policy critic_greedy` grid deploy (curse re-check), all
  with `--critic-checkpoint runs/critic_boxent_zce_seqoracle_c32/best_spearman.pt`.
