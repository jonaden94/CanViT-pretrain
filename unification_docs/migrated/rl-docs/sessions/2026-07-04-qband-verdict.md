# 2026-07-04 — qband 8-seed verdict (overnight band of the 2026-07-03 recipe)

Continuation of `2026-07-03-deep-study-bn-probe-depth-diag.md`; the band it launched completed 05:49
crockett time. Full result table + per-seed provenance: `docs/qband_results.md` (the canonical record —
this doc only logs what happened and what changed).

## What landed

- `qband_s0..7` all reached 8000 steps (640k forwards). Deploy band mean(t1–t4) val CE
  **0.6853 ± 0.0007**; mIoU_t4 44.97 ± 0.10. Matches the 1M-forward HEAD band within seed noise on both
  metrics at 64% of its compute; beats EG-C2F-c64 on both at every t. The recipe's single-seed promise
  (`qlr2e4_s0`, 0.6856) held under seeds.
- Rev span across the band (bcb9742f → 007f7173) checked commit-by-commit: docs, `throwaway/`, and the
  `seed_report` warming-seed display fix only — one training-code version.
- Best-mean checkpoints land at steps 4k–8k (median 5k), all saved (`keep_every=1000`); the deploy
  selection buys ~0.001 CE over the last-step checkpoint.
- Record updated the same session: `docs/qband_results.md` (new), CLAUDE.md validated-result paragraph,
  README reference-result line, milestones entry, tag `result/qband-8seed-640k` at 007f7173.

## Closing decisions

- Hub: **all 8 qband deploy ckpts published** for reproduction [user 2026-07-04: "we need all of these
  ckpts"] — `canvit/qpolicy-ade20k-c64-t5-qband-2026-07-04-s{0..7}`; flagship (`DEFAULT_QPOLICY_REPO`)
  repointed to band-best s2. The selection-bias concern applies to CLAIMS (always cite the band), not to
  artifact choice [user corrected my "unbiased draw" rationale — both ckpts are draws; serving the
  val-best is ordinary model selection, the same rule as `best.pt`]. The 2026-07-03 repo (`qlr2e4_s0`)
  stays up unmutated.
- `publish_qpolicy.py` and `seed_band.py` graduated `throwaway/` → `tools/` [user: "if a script is
  needed repeatedly/regularly it is not throwaway"]; convention encoded in CLAUDE.md.
- README Results section + `docs/figures/policy_comparison.png` committed — the FIRST pin of the
  standing ask ("should be included as an image in GFM"), qband as hero, provenance caption per spec.
  Subsequent image updates stay gated on the user's explicit word (CLAUDE.md Conventions).
## lr probes below 2e-4 (user: "perhaps even lower in the future") — verdict: plateau, keep 2e-4

`qlr1p5e4_s0` and `qlr1e4_s0`, run sequentially after the band (single seed 0, recipe otherwise
unchanged, 640k forwards each, code rev 92088f3e). Best-mean(t1–t4) val CE against the qband's
0.6853 ± 0.0007:

- lr 1.5e-4 → **0.6853 @5k** — dead on the band mean.
- lr 1e-4 → **0.6856 @6k** — inside the band; same plateau reached ~1k steps later (trailed by
  ~0.002 at 1k, ~0.001 at 4k, converged by 5k). Both show the band's late upward wobble past their
  best step.

With lr 3e-4 ~0.001 worse from 2026-07-03, the single-seed picture: the lr plateau spans at least
1e-4–2e-4, with 2e-4 at the fast edge (earliest best step). Default stays 2e-4; going lower buys
nothing and costs convergence speed within the 8k-step budget.

## Behavioral replication on the band (t0 trajectory probe, `throwaway/ckpt_trajectory_probe.py`)

qband_s2 and qband_s5 against the cached 96-scene true-reward maps (`outputs/dueling_cache_c64.pt`);
run at rev 06e29899. Yesterday's qdefault_s0 signatures replicate on both seeds:

- Ranking calibration saturates early: sp_true 0.26 by 1k steps, plateau 0.29–0.32 through 8k.
- Deploy regret keeps improving after calibration flattens: 0.15–0.16 untrained → ~0.11–0.12.
- Dueling V head is live on both: v_corr climbs to ~0.3–0.38 (Pearson vs true scene-mean reward).
- t0 policy goes all-coarse and central by 1k: fine% 62–72 (untrained) → ~0; mean radius 0.56–0.63 → ~0.40–0.48.
- Feature reliance at t0 concentrates in ln_feat (Δsp up to 0.12 when zeroed); every other group ≤~0.02.

So the qband's gains carry the same mechanism as recorded for the recipe's single seeds — nothing about
the band run changed the behavior story.

## Wart sweep (user: "fix your shit, do the collection, don't overdo the model card stuff")

- All 9 Hub cards republished with minimal YAML frontmatter (license: mit = the org standard,
  library_name, pipeline_tag) — the Hub metadata warning is gone. Band cards carry a claims-cite-the-band
  note; the pre-band 2026-07-03 repo's card now links its successor. `publish_policy --note` is the hook.
- HF collection (all 9 policy repos):
  https://huggingface.co/collections/canvit/viewpoint-q-ade20k-glimpse-policies-6a4961198721ee2bea76fc5e
  (one-off `throwaway/create_policy_collection.py`; the API caps descriptions at 150 chars).
- `ckpt_trajectory_probe` graduated → `tools/trajectory_probe.py` (3rd use = recurring), now owning
  `build_cache` (lifted from `dueling_real.py`) and rebuilding `outputs/dueling_cache_c64.pt` when
  missing — the cache is no longer an undocumented single point of failure.
- Preservation convention reconciled: from 2026-07-04 the Hub IS the mechanism; `preserved_ckpts/` +
  its doc frozen as the pre-Hub record.
- The repo had NO LICENSE file (HF cards already declared mit) — **MIT added** [user 2026-07-04:
  "i am the author", copyright Yohaï-Eliel Berreby <me@yberreby.com>], plus license/authors in pyproject.

## LFS scrub + fully-published figure inputs (user directives)

- **`*.png` → git-LFS, history rewritten** (651 commits; png bytes verified identical; no tags or
  doc-cited hashes were in the rewrite zone). Pre-scrub refs: origin branch
  `backup/pre-lfs-scrub-2026-07-04`. New flow (also in CLAUDE.md §Execution): deploy pushes need
  `--no-verify` (bare repo has no LFS endpoint), crockett checkout is skip-smudge (pointer files).
  OPEN: whether GitHub's web UI renders the LFS README image in this PRIVATE repo needs a human look
  (media endpoint rejects API tokens; unverifiable from CLI).
- **The comparison figure depends on no unpushed artifacts now**: measured c64 baseline evals committed
  at `docs/data/measured_baselines/` (summary + manifest, git_rev e8052770 era; seed_report reads the
  repo copy); band + untrained curves come from `training_run/metrics.jsonl` already on each HF seed
  repo. End-to-end regeneration from committed baselines verified on crockett.
- **Untrained inits published**: `training_run/step_000000.pt` uploaded to all 9 policy repos
  (backfill + `publish_policy` now always uploads it).

## throwaway/ full review (user: "should they all be deleted?")

Reviewed all 16 scripts (header + git dates + doc citations). Deleted 13 concluded one-offs whose
findings are recorded in session docs and whose code lives in git history at the cited revs:
ablation_scan, bn_mode_agreement, create_policy_collection, dagger_ab, dueling_real, gate_study,
plot_coarse_fine_violinline, q_calibration, q_reward_landscape(+_traj), reward_corr,
reward_transform_stats (findings 2026-07-03), run_traj (superseded by tools/training_curves +
seed_report --overlay + trajectory_probe's Δsp), viz_t0_channels. Promoted 2:

- `tools/perpetual_sweep.py` — the parked sweep supervisor CLAUDE.md/README point at (recurring if HP
  search resumes); module-level loop wrapped in main() so importing it is side-effect-free.
- `scripts/estimator_probe.py` (was e2e_gumbel_overfit.py) — the rebuttal-grade estimator ablation
  (pathwise ST-Gumbel fails; REINFORCE ≈ Q-regression at matched k); docstring carries the verdict +
  session pointer; typed Literal estimator arg; now in lint/typecheck scope.

`throwaway/` is now empty except `action_data/` (untracked local cache from the 06-19 action analysis).
