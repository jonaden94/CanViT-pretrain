# 2026-06-18 (eve) — collapse to ONE frontend, audit, docs to README

Continues `2026-06-18-curated-band-and-frontend-redesign.md`. The session deleted the canvas-vs-curated fork,
unified the input plumbing, made the candidate index structural, removed the dead qcorr subsystem, moved the
method into the README, ran liveness + staleness audits, and pruned the memory store. HEAD `d074ea4`;
local == origin == deploy == crockett (verify: `git rev-parse HEAD` here and on crockett).

## What changed (the regime)

- **One `Frontend`, one `StateEncoder`** [user: "commit to it, delete the fork"]. Removed `CanvasFrontend`,
  the `Frontend` base class, `qnet_input`, `input_mode`, and the canvas-only knobs (dropout, input_norm,
  frontend_mlp, entropy_channel). `curated_dim` eliminated (never swept, always == width). `StateEncoder` is the
  ONE place a canvas state becomes net input (built once, `reset()` at t0, init reference cached). Renames:
  GROUP_NAMES→FEATURE_GROUPS, curated_*→feature_*/`init_reference`. The frontend arch itself is unchanged from
  the `curated128` design (per-group proj→LN(affine)→gate→sum + token-MLP).
- **Flat candidate index is now structural** [user: "should be IMPOSSIBLE to silently get wrong"]. The random
  sampler draws `randint(n_candidates)` instead of `scale*cells+cell`; everything indexes one layout (`vp_flat
  = candidate_viewpoints().reshape(-1,3)`), consistent by reshape semantics. `vp_cells` dropped.
- **Docs:** method (action space, **γ=0** immediate fractional-CE reward, the 6 derived feature groups,
  Frontend+U-Net+grid_sample readout, K=1 + global-z training, deploy) is now in the README; CLAUDE.md "How it
  works" points to it (no duplication of defaults/module-tree). README has a **Known warts** section.
- **Banned words** [user, emphatic]: never use "honest"/"genuine"/"load-bearing" → memory `banned-words.md`.

## Equivalence / reproducibility

The refactor is behavior-preserving for the trained model — same arch, features, reward, objective, optimizer.
The ONLY difference vs the run's code (`adaa231`) is the random-exploration RNG stream (one randint vs two;
same uniform distribution). So `curated128_s0` is representative of HEAD but not bit-identical; the reproducible
reference going forward is a fresh run/band under HEAD [user: doesn't care about seeds changing].

## Warts / FIXMEs (tracked in README)

- Training reward scored at **128² pixels** (`score_res`): masks downsampled, logits upsampled, ~2× throughput,
  ranking preserved (Spearman ≈0.999). Downsampling masks is a compromise — move to full 512² if it can run
  without a throughput hit. (The one remaining tracked wart; in README "Known warts".)
- **Dormant knobs** (wired + valid but NOT exercised by the default or the live sweep `--search`): `t0_mode=riid`
  + `scale_min`, `compile`, `augment`. Levers, not dead code — kept pending an explicit keep/cut decision.

## Live state / next

- **`curated128_s0` DONE** (git_rev `adaa231`, full 1M / step 12500): endpoint ce_t4 **0.6665**, mIoU_t4 44.86 —
  inside the old band's per-seed spread (0.6644–0.6667). The single-frontend arch reproduces the band; one seed
  is not a band, so:
- **8-seed band under HEAD is RUNNING** — `seedband_s0..7` via `throwaway/seed_band.py` (prefix `seedband_s`,
  sequential, one GPU). seed 0 smoke passed (step-0 ce_t4 0.7062 = the t0 floor). Completion poll `blwbqkf2k`
  re-invokes on finish. **When it lands:** `python -m canvit_pytorch_rl.tools.seed_report --prefix seedband_s`;
  if ≥ the old band, tag `result/qpolicy-8seed-band` + update `docs/old_frontend_band_results.md`. Then the feature
  ablations (`throwaway/ablation_scan.py`) on the new arch, and/or fill the GPU with `perpetual_sweep.py`
  (study `qpolicy_c64_t5_ce`).
- Deploy reminder: `git push origin main && git push deploy main`, then on crockett
  `git fetch origin && git reset --hard origin/main && git rev-parse HEAD` (the fetch is mandatory —
  reset-without-fetch is a silent no-op; see CLAUDE.md §Execution).

## Audits this session (the new discipline: not just valid — LIVE and CURRENT)

- **Dead-subsystem (liveness) audit.** The qcorr/reward-map precompute was valid-but-dead at c64 (gated on a
  `None` from an un-produced artifact; objective is val CE not qcorr). Removed it — kept only
  `reward_maps.candidate_rewards` (feeds the value-map filmstrip). Swept the rest: no other dead subsystem; all
  config fields are read; vulture clean. Lesson encoded: "is it clean" now also asks "is it reached?".
- **Throwaway cleanup.** Deleted the 4 scripts that can't run under HEAD (`overfit_batch`, `synth_fixation`,
  `migrate_qnet_keys`, `migrate_q_ckpts`); fixed the live ones (`seed_band`, `perpetual_sweep` [GPU-mem gate,
  study `qpolicy_c64_t5_ce`], `ablation_scan` `--feature-groups`). `git grep` had missed these (throwaway is
  tracked but my check used `git grep` which skips it) — use `rg` for untracked-inclusive sweeps.
- **Staleness sweep.** Fixed current-state claims in CLAUDE.md (curated128 done, not "validating"), the
  `reward_maps`/qcorr line, the deleted-doc dangling links in milestones, and banned-word residue.
- **Memory store pruned** (`~/.claude/.../memory/`, outside the repo): deleted 9 dead-regime memories
  (flow/critic/grid/gridcorr + redundant reward-ce-iou), removed 2 stale `[ACTIVE]` inline index entries,
  repaired all dangling `[[wikilinks]]`, indexed 2 orphaned-but-valuable directives (`gpu-never-idle`,
  `dont-kill-working-runs`), de-staled survivors (anchor → SUPERSEDED). Now 35 files, all indexed, 0 dangling.

## Open / next (cold-start)

1. **Band verdict** (poll `blwbqkf2k`): aggregate `tools.seed_report --prefix seedband_s`; if ≥ old band, tag
   `result/qpolicy-8seed-band` + update `docs/old_frontend_band_results.md` (the loadable reference for HEAD).
2. **Dormant-knobs decision**: cut or keep `t0_mode=riid`+`scale_min` / `compile` / `augment` (user call).
3. **score_res wart**: try full-512² reward scoring if it fits the throughput budget.
4. Then: feature ablations on the new arch (`ablation_scan`), or fill GPU with `perpetual_sweep`.
