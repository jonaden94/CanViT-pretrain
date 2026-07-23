# 2026-06-16 — deploy-ensemble probe (null) → twin-Q is the new DEFAULT; datetime run-dirs; GPU-discipline

Continues `2026-06-15-refactor-to-q-and-ce-sweep.md` (the c64/T=5 CE sweep, synthesis defaults, the optdyn
sweep). Today: a deploy-ensemble probe (null), then **twin-Q (clipped-double-Q) as the canonical default**,
run-dir datetime prefixes, and two GPU-discipline corrections.

## Deploy-ensembling the overnight nets — NULL (don't re-try expecting gains)

`throwaway/ensemble_eval.py`: aggregate the 5 overnight `t5_c64_ce_f1000000` `best.pt` Q-maps ACROSS nets
before argmax (min / mean / median, raw + per-state z-scored), full-val T=5. Best (min-raw) **tied** the best
single net — 45.02 vs 44.99 mIoU@t4, CE 0.6659 vs 0.6662 — inside the noise floor; z-scored variants slightly
worse. Mechanism: the nets AGREE on the good viewpoints (flat landscape, correlated learners) → little
per-net overestimation for the ensemble to cancel. So **deploy-ensembling distinct nets doesn't help here**.
(solo trial0003 reproduced its stored 44.99 exactly → the eval is sound.)

## Twin-Q (clipped-double-Q) is the new DEFAULT [user 2026-06-16]

`QConfig.n_critics` (default now **2**): N distinct-init `ViewpointQNet` critics in `EnsembleQNet`. `forward`
= per-state **min** (`ensemble_agg`, default `min`) = the rollout + deploy policy — the on-policy DAgger
glimpses AND the deploy argmax rank a viewpoint high only when the critics AGREE. `forward_all` = each
critic's Q, for the loss: every critic regresses the SAME measured fractional-CE target — no bootstrapping,
so the min is ONLY the policy, never the target (the correct adaptation of clipped-double-Q here). Distinct
init via `manual_seed(seed+1+i)` per critic. `n_critics=1` stays byte-identical (bare `ViewpointQNet`).

- Diagnostics: `train_corr`/`corr_dN`/`predstd` track the MEAN-critic fit (not the biased-low min);
  **`critic_qspread`** = per-action critic disagreement (≈0 ⇒ min is a no-op). At twin step ~100 it's ~0.043
  → the critics differ, so the min does real work.
- **Validation** (200k forwards, new defaults): `twin_min_nd_200k` last.pt t0–t4 =
  **39.60 / 42.35 / 43.84 / 44.45 / 44.64** (CE@t4 0.6685) — ≈ EG-C2F-c64 at t4 and ahead at t1–t3, at 1/5 of
  the 1M budget. User decided twin on the trajectory + the sound theory; a single-vs-twin ablation was
  explicitly NOT pursued (and the capacity-vs-min / data-order confounds explicitly set aside).
- `q.eval` reconstructs `EnsembleQNet` when ckpt `n_critics>1` → twin `best.pt` is standalone-evaluable /
  figure-4B-able. `GreedyQPolicy` net type widened to `ViewpointQNet | EnsembleQNet`.
- Perpetual sweep → fresh **twin** study `t5_c64_ce_twinq` (n_critics not searched → uses default 2 ⇒ every
  trial is twin); seed = the canonical twin at 1M. The single-critic `t5_c64_ce_optdyn` study is retired.

## Datetime run-dir prefix [user 2026-06-16]

`RunConfig.__post_init__` prepends `YYYYMMDD-HHMMSS_` (UTC, idempotent) to `run_name` ⇒ every run dir (sweep
trials + manual) sorts chronologically and never collides. `tools/sweep_report` glob → `*{study}__trial*`
(leading `*` = the prefix). Old non-timestamped dirs are grandfathered (referenced by exact name still works).

## GPU-discipline corrections [user, emphatic]

- **Never idle the GPU.** "Holding for direction" / "the sweep is low-value" do NOT justify idle — always run
  the best-available job. (I left it idle after an eval → corrected.) [memory: gpu-never-idle-even-when-holding]
- **Evals run CONCURRENTLY with the sweep** — never kill the sweep for an eval (sweep ~17 GB of the 4090's
  24). Only a competing TRAINING run needs a staged swap. (I'd killed the sweep for the ensemble eval — wrong.)
- A "why are you doing X?" question asks for REASONING, not "stop X" — don't invent a rule the user never
  stated (I confabulated a "don't relaunch the flat sweep" rule).

## State (handoff)

- **Twin sweep LIVE** on crockett: study `t5_c64_ce_twinq_f1000000`, `n_critics=2`, 7-dim optimizer+capacity
  `--search` (lr, weight_decay, adam_beta1/2, width, block_layers, frontend_mlp); design dims +
  target_momentum pinned via QConfig. **~198 glimpse-forwards/s, ~84 min per 1M trial, ~91% GPU util** (twin
  ≈ single — the 2nd critic is a tiny U-Net vs the frozen CanViT-B that dominates). Log `/tmp/sweep_sup.log`;
  supervisor `throwaway/perpetual_sweep.py`. crockett checkout @ `5ca8a51` (the running code); origin/main is
  ahead by doc + q.eval commits, NOT reset under the live sweep.
- NEXT: watch the twin sweep trials (judge by val CE@t4, report mIoU; the trial0 seed is the canonical twin
  at full 1M). README/CLAUDE/milestones updated to twin-default this session.

## train/eval horizon fix — train_horizon = deploy horizon (no t5 overshoot) [user 2026-06-16]

Off-by-one found + fixed: `train_horizon=5` took 5 learned actions (transiently reaching t5 to score the d4
reward) but the eval/objective deploys only to t4 — so 1/5 of supervision (d4, the decision at t4→t5) trained
a transition never deployed. Now **`train_horizon=4`** = the deploy horizon: the rollout takes 4 actions
(d0..d3 → t4), `eval_horizon = train_horizon + 1` (states t0..t4), objective `val_ce_t4` (unchanged); train and
eval both end at t4 — no t5, `corr_dX` only d0..d3. Same 1M budget ⇒ ~12500 steps × 4 deployed decisions
(~50k) vs old ~10416 × 5 with 1/5 wasted (~41.7k deployed) = **~20% more capacity on the decisions we eval**
[user]. Not a numbers bug (state-conditioned net), just better budget allocation. Fresh sweep study
`t5_c64_ce_twinq_hfix` (regime changed); future train_horizon=4 runs differ from the documented 5-action
anchors. The running `t5_c64_ce_twinq` sweep is on the OLD 5-action regime — left running per [user: don't
kill]; switch to the fixed study at the next deliberate relaunch.

## Compaction handoff — 2026-06-16 ~13:30 crockett (CURRENT authoritative state — read first)

**Live:** twin sweep `t5_c64_ce_twinq` (study `t5_c64_ce_twinq_f1000000`) on crockett, **OLD 5-action regime**
(train_horizon=5, pre-horizon-fix code @ `5ca8a51`). **trial0** = first canonical twin at 1M, ~step 5515 of
10416 (~halfway, ~45–50 min to finish at ~1.6 steps/s), 95% GPU. Log `/tmp/sweep_sup.log`.

**LOCKED PLAN [user — do not deviate]:**
1. **Do NOT kill trial0.** Let it finish (step ~10416).
2. Then **analyze** trial0's full 1M deploy curve t0–t4 vs EG-C2F-c64 `39.60/42.22/43.30/44.04/44.65` and the
   single-critic 20k anchor `39.59/42.73/43.97/44.57/44.91`. Noise-aware (one trial ≠ trend); FLAG it's the
   OLD 5-action regime (the horizon fix supersedes it).
3. Then **kill the sweep + relaunch the FIXED regime**: the horizon fix (`train_horizon=4`, study
   `t5_c64_ce_twinq_hfix`) is committed (`7580139`) + pushed but NOT deployed. Switch = kill the sweep
   process group on crockett, `git reset --hard origin/main`, relaunch `throwaway/perpetual_sweep.py`, verify
   step-0 sane + `corr_d0..d3` only (no d4) = fix is live.

**Immediate pending [user]:** the **step-6000 eval** (lands ~13:33) → give the "full picture vs our other
numbers." Pull eval rows from `runs/*t5_c64_ce_twinq_f1000000__trial0000/metrics.jsonl`.

**Watcher NOT armed** (Bash classifier flaky 2026-06-16). To catch trial0 completion: `runs/*..._trial0001`
dir appears, or trial0 metrics reaches step 10416.

**Repo:** all committed/pushed. Session commits: `f00c941` (q.eval loads twin ckpts), `2e31fb8` (docs→twin
default), `30f1fa9` (throwaway 44→3), `7580139` (horizon fix), `073a93b` (horizon record). Tags:
`checkpoint/q-defaults-2026-06-16`, `pre-throwaway-cleanup-2026-06-16`. **Deploy caveat:** do NOT `git reset`
the crockett checkout under the live sweep — only reset when executing the switch (step 3).

**trial0 step-6000 read (~576k fwd, 5-action, PRELIMINARY — finalize at step ~10416):** t0–t4 =
`39.60 / 42.5 / 43.5 / 44.3 / 44.55`, ce_t4 `0.6707`, `critic_qspread` ~0.033 (critics differ → min not a
no-op). t4 is ~flat over steps 2000–6000 (44.55→44.41→44.55; ce ~0.671) — but 3 noisy eval points are NOT a
plateau verdict [user: never overclaim; WAIT for the full ~10416 steps before concluding anything — the
LR-decay tail (cosine to ~0) may still refine it; cf. training-dynamics-patience].
- vs **EG-C2F-c64** (42.22/43.30/44.04/44.65): ahead t1–t3 (+0.2–0.3, above the ±0.2 deterministic-baseline
  floor; t1 +0.28 = the cleanest apples-to-apples point), **~tied at t4** (−0.10).
- vs **single-critic anchors** (20k 44.91; overnight best trial3 44.99/0.6662): **below at every t** (~0.3–0.45).
- **Flag:** the new-defaults runs (twin trial0 + twin_min_nd_200k, ~44.55–44.64) underperform the overnight
  single trials (44.83–44.99) they were *synthesized* from → the synthesized default COMBO MIGHT be suboptimal
  (a HYPOTHESIS to test, NOT a conclusion — trial0 isn't done + single-seed). Twin-vs-single itself is ~tied +
  confounded (killed single_nd_200k 44.69 @1k ≈ twin 44.73 @1k). Snapshot read: clears EG-C2F at t1–t3, ties
  t4; at ~576k it trails the best single numbers. JUDGE NOTHING until trial0's full ~10416-step curve lands.
