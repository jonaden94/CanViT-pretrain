# 2026-06-15 night → 2026-06-16 — refactor to viewpoint-Q, budget-in-forwards, CE-judged perpetual sweep

Big session. Tag `pre-refactor-2026-06-15` preserves the old tree. Work done on branch
`refactor/codebase-cleanup` (commits f67460a / a5cecee / 1fc8bc8 + the budget/objective/docs commit),
merged `--no-ff` into main.

## The 20k run finished and PLATEAUED past ~1M forwards

`grid_t5_aligned_2scale_c64_20k` (c64, 2-scale (0.5,0.25), train_horizon=5, prime_on_policy=0.5, 20k
steps) finished. Full-val T=5 (paper protocol), via the migrated ckpts + new `q.eval`:

| ckpt (forwards) | t0 | t1 | t2 | t3 | t4 |
| step_10000 (~1M) | 39.59 | 42.94 | 44.05 | 44.55 | 44.94 |
| step_15000 (~1.4M) | 39.59 | 42.75 | 43.87 | 44.48 | 44.86 |
| step_20000 (~1.9M) | 39.59 | 42.73 | 43.97 | 44.57 | 44.91 |
| EG-C2F-c64 | 39.60 | 42.22 | 43.30 | 44.04 | 44.65 |

Beats EG-C2F-c64 at every t1–t4, but the within-budget checkpoint (~1M forwards = step_10000) is as good
or better than 2×; the second 1M bought nothing on val (t1 even slipped). → the **1M-forward cap**.

**Does the policy benefit from >1M forwards? NO — judged by VAL CE ALONE.**
⚠️ **The train-vs-val gap is a CONFOUND** [user, 2026-06-16; corrects an earlier wrong read]: the frozen
PROBE is already overfit to ADE train, so `train_ce` < `val_ce` is mostly the PROBE (it segments train
images better regardless of the policy), NOT a policy-overfit signal. Do NOT compare train vs val deploy
metrics, and "the gap widens" means nothing here. The only clean signal is the **val trajectory's shape**:
`val_ce_t1_mode` decreases to ~0.713 by ~7–10k (≈1M forwards) and does NOT improve after — the BEST
(lowest) val CE is at an INTERMEDIATE step (~10k), not the last (11k–20k sit 0.713–0.716; val mIoU flat
~42.7). So **>1M forwards buys no val improvement** → the 1M cap holds. (Whether val actively DEGRADES past
10k — true overfit — is within the val-trajectory wobble; the safe claim is "best-not-at-last, no benefit
past ~1M".) Push val CE via HPs/regularization (the sweep), not more steps. [memory:
probe-overfit-confounds-train-val.]

**⚠️ LR-schedule caveat [user 2026-06-16]:** the schedule (warmup_frac×steps → cosine decay) is tied to the
step count = `budget_forwards/(batch×(1+train_horizon))`. The 20k run's `step_10000` reference was at LR-PEAK
(warmup end, UNDECAYED); a proper 1M run (the sweep trials, steps≈10416) does its FULL warmup+decay within 1M
(ends decayed). So sweep trials are **not** a reproduction of `step_10000` (≈44.94 @t4) — they're the correct
budget-respecting recipe and may land higher or lower; and the "no benefit past 1M" read leaned on a peak-LR
checkpoint, so the budget-respecting sweep trials are the real 1M test. The sweep is internally consistent
(all trials pin H=5 → identical schedule).

## Clean-break refactor: `grid` → viewpoint-Q (`canvit_pytorch_rl.q`)

`ViewpointQNet` predicts Q(state, viewpoint); `GreedyQPolicy` argmaxes. Layout `q/` + `baselines/` +
`tools/` + shared root substrate (`config, data, metrics, scoring, canvas_ops, harness, plots`). Renames:
`GridValueNet`→`ViewpointQNet`, `GridPolicy`→`GreedyQPolicy`, `grid_viewpoints`→`candidate_viewpoints`,
config `grid`→`centers_per_axis`, metric `gridcorr`→`qcorr`. Dropped: the misaligned `direct` readout
(always aligned), the unused `nflows` dep, dead `scripts/`, the stale `plot_training` (→ `tools/training_curves`
on the real schema). NO in-code back-compat shims.

**VERIFIED behavior-preserving:** new `q.eval` reproduces old `grid_eval` to 4 decimals on step_15000 AND
step_20000, after `throwaway/migrate_q_ckpts.py --apply` rewrites pre-refactor ckpt metadata
(`grid`→`centers_per_axis`, drop `policy_arch`; refuses non-aligned ckpts).

## Budgets in glimpse-forwards; judge by val CE at t4

- `QConfig.budget_forwards` (1M) replaces `steps`; the trainer derives step count =
  `budget_forwards // (batch_size × (1+train_horizon))`. q.optuna sweeps `--base-forwards`. [user: "step
  counts are meaningless, budgets need to be in glimpse forwards".]
- **Objective = MINIMIZE val CE at the deploy endpoint** (t4 for T=5) [user]. The trainer now adds a
  `rollout_eval` (GreedyQPolicy via run_episode → per-t deploy mIoU/CE on val AND train), sets a stable
  `metrics["objective"]` = `val_ce_t{eval_horizon-1}`; both `best.pt` and the optuna objective minimize it.
  qcorr/mIoU stay as diagnostics.
- optuna dedup [user]: `FROZEN` (which re-listed the QConfig defaults) removed — non-swept HPs now fall
  back to `QConfig` field defaults (`dataclasses.fields`, no construction). SPACE (ranges) stays.

## Perpetual CE sweep

`throwaway/perpetual_sweep.py` runs `q.optuna` (study `t5_c64_ce`) forever — disk-gated (stops <5 GB free:
in-study `disk_gate` callback + supervisor relaunch), crash-resilient, detached (`nohup setsid`). Broad
`--search` (lr, weight_decay, betas, target_momentum, warmup_frac, width, block_layers, t0_mode,
prime_on_policy, entropy_channel, frontend_mlp). The seed (trial 0) = the QConfig defaults (carried-forward
from the OLD c32/t1/qcorr regime — being re-tuned here for c64/T=5/val-CE). **Restarted with the
budget_forwards + val-CE-at-t4 objective after the merge.**

## CLAUDE.md rewritten from scratch [user]

577 → ~210 lines: priority-ordered (hard rules first), terse (cut justifying prose, kept directives),
`@README.md` import + path-refs to milestones/sessions, contradictions resolved (non-finite guard vs
syncs; out-of-work sweep vs 1M cap), cruft cut (git_rev mechanism, flow/critic framing, Codex anchors,
bootstrap-CI, numbered "RULE N"), updated to c64/viewpoint-Q/1M-forwards/judge-by-CE. Over-pruned on the
first pass then restored the real conventions (zooming-below-native + candidate-data design, the stack
entry points + (y,x)∈[-1,1]² coord convention, efficiency-curves-headline, both eval protocols,
crockett-pauses, t1-apples-to-apples). New standing rule: a stale research record is an OPERATIONAL
FAILURE — update milestones/session/provenance-comments AS each result/regime-change lands.

## Repro

- Canonical run (defaults = c64/T=5/1M): `python -m canvit_pytorch_rl.q.train --run-name <name>`.
- Eval: `python -m canvit_pytorch_rl.q.eval --ckpt-run <name> --ckpt-name last.pt --n-timesteps 5`.
- Sweep: `nohup setsid uv run python throwaway/perpetual_sweep.py > /tmp/sweep_sup.log 2>&1 &` (crockett).

## 2026-06-16 — first sweep results (FLAT); defaults re-tuned by synthesis; sweep redesigned

First 1M-forward CE sweep (study `t5_c64_ce_f1000000`, 7 trials) landed FLAT + single-seed (top-5 val_ce_t4
0.6656–0.6706; CE vs mIoU disagree on the winner). Decisions + the 7-trial table: `docs/milestones.md`
(2026-06-16 entry). In brief:
- Defaults re-tuned by SYNTHESIS of the user's priors + the data (NOT by copying the CE-winner) [user]:
  riid / prime 0.5 / warmup 0.1 / wd 1e-2 / lr 7e-5 / block_layers 3 / entropy off — see `q/config.py`.
- Sweep redesigned [user]: new study `t5_c64_ce_optdyn` pins the design dims (warmup/entropy/riid/prime) +
  target_momentum (0.997, infra knob, not tuned [user]) via QConfig defaults, and `--search`es the
  optimizer+capacity core (lr, wd, both betas, width, block_layers, frontend_mlp). Old 12-dim study retired.
- entropy_channel=off rests on the c64 trials leaning that way + user pref, NOT a single-variable A/B (the
  new study pins it off). Enqueue an ec=True sibling of trial0 if a number on its cost is wanted.

## Current state (handoff, 2026-06-16 ~10:23 crockett time)

- **Perpetual sweep LIVE** on crockett: supervisor bg-pid 2354219, study `t5_c64_ce_optdyn_f1000000`
  (objective = min val CE at t4; 1M forwards ≈ 10416 steps/trial; eval_every 2000; 7-dim optimizer+capacity
  `--search` = lr/weight_decay/adam_beta1/adam_beta2/width/block_layers/frontend_mlp; design + target_momentum
  pinned via QConfig). Log `/tmp/sweep_sup.log`; optuna `runs/q_optuna.db`; mlflow `canvit-q` (:5500). Trial0
  = new-defaults seed; step-0 verified sane (objective=val_ce_t4 0.7355 untrained, val mIoU t0 39.60); learns
  (train_corr climbs), grad norms ~1.5–3 (clipped), ~2 steps/s (~85 min/trial), sole GPU proc ~15 GB.
- crockett checkout `main` @ c3daa6a (defaults tagged `checkpoint/q-defaults-2026-06-16` @ 9416ed4; c3daa6a
  adds only the target_momentum pin). The checkout's `origin` = the local bare `~/repos/CanViT-PyTorch-RL.git`.
- **NEXT (monitor + analyze):** as `t5_c64_ce_optdyn` trials complete (`tools/sweep_report --study
  t5_c64_ce_optdyn_f1000000` or `runs/<trial>/metrics.jsonl`), check whether any config robustly beats the
  trial0 seed BEYOND single-seed noise (`tools/metric_stats`; a number is not a trend). Judge by val CE,
  report mIoU; deploy refs: EG-C2F-c64 t1–t4 42.22/43.30/44.04/44.65 and the 20k within-budget ckpt
  42.94/44.05/44.55/44.94.
