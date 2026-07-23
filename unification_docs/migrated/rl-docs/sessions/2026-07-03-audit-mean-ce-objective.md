# 2026-07-03 — repo audit: the mean(t1–t4) objective lands in code; consistency fixes

Full-repo audit session (user: "find things that are bad, contradictory, and fix them"). No GPU work; all
changes are code/doc consistency, verified by `uv run just` (green before and after). Branch
`audit/2026-07-03-fable`.

## Headline: the 2026-06-24 objective directive was recorded but never applied

[user 2026-06-24, emphatic]: "mean(t1–t4) [val CE] — this is the only thing i care about." The post-mortem
(`2026-06-24-perpetual-sweep-postmortem.md`) and the assistant memory both recorded it — with an explicit
"any future `q.optuna` sweep must set the objective to mean(t1–t4) CE" — but the code was never touched:
`q.train.evaluate()` still set `objective = val_ce_t4`, so **`best.pt`, optuna trial ranking, AND pruning all
selected on the endpoint-only metric the user rejected**, while the deploy-ckpt rule
(`tools.seedband_io.best_mean_ckpt`) already used the mean. Root cause of the gap: the directive was written
into docs/memory as a note-to-future instead of being applied the same session (the CLAUDE.md "encode it the
SAME session" rule exists for exactly this).

**Fixed:** `objective` = mean val CE over the learned steps t1..t{train_horizon} (`q/train.py`); comments/
docstrings updated in `q/config.py`, `q/optuna.py`, `q/eval.py`, `tools/sweep_report.py`,
`throwaway/perpetual_sweep.py`, README, CLAUDE.md (the judging rule is now a hard-rule bullet there).

Consequences to carry:

- **`best.pt` semantics changed** for runs trained at ≥ this commit: best-by-mean(t1–t4) — now identical to
  the deploy selection rule. Pre-change ckpts' `best.pt` = best-by-`val_ce_t4` (noted in `QEvalConfig`).
  Every eval row logs all `val_ce_t*`, so either rule is recomputable offline for any run, old or new.
- **Never resume a pre-change optuna study** — objective drift fools TPE (same class of bug as the 2026-06-13
  fp32/bf16 study split). `perpetual_sweep.py` docstring says to bump `STUDY` before any relaunch.
- `tools.sweep_report` now reads per-t `val_ce_t*` keys instead of the recorded `objective`, so old and new
  runs rank under ONE rule (deploy eval per trial = best mean-CE, the `seedband_io` rule; endrk = that same
  eval ranked by t_end alone). Old-run reports therefore change: the "deployed" eval per trial may move.
- With `train_horizon=1` the mean degenerates to `val_ce_t1` — t1-only studies are unaffected by construction.

## Secondary fixes (same audit)

- **Silent-failure gap:** `keep_every`'s own comment said "must be a multiple of eval_every" but nothing
  enforced it; a violation silently saved NO `step_*.pt` (the save sits inside the eval branch). Now asserted
  at `train()` entry.
- **`evaluate_q(selection=False)` scored full-val t0 mIoU and discarded it** (the caller takes only the
  argmax spreads from val), contradicting its own docstring. t0 scoring now happens only under
  `selection=True` (train slice). No metric changes — the discarded value was never emitted.
- Wrong comments fixed: `q/eval.py` called `GreedyQPolicy` "stateless: step(t, state) is pure" (it holds the
  rolling prev-state reference); `q/config.py` said `eval_every=None` → "step-0 + end" (code runs only the
  final eval — step-0 eval is inside the `eval_every is not None` gate).
- `q.eval` loaded the ckpt from a hardcoded `"runs/"` instead of `cfg.out_root`.
- `baselines.figure4b`: duplicate legend entries when several runs share a policy — now one entry per policy.
- Stale docs: CLAUDE.md still described the perpetual sweep as a live overnight process (stopped 2026-06-24,
  HP plateau) — now states the stop, points at the post-mortem, and records the relaunch conditions
  (fresh study + ~250k-forward ranking trials). README's "validated band" pointer went to
  `old_frontend_band_results.md`; the validated headline is the HEAD band → `head_band_results.md`.

## Not changed (considered, rejected)

- MedianPruner aggressiveness (87% prune rate, 189/202 at first eval): real, but the sweep is stopped and any
  retune would be speculative without a live study; the post-mortem records the observation.
- Historical docs (`sweep_sets.md`, `preserved_checkpoints.md`) still say "best val_ce_t4" — they describe
  what those frozen artifacts ARE; rewriting records to the new rule would falsify provenance.
- `tools.seed_report --final-step 12500` duplicates the 1M/(16×5) derivation as a CLI default — left; it is
  labelled and CLI-overridable.
