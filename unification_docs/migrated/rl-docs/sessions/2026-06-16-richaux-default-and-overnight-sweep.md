# 2026-06-16 (night) — rich-aux promoted to DEFAULT; CE/mIoU nuance; checkpoint preservation; overnight richaux sweep

Continues `docs/sessions/2026-06-16-rich-aux.md`. This session: judged richaux rigorously, made it the default
recipe [user], built the checkpoint-preservation discipline, and set the overnight GPU work to an HP sweep of
richaux. Key commits: `c448908` (default-flip), `9b18b22` (preserved table), `8bfe1be` (sweep pivot),
`68985cb` (ckpt_meta tool).

## Headline changes

- **rich-aux is the DEFAULT recipe** [user: "i want rich aux by default and the 20k run's stuff by default"].
  `QConfig` defaults flipped: `rich_aux=True, n_critics=1, lr=3e-4` (`c448908`). Kept `t0_mode=full_scene`
  (prior call, NOT the 20k run's riid) and `budget_forwards=1M` (hard cap, NOT the 1.6M override). Base/twin
  retained as the alternative: `--no-rich-aux --n-critics 2 --lr 7e-5`. README + CLAUDE + config docstring
  updated; `uv run just` green (rich_aux=False still byte-identical to base, pinned by test_q).
- **`t0_mode` default → `full_scene`** earlier this session (`57be174`): at the t1 selection point the one
  t0_mode sweep showed no riid advantage (riid's only edge was at t4 and flipped sign vs t1 = noise).

## The rigor findings (CE *and* mIoU — judge by CE, but always read both [user emphatic])

- **The anchor's 42.94 is a single-seed PEAK, not a bar.** `grid_t5_aligned_2scale_c64_20k`'s own 20-eval t1
  trajectory: mean 42.63, max 42.91 @step10000 — and step10000 IS the budget-matched checkpoint we'd defined
  as "the anchor". A within-run eval trajectory is deterministic-per-ckpt and is NOT a reliability estimate;
  reliability needs independent seeds. So the apparent rich-vs-anchor t1 gap was band-vs-peak.
- **CE shows clear learning where t1 mIoU looked like noise.** richaux_q_20k val CE improves ~monotonically
  across all t (ce_t1 0.7200→0.7158, ce_t4 0.6724→0.6676) while t1 *mIoU* wandered in a noisy 42.2–42.6 band.
  CE↔mIoU disagree (ρ≈0.26): at t1 the anchor is marginally better on CE (mean 0.7159 vs 0.7187).
- **best-by-CE ≠ best-by-mIoU (per horizon, different steps).** New standing convention [user]: per run, keep
  AND document best-by-CE and best-by-mIoU at each horizon + overall. richaux_q_20k:
  - `best.pt` (lowest val_ce_t4) = **step 18000**, CE 0.6676, mIoU 42.42/43.79/44.52/44.76.
  - **step 13000** (CE 0.6678, ~tied) is BETTER on mIoU at 3/4 horizons (42.53/43.86/44.39/44.84) and beats
    EG-C2F by larger endpoint margins (+0.31/+0.19 vs 18000's +0.20/+0.11). **13000 is the better deploy ckpt.**
  - vs EG-C2F-c64 (42.22/43.30/44.04/44.65): both beat it at every t; vs anchor: both trail everywhere.

## Checkpoint preservation (discipline established [user emphatic])

Backups → `crockett:~/projects/CanViT-PyTorch-RL/preserved_ckpts/` (gitignored; `docs/preserved_checkpoints.md`
is the tracked record). **STEP-LABEL files and NEVER overwrite** — `best.pt`/`last.pt` are rewritten in-place
as a run improves, so capture each new best before it's clobbered; verify the real step with
`python -m canvit_pytorch_rl.tools.ckpt_meta --paths ...` (don't trust mtime). Future headline runs: `keep_every>0` (richaux ran
`keep_every=0`, so its best-by-mIoU ckpt at step 14000 was never saved — unrecoverable).

Preserved so far:
- `anchor_grid_t5_aligned_2scale_c64_20k/` — full scaling curve (step_005k/010k/015k/020k + best + last).
- `richaux_q_20k/` — `best_step13000.pt` (CE 0.6678, best mIoU deploy), `best_step18000.pt` (CE 0.6676,
  lowest CE), `last_step17000.pt`, `last_step19000.pt`. All strict-load clean under current code.

## Behavioral analyses (caveat: on `last.pt`, NOT a best ckpt — different steps!)

- **No-revisit / dispersion** (`throwaway/revisit/`): the greedy policy disperses glimpses far more than a
  scale-matched random null (consecutive box-IoU 0.077 vs 0.257), exact revisits <1%, and avoids the
  JUST-covered region most (lag-1 IoU 0.077 vs lag-2/3 ~0.20 = recency/coverage signature). Placement is
  image-adaptive (t1 norm-entropy 0.77, 164/512 distinct cells), not a fixed scan. **BUT this ran on
  `last.pt` ≈ step 5000 (mid-training), not the deploy-best ckpt.**
- **Synthetic fixation** (`throwaway/synth_fixation.py`, `--two-corner`): with two real ADE patches in
  opposite corners it visits both sequentially (t1→TL, t2→BR) but with COARSE scale-0.5 glimpses (not fine
  0.25), then drifts to empty space (coverage effect). Ran on `last.pt` ≈ step 17–18k.
- **OPEN [offered, awaiting user]:** re-run dispersion + fixation on `best_step13000.pt` so the behavioral
  story and the deploy ckpt are the same policy. Light inference; do it in a clean GPU window (NOT colliding
  with the sweep).

## Overnight GPU: richaux HP sweep [user: "allow the HP tuner to keep tuning richaux"]

`throwaway/perpetual_sweep.py` → study **`richaux_c64_t5_ce`**, q.optuna, `--search lr weight_decay
adam_beta1 adam_beta2 width block_layers frontend_mlp`, 1M budget/trial, MINIMIZE val_ce_t4. q.optuna reads
`rich_aux`/`n_critics` from the **on-disk QConfig defaults** (does not pass them) → the richaux config flip is
DEPLOYED on crockett (surgically, no HEAD move; the live run was unaffected). **GPU-gated**: waits for
richaux_q_20k to finish before any launch (one-GPU rule; else OOM). The rich-vs-base seed band was DROPPED
(user confident in richaux); `throwaway/seed_band.py` retired.

## Cold-start handoff (state at ~step 19000)

- **Live:** `richaux_q_20k` finishing (~step 19000/20000; best.pt=step18000 CE 0.6676). When it ends, the
  sweep supervisor (`pid 2701136`, `/tmp/sweep_sup.log`) auto-starts trials → `runs/*richaux_c64_t5_ce*`.
- **git:** clean; origin == deploy == `68985cb`+ (this doc will bump it). crockett checkout: HEAD 57be174 but
  config.py + throwaway/* surgically updated to current (richaux defaults live on disk).
- **Monitors:** `bfridnlbz` (richaux per-eval, fires at 20k), sweep-health monitor (errors + trial count).
- **TODO at richaux completion:** preserve `last_step20000.pt` (+ `best_step20000.pt` only if it beats
  0.6676), refresh `docs/preserved_checkpoints.md` (hand-maintained; pull numbers via `tools.ckpt_meta` /
  `tools.sweep_report`), brief completion note.
- **Open:** step-13000 behavioral re-analysis (above); `keep_every>0` for future headline runs; deploy the
  config flip via a clean `git reset --hard origin/main` on crockett once the sweep is between trials (the
  surgical checkout left the index dirty but functionally correct).
