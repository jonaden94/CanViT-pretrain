# 2026-06-17 — remove ensembling, unify ViewpointQNet, drop "richaux"; sweep analysis; seed band

Big code-clarity refactor (NO behavior change, verified bit-identical) + the matched-budget seed band the
goal has been missing. Key commits: `973f5be` (remove ensembling + unify), `b348520` (drop richaux →
input_mode), `4b03996` (docs), `2ac9ff9` (sweep_report --full), `5432aad` (crockett-always-latest directive).

## Code changes
- **Removed twin-Q / clipped-double-Q ensembling**: `EnsembleQNet`, `n_critics`, `ensemble_agg`, per-critic
  loss, grad_norms critic-stripping, `throwaway/ensemble_eval`. Single net always.
- **ONE `ViewpointQNet`**: pluggable `frontend` (`CanvasFrontend` | `CuratedFrontend`, both `: Frontend`) →
  shared ConvNeXt body; `build_qnet` is the sole constructor; polymorphic `frontend.log()` (no isinstance).
- **Dropped "richaux"**: `rich_aux:bool` → `input_mode: Literal["canvas","curated"]="curated"`; `rich_dim`→
  `curated_dim`; `RichFrontend`→`CuratedFrontend`; `rich_*`→`curated_*`. ckpt schema `rich_aux`/`rich_dim` →
  `input_mode`/`curated_dim`. (`input_mode` avoids clashing with the swept `frontend_mlp`. Live study name
  `richaux_c64_t5_ce` kept — persistent sqlite ID.)
- **Migration** `throwaway/migrate_qnet_keys.py`: pure key-rename + metadata translation; strict-load + finite
  forward is the gate. All 15 preserved ckpts migrated to `<stem>_unified.pt` beside originals (originals
  only load under pre-2026-06-17 code; load the `_unified.pt` now). See `docs/preserved_checkpoints.md`.

## Verification (loading old ckpts in new code — user-required)
- **Q-net BIT-IDENTICAL** old↔new (amp/backbone/probe-free fixed input): curated sum −71.74588112 ==
  −71.74588112; canvas −468.53779167 == −468.53779167. Unification provably lossless for the policy net.
- `q.eval` of migrated ckpts reproduces documented per-t mIoU/CE to the bf16-backbone floor (trial0000 t4
  44.78 vs doc 44.95; full-fp32 `--no-amp` 44.73 — *lower*, so bf16 is NOT a consistent loss). Anchor (canvas)
  near-exact (t1 42.91, t4 44.93). **Probe is already fp32** (`scoring.head_logits` forces it); only the
  backbone canvas integration is bf16.

## Sweep analysis (`richaux_c64_t5_ce`, 13 trials, all seed=0 — `tools.sweep_report [--trajectories|--full]`)
- **train>val is a fixed PROBE confound, not overfit**: t0 mIoU trn/val = 43.47/39.60, **Δ+3.87 identical in
  every trial** (no policy at t0). At t1 the gap is ~+4.5; the excess over +3.87 (~+0.3 underfit → +0.7
  well-fit → +0.95 w256) is the only real policy train-advantage. Val never regresses → judge by val alone.
- **t1 saturates by 2k steps (160k fwd); t2–t4 keep improving** — training sequences glimpses more than it
  improves the single t1 pick (relevant to t1-selection-headroom).
- **lr is the only axis**: ≤~3e-5 underfits train AND val (trial0002 lr6.9e-6, full run, stalls at ce_t4
  0.6756 / mi 44.2, low train_corr); 5.5e-5–3e-4 all hit the SAME plateau **ce_t4 ≈ 0.6645, mi_t4 ≈ 44.9**.
  wd/β/block_layers/frontend_mlp flat. w256 = same floor, earlier/overfit, slower wall (~2h vs w128 1h49m).
- **The default (lr3e-4, w128, bl3) is the balanced winner, still-improving at 1M, cheapest** — sweep CONFIRMS
  it, beats nothing beyond noise. The plateau beats EG-C2F-c64 (44.65) by ~0.3 mIoU — but at SINGLE seed.

## Seed band (LIVE) — the open question is RELIABILITY, not HPs
Killed the (confirmed-plateaued) HP sweep; launched `throwaway/seed_band.py`: `q.train` at QConfig defaults
`--seed 0..7`, 1M forwards each, SEQUENTIAL (one GPU), crash/pause-resilient via `runs/.seedband_done/<seed>`.
Runs `seedband_curated_s{0..7}`. Log `/tmp/seed_band.log`. Aggregate per-t mean±std with `tools.seed_report`.
Answers: is ce_t4 0.6645 / mi_t4 44.9 reliable across seeds, and does the EG-C2F margin survive seed noise?

## Cold-start handoff
- **crockett HEAD = `origin/main` ALWAYS** [user 2026-06-17, CLAUDE §Execution]: `git reset --hard` after every
  push (safe under a live run — code in memory + git_rev at launch; only re-exec'd supervisors pick up new
  code). Don't push a TRAINING-behavior change mid-seed-band (it must stay one code version).
- **Live**: seed band on crockett (`/tmp/seed_band.log`); seed 0 training. crockett checkout `5432aad`, clean.
- **Next**: let the band finish (~8×~1.8h); `tools.seed_report` for the band; preserve the best seed; update
  CLAUDE "The target" + `docs/preserved_checkpoints.md` with seeded mean±std (replaces the single-seed peaks).
