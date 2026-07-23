# 2026-06-18 — curated seed band (beats EG-C2F, seeded), gate study, curated-frontend redesign

Continues `docs/sessions/2026-06-17-unify-net-and-seed-band.md`. Two threads: (A) **finished + enshrined the
8-seed curated reliability band** (the goal: viewpoint-Q beats EG-C2F, now seeded); (B) **redesigned the
curated frontend** per a live user architecture review (in flight). Everything pushed; crockett kept at
`origin/main`.

## A. The 8-seed curated band — DONE, canonical record = `docs/old_frontend_band_results.md`
- Band (OLD curated arch, BN+gate+shared 2052→32 proj) finished overnight: 8 seeds × 1M. **Matched endpoint
  (final eval): ce_t4 0.6657±0.0008, mIoU_t4 44.76±0.11; deploy (best.pt): 0.6646±0.0008 / 44.91±0.15.**
- **EG-C2F-c64** (deterministic, `baselines.evaluate`, it DOES log CE — the old "lacks CE" note was stale):
  CE 0.7258/0.7004/0.6828/0.6707, mIoU 42.22/43.30/44.04/44.65.
- **Result: curated beats EG-C2F on CE at every t1–t4 by ~6–13σ of the band** (margin +0.0107/0.0117/0.0077/
  0.0050; largest at t1). mIoU margin small (+0.11 endpoint / +0.26 deploy) — CE↔mIoU misalignment, judge by CE.
  The single-seed "44.94 peak" was best-of-trajectory optimism; honest seeded endpoint 44.76.
- **Tags (pushed):** `result/curated-8seed-band`→`55fda96` (band arch; **ckpts ONLY load at this tag**),
  `result/egc2f-c64-baseline`→`e805277`. **Ckpts:** best seed `preserved_ckpts/seedband_curated/s3_*` (ce_t4
  0.6636); all 8 in crockett `runs/*seedband_curated_s{0..7}`.
- Migrated all 15 earlier preserved ckpts (anchor/richaux) → `*_unified.pt` (load under current code; the band
  ckpts are NOT unified-migratable — different arch lineage → use the tag). Verified bit-identical net + q.eval
  reproduces docs to the bf16 floor; **probe is fp32** (`scoring.head_logits`), only backbone is bf16.

## B. Curated-frontend redesign (user-driven; IN FLIGHT)
Sequence of arch states (all `input_mode='curated'`; the frontend is the ONLY curated-vs-canvas difference):
1. **OLD (band, tag `result/curated-8seed-band`):** concat 6 groups → `[B,2052,32,32]` → BN → ×per-group gate
   → SHARED `proj(2052→32)` → `out(x+mlp(x))` (mlp NOT pre-norm) → `out(32→128)`. User: the 1/1/1/1/1024/1024
   concat "gives very unequal roles."
2. **per-group-sum (no-op, abandoned):** each group→proj(→32)→**sum**. **VERIFIED algebraically identical to
   the shared proj (1.9e-6)** — `Σ_g W_g·feat_g = W_concat·feat`. A reparametrization, NOT a real change.
3. **per-group LN sum (`groupln_s0`, abandoned):** group→proj→**LayerNorm**→sum + pre-norm `_TokenMLP`. The LN
   IS the non-trivial part (nonlinear, breaks the equivalence). I WRONGLY claimed LN caps content to 1/6 energy
   — **wrong: LN has elementwise affine (γ,β), so per-channel/per-group importance is restorable** (user
   corrected). Only per-spatial-token magnitude within a group is normalized (user: non-issue).
4. **CURRENT (`curated128_s0`, code `adaa231`):** per group `proj(size→curated_dim)` → `LayerNorm2d(affine)`
   → `× per-group gate` → **sum** → pre-norm `_TokenMLP` → `out(→width)`. **`curated_dim` 32→128** (the 32 was
   a narrow bottleneck, user). Gates kept (harmless, interpretable; `frontend.log()` = `gate_*`). BN on inputs
   kept. `GROUP_NAMES` now lives in `q.config` (single source); `curated_groups` selects a subset (ablation).
- **Early read (groupln_s0, the dim-32 version):** tracked slightly BEHIND old arch (ce_t4 @1k 0.6813 vs
  0.6745; @2k 0.6685 vs 0.6665 — gap shrinking). curated128 (dim-128+gate) should match/beat; **validating**.

## Gate study (OLD-arch band ckpts; `throwaway/gate_study.py`)
8-seed gates (tight): **ent 1.13↑, ent_delta 1.06↑** (favored), **feat_delta 0.70↓, cos_prev 0.88↓**
(suppressed), ln_feat/cos_init ≈1. Per-channel effective gain: entropy dominates (0.758, ~2× ent_delta, ~7×
feature channels). **Group totals are UNTRUSTWORTHY from weights** (depend on data covariance/redundancy of the
1024-d groups — `√Σeff²` 3.8 vs coherent `Σeff` 116 for ln_feat) [user flagged]. Definitive importance = the
`curated_groups` ablation (retrain without a group), NOT gates. `throwaway/ablation_scan.py` (entropy-only,
drop-feat_delta, …) was built but the run was on the OLD arch (confounded) → killed; **re-run on the new arch.**

## Tools added/changed this session
`tools.seed_report` (per-t mean±std; matched-endpoint + deploy bands), `throwaway/run_traj.py` (single-run
trajectory + per-group readout), `tools.sweep_report --full` (train+val CE+mIoU per step), `throwaway/
{gate_study,ablation_scan,migrate_qnet_keys}.py`, `throwaway/seed_band.py` now parametrized
(`--prefix --input-mode --lr`).

## Standing lessons / directives reaffirmed
- **crockett ALWAYS at `origin/main`** — `git reset --hard` after every push. **git ops NEVER disturb a running
  process** (code in memory; git_rev captured at launch) — stop hedging this; only a SUPERVISOR's NEXT spawn
  picks up new code [[git-ops-never-touch-running-process]].
- **pkill -f over ssh self-kills** if the target string appears ANYWHERE in the remote command (incl. an
  `echo "...run_name..."` or a launch line) — use ONLY the bracketed `[x]pattern`, nothing else; confirm in a
  separate ssh [[pkill-f-self-match-over-ssh]].
- Judge by **val CE**; report mIoU too. **Matched-endpoint (final eval) is the rigorous band**; deploy (best.pt)
  is optimistic. **work autonomously, no questions** [[work-autonomously-no-questions]].

## Cold-start handoff (state NOW)
- **Live:** `curated128_s0` training (code `adaa231`; crockett HEAD `29db804` = docs/tags on top, same q/ code).
  Monitor `bweiuaa4i` (completion/stall). Compare its endpoint to the band **0.6657±0.0008**; if ≥ → curated128
  becomes the curated default → seed-band it + tag; else revert to the band arch (tag stands).
- **mlflow** tunnel `brqvu9ncg` (local :5500 → crockett :5500), experiment `canvit-q`.
- **git:** clean; origin == deploy == crockett == `29db804` (+ this doc). Tags pushed.
- **Open / next:** (1) curated128 verdict; (2) feature ablations (entropy-only / drop-feat_delta) on the NEW
  arch via `ablation_scan` + `curated_groups`; (3) curated-vs-canvas band (the canvas band was killed,
  never finished — still open); (4) CLAUDE "The target" / curated-frontend block to update once curated128 lands.
