# 2026-06-18 (late) — keep_every default, curated→old-frontend rename, seed_report = the band tool

Continues `2026-06-18-single-frontend-refactor.md`. No training-code changes this session — config default, a
big naming cleanup, and graduating the band plotting into a proper tool. HEAD `c98c7bd`; local == origin ==
deploy == crockett (verify `git rev-parse HEAD` both ends).

## What changed
- **`keep_every` defaults to 1000** (`q/config.py`) — a `step_*.pt` every eval (~22MB each) so the
  best-by-mIoU step (≠ best-by-CE) isn't lost [user: "they arent that big"]. Seeds 0,1 of the HEAD band ran
  before this; seeds 2–7 keep intermediates.
- **"curated" → old-frontend / derived / hand-picked** (commit `773dace`). The word was overloaded across 3
  senses; "curated band" wrongly implied a feature diff (both frontends use the SAME derived features — the
  real diff is the frontend projection). Renamed docs `old_frontend_band_results.md`, `sweep_sets.md`; all
  inbound links fixed. **Frozen artifacts keep their on-disk `curated` names** (tag `result/curated-8seed-band`,
  dirs `seedband_curated_s*`, `preserved_ckpts/seedband_curated/`) with a "curated == old-frontend" note.
  Memory: [[name-by-distinguishing-axis]].
- **Checkpoint selection rule** [user emphatic]: deploy ckpt = per-seed **best MEAN val CE across t1..tEnd**,
  NOT t4-alone (removed: "best-t4 is shit") and NOT privileging the last step. Say "last", never "matched
  endpoint". Rule lives once in `tools/seedband_io.py`. Memory: [[select-by-mean-t1-t4-ce]].
- **`tools/seed_report` is now THE band tool** — text table (always BOTH CE+mIoU per t) + `--plot PATH` figure.
  Graduated out of `throwaway/` [user: "NOT THROWAWAY ... proper clean tool"]; the separate `band_plot` was
  merged in (they differed only in output medium). Shared readers/selection in `seedband_io`.
- **Figure styling follows `~/code/CanViT-paper-exporter`** (`style.POLICY_STYLES`): per-policy colors
  (EG-C2F green / C2F blue / F-IID near-black), Viewpoint-Q a distinct pink hero; clean lines (no markers),
  CI fills, linestyle = solid measured / dashed paper; paper Table 4 **±95% CI** (`TABLE4_CI95`) shown for the
  stochastic baselines. **mIoU panel only by default** (`--ce-panel` adds CE). Legend names the learned policy
  "Viewpoint-Q (trained, early-stop on mean t1–t{end} val CE)" vs "(untrained)" — NOT "ours" (we're the authors).

## Live state / next (cold-start)
- **HEAD band `seedband_s*`** (`throwaway/seed_band.py`): 3/8 done (s0,s1,s2), s3 running, supervisor PID 3766408.
  best-mean(t1–4)-CE band (n=3): mIoU t1–t4 `42.73 / 43.90 / 44.54 / 44.89`, CE `0.7141 / 0.6880 / 0.6744 / 0.6653`
  — beats EG-C2F at every t. Regenerate any time: `python -m canvit_pytorch_rl.tools.seed_report --plot outputs/band.png`
  (run ON crockett; fetch the PNG with `ssh crockett 'cat .../outputs/band.png' > local.png`).
- **Measured c64 baselines for the CE panel + solid mIoU lines**: deterministic-policy evals running on **CPU**
  (GPU full with the band — concurrent GPU eval OOMs, band unaffected). Run names `c2f_c64_t5`, `fiid_c64_t5`
  (`baselines.evaluate --policy {coarse_to_fine,full_then_random} --canvas-grid 64 --n-timesteps 5 --device cpu`);
  ~40s/batch ×125, sequential. When their `summary.json` lands, seed_report auto-picks them up (solid lines,
  CE panel) — **regenerate + resend the figure**. EG-C2F already measured (`egc2f_c64_t5_ce`).
- **When all 8 seeds finish**: aggregate `seed_report` (the best-mean band IS the headline now), tag
  `result/qpolicy-8seed-band`, and write a HEAD-band record doc (the analogue of `old_frontend_band_results.md`
  for the per-group-LN frontend). Completion poll `blwbqkf2k` may still be armed; the autonomous loop also checks.
- Still open from before: dormant knobs (t0_mode/scale_min, compile, augment) keep/cut; the score_res 512² wart.

## Memory added/updated this session
`select-by-mean-t1-t4-ce`, `name-by-distinguishing-axis` (both indexed in MEMORY.md); `always-judge-ce-and-miou-both`
de-staled for keep_every-default; `best-1m-anchor` / `MEMORY.md` curated→old-frontend.
