# 2026-06-14 — Refactor, the train-slice bug, and per-position target normalization

Continuation of the grid value-policy work. This session did a big cleanup, then a deep
investigation that found a real data bug, then re-architected the target normalization.
Single entry point to resume cold.

## UPDATE — 2026-06-14 evening: eval-mode bug, per-position/riid MISMATCH, global-z restored

Three findings AFTER the refactor below; they DEMOTE per-position norm from the headline.

1. **eval-mode bug (dropout/BN ran during eval).** The probe-style input stem added Dropout2d +
   BatchNorm2d, but `evaluate_grid`/`value_map_figure` never put the value-net in `eval()` — so
   eval ran with dropout active + BN on batch stats, silently corrupting every eval metric.
   Harmless before the stem (GroupNorm only) → it slipped in with the stem. Fixed with an
   `eval_mode` context manager wrapping all eval forwards. Caught BEFORE any stem run reported a
   number. (commit 71553bf)

2. **per-position norm has a train/normstats t0 MISMATCH → high loss/grad (NOT bad init).**
   normstats (per-cell mean/std) are precomputed from FULL-SCENE t0 (`reward_maps`), but training
   draws targets from RIID t0 (`objective` sets t0_mode=riid). RIID is a degraded start state →
   larger, more-variable fractional-CE reduction → `target=(frac_riid-mean_fs)/std_fs` has std
   ~1.4-2.3 (LOGGED `target_std`), not 1 → E[target²]≈2-5 → train_loss 2-6, grad_norm ~100. The
   logged `train_loss_norm`≈1.0 (= loss/target_var) PROVED the net sat at the predict-the-mean
   baseline — i.e. the high loss was TARGET MIS-SCALING, not net init/scaling. Diagnosed from
   grid_s10's own logged metrics (target_std is the instrument).

3. **global-z restored as the DEFAULT normalization (commit 71553bf).** trial0009 used online
   bias-corrected EMA (`RunningNorm`), which normalizes the ACTUAL training stream → unit variance
   regardless of t0_mode. New config `target_norm: "global_z" (default) | "per_position" (opt-in,
   pending riid-derived normstats)`. Reuses the `normstats is None`==global-z convention (no dup).
   With global-z the reproduction shows target_std~1.0, train_loss~0.6, grad_norm~15-25 — mismatch gone.

**Reproduction RUNNING:** `grid_repro_t0009_h5000` — global-z, riid t0, **stem OFF** (`--no-input-norm
--dropout 0.0` to match trial0009's no-stem arch), lr/wd/b2/mom = trial0009, h5000, eval_every 500.
@ step 1000: val_gridcorr 0.154, train_gridcorr 0.151 (GAP CLOSED → slice fix confirmed), val_miou_t0
0.3853 (= paper 38.5%, eval sanity). Target: val_gridcorr ≈ **0.31355** at step 5000.
[crockett:runs/grid_repro.log, mlflow canvit-grid, commit 71553bf]

**So:** under the DEFAULT (global-z), `val_gridcorr` = the absolute landscape corr exactly as in
trial0009 (HEADLINE #2's "metric CHANGED / residual" applies ONLY to target_norm=per_position).
Open: fix per-position by deriving normstats from riid-t0 frac (or train full_scene), THEN test if
removing the per-cell center bias beats global-z's ~0.31.

## HEADLINE FINDINGS (most important — read first)

1. **THE TRAIN-SLICE BUG (root cause of the "train-val gridcorr gap").** The train-eval slice
   and its reward map used `Subset(ds, range(N))` = the **first N training images**. ADE20K is
   **ordered** (by scene), so first-N is a systematically harder, more-homogeneous subset.
   Measured: train t0 mIoU **first-256 = 22.9% vs RANDOM-256 = 36.4% vs val 38.5%**. The whole
   "train_gridcorr 0.46 >> val 0.31 = overfitting" story was an ARTIFACT — a constant
   mean-landscape predictor (no net) reproduced the gap (0.44 train / 0.28 val). **val_gridcorr
   was ALWAYS faithful** (full val, no subset), so every val result/ranking to date stands.
   FIX: `data.representative_subset(n_total, n, seed=1234)` — fixed random subset, used by BOTH
   `grid_train`'s train_eval_loader AND `reward_maps`' training slice (deterministic -> aligned).
   Requires regenerating the trainslice reward map.

2. **THE METRIC CHANGED — old gridcorr numbers are NOT comparable.** We added **per-(position,
   scale) target normalization** [user]: precompute per-cell mean/std [V] of the fractional-CE
   target over the representative train slice (saved as a `normstats` artifact, REQUIRED), and
   z-score the regression target per cell. The net now learns the **image-specific deviation**,
   not the dominant per-position center bias (a constant predictor of the mean landscape scored
   ~0.3 -> the OLD gridcorr was center-bias-dominated). Deploy argmax **denormalizes**
   (pred*std+mean) to recover the absolute-best viewpoint. **Metric names (c016056):**
   `val_gridcorr` = ABSOLUTE landscape corr — UNCHANGED meaning, the optuna objective, **directly
   comparable to grid_s8 trial0009 0.3136** (eval denormalizes pred first; must match it [user]).
   `val_gridcorr_residual` = the NEW per-position-residual corr (image-specific conditioning
   diagnostic; ~0 => net not conditioning — the real open problem). Ckpts store norm_mean/std.

3. **augment was a phantom fix.** Enabled augmentation + dropout/stem to fight the "overfit" —
   which was the slice artifact. augment also slowed convergence and hurt val_gridcorr at h2000.
   Now **augment OFF by default**; **ln/bn input stem KEPT** [user]; dropout 0.1 kept (mild).

## BEST RESULT TO DATE (OLD metric — faithful, val is full-val)
`grid_s8_h5000__trial0009`: **val_gridcorr 0.3136** / spearman 0.3191 / mIoU 0.411. HPs:
lr 8.251e-5, wd 0.011924, adam_beta2 0.95, target_momentum 0.997, h5000. These are now the
**default GridConfig + optuna FROZEN seed**. (grid_s6 trial3/8/9 ~0.308 also faithful.)
NOTE: not comparable to grid_s10's residual gridcorr.

## THE REFACTOR (committed, gate green throughout)
- `canvas_ops.py` — extracted candidate_ce/candidate_canvas/full_scene_state so the grid path
  doesn't import the flow stack.
- **Pruned the flow/RWR/critic lineage**: 14 modules, ~2775 LOC deleted (tag
  `archive/flow-rwr-critic` for recovery: `git checkout archive/flow-rwr-critic -- <path>`).
  `evaluate.py` decoupled to baseline-only. Repo 34 -> 20 modules, 4704 -> ~1870 LOC.
- `per_image_ce` moved to `scoring.py`; dead `harness.aggregate_per_t`, `metrics.spearman` removed.
- Contract tests `test_grid.py`: scale-major flat-index convention, RunningNorm bias-correction,
  grid_viewpoints safe-box.
- `pypatree` added to dev deps (`uv run pypatree` works).

## ARCHITECTURE (grid_net.GridValueNet) — current
- **Probe-style input stem** `_InputStem` (ln-dropout-bn) at the canvas->UNet interface,
  mirroring SegmentationProbe's front-end. input_norm gates ln+bn (params); dropout=0 +
  input_norm=False == pre-stem arch (parameterless no-op -> old ckpts load).
- **`_TokenMLP`**: per-canvas-token transformer-style PRE-NORM residual MLP (x + mlp(LN(x))),
  optional (`frontend_mlp`), before the UNet.
- **`block_layers`**: conv layers per UNet block (depth knob).
- Sweepable via optuna SPACE: lr, weight_decay, adam_beta2, target_momentum, width{64,128,256},
  entropy_channel, augment, input_norm, frontend_mlp, block_layers. **dropout FIXED 0.1** (never
  swept) [user]. `--search lr` default; `--search lr width frontend_mlp ...` to open arch.

## grid_optuna mechanism
`FROZEN` (best-config pin per HP, doubles as seed) + `SPACE` (search ranges); `--search` lists
which to sweep, rest pinned. base_steps default 5000, rungs 1 (ladder didn't help val).
eval_every=None in search (endpoint-only; periodic full-val eval is pointless overhead [user]).

## CURRENT RUN (live at compaction)
- **Chained regen -> grid_s10**, commit **582c399**, seed 6, on crockett (checkout reset to 582c399).
- regen: `reward_maps --canvas-grid 32` writes representative val+trainslice maps + normstats
  (`runs/reward_maps/{grid_validation,grid_trainslice,normstats}_s0.5-0.25_g16_r128_c32.pt`).
- then `grid_optuna --study grid_s10 --seed 6 --base-steps 2000`: h2000, per-position norm,
  augment OFF, ln/bn stem ON, lr-only sweep around trial0009. Logs: runs/regen.log, runs/grid_s10.log.
- Watch: http://localhost:5500/#/experiments/4 (canvit-grid).

## EXPECTED grid_s10 OUTPUT + branch plan
- **train_gridcorr ≈ val_gridcorr** now (representative slice) -> confirms the gap was the slice
  artifact. If train STILL >> val -> slice wasn't the only cause; investigate.
- **val_gridcorr (residual) lower than old 0.31** (different metric). If ~0 -> net barely
  conditions on the image (the real problem) -> arch matters -> the categorical sweep below.
  If meaningfully >0 -> per-position norm unlocked image-specific learning.
- mIoU ~0.41 still (denormalized argmax; center-bias argmax already decent).

## NEXT (user-directed)
- **Sweep arch categoricals**: frontend_mlp, entropy_channel, width 64 (half), block_layers,
  input_norm — `--search lr width frontend_mlp entropy_channel block_layers`.
- Generalization probe (throwaway/generalization_probe.py, written, NOT yet run): synthetic
  salient-patch-at-varied-positions test of conditioning; reconstruct arch from ckpt config.
- value_map_figure still shows absolute TRUE vs normalized PRED (minor viz inconsistency under
  per-position norm) — pass normstats to show residual + denorm the pred argmax mark.

## STANDING DIRECTIVES added this session [user]
- NEVER jump to conclusions; reread code to hunt issues; ≥3 hypotheses + the distinguishing test.
- ALWAYS background long jobs; while they run, reflect on expected output + branch plans.
- val_gridcorr AND train_gridcorr ALWAYS at full_scene t0 (even though train uses riid t0) — both
  already full_scene in evaluate_grid; the gap between them is the like-for-like overfit measure.
- Precompute expensive things separately (don't redo each run); store provenance in ckpts.
- Keep GPU usefully busy at all times; fast 2k-step iteration (useful results at 2k).
- All mlflow runs store git_rev (already via TrainLogger).

## MISTAKES THIS SESSION — guards
1. **Jumped to "generalization gap -> arch/reg fix"** without verifying t0 modes / the slice. The
   gap was a biased-slice artifact. GUARD: verify the eval-set construction before diagnosing.
2. **Tried to kill grid_s8 on a GPU-utilization question** (not a kill request) — blocked, correctly.
   Then trial0009 turned out to be a new high. GUARD: don't kill on ambiguous signals.
3. **rg filter `-v "^\["` ate the key gridcorr output lines** (they start with `[`) — confused a
   read. GUARD: don't filter primary output; read full.
4. **ssh exit-255 mid `pkill;reset`** left the checkout un-reset (HEAD stale). GUARD: split ssh
   calls; verify HEAD after.

## EVENING ARC (2026-06-14, autonomous): BN noise → ConvNeXt arch → WHERE-underfitting

Chronological; see memories `grid-residual-conditioning`, `grid-where-underfitting` for detail.

1. **Init grad-norm pathology + residual fix.** The hand-rolled conv U-Net had init grad ~120
   (amplification 207x; a ViT's is ~0.1), encoder-dominated. Cause (throwaway/grid_init_analysis.py):
   the backbone `get_spatial` input is **std 186, absmax 4428** (DINOv3 artifact tokens). Residual
   pre-act blocks + zero-init fixed it to ~6 — but ONLY with input norm (the identity shortcut
   propagates the input; residual + no-stem EXPLODES to grad 50k). [user: "ln+bn everywhere".]
2. **BatchNorm = the val_gridcorr eval-noise (isolated CPU test, throwaway/bn_eval_stability.py).**
   BN running stats are wrecked by the std-186 outliers -> eval-output-std VOLATILITY 5.3 (vs 0.0
   for LayerNorm/GroupNorm). That was the val_gridcorr -0.035 spikes. [user: "i never use groupnorm"]
   → went LayerNorm everywhere.
3. **Arch = timm ConvNeXt-V2 blocks + LayerNorm2d** [user: "import the convnextv2 library"]. Battle-
   tested init (layer-scale ls_init=1e-6 near-identity + GRN). Isolated CPU validation
   (throwaway/grid_arch_cpu.py): [16,1025,32,32]->[16,2,16,16], **init grad 2.67** (was 120),
   train-vs-eval gap **0.0**, 3.06M params. Validated on real data in grid_s13: grad ~4-5.
4. **The ceiling is NET underfitting the WHERE (throwaway/landscape_headroom.py).** val_gridcorr
   ~0.30 ≈ the 0.28 center-floor. Landscapes are 97% image-specific; target is 50% image-LEVEL /
   50% within-image (WHERE to look); 3x reward headroom. Net learns the LEVEL (train_corr 0.65),
   underfits the WHERE. Lever: per-IMAGE-centered target / K>1 / capacity. (Conditioning + noise
   were fixed but did NOT raise val_gridcorr — consistent: trainability was never the ceiling.)

**Defaults now @ commit b22acef+ (residual->ConvNeXt arch, LayerNorm2d, dropout 0, global-z).**
Commits: 71553bf (eval-mode + denorm + global-z), 21ade18 (residual), 12970e6 (ConvNeXt),
b22acef (timm dep). grid_optuna gained `--eval-every` (best-of-trajectory for long noisy runs).

## CURRENT RUN (cold-start resume — NO archeology)
- **grid_s13** ConvNeXt sweep: `crockett:~/projects/CanViT-PyTorch-RL`, commit **b22acef**,
  `grid_optuna --study grid_s13 --seed 6 --base-steps 20000 --eval-every 5000 --search lr width
  frontend_mlp entropy_channel block_layers`. ~34-40 min/run, 10 trials ≈ 5-6h. Log
  `runs/grid_s13.log`; mlflow :5500 exp canvit-grid. Watch: does val_gridcorr beat 0.30?
- Monitor: `ssh crockett 'cd ~/projects/CanViT-PyTorch-RL && python3 -c "import ast;[print(d.get(\"step\"),d.get(\"val_gridcorr\")) for l in open(\"runs/grid_s13.log\") if \"val_gridcorr\" in l for d in [ast.literal_eval(l.split(\"INFO \",1)[1])]]"'`
- **NEXT lever — must RESPECT K=1 + global-z + fractional-CE** [user 2026-06-15: per-image-centered
  target / K>1 / pairwise ranking is "ABSOLUTELY FUCKING NOT" — it undoes the K=1 keystone that
  AVOIDS per-scene normalization]. So the WHERE must be reached via: (a) capacity/arch — grid_s13
  ConvNeXt h20000 is the running test; (b) richer LOCATION-resolved input features (per-cell probe
  uncertainty, box-sampled features) so the net can resolve which cell helps; (c) better/longer
  optimization to escape the easy image-LEVEL solution. Decide after grid_s13's early trials.
