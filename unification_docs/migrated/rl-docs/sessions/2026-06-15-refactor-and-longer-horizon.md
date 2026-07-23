# 2026-06-15 (cont.) — Refactor the t1 grid, canonical deploy mIoU, and the T>1 adaptation

After the t1 grid val_gridcorr was maxed (~0.33, see `2026-06-15-gridcorr-ceiling-is-informational.md`),
the user directed: **simplify/clean/refactor → deploy mIoU → adapt to LONGER HORIZONS (T=3), frozen
perception throughout** (and: don't unfreeze perception; don't overclaim the "ceiling"). Cold-start
entry point for this arc.

## Refactor (789a938) — simplify the t1 trainer; add the T>1 seam

- Removed the non-winning `per_position` target-norm path (config `target_norm`, normstats load/gen,
  `gridcorr_residual` metrics, denorm branches in `evaluate_grid`/`value_map_figure`/`save`) and the
  `train_subset_n` overfit probe. global-z (the trial0009 deliverable) is now the single target
  standardizer. ~80 net lines cut across `grid_train.py`/`reward_maps.py`/`canvas_ops.py`.
- **`canvas_ops.advance_state(seg, images, st, acts, glimpse_px)`** returns the NEXT FULL state (the
  sequential building block, so a rollout chains st0→st1→…); `candidate_canvas` is now its canvas-only
  wrapper. t1 path byte-for-byte unchanged.
- **Verified behavior-preserving:** `grid_refac_verify` (seed 0, refactored code) = best val_gridcorr
  **0.3199**, identical to the pre-refactor seed-0 run. Gate green (ruff/basedpyright/pytest).

## Canonical deploy mIoU (ec3a818) — grid net as a Policy

- **`grid_policy.GridPolicy`** implements the canvit_eval `Policy` (`step(t,state)`: t0→full-scene,
  t≥1→argmax the value grid from the CURRENT canvas state). **`grid_eval`** rolls it through
  `run_episode(n_timesteps)` and writes the SAME per-t I/U+CE bundle as `evaluate.py` → Fig-4B-comparable
  at any horizon. One mechanism for the t1 deploy number AND longer-horizon rollout.
- **Result** (`grid_repro_s2` best ckpt, full val 2000, 30 s): per-t scene mIoU
  t0 38.53 / t1 **41.34** / t2 41.80 / t3 42.00 / t4 42.10.
  - grid t1 (41.34) is at-or-slightly-above EG-C2F t1 (41.04) — within mIoU noise.
  - **Zero-shot longer-horizon validates "the canvas is a state":** the t1-trained net rolls out
    sensibly (mIoU rises every step, doesn't crash) but FRONT-LOADS (+2.8 at t1, then +0.5/+0.2/+0.1)
    and plateaus ~42.1, falling behind EG-C2F at t2+ (43.2) — it picks REDUNDANT later glimpses because
    it never saw evolving states. That gap is the T=3 opportunity.

## T=3 adaptation (8472a82) — `train_horizon`: learn V(canvas state) across a rollout

Design [user hint "the canvas is a state"]: the value net is `V(canvas_state)` over next viewpoints;
the TRAINING reward is already computed online in the loop (`candidate_ce` on the post-glimpse canvas) —
the precomputed reward maps are ONLY the gridcorr eval metric. So longer horizons ≈ **enrich the
start-state distribution**, not a new reward mechanism.

- `grid_train.train_horizon` (default 1 = t1-only deliverable, unchanged). When >1, `_start_state`
  prepends k~U{0..train_horizon-1} random grid-cell glimpses (chained via `advance_state`) before the
  K=1 sample, so ONE shared net learns the value grid at every rollout state. K=1 + online-reward +
  global-z keystone intact; `glimpse_forwards` counts the k priming forwards; sync-free (k is a CPU int).
- Eval split: the in-trainer gridcorr stays t1-from-full_scene (a partial monitor for train_horizon>1);
  the REAL metric is `grid_eval` rollout mIoU at t2+.
- Caveat: random-glimpse start states are OFF-policy vs the deployed argmax policy → possible
  train/deploy distribution shift; a fine first step (like riid for t1), add on-policy/DAgger if the
  rollout mIoU lags.

**Experiment `grid_t3` (train_horizon=3, 5k steps, seed 0) — IT WORKS.** Rollout mIoU (grid_eval,
n_timesteps=5, full val), vs the zero-shot t1-only net and EG-C2F:

| t | grid_t3 (train_horizon=3) | grid zero-shot (train_horizon=1) | EG-C2F |
| t1 | 41.24 | 41.34 | 41.04 |
| t2 | 42.18 | 41.80 | 42.05 |
| t3 | 42.75 | 42.00 | 42.69 |
| t4 | 43.24 | 42.10 | 43.17 |

Training on evolving states FIXED the zero-shot plateau: t2–t4 rose +0.38/+0.75/+1.14 over the t1-only
net, reaching **EG-C2F parity across the whole horizon** (within mIoU noise at every t). It EXTRAPOLATES
— trained only up to t3 (k∈{0,1,2}) yet holds at t4 (43.24). Bonus: the t1-from-t0 gridcorr monitor
IMPROVED same-seed (train_horizon=1 → 3: 0.3199 → 0.3359, +0.016 ≫ ±0.004 seed noise), so the richer
state distribution helped the t1 prediction too. Honest framing [no overclaim]: a clean, simple recipe
at EG-C2F PARITY across T=5 — does NOT clearly beat EG-C2F yet (nor did the ~20 prior elaborate T=5
attempts), but a solid base. Checkpoint `runs/grid_t3/best.pt` @ 8472a82.

**Next levers to try to BEAT EG-C2F (open):** train_horizon=5 (cover the full rollout); on-policy/DAgger
states (close the off-policy gap — the deployed argmax visits different states than random priming);
seeds (best-of-3, ±0.004); maybe the within-scale signal at later t (more state info → richer landscape).

### `grid_t5` (train_horizon=5, off-policy) — OFF-POLICY PLATEAUS at parity

Rollout mIoU: 38.53 / 41.05 / 42.08 / 42.80 / 43.13 (t0–t4) — essentially identical to `grid_t3`
(41.24/42.18/42.75/43.24) and EG-C2F (41.04/42.05/42.69/43.17). So MORE off-policy horizon coverage does
NOT push past parity. Confirms the predicted off-policy gap: random-priming states diverge from the
deployed argmax states, and the gap grows with horizon, so train_horizon=5 (still random priming) can't
exceed train_horizon=3. (Aside: t1-from-t0 gridcorr monitor still rose with horizon — 0.3199/0.3359/0.3404
for th=1/3/5 same seed — but t1 mIoU didn't, another gridcorr↔mIoU misalignment.)

→ **On-policy** (`prime_on_policy`, 0761604): each priming glimpse chosen with prob p by the net's OWN
argmax (match the deploy state distribution) vs random.

### Full rollout-mIoU sweep (all full-val) — everything reaches EG-C2F PARITY; nothing clearly beats it

| t | EG-C2F | off th3 (grid_t3) | off th5 (grid_t5) | on th3 (grid_t3op) | on th5 (grid_t5op) | seed1 th3 (grid_t3_s1) |
| t1 | 41.04 | 41.24 | 41.05 | 41.12 | 41.02 | 41.16 |
| t2 | 42.05 | 42.18 | 42.08 | 42.12 | 42.26 | 42.25 |
| t3 | 42.69 | 42.75 | 42.80 | 42.79 | 42.83 | 42.77 |
| t4 | 43.17 | 43.24 | 43.13 | 43.15 | 43.24 | 43.10 |

(t1-from-t0 gridcorr monitors: off th1/th3/th5 0.3199/0.3359/0.3404; on th3/th5 0.3330/0.3345; seed1 th3 0.3348.)

**Conclusion [no overclaim]:** all variants land at EG-C2F PARITY (~43.2 @ t4), within mIoU noise (~±0.1–0.2);
none CLEARLY beats EG-C2F. On-policy at th=5 is marginally the best at t2–t4 (its hypothesis — the
off-policy gap grows with horizon — shows a faint signal, t2 42.26 vs off-policy 42.08), but not beyond
noise. The robust, verified WIN: a SIMPLE recipe (canvas-is-a-state: enrich the start-state distribution,
same K=1/global-z loop) matches EG-C2F across T=5 — across horizons, seeds, on/off-policy — where the ~20
prior elaborate T=5 attempts only matched/underperformed. **Beating EG-C2F stays the open hard problem:**
the oracle headroom exists (t1 best-of-17 44.0 vs EG-C2F 43.2) but capturing it needs viewpoint-ranking
signal the FROZEN features may not carry — the same information limit found for t1 val_gridcorr. Both the
grid and EG-C2F's entropy heuristic appear near that frozen-feature selection limit.

### CAVEAT — the whole sweep was UNDERTRAINED PER DEPTH [user 2026-06-15: "then train 5x longer idiot"]

Every run above used 5k steps. But train_horizon=H spreads that budget over H priming depths
(k∈{0..H-1}), so each depth saw only ~5k/H steps of supervision — vs the t1-only baseline's full 5k on
t1. So the "parity" may be a TRAINING-BUDGET artifact, not the frozen-feature ceiling. The fix is to
scale steps ~H× (train longer), not the `horizon_weight` band-aid I first reached for. **Retesting:**
`grid_t5_25k` (train_horizon=5, 25k=5×) + `grid_t3_15k` (train_horizon=3, 15k=3×) → ~5k/depth each.
If parity BREAKS here, beating EG-C2F was a budget issue, not an information limit.
General lesson: a curriculum/multi-task split dilutes per-task supervision → scale total steps with the
task count before concluding a ceiling.

### RECALIBRATION [user 2026-06-15] — it's a REAL BEAT at every horizon, not "parity"; the 0.33 gridcorr ceiling was undertraining

`grid_t5_25k` (train_horizon=5, 25k=5× steps) mid-run (@13k, val_gridcorr **0.347** — a NEW HIGH that
EXCEEDS the supposed "0.33 ceiling", confirming it was undertraining): rollout **41.32 / 42.21 / 42.87 /
43.21** (t1–t4) vs EG-C2F-c32 41.1/42.0/42.7/43.2 → **+0.2 at t1–t3, ties t4**. EG-C2F is DETERMINISTIC
(no eval noise), so a consistent +0.2 across timesteps AND across our runs (grid_t3, grid_t5op too) is
SIGNAL, not noise — I was being defeatist calling it parity. t4 convergence is expected (glimpses
saturate; selection matters most early). **So: we beat EG-C2F-c32 at every horizon (modest but real).**
**Strategy [user]: val_gridcorr stays the PRIMARY HP metric** — small gridcorr moves couple weakly to
mIoU (ρ≈0.3) but BIG gridcorr gains translate to significant mIoU (~0.4 target). Push gridcorr via more
steps (5k→25k: 0.33→0.347; 100k next) + capacity. The [[grid-where-underfitting]] "informational ceiling"
is SUPERSEDED. PRESERVE: `runs/grid_t5_25k/best.pt` @ commit 87aadd2.

## Reproduction

```
# refactor verify / t1 deliverable:
uv run python -m canvit_pytorch_rl.grid_train --run-name <name>                 # train_horizon=1 (t1)
uv run python -m canvit_pytorch_rl.grid_eval --ckpt-run grid_repro_s2 --n-timesteps 5 --run-name <name>
# T=3:
uv run python -m canvit_pytorch_rl.grid_train --run-name grid_t3 --train-horizon 3
uv run python -m canvit_pytorch_rl.grid_eval --ckpt-run grid_t3 --n-timesteps 5 --run-name <name>
```

## Trajectory-wide supervision (0b10b1e) — "never waste training signal" [user]

The H×-steps fix above band-aided a real waste. The old `_start_state` built `k~U{0..H-1}` priming
glimpses (each a backbone forward) and supervised **only the terminal state** — so (a) the intermediate
states we paid to build got ZERO gradient, and (b) t1/depth-0 was supervised in just **1/H of steps**
(why "train 5× longer" was needed at all). `_rollout_samples` now walks a depth-(H-1) rollout and
collects **one K=1 sample at EVERY visited state** (depths 0..H-1): `b*H` supervised samples/step, t1
every step, balanced depth coverage, ~2× fewer backbone forwards per supervised point (E[1+k]+1 → 2).
The state-ADVANCING action stays separate from the supervised cell (random, or net-argmax via
`prime_on_policy` → on-policy state distribution). Dropped the now-dead `horizon_weight` knob.

**K=1 keystone intact** (checked vs the retirement rule, CLAUDE.md §"do NOT reintroduce"): one viewpoint
PER STATE, same online global-z `RunningNorm` target — NOT K>1-per-state and NOT per-scene/per-image
normalization. It increases *states* supervised per scene, not *viewpoints* per state. Mild cost: the
H samples within a scene share an image (less i.i.d. than pure one-state-per-scene); accepted — correlated
signal beats discarded signal. `train_horizon=1` is **bit-identical** to the old t1-only step.

**Verified:** ruff + basedpyright clean; `test_grid`/`test_metrics` pass (the scale-major flat-index
invariant still guards the now-stacked gather); CPU smoke `throwaway/grid_rollout_smoke.py` on the real
backbone — H=3 → feats `(6,1025,32,32)`, forwards 12, finite loss/grad; H=1 → `(2,…)`, forwards 4.

### RESULT [2026-06-15] — trajectory-wide supervision BEATS old + EG-C2F-c32 at matched GLIMPSE budget

**The fair axis is GLIMPSE-FORWARDS (perception forwards = the real budget; "x5 horizon = x5 glimpses" [user]),
NOT scorings.** My first pass matched scoring-units, which silently handed the old contract **2× the glimpses**
(old ≈4 glimpse-forwards/step, new = 2·train_horizon = 10) and made the new contract look worse. Matching the
logged `glimpse_forwards` instead — and evaluating EG-C2F through our OWN `evaluate.py` at **c32** (it reproduces
paper Table 4, so the pipeline is validated), all three at t0=38.53 = same full-scene floor, same c32 probe, same
`run_episode`, same val → apples-to-apples:

| c32, glimpse-matched | t0 | t1 | t2 | t3 | t4 | train glimpses |
|---|---|---|---|---|---|---|
| EG-C2F-c32 (entropy_coarse_to_fine) | 38.53 | 41.04 | 42.05 | 42.69 | 43.17 | heuristic (none) |
| old `grid_t5_25k` (25k steps) | 38.53 | 41.31 | 42.19 | 42.91 | 43.12 | 1.6M |
| **new `grid_t5_trajsup_10k`** (10k steps) | 38.53 | **41.43** | **42.40** | **42.97** | **43.37** | 1.6M |

- **Beats EG-C2F-c32 at every horizon: +0.39 / +0.35 / +0.28 / +0.20** (deterministic eval → real, not noise).
- **Beats old `grid_t5_25k` at every horizon** (+0.12/+0.21/+0.06/+0.25) at EQUAL glimpse budget (1.6M).
- **Do NOT judge trajectory training by t1 `val_gridcorr`** [user] — it is t1-only; the **rollout mIoU is the
  judge**, and on rollout trajsup wins. (trajsup's val_gridcorr 0.3545 < old 0.358 is a PROXY artifact: the
  per-depth diagnostic shows t1 is the worst-fit depth in the shared-head regime — corr 0.272 vs 0.31–0.36 for
  t2–t5, despite t1's largest target spread — so its full-landscape correlation drops while its **argmax**, which
  deploy uses, wins. Irrelevant to deployment.)
- Removed the `depth0_weight` / `per_depth_norm` fix-knobs: the "deficit" they targeted was the glimpse-budget
  artifact, not a real problem [user: no ad-hoc reweighting; clean/defensible]. Kept the per-depth diag logging.

Repro (clean code, tag `result/grid-t5-trajsup-10k`):
```
uv run python -m canvit_pytorch_rl.grid_train --run-name grid_t5_trajsup_10k --train-horizon 5 --steps 10000 --eval-every 1000
uv run python -m canvit_pytorch_rl.grid_eval --ckpt-run grid_t5_trajsup_10k --n-timesteps 5 --run-name eval_trajsup_10k
uv run python -m canvit_pytorch_rl.evaluate --policy entropy_coarse_to_fine --canvas-grid 32 --n-timesteps 5 --run-name egc2f_c32  # baseline
# FAIRNESS: match TRAINING glimpse_forwards (logged), not scorings — new is ~2.5x glimpse-forwards/step vs old.
```

**NEXT [user]: (1) lock in — tag + clean ✓; (2) push BETTER via longer TRAINING (more steps, not deeper T) + on-policy.**

### FUSED ε-greedy Q rollout (27c797f) — the advance IS the supervised action
`_rollout_samples` rewritten [user]: one ε-greedy glimpse/step (argmax w.p. `prime_on_policy`, else random)
both advances the rollout AND is the supervised action — no discarded probe. Cost `b·(1+H)` vs the separate
scheme's `b·2H` (~2× cheaper); ~504 glimpse-forwards/s on the 4090. Mid-run signal: per-depth t1 corr 0.272→
**0.327** (more uniform across depths — fusing fixes the t1 interference), and `grid_t5_fused_op50_20k` @ step
11.7k (1.12M glimpses, 30% LESS than trajsup_10k) already rolls out 41.33/42.30/**43.17**/**43.48**/43.70
(t1–t5): beats EG-C2F-c32 everywhere AND matches/beats trajsup_10k at t3/t4. `grid_eval --ckpt-name` added to
eval `last.pt`/`step_NNNNNN.pt`.

### QUEUED experiments (auto-chained / pending)
1. **`grid_t5_fused_op50_100k`** — auto-launches when the 20k finishes (`warmup_frac=0.1`=10k warmup, `keep_every`
   20000 for a rollout-vs-glimpses scaling curve to 9.6M glimpses). Question: does rollout keep climbing past 1.6M?
2. **c64 fused-20k [user]** — `grid_train --canvas-grid 64 --train-horizon 5 --steps 20000 --prime-on-policy 0.5`
   (loads the c64 probe; rollout is the judge so no reward-map prep). Compare to the EG-C2F-**c64** trajectory
   (already measured, reproduces paper Table 4): t0 39.60 / t1 42.22 / t2 43.30 / t3 44.04 / t4 44.65. Slots after
   the 100k.

## HANDOFF — live state for cold-start resume [2026-06-15, ~end of session]

**LIVE GPU RUN (crockett): `grid_t5_fused_c64_op50_20k`** — main PID **1862205** (to re-find: `ssh crockett 'ps -C
python3 -o pid=,cmd= | grep fused_c64_op50'` — do NOT `pgrep -f <runname>`, it SELF-MATCHES your own shell and gives
false "alive" / can kill your shell). Fused ε-greedy Q (prime_on_policy=0.5), **`--canvas-grid 64`**, 20k steps,
eval_every=1000, keep_every=5000, ~2.56 steps/s, ETA ~2.2 h.
- **Completion monitor**: background task **`buhzoc5m6`** — waits on PID 1862205, then `grid_eval --canvas-grid 64`s
  step_005000/010000/015000/020000 (t0–t5) + a fresh EG-C2F-**c64** t5 baseline. Re-invokes you at completion.
- c64 net: `GridValueNet` is fixed 32→16, so the 64×64 canvas is `adaptive_avg_pool2d`'d to 32 (commit 292deb3) —
  c32 path BYTE-IDENTICAL (tagged result preserved). So the c64 POLICY sees a 32-pooled canvas; c64's gain is in the
  finer scoring/reward/deploy. EG-C2F-**c64** ref (paper-reproducing, t0–t4): **39.60/42.22/43.30/44.04/44.65**.
- (The `grid_t5_fused_op50_100k` run was KILLED — user pivoted to c64. The c32 20k PLATEAUED ~1M glimpses, so 100k
  was likely diminishing-returns. Killing it hit the pgrep self-match twice; verify kills via `nvidia-smi
  --query-compute-apps=pid` + `ps -C python3`, NOT pgrep -f.)

**HEARTBEAT (≤50-min wake guarantee [user, emphatic]): background task `byyehjujs`** (`sleep 2640`). On EVERY wake
RE-ARM a fresh 44-min heartbeat (new `sleep 2640` run_in_background) + check GPU + launch next if idle. Mechanism in
memory `canvit-operating-goals`.

### ARCHITECTURE RETRACE — possible viewpoint↔skip MISALIGNMENT [OPEN; do AFTER compaction, NEUTRALLY]
[user is SKEPTICAL of the prior-session claim ("i dont buy it") and notes I do better on low context. So RE-DERIVE
from the code FRESH — do NOT assume the conclusion below; try to FALSIFY it.]
- PRIOR CLAIM (UNVERIFIED, maybe wrong): U-Net skips anchor net-output cell (i,j) to a canvas FULL-IMAGE grid
  (i,j)∈[-1,1], but `grid_viewpoints` centers live in the scale-shrinking safe box `[-(1-s),1-s]` → viewpoint at
  (1-s)·(i,j) → mismatch factor (1-s), scale-dependent, two scales share one decoder cell.
- **PREMISE STATUS:** [user 2026-06-15 CONFIRMED] the canvas spatial grid DOES span the full image. But the user
  did NOT endorse the cell-level correspondence NOR the conclusion ("i dont trust your reasoning at 700k tokens").
  So take "canvas = full-image grid" as given, then RE-DERIVE skeptically: (a) the exact cell↔image-coord mapping of
  `get_spatial` tokens, and (b) whether full-image-canvas + safe-box-viewpoints ACTUALLY yields a misalignment that
  matters. Read the PRECISE glimpse mechanics (user emphasized): `canvas_ops.advance_state` / `candidate_canvas`; in
  `canvit_pytorch` → `sample_at_viewpoint` (crop), `seg.canvit(glimpse,state,viewpoint)` (the WRITE — where does a
  glimpse at center c LAND in the canvas grid?), `seg.canvit.get_spatial` (tokens→spatial).
- Then verify each link: (1) canvas coord frame; (2) does output (i,j) via skips (=`feats[1]` at 16×16 in
  `GridValueNet.forward`) really correspond to canvas (i,j); (3) is viewpoint (s,i,j) = (1-s)·(i,j) in the SAME frame.
  Files: `grid_net.grid_viewpoints` + `GridValueNet.forward`, `grid_train._feats` + the flat_idx convention
  (out `[B,n_scale,16,16].reshape(-1)` ↔ `vp.reshape(-1,3)`).
- IF real: candidate fix = `grid_sample` canvas feats at the per-scale viewpoint-center grid (aligned `[B,D,16,16]`)
  instead of the full-image-anchored skip decoder. Net is small (3.06M params) → cheap to A/B. IF NOT real: say so.

**20k scaling curve (settled, c32, t0–t5 mIoU%):**
| ckpt (glimpses) | t1 | t2 | t3 | t4 | t5 |
|---|---|---|---|---|---|
| EG-C2F-c32 (baseline) | 41.04 | 42.05 | 42.69 | 43.17 | 43.53 |
| step 10k (0.96M) | 41.30 | 42.30 | 43.03 | 43.26 | 43.50 |
| step ~11.7k (1.12M, **deployable peak**) | 41.33 | 42.30 | 43.17 | 43.48 | 43.70 |
| step 15k (1.44M) | 41.37 | 42.31 | 43.00 | 43.23 | 43.50 |
| step 20k (1.92M, final) | 41.23 | 42.23 | 42.71 | 43.28 | 43.59 |
