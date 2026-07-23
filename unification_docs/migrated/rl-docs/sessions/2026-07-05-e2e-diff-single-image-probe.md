# 2026-07-05 — E2E differentiability on ONE image: precision ruled out, pathwise variance measured

**Question** [user]: why can't end-to-end differentiability train a policy even on one image — and
could the earlier failures have been numerical-precision artifacts? Try continuous-viewpoint
optimization including the Gaussian reparameterization trick. Do not take old notes (the 2026-07-03
ST-Gumbel verdict, dataset-findings entry 5 roughness) at face value.

**Setup**: `throwaway/2026-07-05_e2e_probe.py` on branch `probe/e2e-diff` (commits `5508a5d`, `91ba5c7`),
run on crockett. One representative val scene (`ADE_val_00001776`, dataset idx 1775, a records/archive
room), t1 only, fixed scale 0.25, score_res 128, fp32 gradients. Artifacts (figures + tensors +
`results.json`): crockett `~/projects/CanViT-PyTorch-RL/outputs/e2e_probe/ADE_val_00001776/`, synced to
the laptop at the same relative path. The viewpoint path is differentiable end to end: grid_sample
(pixels), VPE (RFF), RoPE positions — no discretization anywhere.

## Findings (single image, single seed — mechanism probe, not a generalization claim)

1. **Precision is NOT the culprit.** Full 64×64 center landscape computed fp32 AND bf16:
   mean |fp32−bf16| = 0.0018 CE (1% of the 0.183 landscape spread), max 0.011, diff map unstructured.
   1024-point line scans: the sub-cell wiggles COINCIDE between precisions (adjacent-point mean |ΔCE|
   0.0027 fp32 vs 0.0029 bf16). The roughness in dataset-findings entry 5 is real structure, not bf16
   noise — that entry survives an fp32 re-measurement on this scene.
2. **The backprop gradient is correct but describes a microscopic neighborhood.** Gradcheck vs central
   finite differences: cosine 0.10 at h=1e-2, 0.54 at 1e-3, 0.94 at 1e-4. Mean |∇CE| ≈ 6.8 — set by
   the high-frequency ripple; the macro slope toward the basin is ~0.2–0.5, i.e. the gradient is ~30×
   ripple over signal at decision scale.
3. **Landscape geometry is benign under smoothing.** Gaussian-blurring the measured fp32 landscape
   (σ 0.05→0.4): the argmin stays inside the true basin at every σ (`blurred_landscape.png`). An
   idealized smoothed optimizer WOULD find it — so any failure is estimator-side, not geometry.
4. **Matched-budget arms (16 forwards/step × 300 steps), fixed scale 0.25**: t0 CE 0.5255;
   best-of-512 discrete 0.3518 (s=0.25 subset 0.3761); 64² landscape min 0.3615.
   Random search **0.3599** (≈ the continuous optimum; beats the 16×16 discrete grid at the same scale,
   consistent with sub-cell structure). Direct GD: final min 0.4001, median 0.5011 — particles
   random-walk in the ripple. Reparam (single μ, σ cosine 0.5→0.02, Adam 3e-2): μ migrates TOWARD the
   basin but stalls at its edge; final μ CE **0.4787**. Plain random search dominates both gradient arms.
5. **The mechanism, measured** (`variance.png`, 256 samples per (μ, σ), 3 μ × 5 σ, both estimators of
   ∇_μ E[CE(μ+σε)] at identical samples): per-sample gradient SNR (‖E g‖ / mean per-coord std) is
   **0.3–0.9 for score-function** (REINFORCE, in-batch mean baseline) vs **0.01–0.15 for pathwise** —
   5–40× worse, i.e. 25–1600× more samples for equal gradient quality. Pathwise std is 12–20 (set by
   the rough pointwise |∇CE|, and it GROWS with σ); score-function std is 0.08–0.5 (set by the CE value
   spread / σ). At σ=0.2 in the basin, 256 pathwise samples cannot even resolve the smoothed gradient's
   direction (‖E g‖ 0.16 vs SE ≈ 0.9). This is the standard variance dichotomy: pathwise variance scales
   with E‖∇f‖² of the integrand, score-function with Var f — and this integrand is rough in derivative
   but tame in value.

## Verdict

E2E differentiability fails here for a measured, intrinsic reason: the CE(viewpoint) landscape has
high-frequency real structure whose pointwise gradients are ~30× the useful macro slope, so ANY
pathwise estimator (plain GD, reparam at any σ tried) drowns; smoothing fixes the objective's geometry
but not the estimator's variance. Score-function credit (REINFORCE) and the deployed measured-reward
K=1 regression sidestep it by using CE VALUES, not CE derivatives. Confirms the 2026-07-03 conclusion
via an independent route (continuous coordinates, fp32, no straight-through), refutes the precision
hypothesis quantitatively, and adds the estimator-variance mechanism the earlier probe only gestured at.

Caveats: one scene, one seed, one σ-schedule/lr for the reparam arm; sample clamping at the box edge
slightly biases the σ=0.4 variance rows; t1 only.

Figures (all under `outputs/e2e_probe/ADE_val_00001776/`): `scene.png` (image + landscape overlay +
picked glimpses as real boxes), `landscape.png` (fp32/bf16/diff), `linescan.png`, `gradcheck.png`,
`opt.png` (curves + trajectories), `blurred_landscape.png`, `variance.png`.

---

# Afternoon extension: the two-scene estimator race

**Question** [user]: apples-to-apples, which learning signal (production reward-map regression vs
REINFORCE vs GRPO vs SAC-style vs pathwise vs flows) learns good t1 viewpoints fastest — and can any
actor-based method match the production recipe? Instrument: `throwaway/2026-07-05_estimator_race.py`
(final rev on `probe/e2e-diff`). Two scenes picked for anti-correlated landscapes (rank 0
`ADE_val_00001776`, rank 4 `ADE_val_00001905`, ρ=−0.17); t1; scale fixed 0.25 (flow3d frees it);
tabular per-scene params (isolates the estimator); every method k=8 true-CE forwards/scene/step,
1200 steps; deploy CE evaluated off-budget every 25 steps. ~330 forwards/s ≈ 60 s/arm (GPU pegged,
MFU ≈ 0.4–0.5; the canvas cross-attention dominates per-glimpse FLOPs). Artifacts:
`outputs/e2e_race*/` on crockett + laptop (race.png curves, diag_*.png, flow_density_*.png, race.pt).

## Race findings (seed 0 unless noted; hot-LR run: lr 3e-2/5e-2/1e-2, no warmup, no clip)

- **qreg (production miniature) walks to the exact 16×16 grid optimum on both scenes.** So do
  reinforce_grid and grpo_grid once the policy is grid-parameterized [user's fix for the exploration
  trap] — cell-identical argmax, ~same speed. **Actor-based matches production** on this bandit.
- **grpo ≡ reinforce bit-identically under Adam** (group-std is a per-step gradient rescale that Adam
  absorbs). Structurally GRPO needs k≥2 samples/scene — it cannot operate at the deployed K=1; the
  frac-CE reward + global per-depth z IS the K=1-compatible baseline [user's design point confirmed].
- **Unimodal Gaussian actors (reinforce/grpo/sac) get trapped** when the basin is far from init
  (scene 0): mode-seeking + σ-collapse. sac (reparam through a learned interpolated critic) partially
  escapes: the critic aggregates global information.
- **Pathwise-through-world is last at every budget** (and 4× wallclock/sample): 18.8% captured vs
  qreg 83.6%; flow_pathwise −12%. Confirms the morning's SNR verdict at learning level.
- **Random search wins the tabular game outright** (~101% of the 64² oracle, continuous sampling) —
  in a per-scene-table world, search ≥ learning; learning earns its keep only via cross-scene
  generalization (the next rung: shared net over t0 state features, held-out scenes).
- **flow (2-D RealNVP actor, score-function) is the only learner beating the grid ceiling**
  (0.3732 < 0.3761 on scene 0, sub-cell). flow3d (free scale ∈ [0.05,1]) found the day's best action
  on scene 0 (0.3505, beats even the 2-scale discrete-512 best 0.3518) but diverged on scene 1.

## Instability forensics (flow3d scene 1) — measured causes, not vibes

1. First blowup (no clip, lr 1e-2): sample cloud saturated into the corner at scale→1 (density plot),
   deploy CE 0.6883 > t0. **Grad-clip 1.0 (house rule, initially omitted) stopped the divergence**;
   entropy-estimator ablation (exact-logdet vs none vs biased-detached) then showed near-identical
   finals ≈0.61 — the runaway was clip-fixable, but all variants stayed mode-stuck.
2. **Raw-space logdet entropy is wrong under a squash**: |raw| exploded to ~550 while the "entropy"
   bonus climbed — raw entropy is unbounded and the tanh hides it, so the bonus PAYS for saturation
   (sample cloud degenerated to a filament). Fix: **action-space entropy = flow logdet + squash
   jacobian** (SAC's log(1−tanh²) correction, generalized to the scale sigmoid) — the jacobian's
   log→−∞ at saturation is the restoring force.
3. **lr 1e-2 was the NaN cliff** [user called it]: with action entropy the run was HEALTHY (grad<1,
   |raw|↓, scene-1 deploy ~0.48, best of any flow) then went non-finite ~step 130. At lr 1e-3 +
   warmup-hold + clip: **no NaN, flow 0.4922 / flow3d 0.4958 on scene 1 ≈ the grid best 0.4921** —
   the first continuous actors to crack scene 1.
4. CPU harness (`scratchpad/flow_cpu_test.py`, seconds/iteration — [user: check flows on CPU first])
   verified: inverse consistency exact; scale conditioning learns exactly; **mode collapse is
   intrinsic** (equal-bump synthetic → 100% mass on one mode) and **mode CHOICE is ~coin-flip**
   (unequal bumps: 2–3/5 commits to the better mode regardless of entropy schedule 0.01→0.3-anneal).
   Global-comparison methods (value table / grid softmax) win multimodal landscapes because they score
   all modes in parallel; continuous densities hill-climb locally and cannot compare modes they no
   longer sample.

## Process lessons (cost real time today)

- Artifacts must save per-arm/incrementally — the first race died with zero artifacts (fixed same day).
- A completion watcher keyed only on the success file is blind to crashes — always include failure
  signatures (a tyro CLI error and a NaN run each sat unnoticed behind such watchers).
- CPU-first for anything model-free (flow math, estimator logic): seconds per iteration vs a GPU arm.
- Grad clip 1.0 and warmup are house rules for a reason; "throwaway" does not exempt training loops.

---

# Afternoon II: pg promoted to a first-class method; codebase restructure; HANDOFF

## What landed (all on `probe/e2e-diff`, merged to main at end of session)

- **`flow/` module**: conditional RealNVP + `tanh_box` squash with exact action-space entropy,
  distilled from the day's probes; docstring carries usage + measured footguns; 8 CPU tests
  (inverse/logdet/squash-jac vs autograd, MLE bimodal fit, conditional mode steering, hot-LR
  REINFORCE stability regression) saving density heatmaps to `outputs/flow_tests/`. NOT wired into
  any production run — standalone deliverable [user request].
- **pg training method** (user goal: directly optimize reward, actor-based, toward eventual
  perception unfreezing): the SAME ViewpointQNet, candidate readout read as categorical logits,
  ON-POLICY softmax rollout sampling, loss -(z·logπ) − entropy_bonus·H. The per-depth global-z
  frac-CE reward machinery is UNCHANGED and doubles as the REINFORCE advantage (E[z]≈0/depth = the
  K=1-compatible baseline; GRPO needs k≥2/scene and its group-std does nothing under Adam — race
  evidence). Deploy stays argmax → ckpts/eval/publish/seed-band tooling all shared.
- **`credit="return"` variant** (committed, not yet run): action credit = Σ_{t≥s}(CE_t0−CE_t)/CE_t0,
  whose objective IS mean CE over timesteps = the pretraining objective = the judging metric. The
  frozen restriction of the full unfreezing objective: dJ/dθ = score-function term + direct dCE/dθ
  term, coefficients fixed by calculus, NO loss weights [user: no weighted multi-term losses]. On
  unfreezing, switch the z to subtract-only (division rescales policy vs perception components on
  shared params) — rule recorded in the config comment.
- **Restructure** [user: "actor under q makes no sense; optuna isn't q; net isn't q-specific"]:
  `q/` → `policy/` (net, features, deploy=GreedyQPolicy, eval CLI) + `training/` (config, rollout,
  train, eval_loops, stats, viz, reward_maps) + `search.py` (Optuna). `QConfig`→`TrainConfig`;
  method is a sum type `objective: QReg | PG` (QReg: prime_on_policy, dueling; PG: entropy_bonus,
  credit) — pg+dueling and qreg+entropy unrepresentable; `cfg.dueling` is a derived property.
  Entry points now `canvit_pytorch_rl.training.train` / `.policy.eval` / `.search` (README+justfile
  updated; docs/sessions left as history). **Hub verified post-refactor**: the flagship
  `canvit/qpolicy-ade20k-c64-t5-qband-2026-07-04-s2` loads via `policy.net` and forwards finite
  (5,670,800 params, [2,2,16,16]) — config.json/safetensors are path-agnostic, no repush needed.

## Live run state at handoff [2026-07-05 ~13:45 crockett time]

- **pg 3-seed band running on crockett**: pg_s0 (started ~13:08) then pg_s1, pg_s2 chained, full
  recipe budget (640k forwards, 8k steps, lr 2e-4, entropy_bonus 0.01, credit=immediate, dueling
  off), logs /tmp/pg_s{0,1,2}.log, runs/<stamp>_pg_s*. **Launched at rev `b7936f8` via the OLD
  module path `canvit_pytorch_rl.q.train`** — crockett's checkout is DELIBERATELY held at b7936f8:
  resetting to main before pg_s2 launches would crash the chain (q/ no longer exists on main).
  **After pg_s2 finishes: `git fetch origin && git checkout main && git reset --hard origin/main`
  on the crockett checkout; all future launches use `--objective:pg` under training.train.**
- **pg_s0 trajectory vs qband_s0** (objective = mean t1–t4 val CE): 0.7180 (init, identical) →
  0.6990 @1k (qband 0.6911) → 0.6932 @2k (qband 0.6890) — gap halved, policy_entropy stable ~2.6
  (no collapse), val mIoU improving every t. Verdict criteria: within seed noise of qband
  (0.6855±0.0004, 8 seeds) by 8k = actor framework validated at recipe scale.
- MLflow: server on crockett :5500; laptop tunnel `ssh -f -N -L 5500:localhost:5500 crockett`
  (verified). Experiment canvit-q; pg runs carry kind=pg.

## Open threads, in priority order

1. Read the pg band when it lands (tools.seed_report works on pg runs — shared ckpt format).
2. Launch `credit=return` (the mean-CE objective) — the run that can in principle BEAT qreg, since
   qreg's myopic per-step target is a proxy for the trajectory objective.
3. If on-policy coverage is the residual gap: bounded-ratio mixture behavior (½π+½uniform, IS
   ratio ≤ 2) — unbiased same-objective PG with qreg-style coverage. AWR (weighted MLE on random
   samples, no ratios) is the fallback family [discussed, not built].
4. Unfreezing ladder: probe first (add the direct CE term = calculus, not a weighted loss), then
   backbone at tiny LR. Prerequisite: switch advantage z to subtract-only.
5. Flow policies: reliable at cold LR on the single-scene probes (flow3d 3 seeds, scene-1
   0.443–0.463 vs grid best 0.492); mode CHOICE remains exploration-determined — grid/categorical
   for global comparison + flow for sub-cell/scale refinement is the architecture if ever needed.

## Wasted-time ledger (same-day guards)

- First estimator race died with zero artifacts (end-only saves) → per-arm incremental save+replot.
- Two watcher blind spots (success-only patterns) sat on a tyro CLI error and a NaN run → watchers
  now match failure signatures; memory: always read partial logs.
- Two accidental concurrent GPU launches (23.2/24 GB) → check `pgrep` BEFORE every launch.
- pkill -f over ssh killed its own shell (hard-rule violation, no harm done) → kill by explicit PID.
- flow3d NaN cliff at lr 1e-2 burned ~3 GPU arms before the CPU harness (seconds/iter) localized
  everything → CPU-first for model-free logic, always.

## Afternoon III — pg_s0 verdict, entropy collapse, first pg publication, pg sweep [landed same day]

- **pg_s0 finished (640k forwards, 8k steps, run `20260705-170816_pg_s0`, rev `b7936f8`): best
  objective 0.6869 @5k** vs qband 0.6853±0.0007 — 0.0016 above the band mean (~2σ of the band's
  seed spread) on the FIRST untuned actor run. Deploy ckpt = the 5k step. Per-t at best: CE
  0.7157/0.6886/0.6764/0.6669, mIoU_t4 44.76. User killed pg_s1/pg_s2 (chain parent PID killed
  before s1 launched) in favor of HP search; crockett checkout reset to origin/main (`948a834`)
  after the run exited.
- **Late instability diagnosed (measured, cause not confirmed)**: val objective bottoms @5k then
  degrades (0.6898 @6k, 0.6909 @7k, 0.6925 @8k, ~10x band σ) while policy entropy collapses
  (median 3.4 → 0.84 by 5–6k; 0.20 at 8k) and pre-clip grad norm grows (median 3.3 → 8.8, spikes
  to 115; taken_logp reached −25.6). Late windows: corr(grad_norm, taken_logp) = −0.71 — spikes
  arrive when the sampler draws actions the peaked policy assigns ~e^{-25} probability, the
  1/π(a) score-function variance signature. Hypotheses (H1 entropy_bonus=0.01 too weak → variance
  blow-up; H2 constant-hold lr too hot for the peaked regime; H3 on-policy state narrowing) are
  disentangled by the sweep below. Figure: `outputs/pg_s0_instability.png` (laptop). Deploy
  selection sidestepped the slide (best.pt @5k, pre-collapse).
- **First pg policy published**: https://huggingface.co/canvit/pgpolicy-ade20k-c64-t5-2026-07-05-s0
  (best.pt @5k + init + manifest + metrics; method-aware card). Verified from-Hub on the laptop:
  5,539,343 params (qband arch minus the 131,457-param dueling V-head), finite forward. Publishing
  machinery was BROKEN post-refactor — `replace(cfg, dueling=...)` with dueling now a derived
  property crashed all three ckpt loaders (policy.eval, tools.publish_policy,
  tools.trajectory_probe); fixed via `policy.net.objective_from_ckpt` (`d9328e6`), hub round-trip
  test now covers both arches (`948a834`).
- **pg HP sweep running on crockett**: study `pg_c64_t5_ce_f400000` (fresh; sqlite
  runs/q_optuna.db), supervisor `tools.perpetual_sweep` (log /tmp/sweep_sup.log), 400k-forward
  trials, `--search lr weight_decay entropy_bonus credit`, seed trials = defaults, credit=return,
  entropy_bonus=0.03, and 0.03+return (`518b272`). trial0000 launched 14:26 crockett time.
- **Entropy floor landed** (`8d34cdf`, user-directed: "targeting a minimum entropy of 1"):
  `PG.entropy_target` = SAC-style dual ascent on alpha (log-space integrator, alpha in [1e-4, 10] —
  a tight cap binds and defeats the floor, measured in test_training's toy; fixed-bonus default
  unchanged so the sweep study's semantics hold). **pgtarget_s0 running on crockett** (launched
  14:34 crockett time, rev `8d34cdf`, full 640k budget, seed 0, entropy_target=1.0, log
  /tmp/pgtarget_s0.log): the A/B vs pg_s0 (fixed 0.01) — if H holds ~1 and val improves past 5k,
  the floor rescues the late steps the collapse cost. Sweep trial0000 (pg-defaults, pg_s0's config
  at 400k) was killed for it; supervisor relaunched, GPU-gated behind the run, resumes trial0001
  after. Re-enqueue the defaults seed manually if its short-budget anchor is ever wanted.
- **pgfloor_s0 finished (rev `4a24a3d`, run `runs/20260705-*_pgfloor_s0` on crockett): best objective
  0.6857 @7k** — vs pg_s0's 0.6869 @5k and the qband's 0.6853±0.0007: the actor lands WITHIN ~1σ of
  the qreg band mean, on one seed. Evals 0.6983/0.6895/0.6888/0.6908/0.6892/0.6877/0.6857/0.6883
  (@1k..8k). **The floor engaged and cycled** (windowed data — 1k-grid eval snapshots hid this, and an
  earlier note here wrongly said "barely engaged" off them): 206/2668 windows below H=1.0 spanning
  steps 2600–7993, min H 0.639 @6436 with alpha risen to 0.038 there; each dip drove alpha up and H
  recovered, hovering ~1.17 with alpha ~0.010 post-7k. No terminal collapse (pg_s0 ended at H 0.20).
  Caveat that stands: pg_s0 vs pgfloor_s0 is a CROSS-CODE comparison (b7936f8 vs 4a24a3d; the refactor
  plausibly shifted RNG consumption), so floor-causes-the-gain is plausible, not confirmed — a
  same-code fixed-vs-floor A/B (or the band) settles it. What stands regardless: an actor at recipe
  scale matches the qreg band on one seed. **Floor recipe made the PG default** [user 2026-07-05:
  "the new recipe seems stabler so should be defaults"].
- Ops: the sweep supervisor relaunched from inside a watcher's ssh died with that session → GPU sat
  idle ~2 min after pgfloor_s0 finished (caught by the hourly loop). Relaunched via `ssh -f` from a
  dedicated call; trial0001 (credit=return seed) taking the GPU. Guard: never launch a daemon from
  inside a compound watcher ssh — dedicated `ssh -f` only.
- **credit=return first read (old-study trial0001, fixed bonus 0.01, 400k forwards): 0.6907** —
  behind immediate credit's comparable-budget anchor (pg_s0 0.6879 by 320k). Entropy ran much
  higher (5.02 @1k vs ~2.5 immediate): the future-sum credit is noisier per sample, slowing
  commitment. One seed; the floor-default study re-tests return credit as a seed trial.
- Old sweep stack retired at the trial0001 boundary (killed by PID); supervisor relaunched on study
  `pgfloor_c64_t5_ce_f400000` (floor defaults; seeds: defaults / credit=return / target1.5 /
  target1.5+return) — trial0000 took the GPU 16:59 crockett time.
- **Sampler-mode test ran 2026-07-06** (phase_sampler, written 2026-07-05 but never executed; artifacts
  `outputs/e2e_probe/ADE_val_00001776/sampler.{pt,png}`, laptop+crockett): at scale 0.25 (1:1 sampling)
  **bicubic grid_sample cuts the pointwise |dCE/d(center)| ~8x (mean 9.29 -> 1.16, median 5.14 -> 0.99)
  with the CE VALUES unchanged** (scan mean 0.4790 vs 0.4800; value-ripple identical — the pathology was
  bilinear's piecewise-constant DERIVATIVE, the classic STN gradient problem [user's hypothesis]).
  Against the 0.2–0.5 macro slope, ripple-to-signal drops from ~30:1 to ~2–5:1 — the morning-of-07-05
  "pathwise is dead" verdict was largely a SAMPLER artifact at this scale, not intrinsic to the network.
  At scale 0.5 (2x decimation) bicubic helps the median (5.50 -> 1.55) but not the mean (4.34): the
  heavy tail is real aliasing; AA+bicubic did NOT combine cleanly BUT the grad stats there are over only
  4 probe points (n cut for GPU-sharing) — the 0.5 interactions are unresolved, don't conclude on them.
  Follow-up that would settle usefulness: re-run phase_variance (pathwise SNR) under bicubic — if SNR
  rises ~8x it lands in score-function's 0.3–0.9 range and pathwise re-enters the design space (e.g.
  for the eventual unfreezing, or hybrid estimators). Not run; GPU owned by the pg sweep.
- **Bicubic full-path re-run (2026-07-06, `outputs/e2e_probe_bicubic/ADE_val_00001776/`, monkeypatched
  grid_sample, particles 8 vs bilinear's 16): REPARAM NOW FINDS THE BASIN — final mu CE 0.390 vs
  bilinear's 0.479 stall** (random-search control unchanged 0.359 vs 0.360; GD best particle
  0.400 -> 0.377, median still ~0.48 = local minima, not noise). The wide-sigma SNR metric did NOT
  move (path 0.02–0.15) — resolved: at sigma 0.1–0.2 the sample cloud spans ±50–100 px and path_std
  (16.1 -> 10.7 at basin_edge) is dominated by GENUINE gradient-field heterogeneity across the
  neighborhood, which step-averaging handles; bilinear's per-point interpolation ripple, which
  step-averaging could NOT fix, is what bicubic removed. So "pathwise is dead" was substantially a
  sampler artifact at 1:1 scale; the E2E-differentiability door reopens (bicubic costs one mode arg;
  scale-0.5 aliasing tail still unresolved). Two-scene race with bicubic pathwise arm: running.
- **Bicubic race (2026-07-06, ranks 0+1 — NOT yesterday's 0+4 pair; `outputs/e2e_race_bicubic/`):
  captured-% of oracle gain: random 101, qreg 89, pathwise-bicubic 26, gaussian-reinforce 21.
  The ESTIMATOR gap is closed** (bilinear pathwise was strictly last; now at parity with the
  same-parameterization score-function arm) — the remaining gap to qreg is the unimodal Gaussian's
  mode-seeking, the SAME failure the grid parameterization fixed for score-function yesterday, and
  pathwise costs 4x wallclock/sample (backward through the backbone). Overnight verdict for "make
  grid_sample work": bicubic makes pathwise gradients USABLE (single-scene reparam 0.479 -> 0.390;
  race parity); to make them WIN they need a multi-modal/global parameterization (grid-of-Gaussians,
  flow, or pathwise as a refinement/perception term on top of categorical placement — the unfreezing
  use case, where the policy stays score-function and only perception needs d(CE)/d(params)).
- **Sub-cell pathwise refinement works (2026-07-06, `throwaway/2026-07-06_subcell_refine.py`, bicubic,
  30 GD steps from the 16x16 grid argmax)**: scene 1776 0.3740 -> 0.3706, scene 432 0.6495 -> 0.6450 —
  centers moved ~half a cell half-width; ~0.003–0.005 CE below the discrete ceiling at 30 fwd+bwd
  (random search needs ~1000s of forwards for the same neighborhood). Captures a fraction of the
  ~0.016 measured sub-cell headroom; GD overshoots past its best. NOTE [user correction 2026-07-06]: this needs the MASK
  (true CE), so it is TRAIN-TIME machinery (refined-action distillation targets) or needs a learned
  CE-predictor to refine against at deploy — never raw deploy-time GD. THE OVERNIGHT ANSWER to "make grid_sample work": bicubic + categorical-place-then-pathwise-
  polish is a working hybrid at the mechanism level; next rungs are t>1 states, scale refinement, and
  whether a shared refinement head amortizes the GD steps.
- **Amortized pathwise training on real data: NEGATIVE at 1k steps, two head designs**
  (`throwaway/2026-07-06_pathwise_train.py`, actual ViewpointQNet trunk, real ADE20K, bicubic,
  t1/scale-0.25, batch 8, lr 2e-4, 256-image diagnostic val slice — NOT the protocol eval).
  v1 soft-argmax + tau-anneal: fails by construction (1/tau grad blowup, measured 11 -> 165; trains the
  averaged center, deploys argmax); val t1 CE 0.6886 -> 0.7020. v2 fixed-tau mean + annealed reparam
  jitter (the single-scene WINNING recipe): grads stable (6–33) but no learning — 0.6909 -> 0.6991 ->
  0.6940, never below init. The single-scene pathwise success does NOT transfer to the amortized
  setting at this budget: per-scene persistent optimization could average the landscape heterogeneity
  over steps; an amortized net gets ONE gradient evaluation per new scene (K=1 pathwise), which the
  post-bicubic ~2–5:1 residual heterogeneity still swamps. NOT a "can't work": untried — lower lr /
  longer horizon, direct-regression head, per-image multi-start, and the hybrid that already works
  (score-function placement + pathwise sub-cell refinement, measured 2026-07-06) which sidesteps
  amortized-placement-by-pathwise entirely. Score-function credit remains the trainer for placement.
- **Why AVA's end-to-end RL worked and ours doesn't (read `~/code/ava/research/log/`
  `2026-05-01-rl-archeology-session-log.md` — the codex RL-archeology report [user pointer]):**
  AVA's working pathwise/BPTT recipe (Gaussian search 2025-07-15, ModMNIST 2025-07-27) differs from
  our setting on every axis that decides pathwise variance: (1) LOSS — dense smooth distance-to-target
  (near-quadratic in the action) vs our frozen-probe CE with measured high-frequency response;
  (2) PERCEPTION — a small TRAINED CNN (3 layers, CReLU/LayerNorm, spatial-preserving) that co-adapts
  to produce useful gradients vs our frozen 12-block CanViT; (3) APERTURE — their successes used
  patch=image (ModMNIST 40/40) or patch 7-8/16; their patch-2 run FAILED (0.677 vs 0.17) — our
  scale-0.25 glimpse on complex scenes IS the small-aperture regime; (4) their REINFORCE lost and was
  removed — the exact inverse of our ranking, explained by the variance dichotomy: pathwise var ~
  E||grad f||^2 (tiny for distance, huge for probe-CE), score-function var ~ Var f. Their recipe
  details we already rediscovered independently: tanh-squash + rsample, small fixed std, tiny head
  init, warmup+clip, delta/retinotopic actions with range [-2,2]. Today's direct-head coda matches:
  default-init deterministic tanh head saturated and died (center_spread -> 0, all grads 0) — their
  "init tiny + sampling noise" rules exist to prevent exactly this. The fixed point/const controls
  remain to run under a tiny-init sampled head before any further CE attempts.
- **Matched-jitter estimator showdown (2026-07-06, actual arch, real data, direct head tiny-init,
  same mu/sigma=0.1/samples, bicubic; 1k steps, 256-img diagnostic slice):**
  scale 0.25: pathwise 0.6926->0.7039, reinforce 0.6926->0.7012 — BOTH degrade (a unimodal Gaussian
  head is the wrong policy class for rough multimodal placement, independent of estimator).
  scale 0.75: pathwise 0.6703->0.6674 (converged by 250), reinforce 0.6703->0.6656 (improving to 750).
  ANSWER [user question]: matched-jitter pathwise + bicubic does NOT outperform reinforce — near-tied
  at large aperture (delta 0.0018, single seed, sub-noise), pathwise's one edge is speed (3x fewer
  steps to plateau). Aperture story confirmed at TRAINING level: both estimators work at 0.75, both
  fail at 0.25 with this head. Sanity ladder passed (const fits, bright optimizes through the real
  glimpse path) — no plumbing bugs; failures are policy-class/landscape, not code.
- Sweep retired [user: "most meaningful things instead of endless sweep"]; study preserved in
  runs/q_optuna.db (15 complete). **pgconf_t12_s0 launched** (trial0012 config at 640k: lr 1.77e-4,
  wd 3.96e-3, floor 0.5; /tmp/pgconf_t12_s0.log) — the full-budget confirmation. SAC/DDPG arm next.
- **trial0012 does NOT confirm at full budget** (`pgconf_t12_s0`, 640k, lr 1.77e-4/wd 3.96e-3/floor
  0.5: best 0.6877 @7k vs 0.6862 at 400k; floor-1.0 pgfloor_s0 = 0.6857) — the low-floor sweep win was
  the predicted short-budget artifact. RECOMMENDED (pending user): revert PG entropy_target default
  0.5 -> 1.0. GPU idled ~1h45 after pgconf ended (no completion watcher) — guard: every long run gets
  a completion watcher at launch. **pgfloor_s1/s2 band running** (floor 1.0 explicit, seeds 1-2
  chained; with pgfloor_s0 -> 3-seed floor-1.0 band; logs /tmp/pgfloor_s{1,2}.log).
- **3-SEED ACTOR BAND MATCHES THE QREG BAND** (pgfloor_s0/s1/s2, floor 1.0, 640k each; s1/s2 at
  current main, s0 at 4a24a3d): best-mean objective 0.6857/0.6859/0.6853 (mean 0.6856±0.0003) vs
  qband 0.6853±0.0007 — per-t CE within qband sigma at EVERY timestep (0.7145/0.6882/0.6748/0.6650
  vs 0.7143/0.6878/0.6741/0.6652); mIoU matched except t3 (44.45±0.10 vs 44.62±0.12). The
  score-function actor at the recipe budget is band-equivalent to Q-regression. Floor comparison at
  full budget: floor-1.0 band 0.6856±0.0003 vs floor-0.5 single seed 0.6877 — REVERT entropy_target
  default to 1.0 recommended (0.5 was the short-budget sweep artifact). pgfloor_s3/s4 extending the
  band; SAC/DDPG arm next.
- 2026-07-07: pgfloor_s3 0.6861, pgfloor_s4 0.6888 (the weak seed) — 5-seed band 0.6864±0.0014 vs
  qband 0.6853±0.0007: overlapping, actor mean ~0.001 behind, s4 widens the spread. Seeds 5-7
  launched (8-seed n-matched band lands today). OPS FAILURE: GPU idle ~00:30-10:45 — the s3/s4 chain
  ended with no queued occupant and the overnight cron queued unfired; guard: never end a chain
  without the NEXT occupant already staged behind it.
- **8-SEED ACTOR BAND COMPLETE (2026-07-07)**: pgfloor_s0..7 (floor 1.0, 640k) best objectives
  0.6857/0.6859/0.6853/0.6861/0.6888/0.6867/0.6874/0.6859 -> **0.6865±0.0012 vs qband 0.6853±0.0007**.
  Per-t CE 0.7153/0.6889/0.6757/0.6661 vs qband 0.7143/0.6878/0.6741/0.6652; mIoU 44.83±0.15 vs
  44.97±0.10 at t4. Honest n=8 verdict: the actor band OVERLAPS the qreg band but sits ~0.0012 behind
  with ~2x the spread — "matches within noise" held only for the first 3 lucky seeds; at n=8 qreg is
  slightly but consistently ahead. **DPG arms: negative** — dpg@0.25 degrades then freezes (0.7031
  flat from 250); dpg@0.75 0.6703->0.6745 (worse than both pathwise 0.6674 and reinforce 0.6656):
  the critic surrogate as actor-gradient source underperforms both direct estimators at this scale
  of training; single configs, untuned. The estimator program is now closed at parity: score-function
  categorical (qreg/pg) remains the placement trainer of record.
- All 8 pgfloor band seeds published [user go]: canvit/pgpolicy-ade20k-c64-t5-floorband-2026-07-07-s{0..7}
  (deploy ckpt + init + manifest + metrics each; band provenance on the cards). Band-vs-qband figure:
  outputs/qband_vs_pgfloorband.png (crockett + laptop). pgsubz ablation: s0 done (subtract-only z),
  s1 running; AA 12-probe sampler measure in headroom.
- **pgsubz_s0 final: best 0.6865** — subtract-only z lands EXACTLY on the pgfloor band mean
  (0.6865±0.0012): the std division is NOT load-bearing; the unfreezing-side credit prerequisite is
  cleared on one seed (s1 running for the second). Dead OOM'd s1 dir pruned; live s1 at ~15.6GB.
- **pgsubz_s1: 0.6860** — subtract-only z on two seeds (0.6865, 0.6860) sits inside the pgfloor band
  (0.6865±0.0012): the std division is confirmed non-load-bearing; unfreezing credit prerequisite
  CLEARED. Seeds 2-3 chained for a 4-seed ablation band.
- **AA verdict (12-probe sampler, scale 0.5, `outputs/e2e_probe_aa12/`)**: pointwise grad
  mean/median — bilinear 6.3/6.9, bicubic 5.2/2.4, aa+bilinear 4.3/2.7, **aa+bicubic 3.1/1.8 (best
  on both)**: the blur+bicubic combo roughly halves the mean and quarters the median vs bilinear —
  a PARTIAL fix (residual tail; scale-0.25 bicubic reaches 1.2) but the reliable ordering multi-scale
  pathwise would need.
- **2026-07-07/08 — Q-Prop + the great rename.** PG.qprop landed (exact-expectation discrete control
  variate: second scorer as critic on the z target; score-function on the residual + exact grad of
  E_pi[Q]; unbiased for any critic). Smoke: 0.6997 @200 steps (band needed ~1k). pgqprop_s0 (640k,
  seed 0): 0.6967/0.6884/0.6887 @1k/2k/3k — ahead of the band at 2k, in-band at 3k; resid_std
  fluctuates 0.67-0.94 (variate intermittently cancelling); verdict at run end, s1 chained (will log
  the new CRITIC Q score-maps panel). REFACTOR [user: "aggressively"]: ViewpointQNet->ViewpointScorer,
  GreedyQPolicy->ArgmaxPolicy, build_qnet->build_scorer, QEvalConfig->PolicyEvalConfig,
  evaluate_q->argmax_stats, value_maps->score_maps artifact, manifest kind policy_eval — merged no-ff
  `639ae41`; Hub loading verified post-rename; all 19 Hub cards re-uploaded (renamed API + fixed
  per-timestep table header); publish template updated (entropy-floor blurb). mlflow `canvit-q`
  renamed IN PLACE to `canvit-pytorch-rl` (id 5, 410 runs; the early-era 112-run experiment that held
  the name is now canvit-pytorch-rl-early-era-to-2026-06). Four concluded throwaways deleted. README:
  pg band in Results; both sweeps' retirement recorded.
- **pgqprop_s0 final: best 0.6861** — inside the pgfloor band (0.6865±0.0012), NOT above it: Q-Prop's
  early speed (0.6884 @2k vs band ~0.690) did not convert to a better endpoint on seed 0; verdict so
  far = band-equivalent with faster convergence. **WASTED-GPU incident (my bug)**: the CRITIC-Q viz
  panel added 07-08 never widened the subplot grid -> pgqprop_s1 crashed at its step-0 eval
  (IndexError, viz.py) and the GPU idled ~22:35-10:05 (~11.5 h). Root cause: deployed a viz change
  no test or smoke exercised (the qprop smoke predated the panel). Guard: any change touching the
  eval/viz path gets a 1-step --eval-every-1 smoke before a chained launch rides on it. Fixed
  (grid widened, arithmetic asserted), crashed run dir pruned, s1 relaunched with a watcher on the
  crash site.
- **Two-seed Q-Prop verdict (2026-07-08): band-equivalent endpoint, faster convergence.** pgqprop_s0
  best 0.6861, pgqprop_s1 best 0.6863 (@8k) — both inside the pgfloor band (0.6865±0.0012), neither
  above it; the early-speed edge (0.688 territory by 2k vs the band's ~4k) is consistent across both
  seeds. Q-Prop = a convergence-speed tool at matched budget, not an endpoint improvement — coherent
  with the recipe plateau being objective-level. pgqprop_s2 chaining (first run with the persisted
  critic_state; ckpt_meta hardened for the new key).

---
Log closed 2026-07-08 (it had grown far past its title: the estimator program, the actor bands, the
floor, Q-Prop, and the rename all landed here). Continues in `2026-07-08-qprop-and-hardening.md`.
