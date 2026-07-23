# 2026-06-16 (afternoon) — repo audit/cleanup, then single-batch overfit + aux-feature/reward-map exploration

Continues `2026-06-16-twinq-default.md`. Two phases today: (1) a full repo audit + cleanup (all committed,
pushed, `just`-green); (2) an exploration thrust on single-batch overfitting, reward-map analysis, and the
design for cleanly adding auxiliary signals + recurrence to the REAL `q/` pipeline. **Exploration findings
below are PRELIMINARY (single noisy 24-scene batch); not verdicts.**

## Phase 1 — repo audit + cleanup (DONE, pushed, green; commits `7b372c2`..`2b9efae`)
Audited every source file. Fixes landed:
- **Stale "BatchNorm" comments** in `eval_mode`/`q.eval`/`policy` — the arch is LayerNorm2d-only (no BN),
  `dropout=0`, `ConvNeXtBlock drop_path=0` (verified max|train−eval|=0) ⇒ `eval_mode` is a **no-op** today;
  comments now say so (kept as the dropout>0 guard).
- **`baselines/figure4b` crashed** on the real mixed `runs/` (no `summary.json` on training dirs, no
  `policy` key on q.eval dirs) → guarded.
- **`grad_norms`** now groups per-submodule by param name, stripping a leading EnsembleQNet `critics.N.`
  (twin runs were collapsing to one `grad_norm_critics`); `training_curves` discovers `grad_norm*` from the log.
- **`harness.base_manifest`** unifies TrainLogger + write_run manifest (fixes train="64"/eval=64 config-repr
  drift); `test_harness.py` pins it.
- **`evaluate()`** dedup: `evaluate_q(selection=False)` on val skips the t1 advance/scoring (rollout already
  has t0/t1); removed duplicate `val_miou_t1_mode` key.
- `sweep_report` horizon-derived-from-data; CLAUDE.md Table-4 → pointer to `paper_reference.py`; `bench`
  `resolved_probe_repo`; **OOM-resilient optuna** (catch `torch.OutOfMemoryError` → prune, not crash-loop) +
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in `perpetual_sweep`.

**trial0** (twin-Q 1M, OLD `train_horizon=5` regime) FINAL: t0–t4 `39.60/42.39/43.66/44.24/44.69`,
ce_t4 `0.6698`, best==final (no overfit). Beats EG-C2F-c64 t1–t3, ties t4; trails the single-critic 20k
anchor (−0.2…−0.34). Superseded by the horizon fix + the exploration.

**NOTE:** the horizon-fixed sweep (`t5_c64_ce_twinq_hfix`, `train_horizon=4`) was NEVER relaunched — the GPU
went to the exploration. It's staged in `throwaway/perpetual_sweep.py` if we want the train_horizon=4 twin
HP baseline.

## Phase 2 — exploration (throwaway/; HEAD `ddb9b80`)
Frame [user]: "can't overfit fast ⇒ can't fit real data"; **CanViT is the bottleneck, not the net**
(throughput = **glimpses/s**, not steps/s); ORACLE (greedy true-best rollout) = the CanViT-bounded ceiling.

### `throwaway/reward_corr.py` — what predicts the t0 reward landscape (pure analysis, CACHED)
**Measure PREDICTIVENESS, not correlation** [user, emphatic — see memory predictiveness-not-correlation]:
how much of the random→oracle reward headroom does *greedy-by-cue* capture (= what a cue-greedy policy gets).
Result (n=48, V=512): random reward 0.0302, oracle 0.1854; **entropy (EG-C2F's cue) captures only ~12%**,
scale ~5%, `|center|` **−8.6%** (central glimpses beat peripheral). ⇒ The reward is **subtle / largely
orthogonal to simple deployable cues** — a policy must learn real structure (why oracle ≫ EG-C2F).
[I initially OVERclaimed "0%/useless" using linear-corr + argmax-match (meaningless, 1-of-512) + current_CE
(GT-derived, invalid at inference) — all wrong; corrected.]

### `throwaway/overfit_batch.py` — T=5 single-batch AUX ABLATION (RUNNING on crockett now)
Variants `{none, cos, delta, dmag, vpe, all}` on a fixed 24-scene batch, **s=0.5 only** (V=256). Aux on the
**LayerNormed canvas**: `cos`=1−cos(LN c_t,LN c_{t−1}); `delta`=`LN(c_t)−LN(c_{t−1})` full vector;
`dmag`=its magnitude; `vpe`=`VPEEncoder(last_vp)` (`[:32]`) broadcast. Per variant: deploy mIoU/CE vs oracle
+ EG-C2F, per-depth `corr_d*`, **free per-depth ε-greedy training mIoU**, reward-landscape heatmap. Oracle
**cached** (`oracle_335143a8d08e.pt`: t0–t4 `26.72/30.39/33.40/34.96/36.06`, ce→0.533; EG-C2F →`30.80`/0.686).
Live to mlflow experiment **`canvit-overfit`**.

PRELIMINARY reads — **single noisy 24-scene batch, hypotheses NOT verdicts** [user: this limited ablation
won't reliably rank the aux]. CORRECTED run (s=0.5, LN-features), deploy t0–t4 mIoU | ce_t4 | fit d2/d3:
```
oracle  26.72/30.39/33.40/34.96/36.06 | 0.533   (EG-C2F: .../30.80 | 0.686)
none    26.72/28.49/29.83/30.62/31.00 | 0.646 | d2=.42 d3=.46
cos     26.72/28.69/30.17/29.92/30.11 | 0.652 | d2=.49 d3=.58
delta   26.72/28.05/29.06/30.42/30.41 | 0.653 | d2=.42 d3=.59
dmag    26.72/28.00/29.13/29.56/29.80 | 0.651 | d2=.47 d3=.51
vpe     26.72/29.11/29.91/30.71/30.99 | 0.636 | d2=.51 d3=.43   <- best CE + best t1
```
- Net fits **t0** trivially (corr~1.0) but only partially fits **deep** states (`corr_d2≈0.42–0.51`) — deep
  rollouts intrinsically harder [user-confirmed]; aux barely move it.
- **vpe (last-viewpoint encoding) is the best on CE** (0.636 vs none 0.646) and t1 — weak but the clearest
  signal; cos/delta ~neutral-to-slightly-better CE; dmag worst. ALL still far below the oracle (CE 0.533,
  t4 36.06) — large unrealized headroom. (`all` + combined plot + heatmaps land when the run finishes.)

## The plan — add aux + recurrence to the REAL `q/` pipeline, CLEANLY [user]
Pivot BACK to the working twin-Q grid policy; the net should get: features (have) + feat-differential +
cos-dissim + entropy map + VPE→trainable-linear + recurrence — added cleanly, gated OFF by default.

**One state-encoding contract:** `qnet_input(seg, st, cfg, *, prev_canvas=None, last_vp=None, hidden=None)`
assembles gated pieces (each a `QConfig` flag; aux-OFF ⇒ byte-identical to today, pinned by a `test_q`
invariant). Three consumers thread `prev_canvas`/`last_vp`/`hidden`: `rollout_samples`, `GreedyQPolicy`,
`evaluate_q`/`rollout_eval`. Net `in_ch` grows + a `nn.Linear(rff_dim, D)` for VPE. Input-channel aux are a
cheap **prior**; a trainable recurrence can LEARN the same step-to-step signal.

**RECURRENCE (corrected design — I made errors the user caught):**
- NOT a 1×1 global bottleneck (zero spatial resolution — can't encode "where I've looked"; wrong).
- The canvas IS spatial memory but **FROZEN** (CanViT's, not trainable/policy-optimized) → a **trainable**
  recurrence adds learned spatial memory. Shape: **per-position shared-weights ConvGRU**, hidden
  `h_t [B,C,32,32]` on the canvas→ConvNeXt feature map ("canvas → ConvNeXt feat map → GRU-ish recur back").
- **Training (the crux):** ON-POLICY ⇒ CANNOT pre-collect-then-BPTT (the visited states depend on the
  recurrence). Do **ONE rollout**: gradient flows through the trainable chain `encode→ConvGRU→head` with the
  hidden **BPTT'd across the 5 steps**; the frozen-backbone advance and `a_t=argmax Q_t` are `no_grad`
  (argmax non-diff ⇒ no gradient through action/state-transition; gradient flows `h_0→…→h_4` through the
  GRU). The SAME recurrent forward both chooses `a_t` and is trained. Separating collection from training is
  ONLY valid OFF-policy (random/oracle behavior rollouts — a deliberate DAgger/expert choice, easier to
  parallelize).

**Execute at LOWER context (fresh).** This touches the working pipeline; per the high-context-degradation
discipline ([[skeptical-of-own-high-context-reasoning]]) do it as a careful, tested, incremental pass
(validate each aux with a fast short-horizon run, judge val CE@t4 + deep-depth fit vs the no-aux twin-Q),
NOT in this enormous session.

## State for resume (cold-start)
- **GPU:** crockett running `throwaway/overfit_batch.py` (HEAD `ddb9b80`, the corrected s=0.5/LN-delta/live-IoU
  ablation), ~25 min, 6 variants → per-variant deploy mIoU/CE + heatmaps + `runs/ablation_trajectory.png` +
  live mlflow `canvit-overfit`. (If still running at resume, read its per-variant table + combined plot.)
- **Repo:** all committed + pushed to origin + deploy, HEAD `ddb9b80`, `just`-green.
- **Caches** (reuse, never recompute): `runs/oracle_cache/` (oracle trajectory per config),
  `runs/analysis_cache/rewardcorr_*.pt` (reward landscape + entropy). [memory: always cache expensive computes.]
- **Ops:** deploy = commit → push origin+deploy → `git reset --hard origin/main` in `~/projects/...` (crockett
  origin = `~/repos` bare = local `deploy`). `pkill -f` SELF-MATCHES the ssh shell whose cmdline contains the
  pattern → use the `[x]pattern` bracket trick AND keep kill separate from any launch of the same filename.
- New memories: predictiveness-not-correlation, dont-be-lazy-find-best-training, pkill-f-self-match-over-ssh,
  staleness-duplication-fluff-never-minor.
