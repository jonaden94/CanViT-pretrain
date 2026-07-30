# P3 — policy machinery ported (CODE COMPLETE 2026-07-22; gate run pending)

## Delivered

**Core (`canvit_pytorch`, commit `524d27c`):** `canvit_pytorch.policy` package —
`ViewpointScorer` (identical arch + config.json schema; published ckpts load;
legacy configs without `action_space` default to the historical `"safebox"`),
`StateEncoder`/feature groups (decoupled from RL TrainConfig; `INTRINSIC_GROUPS`
for probe-free tasks), scoring primitives (ignore_label parameterized — no dataset
dep), the NEW `"fixation"` action space (centers-only grid over the full field,
n_scale=1 — user decision 2026-07-22; candidates sit at score-map cell centers).
New `[policy]` extra (timm). 5 CPU tests incl. hub round-trip + legacy config.

**Pretrain:**
- `train/rl.py` — QReg/PG objective dataclasses, `RunningNorm` (per-depth online
  z), `entropy_floor_step` (SAC-style dual ascent), `qreg_loss`/`pg_loss`
  (+ exact discrete Q-Prop variant). Task-agnostic.
- `ade20k/rl_train.py` — the frozen-model policy trainer recreating the RL-repo
  flagship (`python -m canvit_pretrain.ade20k.rl_train`, tyro): frozen
  `from_pretrained_with_probe` stack, **in-graph rollout** (§4.3 — the selecting
  forward IS the training forward; one forward per state), ε-greedy DAgger (QReg)
  / on-policy sampling (PG), per-depth RunningNorm, fractional-CE reward at
  `score_res=128`, argmax-deploy val (mean full-res CE t1..tH), best/last ckpts +
  `save_pretrained` HF dir, wandb. Canonical recipe defaults verbatim from the RL
  repo (lr 2e-4, wd 1e-2, betas .9/.95, clip 1.0, 640k-forward budget → 8000
  steps, warmup 12.5% then hold, momentum .997, batch 16).
- 3 CPU tests (`ade20k/test_rl_train.py`): QReg + PG uniform; **QReg on a
  foveated model with the fixation action space — first-ever foveated policy
  training step**, runs end-to-end.

## Deliberate deltas vs the RL repo (all recorded decisions)

1. **In-graph rollout replaces collect-detached-then-reforward** (master plan §4.3).
   **BN mode (a) was RETIRED 2026-07-30 — mode (b) is now the default on both paths.**
   The knowingly-approximated train-mode selection turned out to cost **0.19 mIoU t4 at
   matched CE** (exp27 arm A vs arm C, measured at the last step): the scorer's one
   BatchNorm normalized on batch statistics where the reference uses running statistics,
   and the two modes disagree on 45.7% of chosen glimpses. `select_bn_eval=True` restores
   the reference behaviour; `--no-select-bn-eval` / `--rl.no-select-bn-eval` gets mode (a)
   back. The rest of the in-graph rollout (one forward per state) was shown to cost
   nothing once selection was fixed — see doc 15 §A3. `pooled_policy_loss` implements the
   other half of the original deviation and remains available and unvalidated.
2. **`image=` API**: the RL repo called `seg.canvit(glimpse=…)` against the OLD
   core API (it pins upstream m2b3) — the port uses the current keyword and the
   patcher-aware glimpse routing (`consumes_full_image`), so foveated works.
3. Not ported (deferred or dropped): MLflow/justfile/uv (D4), sweeps
   (`search/`, plateaued at seed noise), `unfreeze="probe"` ladder + Q-Prop
   *trainer-level* extras beyond the loss (P4 territory), `keep_every`
   step-checkpoints, `policy.eval` episode runner (→ canvit_eval in P6), flow (out of
   scope).

   **Update 2026-07-30: the figure4b BASELINES are no longer missing.** EG-C2F is ported
   into `harness/eval_viewpoints.py` from `canvit_eval/policies.py` (the implementation the
   published row came from) and validated to max|Δ| = 0.05 against paper Table 4;
   `coarse_to_fine` matches to 0.07. Both are selectable as `--cfg.eval-policy`. Note that
   our `random` is NOT the paper's F-IID (it runs +0.17..+0.42 above that row) — doc 15 §A4.
4. Q-Prop critic: one optimizer over actor+critic params, critic non-dueling —
   simplification; treat qprop as experimental until validated.
5. Data: squish transforms, no augmentation (the RL repo's protocol), its own
   loader in rl_train (train aug pipeline stays probe-training-only).

## Gate — SUBMITTED 2026-07-22: job **15025279** (`--run-name qband-port --seed 0`,
pinned PRETRAIN=7e5afac, PYTORCH=524d27c; log `logs/ade20k/policy-job-15025279.log`;
run dir under `checkpoints/canvit-ade20k-policies/`). Pass = final best val CE
(mean t1–t4) inside the qband band in CanViT-PyTorch-RL `docs/qband_results.md`.

## Gate criteria (original)

1. **Published-ckpt load check**: `ViewpointScorer.from_pretrained(DEFAULT_POLICY_REPO)`
   through the CORE class (needs network; mechanism covered by round-trip test).
2. **Statistical parity**: one seed of the default QReg recipe on this cluster
   (`python -m canvit_pretrain.ade20k.rl_train --run-name qband-port-s0`) must land
   inside the documented 8-seed qband band (mean val CE t1–t4; the RL repo's
   `docs/qband_results.md`). Note: no local reference runs exist (qband ran on the
   m2b3 machine), so the band comes from the docs. A SLURM launcher for this run
   still needs writing (small; pattern = slurm_nhr/ade20k/train_ade20k.sbatch).

## P4 pointers (what this unlocks)

- `PolicySelector`/`MixtureSelector` for the `training_step` seam (joint modes,
  ε-curriculum): the trainer-side selection logic in `rollout_and_loss` is the
  reference; P4 lifts it behind the Selector protocol.
- `policy_feats_detached=False`: remove the `no_grad` around `encoder(...)` in
  `rollout_and_loss` + unfreeze parts of seg + TBPTT-chunk the seg forwards.
- Distill-task policy: swap the reward for per-glimpse distill-MSE reduction and
  `feature_groups=INTRINSIC_GROUPS` (core supports both already).

---

# GATE RESULT (2026-07-23) — PASS. In-graph rollout + BN mode (a) VALIDATED.

Jobs 15025279 (seed 0) / 15025337 (seed 1), PRETRAIN=7e5afac / 2eeaa29,
PYTORCH=524d27c. QReg, c64, T=5, 8000 steps, `probe-ade20k-40k-s512-c64-in21k`
on `canvitb16-...-2026-02-02`. Judged as the reference does: best mean(t1–t4)
val CE.

```
                                mean(t1-4) val CE     source
 qband reference band (8 seeds)   0.6853 ± 0.0007     RL repo docs/qband_results.md
   per-seed spread of that band   0.6845 … 0.6865     (s2 best, s7 worst)
 PORT seed 0 (15025279)           0.6855   (@step 6000)
 PORT seed 1 (15025337)           0.6867   (@step 5000)
 EG-C2F-c64 (entropy-guided       0.6949
   coarse-to-fine, deterministic)
```

- Seed 0 sits **dead center** in the band (+0.3σ of the band mean).
- Seed 1 is +2σ, marginally above the reference's own worst seed (0.6865) — at
  the edge, not outside it in any meaningful sense given n=2 vs n=8.
- Both beat the **EG-C2F baseline by 0.008–0.009 CE (>10σ of the band spread)**,
  reproducing the headline claim the RL repo makes for the learned policy.

**What this validates:** the two deliberate deviations from the RL repo — the
**in-graph rollout** (selection forward IS the training forward; no
collect-detached-then-reforward) and **BN mode (a)** (train-mode rollout
forward, accepting the DAgger deviation) — reproduce the original band. The
fallback (b) (eval-mode-with-grad, master plan §4.3) is NOT needed. **P4b is
unblocked.**

**Gap to close:** the port logs val CE only; the reference band reports CE *and*
mIoU (42.65→44.97 over t1–t4), and mIoU is what the probe runs report, so
policy runs are currently not comparable to probe runs. Add mIoU to
`rl_train.evaluate` (cheap — `mIoUAccumulator` already exists in
`ade20k/metrics.py`). Filed as a P4b prerequisite.

**Baseline availability (answers "do we even have a reference?"):** yes, for the
policy — both the qband band and EG-C2F are documented *numbers* in
`qband_results.md`, with the EG-C2F measurement data under
`docs/data/measured_baselines/egc2f_c64_t5_ce/`. They are not yet *runnable*
here (the baselines module wasn't ported). P6 ports `baselines.figure4b` into
`canvit_eval` so EG-C2F / random / coarse-to-fine can be re-measured on our own
checkpoints instead of cited from the RL repo's runs.
