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

1. **In-graph rollout replaces collect-detached-then-reforward** (master plan
   §4.3). **BN mode (a)** [user]: selection under train-mode BN — strict eval-mode
   DAgger selection knowingly approximated; fallback (b) documented in §4.3.
2. **`image=` API**: the RL repo called `seg.canvit(glimpse=…)` against the OLD
   core API (it pins upstream m2b3) — the port uses the current keyword and the
   patcher-aware glimpse routing (`consumes_full_image`), so foveated works.
3. Not ported (deferred or dropped): MLflow/justfile/uv (D4), sweeps
   (`search/`, plateaued at seed noise), `unfreeze="probe"` ladder + Q-Prop
   *trainer-level* extras beyond the loss (P4 territory), `keep_every`
   step-checkpoints, figure4b baselines + `policy.eval` episode runner (→
   canvit_eval in P6), flow (out of scope).
4. Q-Prop critic: one optimizer over actor+critic params, critic non-dueling —
   simplification; treat qprop as experimental until validated.
5. Data: squish transforms, no augmentation (the RL repo's protocol), its own
   loader in rl_train (train aug pipeline stays probe-training-only).

## Gate (pending — user submits; needs GPU + HF access)

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
