# P4 — joint modes & curriculum (P4a DONE 2026-07-22; P4b IMPLEMENTED 2026-07-23)

## P4a — selector plumbing (DONE)

`train/selector.py` gains, behind the unchanged P1 `Selector` protocol:

- **`PolicySelector`** — featurize the live state (core `StateEncoder`) → score all
  candidates (`ViewpointScorer`) → argmax (deploy) or softmax-sample (PG). FULL
  viewpoints delegate to a wrapped `RandomSelector` (patcher-specific t0 anchor
  logic stays in one place). Caller controls grad/eval context; training aux
  (feats / flat_idx / scores nodes) is stashed on `last_aux` for P4b.
- **`MixtureSelector`** — per-SAMPLE ε-mixture: policy pick with prob `p_policy`,
  else random. `p_policy=0` ≡ today's random training (the off-switch), `1` = pure
  policy; the trainer owns the schedule and sets the float per step (warmup
  curriculum = ramp 0→target). `last_mask` records policy-chosen rows (credit).

3 tests (`train/test_selectors.py`, fake net/encoder). Suite 71 green; parity
digest still `9a0100a1…` (the additions are dead code until injected).

Immediate use unlocked even before P4b: **deploying a trained frozen policy as
the glimpse source of a distill/probe run** — pass
`PolicySelector(net=ViewpointScorer.from_pretrained(...), encoder=StateEncoder(...),
mode="argmax")` (inside no_grad, net.eval()) to `training_step(selector=...)` or
wire it into the ade20k trainer. Not yet exposed as config — P4b adds the flags.

## P4b — joint task+policy training (DEFERRED pending the P3 gate, deliberately)

Scope: `use_rl`/`rl_weight`/`policy_feats_detached`/ε-schedule config surface;
in-graph policy loss terms inside `training_step`'s chunked loop (selector aux →
qreg/pg loss added to the task loss); distill-task reward (per-glimpse
fractional distill-MSE reduction, INTRINSIC_GROUPS features); param groups for
{backbone, head, policy}.

Why it was deferred: P4b builds directly on the in-graph + BN-mode-(a) selection
semantics whose validation IS the P3 gate (job 15025279 vs the qband band). That
gate PASSED (p3-notes.md), so P4b was unblocked and implemented.

## P4b — IMPLEMENTED (2026-07-23), CPU-validated; SLURM run pending (user's call)

Joint task+policy training as a mode of the pretraining loop, OFF by default
(`cfg.rl.use_rl=False` => byte-identical to pre-P4b; parity probe still
`9a0100a1a3de3acd`). New/changed:

- **`train/config.py` `JointPolicyConfig`** (`cfg.rl`): `use_rl`, `rl_weight`,
  `feats_detached`, `keep_random_branch`, objective (`qreg`/`pg`) + knobs,
  `prime_on_policy` + `policy_warmup_steps` (ε-curriculum), action-space/net knobs
  (`centers_per_axis`/`scales`/`width`/`block_layers`/`feature_groups`=INTRINSIC),
  `target_momentum`, `policy_lr`, `policy_weight_decay`.
- **`train/joint.py`** — `JointPolicy` orchestrator + `build_joint_policy` factory.
  Owns the policy/random selectors, the scorer, per-depth `RunningNorm`s, the PG
  dual `log_alpha`; `glimpse_loss` forms the qreg/pg loss for one glimpse;
  `branch_selector`, `set_prime_for_step`, `broadcast`/`allreduce_grads` (DDP),
  `state_dict`/`load_state_dict` (sidecar). Encoder is probe-free (INTRINSIC groups)
  via a `SimpleNamespace(canvit=core_model)` shim.
- **`train/step.py`** — `joint` param (default None => parity path). Policy branches
  are FULL-anchored; each t>=1 glimpse's policy loss (fractional distill-MSE-reduction
  reward, detached, per-depth standardized) is added to the chunk's `chunk_combined_loss`
  BEFORE its TBPTT backward, so it backprops with the task loss (one backward, mode (a)).
- **`train/selector.py`** — `PolicySelector` gains ε-greedy (`prime_on_policy<1` mixes
  a random candidate; =1.0 default = pure argmax, consumes no RNG => P4a parity) and
  `feats_detached` (encoder under no_grad => scorer-only gradient).
- **`train/task.py`** — `DistillTask.per_image_loss` [B] (the reward's raw material).
- **`policy/features.py` (core)** — `StateEncoder` now stores its prev/init references
  DETACHED. Required: else the retained graph is re-entered across TBPTT chunks/steps
  ("backward through the graph a second time"). No-op under the frozen no_grad callers
  (rl_train, eval); their tests stay green.
- **`train/loop.py`** — builds the joint policy on the raw model, adds a `{model, policy}`
  optimizer param-group split, runs the ε-curriculum, passes `joint` to `training_step`,
  AllReduces+clips the scorer grads, logs `policy_loss`/`reward_frac`/`prime_on_policy`,
  and writes/loads a `<ckpt>.policy.pt` sidecar (scorer + running-norms + log_alpha) so
  the CheckpointData schema is untouched.

Decisions (user, 2026-07-23):
- Glimpse source in policy mode = the policy's discrete candidate grid (ε-greedy /
  on-policy). NOT the fork I initially framed — there is no contradiction: random
  glimpses stay continuous (RL off), policy glimpses use the grid the scorer expects.
- **Distill coverage**: both supported; DEFAULT `keep_random_branch=False`
  (all branches are policy branches). `True` retains a pure-random distill-only branch.
- **Policy grad -> backbone**: both supported; DEFAULT `feats_detached=True`
  (scorer-only). Coupled (`False`) reshapes the backbone.

DDP: the scorer is NOT DDP-wrapped (its forward is deep in the rollout), so its params
are broadcast once and its grads AllReduced by hand (the head-√N pattern task.py
documents). **Coupled (`feats_detached=False`) + DDP is asserted UNSUPPORTED** — that
gradient path runs through the unwrapped core model and would bypass AllReduce; use the
default `feats_detached=True` for multi-GPU. RunningNorm stays per-rank by design (rl.py).

Validated: 7 CPU tests in `train/test_joint.py` (qreg+pg train the scorer AND the task,
keep_random_branch, feats_detached gates backbone grad at the selector, action space per
patcher, sidecar round-trip, ε-curriculum), plus core `policy/test_policy.py`, the full
`train` (65) + `ade20k` suites green, and the parity digest unchanged. **NOT yet run on
GPU/DDP** — the real gate (a short joint run: reward_frac trends up, val distill loss not
worse than a distill-only baseline, resume via sidecar) is a SLURM submission, the user's
call. Follow-ups: publish a joint-trained scorer to HF (save_pretrained on the scorer);
per-group LR schedule for the policy (currently shares the model's warmup+constant/cosine).
