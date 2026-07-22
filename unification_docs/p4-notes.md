# P4 — joint modes & curriculum (P4a DONE 2026-07-22; P4b gate-blocked by design)

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

Why deferred: P4b builds directly on the in-graph + BN-mode-(a) selection
semantics whose validation IS the P3 gate (job 15025279 vs the qband band). If
the gate fails and traces to BN mode, the selection path switches to (b)
(eval-mode-with-grad + stat updates, master plan §4.3) — better to know before
wiring it into the production loop.
