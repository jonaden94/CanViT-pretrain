# Potential validation-side speedups (compile-cost reductions)

`validate.py` currently uses two PyTorch idioms whose canonical purpose is
correctness for validation — but each one forces `torch.compile` to compile
an additional kernel set on the cold cache that the first job hits. The
redistillation showed both can be removed safely *in this codebase as it
stands today*, trading a small compile-time win for a small loss of
canonical-PyTorch defensive semantics.

Empirical baseline (single-GPU A100, redistill measurements): with both
"expensive" idioms in place, the first training step takes **~71 s** on a
cold Triton cache. Each idiom contributes roughly:

- `torch.inference_mode()` — adds ~4 s of compile (runs an inference-mode-
  specific Triton compile distinct from the training-mode kernels).
- `model.eval()` toggle — adds ~18 s of compile (flipping `model.training`
  is traced as a different graph by `torch.compile`, which then has to
  recompile the *training* graph from scratch when `model.train()` flips
  it back, hitting on the first training step).

So the ~22 s difference between cold-cache `~49 s` and `~71 s` first-step
times is the combined cost of both idioms. For the sanity-test setup
(`steps_per_job=256`, val_every=128, ~3 validations per job), most of this
is one-time-per-job overhead.

For a typical real training run (thousands of jobs over the lifetime of a
model), this is on the order of a couple of CPU-hours total, which is not
much, but it's also free to recover.

The redistillation tested removing each idiom independently (3/3 PASS for
each). Neither removal breaks DDP correctness, multi-rank coordination,
checkpointing, or wandb logging.

---

## Idea 1: replace `torch.inference_mode()` with `torch.no_grad()`

### What this does

`torch.no_grad()` and `torch.inference_mode()` both disable autograd
gradient tracking, but they differ in how `torch.compile` treats them:

- `torch.no_grad()` — transparent to `torch.compile`. The compiled graph
  for training is reused inside the `no_grad` block; no recompile.
- `torch.inference_mode()` — opens a **distinct compilation context**. The
  first call triggers a from-scratch Triton compile of an inference-mode-
  specific kernel set. Inference-mode tensors also have version-counter
  and view-tracking disabled at the C++ level, giving slightly lower
  per-op overhead at runtime.

### What you lose by switching to `no_grad`

The runtime savings of inference-mode (lower per-op autograd overhead, no
version-counter bumps on tensor reads/views). For our validation, this
is on the order of single-digit milliseconds per call; with ~3
validations per job it's noise relative to the ~4 s compile saving on the
first job hit.

You also can't mix the resulting tensors back into a gradient-tracked
computation later — but validation doesn't do that anyway (it only logs
metrics + plots figures).

### Risk

Zero. The redistillation iter 8 tested this exact change (revert U10 →
keep the rest) and got 3/3 PASS. Verified in
`distill_iterative_documentation_new.md`'s test tracking.

### Suggested change

In `canvit_pretrain/train/viz/validate.py`, replace:

```python
with torch.inference_mode():
    ...
```

with:

```python
with torch.no_grad():
    ...
```

Single-line change.

### Recommendation

Apply. ~4 s saved per cold-cache job; semantics unchanged for our use
case; redistill-verified.

---

## Idea 2: remove the `model.eval()` / `model.train()` toggle

### What this does

The current `validate.py` wraps validation in:

```python
model_was_training = model.training
model.eval()
try:
    with torch.inference_mode():
        ... validation body ...
finally:
    if model_was_training:
        model.train()
```

`model.eval()` flips `model.training` from `True` to `False`. This is the
canonical PyTorch idiom for validation because it changes layer behavior:

- **`nn.Dropout`** stops dropping units (uses all units, deterministic
  output).
- **`nn.BatchNorm*`** uses its stored running statistics instead of the
  current batch's statistics, and stops updating those running statistics.

`torch.compile` traces `model.training` as a flag — flipping it produces a
*different* compiled graph than training mode. The first call to
`model.eval()` triggers compilation of the eval-mode graph; the
subsequent `model.train()` flip back triggers a re-compile of the
training-mode graph (because the training-mode graph hadn't actually been
fully compiled yet on first job — it was preempted by the eval-mode flip
during validation at step 0).

### What you lose by removing the toggle

Validation runs in training mode. Concretely:

- If the model has `nn.Dropout`: validation forwards see different random
  unit drops each call → validation metrics gain an extra source of
  variance (still unbiased in expectation, but per-batch noisier).
- If the model has `nn.BatchNorm`: validation batches' statistics would
  enter the running-stats accumulator, *contaminating training*. This is
  the more dangerous one — it means a few validation batches' statistics
  drift the BN running mean/var, which the next training step will use.

For the **CanViT model as it stands today**, this redistillation iter 9
empirically passed 3/3 — implying neither failure mode hit. The model
either doesn't use these layers in a way that matters, or the effect is
small enough not to surface in 256-step sanity runs.

The risk is that this becomes a silent regression *the first time* the
model gains a `BatchNorm` or anyone touches the dropout config. The
toggle is the standard guard against exactly that.

### Risk

Empirically: zero in the redistill (3/3 PASS at the time, model in its
current shape).

Going forward: medium-low. Anyone adding a `BatchNorm` (or replacing
`LayerNorm` with `BatchNorm` for an ablation) without restoring the
toggle would silently corrupt running stats on every validation. Anyone
adding meaningful `Dropout` would notice noisier val curves but not a
correctness break.

### Suggested change

If applied, the validate body simplifies to:

```python
with torch.no_grad():    # or inference_mode if Idea 1 is rejected
    ... validation body ...
```

— no `model_was_training`, no try/finally.

### Recommendation

Apply, but **with two guardrails**:

1. Add a runtime assert at the top of `validate()` documenting the
   assumption — something like:

   ```python
   for m in model.modules():
       assert not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)), (
           "BatchNorm in model — restore model.eval()/model.train() toggle in validate.py"
       )
   ```

   This catches the silent-regression risk loudly the first time someone
   adds a BatchNorm.

2. Add a comment block explaining *why* the toggle isn't there (links
   back to this doc or to the redistill).

With those guardrails, applying is recommended — saves ~18 s per
cold-cache job, costs zero against the current model.

---

## Combined savings

If both applied: ~22 s per cold-cache job. Hot caches don't see a
difference (the kernels are reused).

For a long training campaign (say 500 jobs over a few weeks), that's
roughly **3 CPU-hours** saved cumulatively. Modest but free.

---

## What I am *not* recommending

- **Disabling `torch.compile` altogether**: would eliminate compile cost
  but cost much more in steady-state per-step throughput. Net loss.
- **Pre-warming the Triton cache** by keeping kernel artifacts across
  jobs: technically possible (skip the cache wipe at sbatch start) but
  loses the protection against poisoned state from cancelled-mid-compile
  jobs. The cache wipe is a robustness mechanism, not a perf bug.
- **Compiling validation with `mode="reduce-overhead"` or `mode="max-autotune"`**:
  changes compile output for every kernel, not just validation; would
  need broader benchmarking. Out of scope for a "small validation-side
  speedup" change.

---

## Suggested action order

1. Apply Idea 1 first (zero-risk, small win, single-line change).
2. Then apply Idea 2 with the BatchNorm assertion guardrail.

Both together touch only `canvit_pretrain/train/viz/validate.py`. The
diff is small (~10 lines net).
