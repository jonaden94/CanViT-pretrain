# exp23 — cutover fidelity A/B: current unified stack vs exp22's old stack

**Question this answers:** does the code I actually want to ship (the unified repo at
current HEAD, driven through the harness) reproduce exp22's distill training? If the
paired curves overlay, the cutover is numerically safe end-to-end against the *real*
production baseline — not just against a same-pin sibling.

## The four runs (two pairs)

| script | patcher | side | entry point | pins (pretrain/pytorch/fovi) |
|---|---|---|---|---|
| `uniform16-ti-oldloop.sh` | uniform | OLD | `python -m canvit_pretrain.train` | **fe24aa1 / 3277048 / c399d3b** (exp22's exact stack) |
| `uniform16-ti-harness.sh` | uniform | NEW | `python -m canvit_train.harness.run distill` | **24a8500 / 017ce9b / c399d3b** (current HEAD) |
| `fovi-ti-oldloop.sh` | foveated | OLD | `python -m canvit_pretrain.train` | fe24aa1 / 3277048 / c399d3b |
| `fovi-ti-harness.sh` | foveated | NEW | `python -m canvit_train.harness.run distill` | 24a8500 / 017ce9b / c399d3b |

Everything else is held identical across a pair: `seed=0`, `peak_lr=4e-4`,
`batch=64`, `warmup_steps=100_000` (config default), constant LR after warmup
(`cosine_total_steps=None`), same `WEBDATASET_DIR`/`VAL_DIR` from `.envrc.grete`, same
teacher-init, same fovi geometry. **The only variable within a pair is the training
code + entry point.** All four are fresh from step 0.

## Why these exact choices

- **Fresh re-run of the old side, not the historical exp22 curves.** exp22 was a
  2M-step run with LR drops; this is a clean `0 -> ~205k` constant-schedule window with
  seed/schedule/logging pinned identically to the new side. Overlaying old wandb curves
  would confound the schedule.
- **New side pins pretrain `24a8500` (HEAD), NOT `bc63eee`.** `24a8500` adds the
  matplotlib-DDP and `torch._functorch.config.backward_pass_autocast="off"` fixes that a
  **compiled** run needs (`Config.compile=True` by default). A compiled run at `bc63eee`
  would hit the gradient-corruption bug the pre-cutover audit found (doc 12).
- **Bundles the whole stack, on purpose.** New = current pretrain + current pytorch;
  old = exp22 pretrain + exp22 pytorch. This is the "does the shippable code reproduce
  exp22" question. The one link code-reading alone can't certify — that `fe24aa1`'s
  inline viewpoint-selection is numerically identical to HEAD's `RandomSelector` (the
  byte-parity digest only proves harness==selector *at HEAD*) — is exactly what these
  runs measure against the true baseline.
- **205k, not 100k.** `warmup_steps=100_000`, so 100k is *pure warmup*; 205k = 100k
  warmup + ~105k constant-LR plateau. The plateau is where any divergence compounds, so
  it's the discriminating half.

## Reading the result

Overlay each pair (`*-oldloop` vs `*-harness`) in wandb project `exp23`:
- **train loss** and **val reconstruction** curves.
- Same seed + same deterministic shard schedule => if the code is equivalent, the
  curves should sit essentially on top of each other. They will **not** be byte-identical
  (GPU-atomic / `torch.compile` nondeterminism), so judge by "coincide within
  run-to-run noise," not exact equality. A persistent, growing gap on the plateau is the
  signal that something in the current stack changed distill behavior — then decompose
  (swap one pin at a time) to localize it.

## Logistics

- `%1` (one job at a time) is **required** — each job resumes the previous job's
  checkpoint (`STEPS_PER_JOB=8192`, 25 jobs). So each run is inherently sequential:
  ~25 x (≤2h + queue) ≈ 1–2 days wall per run. The four runs are independent array
  submissions and can run concurrently.
- Start with the **uniform16-ti pair** to de-risk cheaply (uniform is the clean
  teacher-init best case), then submit the fovi-ti pair.
- `8192 x 64` images/job is a whole number of shards (exp22-proven), and 25 whole jobs
  keeps `batch x steps` a shard multiple — clean resume throughout.

## Submit (owner runs these; nothing here submits on its own)

```bash
cd /user/henrich1/u25995/jonathan/repos/CanViT-train
bash slurm_nhr/runs/exp23/uniform16-ti-oldloop.sh
bash slurm_nhr/runs/exp23/uniform16-ti-harness.sh
# then, once the uniform pair looks right:
bash slurm_nhr/runs/exp23/fovi-ti-oldloop.sh
bash slurm_nhr/runs/exp23/fovi-ti-harness.sh
```
