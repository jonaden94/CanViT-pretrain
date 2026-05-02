# Pluggable experiment tracker (Comet / wandb / none)

This document describes how the codebase abstracts experiment tracking
behind a single interface so the training loop never imports `comet_ml` or
`wandb` directly. The goal is reproducibility: a reader who follows this
doc should be able to add the same dispatch layer to a codebase that
currently uses one tracker hardcoded everywhere, and end up with
runtime-selectable Comet / wandb / no-op logging that works under DDP.

The starting point assumed by this doc is a codebase where the training
loop directly creates a `comet_ml.Experiment` (or any other tracker
client) and calls its methods inline (`exp.log_metric`, `exp.log_image`,
etc.). The endpoint is HEAD's design: a thin `Tracker` wrapper plus a
`make_tracker(...)` factory selected by `cfg.tracker`.

---

## 1. Why an abstraction layer

Three orthogonal concerns motivate moving the tracker behind a wrapper:

1. **Selectable backend at runtime.** Different runs may want Comet (rich
   metric viewer + project history) or wandb (different plotting, team
   workflow, or just personal preference). Hardcoding one means a
   substantial diff to switch.

2. **No-op behavior for non-main DDP ranks.** Without a wrapper, every
   `exp.log_*` call site has to be guarded with `if ddp.is_main():`,
   which is dozens of conditionals scattered across `loop.py` and
   `viz/*.py`. With a wrapper, non-main ranks receive an empty `Tracker()`
   and every method is a silent no-op — call sites stay unconditional.

3. **No-op for smoke tests.** `tracker="none"` disables logging entirely,
   useful for quick local verification or CI runs that shouldn't pollute
   project dashboards. Same wrapper, no real backend.

These three modes — Comet / wandb / none — are dispatched by a single
factory, and after the factory returns, the rest of the code is identical
regardless of choice.

---

## 2. The `Tracker` wrapper

**File: `canvit_pretrain/train/tracker.py`**

A thin class that holds an optional Comet experiment and an optional
wandb run, fanning each `log_*` method out to whichever backend is set:

```python
class Tracker:
    def __init__(
        self,
        comet_exp: comet_ml.CometExperiment | None = None,
        wandb_run: Any | None = None,
    ) -> None:
        self._comet = comet_exp
        self._wandb = wandb_run

    def log_metric(self, name: str, value: Any, step: int | None = None) -> None:
        if self._comet is not None:
            self._comet.log_metric(name, value, step=step)
        if self._wandb is not None:
            self._wandb.log({name: value}, step=step)

    # ... log_metrics, log_image, log_curve, log_parameters, end ...
```

Both being `None` is a valid state — that's what non-main DDP ranks and
`tracker="none"` produce, and it makes every method a silent no-op.

Both being non-`None` is also formally valid (would log to both
simultaneously) but the factory in §3 doesn't produce that state — the
selector exposes "comet | wandb | none", picking exactly one (or
neither).

### 2.1 The full method surface

The wrapper exposes exactly the surface the training loop and `viz/*`
historically used. Each method maps onto each backend as follows:

| Method | Comet | wandb |
|---|---|---|
| `log_parameters(params: dict)` | `experiment.log_parameters(params)` | `run.config.update(params, allow_val_change=True)` |
| `log_metric(name, value, step=)` | `experiment.log_metric(name, value, step=step)` | `run.log({name: value}, step=step)` |
| `log_metrics(metrics: dict, step=)` | `experiment.log_metrics(metrics, step=step)` | `run.log(metrics, step=step)` |
| `log_image(image, name=, step=)` | `experiment.log_image(image, name=name, step=step)` | `run.log({name: wandb.Image(image)}, step=step)` |
| `log_curve(name, x=, y=, step=)` | `experiment.log_curve(name, x=x, y=y, step=step)` | render PNG via matplotlib → `run.log({name: wandb.Image(img)}, step=step)` (no native equivalent) |
| `get_comet_id() -> str | None` | `experiment.get_key()` | `None` |
| `get_wandb_id() -> str | None` | `None` | `run.id` |
| `get_key() -> str` | `get_comet_id()` (fallback) | `get_wandb_id()` (preferred) |
| `end()` | `experiment.end()` | `run.finish()` |

Both backends accept `PIL.Image` natively, so `log_image` doesn't need
backend-specific image plumbing — the caller hands in a PIL image and
the wrapper passes it through.

### 2.2 The wandb `log_curve` workaround

wandb has no native time-series-style line-plot API analogous to
`comet.log_curve(x, y)`. The wrapper renders the curve as a PNG via
matplotlib and logs it through `wandb.Image`:

```python
def _curve_as_pil(name: str, x: list, y: list) -> Any:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o")
    ax.set_xlabel("timestep"); ax.set_ylabel(name); ax.set_title(name)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig); gc.collect()
    buf.seek(0)
    img = PILImage.open(buf)
    img.load()  # force decode before buf goes out of scope
    return img
```

Each curve update lands as a new image (versioned by the `step` argument),
similar to how PCA figures are logged. Note the consequence:
**curve updates on wandb consume image storage**, not native time-series
storage. The `_CURVE_BUDGET` cap (§5) keeps this bounded.

### 2.3 ID accessors and `get_key()`

`get_comet_id()` and `get_wandb_id()` return the resume identifier for
their respective backend, or `None` if that backend isn't active. They are
called at checkpoint save time (§7) so each is stored independently in
the checkpoint.

`get_key()` returns a stable identifier for any active run, preferring
the wandb id (since wandb is the default backend in HEAD) and falling
back to comet, with the literal string `"no-tracker"` as the final
fallback (used when both backends are `None`).

---

## 3. The `make_tracker(...)` factory

Same file, ~line 110.

```python
def make_tracker(
    *,
    tracker: str,
    is_main: bool,
    is_seeding: bool,
    run_name: str,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_dir: Path | None,
    prev_comet_id: str | None,
    prev_wandb_id: str | None,
) -> Tracker:
    if not is_main or tracker == "none":
        return Tracker()
    if tracker == "comet": ...
    if tracker == "wandb": ...
    raise ValueError(...)
```

Three branches:

### 3.1 No-op fast path

```python
if not is_main or tracker == "none":
    return Tracker()
```

If the calling rank isn't main, or if the user explicitly selected
`"none"`, return the empty `Tracker` and skip all backend setup — no
network calls, no API keys checked, no run created.

### 3.2 Comet branch

```python
comet_cfg = comet_ml.ExperimentConfig(auto_metric_logging=False)
if prev_comet_id is not None and not is_seeding:
    log.info(f"Continuing Comet experiment: {prev_comet_id}")
    exp = comet_ml.start(experiment_key=prev_comet_id, experiment_config=comet_cfg)
else:
    if is_seeding and prev_comet_id:
        log.info(f"SEED mode: creating new Comet experiment ...")
    else:
        log.info("Creating NEW Comet experiment")
    exp = comet_ml.start(experiment_config=comet_cfg)
return Tracker(comet_exp=exp)
```

- `auto_metric_logging=False` because we control all metric logging
  explicitly through `log_metric` / `log_metrics`. Comet's auto-logging
  would add untyped duplicates.
- Resume condition: `prev_comet_id` provided AND `not is_seeding`. The
  reverse-not-resume case (a fresh run, or a SEED-mode run discarding
  the seed checkpoint's tracker continuity) creates a new experiment.

### 3.3 wandb branch

```python
assert wandb_project, "tracker='wandb' requires --wandb-project"
if wandb_dir is not None:
    wandb_dir.mkdir(parents=True, exist_ok=True)
kwargs = {"project": wandb_project, "name": run_name}
if wandb_dir is not None:
    kwargs["dir"] = str(wandb_dir)
if wandb_entity:
    kwargs["entity"] = wandb_entity
if prev_wandb_id is not None and not is_seeding:
    log.info(f"Resuming wandb run: {prev_wandb_id}")
    kwargs["id"] = prev_wandb_id
    kwargs["resume"] = "allow"
else:
    log.info("Creating NEW wandb run" if not (is_seeding and prev_wandb_id)
             else f"SEED mode: creating new wandb run ...")
run = wandb.init(**kwargs)
return Tracker(wandb_run=run)
```

- `wandb_project` is asserted required — wandb's `init()` requires it,
  and we want a clear error rather than wandb's silent fallback to a
  default project.
- `wandb_dir` is created if given and passed as `dir=` to `wandb.init`.
  This is where wandb stores its run files locally before/while syncing.
  In this codebase the default is a per-user shared-storage path
  (see §6).
- `wandb_entity` (team or user) is optional; wandb falls back to the
  authenticated user's default workspace if unset.
- Resume uses `id=prev_wandb_id, resume="allow"` (accepts either a
  brand-new init or a resumed one — robust to a partial first save).

### 3.4 SEED mode

`is_seeding` is true when the run is initializing model weights from a
*seed* checkpoint (not resuming a previous training continuation —
`--seed-ckpt`). In that case, the seed checkpoint may have its own
tracker IDs (the IDs of *whoever produced the seed*), which we don't want
to continue. Both the Comet and wandb branches detect this and create a
fresh tracker run while logging the discarded ID for traceability.

---

## 4. The `log_figure` helper

**File: `canvit_pretrain/train/viz/comet.py`**

A small backend-agnostic helper for sending matplotlib figures through
the wrapper:

```python
def log_figure(exp: Tracker, fig: Figure, name: str, step: int) -> None:
    try:
        with io.BytesIO() as buf:
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            buf.seek(0)
            img = PILImage.open(buf)
            img.load()  # decode while buf is alive — both backends accept PIL
        exp.log_image(img, name=name, step=step)
    except Exception as e:
        log.exception(f"Failed to log figure {name} at step {step}: {e}")
    finally:
        for ax in fig.axes:
            ax.clear()
        fig.clf()
        plt.close(fig)
        gc.collect()
```

Key points:

- Renders to PNG via `BytesIO`, decodes once into a `PIL.Image`, hands
  it to `Tracker.log_image`. Both backends accept PIL natively — no
  per-backend `isinstance` plumbing inside the wrapper.
- `img.load()` forces decode while the BytesIO is still alive (it goes
  out of scope at `with` exit). Without it the PIL image would lazily
  reference a closed buffer.
- The `finally` block is aggressive about matplotlib cleanup
  (`ax.clear()`, `fig.clf()`, `plt.close(fig)`, `gc.collect()`) because
  in long training runs matplotlib is a memory-leak source — its global
  figure registry holds references to every figure unless explicitly
  closed.

The file name `viz/comet.py` is historical (the file existed before
the abstraction) — the contents are now backend-agnostic, despite the
name. It ships only `log_curve` (§5) and `log_figure`.

---

## 5. Curve budget (`log_curve`)

Same file (`viz/comet.py`):

```python
_curve_count = 0
_CURVE_BUDGET = 900

def log_curve(exp: Tracker, name: str, **kwargs) -> None:
    """Log curve with budget enforcement. Skips silently once exhausted."""
    global _curve_count
    if _curve_count >= _CURVE_BUDGET:
        if _curve_count == _CURVE_BUDGET:
            log.warning(f"Curve budget exhausted ({_CURVE_BUDGET}), skipping further curves")
            _curve_count += 1  # only warn once
        return
    try:
        exp.log_curve(name, **kwargs)
        _curve_count += 1
    except Exception as e:
        log.exception(f"Failed to log curve {name}: {e}")
```

The 900-curve cap is in place because:

- **Comet** imposes a hard cap on curve panels per experiment; exceeding
  it errors out the whole logging.
- **wandb** logs curves as images (§2.2), which accumulate as image
  artifacts; 900 PNGs per run is already a noticeable storage cost.

The cap applies process-globally (module-level `_curve_count`), and
exhaustion warns exactly once before silently dropping subsequent
curves. `log_metric` is unaffected — only `log_curve` consumes the
budget.

---

## 6. Configuration

**File: `canvit_pretrain/train/config.py`**

Four new fields:

```python
tracker: Literal["comet", "wandb", "none"] = "wandb"
"""Backend for parameter/metric/figure logging."""

wandb_project: str | None = None
"""W&B project name. Required when tracker='wandb'."""

wandb_entity: str | None = None
"""W&B entity (team or user). Falls back to your default account when unset."""

wandb_dir: Path | None = Path("/mnt/vast-nhr/projects/nib00021/jonathan")
"""Directory wandb writes its run files into. None = wandb's own default (./wandb)."""
```

- `tracker` is a `Literal["comet", "wandb", "none"]` so `tyro` validates
  it at CLI parse time — typos fail loudly.
- Default is `"wandb"`. (Selection rationale is project-specific; the
  important bit is that the default fully drives a working run when
  `WANDB_API_KEY` is set and `wandb_project` is provided.)
- `wandb_dir` defaults to a project-shared path so multiple users on
  the same project see consistent local wandb state. Set to `None` to
  use wandb's own working-directory default.

The Comet path requires no new config — Comet picks up its API key from
the standard `COMET_API_KEY` env var the way it always did.

CLI invocation examples:

```bash
# wandb (default)
... --tracker wandb --wandb-project canvit-pretrain
# comet
... --tracker comet
# disable
... --tracker none
```

---

## 7. Checkpoint integration

**File: `canvit_pretrain/checkpoint/__init__.py`**

`CheckpointData` (the typed-dict for `latest.pt` content) gained both
tracker IDs as independent fields:

```python
class CheckpointData(TypedDict):
    ...
    comet_id: str | None
    wandb_run_id: str | None
    ...
```

Both are stored independently — switching backends across resumes
doesn't lose continuity on the original side; the next resume will
ignore the inactive backend's ID.

`save(...)` accepts both as kwargs and writes them through. The training
loop's two save sites pull both IDs from the active `Tracker` and pass
them in:

```python
save_checkpoint(
    ckpt_path, core_model, ...,
    comet_id=exp.get_comet_id(),
    wandb_run_id=exp.get_wandb_id(),
    ...
)
```

Whichever backend was inactive during this run returns `None` from its
accessor; the inactive slot is therefore `None` in the checkpoint. If a
future resume picks the *other* backend, it sees `None` for its own ID
and falls through to the "create new" branch (§3.2 / §3.3).

`load(...)` reads both fields with `raw.get(...)` so older checkpoints
that predate the new fields still load (their `comet_id` / `wandb_run_id`
keys are absent, `.get()` returns `None`, and the resume logic creates a
new run on whatever backend is active).

---

## 8. Loop integration

**File: `canvit_pretrain/train/loop.py`**

A single dispatch site, replacing whatever inline tracker-creation block
the pre-abstraction codebase had:

```python
from .tracker import make_tracker

# Non-main ranks (and tracker='none') get a no-op Tracker so all `exp.log_*`
# calls below remain unconditional.
exp = make_tracker(
    tracker=cfg.tracker,
    is_main=ddp.is_main(),
    is_seeding=is_seeding,
    run_name=run_name,
    wandb_project=cfg.wandb_project,
    wandb_entity=cfg.wandb_entity,
    wandb_dir=cfg.wandb_dir,
    prev_comet_id=prev_comet_id,
    prev_wandb_id=prev_wandb_id,
)

exp.log_parameters(flatten_dict(asdict(cfg)))
```

After this point, the rest of `loop.py` and all of `viz/*.py` use only
the `exp.*` surface. No `comet_ml` or `wandb` symbols appear elsewhere
in the codebase. Specifically:

- `exp.log_metric(name, value, step=step)` for individual metrics.
- `exp.log_metrics(metric_dict, step=step)` at the end of each step.
- `exp.log_image(pil, name=name, step=step)` (rarely used directly —
  most figure logging goes through `log_figure`).
- `log_figure(exp, fig, name, step)` for matplotlib figures.
- `log_curve(exp, name, x=x, y=y, step=step)` for line plots.
- `exp.get_comet_id()` / `exp.get_wandb_id()` at save sites.
- `exp.end()` at end of `train()`.

`prev_comet_id` and `prev_wandb_id` come from the loaded checkpoint
(both `None` for fresh runs).

`is_seeding` is the same flag the rest of the loop uses to distinguish
"resume from latest checkpoint" from "initialize from a seed checkpoint";
it controls SEED-mode behavior in `make_tracker` per §3.4.

---

## 9. Environment requirements

**Comet**: `COMET_API_KEY` set in the environment. Comet's library
picks it up automatically from there.

**wandb**:

- `WANDB_API_KEY` set in the environment (typical workflow: store the
  key in `~/wandb_api_key.txt`, source it via the shell rc).
- On networks with restricted egress (e.g. compute clusters behind a
  proxy), set `HTTPS_PROXY` so `wandb.init` can reach the wandb backend.
  In this codebase that's
  `export HTTPS_PROXY=http://www-cache.gwdg.de:3128` for Grete.
- `wandb.init` writes local run files under `wandb_dir` (or `./wandb` if
  unset). Make sure the path is writable.

**`tracker="none"`**: no environment requirements.

---

## 10. Reproduction checklist

Starting from a codebase with hardcoded inline tracker setup
(typically a single `comet_ml.Experiment(...)` or `wandb.init(...)`
call followed by direct `exp.log_*` calls):

**New file:**

- [ ] `canvit_pretrain/train/tracker.py` (§2 + §3) — the `Tracker`
      class plus `make_tracker(...)` factory plus `_curve_as_pil` helper.

**Edits to `canvit_pretrain/train/viz/comet.py`** (or whatever module
holds your figure-logging helpers):

- [ ] Re-type the `exp` parameter from `comet_ml.Experiment` to
      `Tracker` (§4).
- [ ] In `log_figure`, render once to `PIL.Image` and call
      `exp.log_image(img, name=name, step=step)` so the same code path
      handles both backends (§4).
- [ ] In `log_curve`, enforce the `_CURVE_BUDGET` cap and call
      `exp.log_curve(name, x=x, y=y, step=step)` (§5).
- [ ] If your codebase has any other place that imports `comet_ml`
      directly (or `wandb` directly), retype to `Tracker`.

**Edits to `canvit_pretrain/train/config.py`:**

- [ ] Add `tracker: Literal["comet", "wandb", "none"]` field (§6).
- [ ] Add `wandb_project: str | None`, `wandb_entity: str | None`,
      `wandb_dir: Path | None` fields (§6).

**Edits to `canvit_pretrain/checkpoint/__init__.py`** (or wherever
`CheckpointData` lives):

- [ ] Add `comet_id: str | None` and `wandb_run_id: str | None` to the
      typed-dict (§7).
- [ ] Add matching kwargs to `save(...)` (default `None`).
- [ ] Use `raw.get("comet_id")` and `raw.get("wandb_run_id")` in
      `load(...)` so pre-change checkpoints still load (§7).

**Edits to `canvit_pretrain/train/loop.py`:**

- [ ] `from .tracker import make_tracker`.
- [ ] Replace the inline tracker-creation block with a single
      `make_tracker(...)` call, passing `is_main=ddp.is_main()`,
      `is_seeding`, `run_name`, the wandb config fields, and
      `prev_comet_id` / `prev_wandb_id` from the loaded checkpoint
      (§8).
- [ ] At both checkpoint save sites, pass
      `comet_id=exp.get_comet_id(), wandb_run_id=exp.get_wandb_id()`
      to `save_checkpoint(...)` (§7).
- [ ] Remove any pre-existing `if ddp.is_main():` guards around
      `exp.log_*` call sites — they're no longer needed (§1).
- [ ] Remove direct `import comet_ml` / `import wandb` from `loop.py`.

**`pyproject.toml`** (or your package config):

- [ ] Add `wandb` as a dependency (keep `comet-ml`).

**Environment / sbatch** (whatever launches your training):

- [ ] Export `WANDB_API_KEY` before the python invocation (§9).
- [ ] Export `HTTPS_PROXY` if your compute environment requires it
      (§9).
- [ ] Pass `--wandb-project ...` (and optionally `--wandb-entity ...` /
      `--wandb-dir ...`) on the python CLI when `tracker="wandb"`.

After all of the above, the training loop logs to the active backend
selected by `cfg.tracker`, gracefully no-ops on non-main DDP ranks,
and persists resume continuity across either backend (or both, if
checkpoints alternate between backends across runs).
