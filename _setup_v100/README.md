# V100 Environment Setup

Two virtual environments exist side by side:

| venv | torch build | target hardware | managed by |
|------|-------------|-----------------|------------|
| `.venv` | cu130 | H100 / A100 | `uv sync` (fully locked) |
| `.venv-v100` | cu124 | V100 | `uv sync` + manual torch overwrite |

The only difference is the torch/torchvision wheel. Everything else comes from the
same `uv.lock`.

---

## First-time setup

Run once from the repo root on a node that has internet access (or proxy configured):

```bash
bash _setup_v100/setup.sh
```

This:
1. Creates `.venv-v100`
2. Installs all dependencies from `uv.lock`
3. Overwrites torch/torchvision with cu124 builds

---

## Running training on a V100

```bash
bash slurm_nhr/train-v100.sh
# or with extra args:
bash slurm_nhr/train-v100.sh --run-name my-run
```

---

## After updating dependencies (new package added to pyproject.toml)

Re-run the setup script — it is fully idempotent:

```bash
bash _setup_v100/setup.sh
```

This re-syncs all deps from the updated lock file, then re-overwrites torch with cu124.

---

## Notes

- The torch version in `.venv-v100` may differ from `.venv` (cu124 vs cu130 build).
  All other packages are identical to the lock file.
- `.venv-v100` is gitignored — each machine needs its own setup.
- The code automatically detects GPU capability at runtime:
  - sm_80+ (A100/H100): FlashAttention + bfloat16 AMP
  - sm_70 (V100): standard SDPA + float16 AMP
