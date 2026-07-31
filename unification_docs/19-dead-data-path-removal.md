# 19 — Removing the dead precomputed-feature tar training path

**Date:** 2026-07-31. **Nature:** deletion of a dead code path (−585 lines, +35).

## Read this before touching distill's data loading: "raw images" means three things

The request that triggered this was "get rid of the raw-image training path" — and that
phrase does not map onto one thing. Getting it wrong deletes the path production runs use.

| | what it is | status |
|---|---|---|
| **1. Raw WebDataset shards** | `{webdataset_dir}/train-shuffled/shard-*.tar` carrying **jpg+json only**; the frozen DINOv3 teacher computes targets **on the fly**. Goes through `WebDatasetTrainLoader` (`has_features=False`). | **LIVE — the exp21/exp22 path.** Validated by `harness_run_raw_shards.py`. Never remove. |
| **2. Feature WebDataset shards** | Same loader, shards additionally carry `cls.npy` + `ptch.npy`, so no teacher forward per batch (`has_features=True`). | **LIVE — the current default** (`WEBDATASET_DIR`). |
| **3. Precomputed-feature *tar* path** | A separate `ShardedFeatureLoader` reading `{feature_base_dir}/{teacher}/{res}/shards/` plus mmap'd image tars via `tar_images.py`. Reached only when `cfg.webdataset_dir is None`. | **DEAD — removed here.** |

(1) and (3) both involve "raw images" in some sense, which is exactly the trap. What made
(3) safe to delete is not its name but that **nothing can reach it**:
`harness_train.sbatch:134` requires `WEBDATASET_DIR` with `:?`, so `webdataset_dir` is
never `None` in any launched run, and no `.idx` files (which `tar_images.py` needs) exist
on disk anywhere.

## Removed

- `distill/data/shards.py` (296 ln) — `ShardedFeatureLoader`
- `distill/data/tar_images.py` (110 ln) — `TarImageReader`, `TarIndex`, `load_tar_index`,
  `scan_tar_headers`
- `scripts/bench_dataloader.py` (108 ln) — benchmarked only `ShardedFeatureLoader`
- `distill/data/__init__.py` — `_create_sharded_train_loader`, the `create_loaders`
  if/else dispatch, and the now-unused `start_step` parameter (its only consumer was the
  sharded loader's shard cursor; the WebDataset path resumes via `job_index`)
- `Config` fields: `train_dir`, `feature_base_dir`, `feature_image_root`, `tar_dir`

`create_loaders` now **asserts** `webdataset_dir is not None` with a message naming this
removal, rather than silently falling through to a loader that no longer exists.

## Kept, despite looking removable

- **`Config.train_index_dir`** — reads like a training knob; it is actually the fallback
  source for `val_index_dir` (`distill/data/__init__.py:172-174`), i.e. it feeds
  **validation's** parquet index. Docstring now says so.
- **`Config.webdataset_dir` stays typed `Path | None`** rather than becoming a required
  field. Making it required would break ~5 test sites that construct a bare `Config()`;
  the assert gives the same guarantee without that churn. Same reasoning as `val_dir`.
- **`In1kConfig.train_dir`** — a *different, live* field. in1k's flag count is unchanged
  (107), which is the evidence it was not touched.

## Verification

A deletion is not a move, so the digest gates alone are not sufficient — the full suite
was re-run rather than skipped.

| gate | result |
|---|---|
| full test suite | 307 passed (unchanged count) |
| distill parity digest | `9a0100a1a3de3acd` unchanged |
| 4 ade20k/in1k pinning digests | unchanged |
| `ruff --select F,E9` | 8 findings, byte-for-byte the pre-existing set — no new dead imports |
| CLI flag surface | distill **228 → 224**: exactly `--cfg.train-dir`, `--cfg.feature-base-dir`, `--cfg.feature-image-root`, `--cfg.tar-dir`, nothing else. ade20k 108 and in1k 107 unchanged. |
| `create_loaders(Config())` | raises the new assert instead of dispatching |
| launchers / `.envrc.grete` | no reference to any removed knob (`FEATURES_DIR`, `TAR_DIR`, `CFG_TRAIN_DIR`, …) |

The flag-surface delta is the strongest single check: it proves the removal touched exactly
the four intended knobs and no other part of the CLI contract, so no launcher can break.
