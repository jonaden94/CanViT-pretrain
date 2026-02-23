# SA-1B Continual Pretraining

**Goal**: Continually pretrain CanViT-B flagship checkpoint (2M steps on IN21k @ 512px) at **1024px on SA-1B**.
First run: get loss on Comet, verify the model works at higher resolution.

**Branch**: `sa1b` (worktree: `~/code/CanViT-train-SA1B`)
**Last updated**: 2026-02-23 ~12:00 EST, commit `3539567`

---

## Architecture Decisions

### HF Seed Checkpoint
- **Source**: `canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02` on HF Hub
- **Format**: `config.json` + `model.safetensors` (239 keys) — NOT CheckpointData `.pt`
- **Config**: `additive` update mode, `enable_vpe=True`, `gate_bias_init=None`, `teacher_dim=768`
- **Solution**: `hf_seed_ckpt` config option. Downloads from HF, extracts state_dict + config, overrides `cfg.model`, proceeds with seed mode (step=0, fresh optimizer).
- **Grid size change**: 32 → 64. Only 6 standardizer keys mismatch (3 missing for "64", 3 unexpected for "32"). All 233 core weights load perfectly. Handled by existing `strict=False` + regex filter.
- **IMPORTANT**: `cfg.model` defaults are `convex` + `gate_bias_init=-2.0`. The HF model is `additive` + `gate_bias_init=None`. Using defaults crashes. The `hf_seed_ckpt` path MUST override `cfg.model` with the HF config.
- **Smoketested locally**: 2026-02-23, commit `3539567`. `from_pretrained()` → `.state_dict()` → `load_state_dict(strict=False)` → 0 core mismatches.

### Storage Model
| What | Where | Persistence |
|---|---|---|
| SA-1B tars (~10.5 GB each) | `~/projects/def-akrish/datasets/sa1b/tars/` (NFS) | Permanent |
| SA-1B feature shards (~65 GB each) | `$SA1B_FEATURES_DIR/sa1b/dinov3_vitb16/1024/shards/` (NFS) | Permanent |
| Extracted JPEGs (~10.3 GB/tar) | `$SLURM_TMPDIR/sa1b_images/` | Per-job |
| Checkpoints | `$CHECKPOINTS_DIR/{run_name}/` (NFS) | Permanent |

### Shard Loading Strategy
- Feature `.pt` shards on NFS, read with `mmap=True` (cheap, no copy needed).
- Images extracted from tars to SLURM_TMPDIR before training starts.
- `--feature-base-dir "$SA1B_FEATURES_DIR/sa1b"` → shards path auto-constructed: `{base}/dinov3_vitb16/1024/shards/`.
- `--feature-image-root "$SLURM_TMPDIR/sa1b_images"` → extracted JPEGs.
- Shard loader iterates all shards on NFS, starts at `start_shard`, processes `steps_per_job` steps.
- Only images from shards within [start_shard, start_shard + shards_per_job] are accessed.
- **No symlinks needed.** Just extract the right tars.

### Tar Extraction
- File count limit on def-akrish: 500K files. Permanent extraction is impossible (250 tars × 11K images = 2.8M files).
- SLURM_TMPDIR: 3 TB shared across 8 GPUs per node. Conservative budget: ~375 GB.
- Each tar = ~10.3 GB of JPEGs. Budget for ~28 tars per job.
- `tar xf $TAR_PATH --strip-components=1 -C $SLURM_TMPDIR/sa1b_images/ '*.jpg'`
- Extraction is sequential; network-bound (~5 min for 28 tars from NFS).

### Job Array Design
- `%1` concurrency (1 job at a time, like IN21k).
- `steps_per_job` = multiple of `batches_per_shard`. batches_per_shard = 11186 // 64 = **174**.
- 174 × 28 = **4872 steps/job = 28 shards = 28 tars** (~288 GB extracted).
- A helper script (`sa1b/plan_job.py`) reads the checkpoint to determine start_step, outputs tar indices.
- Robust to failures: if task N fails mid-job, restart reads same checkpoint, re-extracts same tars.

### Key Numbers
| Metric | Value | Source |
|---|---|---|
| Images per tar | ~11,186 | tar tvf on sa_000020 |
| JPEG size per tar | ~10.3 GB | tar tvf estimate |
| Feature shard size | ~65-70 GB | sa_000020.pt on def-areynaud |
| batch_size | 64 | config default |
| batches_per_shard | 174 | 11186 // 64 |
| shards_per_job | 28 | 4872 / 174 |
| steps_per_job | 4872 | 174 × 28 |
| SLURM_TMPDIR budget | ~375 GB | 3 TB / 8 GPUs |
| Extracted size/job | ~288 GB | 28 × 10.3 |
| Canvas grid size | 64 | 1024 / 16 |
| Scene resolution | 1024px | Target for SA-1B |
| VRAM estimate | <80 GB | 18.4 GB @ 512/32 × ~4x, plus constant factors |

---

## What's DONE

### Export Pipeline (verified on MIG slice, NOT yet via sbatch)
- `sa1b/export_features.py` — 1 tar → 1 shard. Atomic save. Idempotent.
- `sa1b/export_features.sh` — SLURM array job. `gpu:h100:1`, 96G RAM, 16 CPUs, 10 min.
- `sa1b/submit_export.sh` — Auto-submits missing shards. `--dry-run` supported.
- First shard exported: `sa_000020.pt` (70,394 MB, 11186 images).

### Standardizer Unification (commit `21e9bbc`)
- Training loop uses `model.standardizers(G)` — standardizer state travels with model state_dict.
- `strict=False` on state_dict load + regex validation.

### Download
- `sa1b/download.py` running on Nibi (tmux `sa1b-dl`). ~73 MB/s.
- Non-contiguous tar indices. submit_export.sh handles this.

### Env Setup
- `.envrc.nibi` committed. `SA1B_TAR_DIR`, `SA1B_FEATURES_DIR`, `CANVIT_FLAGSHIP_CKPT` defined.
- `sa1b/sa1b_links.tsv` committed (1000 image tars).
- `sa1b/build_parquet.py` committed.

---

## What's PENDING

### 1. `hf_seed_ckpt` support — IN PROGRESS
- Add `hf_seed_ckpt: str | None = None` to `Config`.
- In `loop.py`: download from HF, extract state_dict + config, override `cfg.model`, seed mode.
- Mutually exclusive with `seed_ckpt`.

### 2. SA-1B training sbatch — NOT WRITTEN
- `sa1b/train.sh`: SLURM array job.
- Preamble: run `sa1b/plan_job.py` to get tar list, extract to SLURM_TMPDIR.
- CLI args: `--hf-seed-ckpt`, `--canvas-patch-grid-size 64`, `--scene-resolution 1024`, `--reset-normalizer`, `--steps-per-job 4872`, `--dataset sa1b`.

### 3. `sa1b/plan_job.py` — NOT WRITTEN
- Reads checkpoint (if any) to determine start_step.
- Computes which shards/tars this job needs.
- Outputs tar indices (zero-padded).

### 4. Submit export jobs
- Need enough tars downloaded (≥28 for first training job).
- `bash sa1b/submit_export.sh` on Nibi.

### 5. First training run
- Submit with `--array=0-0%1` (single task, quick test).
- Monitor Comet for loss.
- Check VRAM usage.

---

## Tensions / Open Questions

1. **Image size vs export size**: Teacher features are exported at 1024px. Student sees images at `scene_resolution` (also 1024px). Could dissociate later (e.g., student at 512px with 1024px features). Deferred for first run.
2. **Shuffling**: SA-1B images within a tar are geographically correlated (contiguous IDs = same region). No shuffle for now — verified visually that 4 sequential images look diverse enough. Revisit if loss curves show issues.
3. **Variable shard sizes**: `ShardedFeatureLoader` assumes uniform shard sizes (line 132). SA-1B shards are ~11,186 but not guaranteed identical. Should be fine for first run.
4. **Batch size at 1024px**: 64 @ 512px uses 18.4 GB. At 1024px/grid64, canvas is 4x larger. ~50-70 GB estimated. Should fit H100 80GB. May need to reduce if OOM.

---

## Changelog

| Date | Commit | What |
|---|---|---|
| 2026-02-23 | (pending) | Add `hf_seed_ckpt` config + loop support, plan_job.py, train.sh |
| 2026-02-23 | `3539567` | Add --max-concurrent to submit_export.sh |
| 2026-02-22 | `21e9bbc` | Unify standardizers: model.standardizers(G) in loop |
| 2026-02-22 | various | Export pipeline (export_features.py/.sh, submit_export.sh) |
