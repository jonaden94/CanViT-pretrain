# P5 — CUDA IN1k classification task (STARTED 2026-07-23)

Fresh CUDA implementation (master plan D2: the specialize TPU/XLA path is abandoned,
not ported), modeled on the `ade20k/` task package. Frozen-backbone linear probe
(default) or full finetune of `CanViTForImageClassification` over a glimpse rollout.

## Built + CPU-validated (6 tests in `in1k/test_in1k.py`, all green)

- **`in1k/config.py`** — `In1kConfig`: `mode` (frozen/finetune), rollout knobs
  (`n_timesteps`/`glimpse_px`/`canvas_grid`), train viewpoint policy (IID random,
  patcher-aware + `foveated_scale`), `eval_policy` (coarse_to_fine default), epoch-based
  optimization (`epochs`/`batch_size`/`peak_lr`/`warmup_epochs`), RandomResizedCrop+flip
  aug knobs, `resize_mode` (val, aspect-preserving), data dirs (env `IN1K_TRAIN_DIR` /
  `IN1K_VAL_DIR`). NUM_CLASSES=1000.
- **`in1k/rollout.py`** — reuses ade20k's `consumes_full_image` / `derive_glimpse_px` /
  `make_random_viewpoints` (they only touch `.canvit`, which the classifier also has);
  `rollout_cls_tokens` returns the per-timestep CLS token (frozen mode runs the backbone
  under no_grad); `eval_viewpoints` (coarse_to_fine / full / random). Identical glimpse
  routing to ade20k + canvit_eval/episode.py.
- **`in1k/metrics.py`** — CE loss + streaming top-1/5 (`TopKAccuracy`).
- **`in1k/data.py`** — TRAIN: WebDataset shards (`jpg` + `json {"label": int}`), decoded
  with train aug; DDP-safe epochs via `resampled=True` + `.with_epoch(n_images //
  (world_size·batch))` (each rank samples independently, same batch count → no uneven-
  shard hang; trades exact-once for statistical coverage). VAL: IN1k ImageFolder +
  canonical preprocess (no val shards in the no-features set).
  **Verified against the REAL shards** (`test_train_pipeline_decodes_real_shards`): decodes
  `jpg`+`json` → (image [B,3,S,S], labels ∈ [0,999]).
- **`in1k/train.py` + `__main__.py`** — DDP epoch loop. `clf.canvit` is stepped directly
  (per the rollout), so the classifier is NOT DDP-wrapped: params are broadcast once and
  grads AllReduced by hand (the P4b-scorer / head-√N pattern). Frozen trains LN+head,
  finetune trains all. Eval = argmax deploy over `eval_policy`, global top-1/5. Saves
  `best-hf` (HF format) + `best.pt`.
- **core `CanViTForImageClassification.from_pretrained_with_new_head`** (CanViT-PyTorch) —
  pretrained backbone + FRESH random LN+Linear head, the classifier-training constructor
  (the model only had the eval-time `from_pretrained_with_probe`). Mirrors the seg model's
  `from_pretrained_with_new_probe`.

## Data (this cluster)

Train: `/user/henrich1/u25995/jonathan/datasets/webdataset-imagenet-1k-no-features/train-shuffled`
(1,281,167 imgs, 313 shards × 4096, pre-resized 512², `jpg`+`json`). **No val split there**
— val needs an IN1k ImageFolder (`IN1K_VAL_DIR`, e.g. the ILSVRC `.../val` canvit_eval
uses via `IMAGENET_VAL`).

## Remaining (NOT done)

- **SLURM launcher** (`slurm/archive/in1k/…`) with commit pinning, `IN1K_TRAIN_DIR`/`IN1K_VAL_DIR`.
- **Val data** wired on-cluster + the **GPU acceptance run**: frozen-probe top-1/5 sane vs
  canvit_eval's frozen-probe baseline (`dinov3-vitb16-...-in1k-512x512-linear-clf-probe`).
- Single-GPU smoke first (amp/compile/rollout) before multi-GPU, same caveat as P4b.
- Not committed yet (WIP).
