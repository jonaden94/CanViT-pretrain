# P2 — ADE20K task on the stable wrapper (CODE COMPLETE 2026-07-22; gate run pending)

## Status

All P2 code is written and smoke-tested. What remains is the **gate run** (§Gate
below) — a SLURM job the user submits. Suite: 65 passed; parity digest unchanged
(`9a0100a1…`).

## Delivered

- **`canvit_pretrain/ade20k/`** (new package): `config.py` (Ade20kConfig, tyro),
  `data.py` (dataset + dinov3 aug + loaders + AdamW/WarmupOneCycleLR + amp),
  `metrics.py` (ce_loss, mIoUAccumulator, ProbeState, eval_probe_on_batch),
  `rollout.py` (patcher-aware `rollout_canvas_hidden` on the WRAPPER —
  `consumes_full_image` routing + `derive_glimpse_px` token guard, both from
  canvit_eval), `train.py` (frozen-backbone loop, pretrain Tracker, per-t scalars),
  `__main__.py` (`python -m canvit_pretrain.ade20k`, tyro CLI verified).
- **Tests** (`ade20k/test_ade20k.py`, 3): uniform pre-crop rollout trains probe /
  frozen backbone untouched; **foveated full-image rollout trains probe — the NEW
  capability specialize never had**; glimpse_px token-count guard fails loudly.
- **Launcher**: `slurm_nhr/ade20k/train_ade20k.sbatch` — single A100, MODEL_REPO
  (hub id or local HF dir) + ADE20K_ROOT env, **commit pinning included** (closes
  unification-status §5.6 for ADE runs; specialize never had it).

## Deliberate deltas vs specialize (recorded, all per plan decisions)

- Wrapper-only: loads via `from_pretrained_with_new_probe` (core commit `2759e18`);
  no raw pretraining model, no DINOv3 teacher dependency (was only needed for
  recon_normalized dims — dropped per D3).
- `features` list / recon_normalized / FeatureSpec machinery: dropped (D3).
- Legacy `finetune` / `init_probe_repo` / `save_resume_state` branches: not ported
  (full-FT arrives via the harness in P4; resume can be re-added if wanted).
- Tracker: pretrain's `Tracker` (wandb scalars per timestep); specialize's
  `log_curve`-renders-PNG-and-uploads path and `log_viz` image logging NOT ported
  (unification-status §5.5 — avoids re-importing the matplotlib/wandb leak).
- Viewpoints: core `random_viewpoints` (identical L²-safe-box law; same call
  specialize made) — val always starts full-scene, train per `train_start_full`.

## Gate (pending — user submits)

Reference: the specialize ADE probe run on exp22-uniform16 best checkpoint
(`.../exp22-uniform16/checkpoints/step-1515520-hf`, specialize launcher
`train_ade20k_uniform16_best.sbatch`, launched 2026-07-22). Command:

```bash
export MODEL_REPO=/user/henrich1/u25995/jonathan/repos/CanViT-pretrain/logs/jon_exp22_full_runs/exp22-uniform16/checkpoints/step-1515520-hf
export ADE20K_ROOT=/user/henrich1/u25995/jonathan/datasets/zhoubolei--scene_parse_150/ADEChallengeData2016
export PRETRAIN_COMMIT=<this commit>  PYTORCH_COMMIT=<core HEAD>  # optional pin
sbatch --export=ALL slurm_nhr/ade20k/train_ade20k.sbatch
```

Pass criterion: per-timestep val mIoU within run-to-run noise of the specialize
reference run (same checkpoint, same protocol). Foveated capability: a second run
on an exp22-fovi checkpoint must train end-to-end (no reference numbers — new).

## Done so far

1. **Core** (`canvit_pytorch`, commit `2759e18`):
   `CanViTForSemanticSegmentation.from_pretrained_with_new_probe(pretrained_repo,
   num_classes, dropout, use_ln)` — loads the pretrained backbone (shared
   `_from_pretrained_backbone` helper, same bare-weight copy as
   `from_pretrained_with_probe`) with a FRESH probe head, for probe training on the
   wrapper (D3). Core suite green (12 passed / 52 hub-or-GPU skips).

## Remaining P2 work (in order)

2. **Port the ADE task into `canvit_pretrain/ade20k/`** from
   `canvit_specialize/training/ade20k/` + `canvit_specialize/datasets/ade20k.py`:
   - `data.py` — the ADE20kDataset + train/val transforms (faithful port; the P2
     gate is reproducing exp22 probe numbers, so keep augmentation identical).
   - `rollout.py` — patcher-aware canvas-feature rollout on the wrapper:
     `consumes_full_image = isinstance(seg.canvit.patcher, (FoveatedPatcher,
     SquarePatcher))` → full image; else pre-crop via `sample_at_viewpoint` with
     `glimpse_px` derived as in `canvit_eval/episode.py:69-92` (patch size/stride ×
     glimpse_grid_size, token-count hard guard). Steps `seg.canvit(...)` directly,
     collects `get_spatial(state.canvas)` per timestep (head applied by trainer —
     wrapper docstring blesses head-only on cached state). `canvas_hidden` ONLY
     (`recon_normalized` dropped per D3).
   - `train.py` — entry `python -m canvit_pretrain.ade20k.train`: frozen backbone,
     per-timestep probe CE (mean over timesteps), AdamW + warmup, per-timestep val
     CE/mIoU, wandb scalars (pretrain conventions; specialize's wandb *image*
     logging is NOT ported — disk viz can come later), checkpoint + HF-format probe
     export (port `probe_ckpt_to_hf_format` logic).
   - viewpoints: RandomSelector-based (FULL t0 when train_start_full, RANDOM rest) —
     same safe-box law as specialize's `random_viewpoints` (verify min/max knobs
     match specialize's `min_vp_scale`/`max_vp_scale`).
   - port `metrics.py` (mIoU accumulator) from specialize.
3. **CPU smoke tests** in pretrain: uniform (pre-crop) + foveated (full-image)
   tiny-model rollouts through the new task; foveated is the NEW capability —
   assert it runs end-to-end and the probe gets grads.
4. **Launcher**: `slurm_nhr/` sbatch for ADE probe training with commit pinning
   (specialize never had pinning — this closes unification-status §5.6).
5. **Gate**: reproduce specialize's exp22 ADE probe numbers (uniform checkpoint)
   within run-to-run noise — needs a SLURM run (user submits); foveated probe
   trains end-to-end on GPU (new capability, no reference numbers).
6. p2-notes final update + master plan row + commit.

## Port facts read from specialize (2026-07-22, verbatim from source)

- **Config** (`training/ade20k/config.py`): `ProbeTrainBase`: ade20k_root (env
  `ADE20K_ROOT` → `$SLURM_TMPDIR/ADEChallengeData2016` → /datasets fallback),
  scene_size=512, batch_size=16, eval_batch_size=32, num_workers=4, peak_lr=3e-4,
  weight_decay=1e-3, warmup_steps=1500, warmup_lr_ratio=1e-6, max_steps=40000,
  grad_clip=inf, dropout=0.1, aug_scale_range=(0.5,2.0), aug_flip_prob=0.5,
  log_every=20, val_every=500, amp=True, tracker=wandb. `Config`: model_repo
  (canvitb16-add-vpe...-2026-02-02), features=["canvas_hidden"], n_timesteps=10,
  glimpse_px=128, canvas_grid=None (derive), min_vp_scale=0.05, max_vp_scale=1.0,
  train_start_full=False, finetune=False (legacy). DROP in port: recon_normalized
  machinery (CanvasFeatureType/FeatureSpec/teacher_repo — D3), finetune legacy path
  (decide: port frozen-probe path only; full-FT comes via the harness later).
- **Dataset** (`datasets/ade20k.py`): `ADE20kDataset(root, split, img_transform,
  mask_transform, joint_transform)` — images/`split`/*.jpg + annotations/`split`/
  *.png; separate transforms (eval; mask: -1 shift, <0 → IGNORE_LABEL=255) or
  joint_transform (train aug — defined in train_canvit.py, STILL TO READ).
  `make_val_transforms(size, mode)`: squish = Resize((s,s)) bilinear+NEAREST for
  mask; NUM_CLASSES=150, IGNORE_LABEL=255, timm IMAGENET mean/std.
- **Still to read before writing the port**: train_canvit.py joint train transform
  (aug: scale jitter (0.5,2.0) + flip 0.5 + crop to scene_size presumably),
  warmup/scheduler construction, val loop + mIoU accumulator (metrics.py),
  probe checkpoint save format, tracker usage; `training/utils.py make_viewpoints`.

## Facts pinned for the port (verified today)

- specialize ADE trainer details: probe = `SegmentationProbe(dim, 150, dropout,
  use_ln)`; loss = `ce_loss` (nearest-resize masks → CE ignore_index=255); per-step
  `n_timesteps=10` viewpoints via `make_viewpoints("random", ...)` (core
  `random_viewpoints`, same L²-safe-box law as pretrain's `Viewpoint.random`);
  backbone frozen via `requires_grad_(False)`; clip on head params only.
- `canvit_eval` glimpse_px derivation: `(grid-1)*stride + patch_size`, grid from
  `model.glimpse_grid_size` (fallback 8), assert `(glimpse_px - patch) % stride == 0`.
- ADE20K data root: specialize reads its own dataset dir; RL repo used
  `canvit_eval.config.ade20k_root` — unify on ONE source in the port (check both
  point at the same files before the gate run).
