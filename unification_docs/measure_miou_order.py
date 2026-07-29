"""Measure the mIoU argmax/upsample ORDER effect on ONE checkpoint.

Both reductions read the SAME rollout and the SAME logits, so the only thing that
differs is how per-pixel labels are produced:

  A) "pretrain" (current): argmax at the 64x64 probe grid, then NEAREST-upsample
     the integer labels to 512x512.   -> canvit_pretrain/ade20k/metrics.py:35
  B) "paper"    (reference): BILINEAR-upsample the logits to 512x512, then argmax.
     -> canvit_eval/tasks/ade20k_seg.py:92 AND canvit_pytorch_rl/scoring.py:44

CE is also reported as a control: it takes its own bilinear path in both cases, so
it MUST come out identical. If CE moves, the harness is wrong, not the reduction.

Usage: python measure_miou_order.py <run_dir> [--limit N]
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from canvit_pytorch import CanViTForSemanticSegmentation
from canvit_pytorch.policy import StateEncoder, ViewpointScorer, head_logits

from canvit_pretrain.ade20k.data import IGNORE_LABEL, NUM_CLASSES, ADE20kDataset, make_val_transforms
from canvit_pretrain.ade20k.metrics import mIoUAccumulator, upsample_preds
from canvit_pretrain.ade20k.rl_train import (
    PolicyTrainConfig,
    advance_state,
    build_action_table,
    ce_from_logits,
    consumes_full_image,
    full_scene_state,
)

p = argparse.ArgumentParser()
p.add_argument("run_dir", type=Path)
p.add_argument("--ckpt", default="best.pt")
p.add_argument("--limit", type=int, default=0, help="0 = full val split")
p.add_argument("--batch-size", type=int, default=16)
args = p.parse_args()

device = torch.device("cuda")
cfg = PolicyTrainConfig(run_name="measure")          # canonical recipe defaults
assert cfg.resize_mode == "squish", cfg.resize_mode  # the band's protocol

ck = torch.load(args.run_dir / args.ckpt, map_location="cpu", weights_only=False)
print(f"ckpt step={ck['step']}  logged val_ce={ck['val_ce']:.4f}  "
      f"logged miou_per_t={[round(m, 4) for m in ck.get('val_miou_per_t', [])]}")

seg = CanViTForSemanticSegmentation.from_pretrained_with_probe(
    pretrained_repo=cfg.model_repo, probe_repo=cfg.resolved_probe_repo).to(device)
seg.eval().requires_grad_(False)

vp_flat, n_scale = build_action_table(seg, cfg)
vp_flat = vp_flat.to(device)
fixation = consumes_full_image(seg)
net = ViewpointScorer(
    canvas_dim=seg.canvas_dim, width=cfg.width, n_scale=n_scale,
    scales=(1.0,) if fixation else cfg.scales, centers_per_axis=cfg.centers_per_axis,
    block_layers=cfg.block_layers, groups=cfg.feature_groups, dueling=True,
    action_space="fixation" if fixation else "safebox").to(device)
net.load_state_dict(ck["net_state"])
net.eval()
encoder = StateEncoder(seg, canvas_grid=cfg.canvas_grid, feature_groups=cfg.feature_groups)

img_tf, mask_tf = make_val_transforms(cfg.scene_size, cfg.resize_mode)
ds = ADE20kDataset(root=cfg.ade20k_root, split="validation",
                   img_transform=img_tf, mask_transform=mask_tf)
if args.limit:
    ds = torch.utils.data.Subset(ds, range(args.limit))
loader = torch.utils.data.DataLoader(ds, args.batch_size, num_workers=4, pin_memory=True)
print(f"val images: {len(ds)}  batch={args.batch_size}  T=t1..t{cfg.train_horizon}")

H = cfg.train_horizon
acc_pretrain = [mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) for _ in range(H)]
acc_paper = [mIoUAccumulator(NUM_CLASSES, IGNORE_LABEL, device) for _ in range(H)]
ce_total, count = torch.zeros(H, dtype=torch.float64), 0

amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.amp)
t_start = time.time()
with torch.no_grad():
    for bi, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        with amp_ctx:
            st = full_scene_state(seg, images, canvas_grid=cfg.canvas_grid, glimpse_px=cfg.glimpse_px)
            logits = head_logits(seg, st.canvas, canvas_grid=cfg.canvas_grid)
        encoder.reset()
        for t in range(H):
            f = encoder(st, logits=logits).float()
            idx = net(f).reshape(images.shape[0], -1).argmax(dim=1)
            with amp_ctx:
                st = advance_state(seg, images, st, vp_flat[idx], cfg.glimpse_px)
                logits = head_logits(seg, st.canvas, canvas_grid=cfg.canvas_grid)

            # ---- the ONLY difference: order of argmax vs upsample ----
            acc_pretrain[t].update(
                upsample_preds(logits.argmax(1), masks.shape[1], masks.shape[2]), masks)
            up = F.interpolate(logits.float(), size=masks.shape[-2:],
                               mode="bilinear", align_corners=False)
            acc_paper[t].update(up.argmax(1), masks)

            ce_total[t] += ce_from_logits(logits, masks, score_res=None).double().sum().cpu()
        count += images.shape[0]
        if bi % 20 == 0:
            print(f"  batch {bi}/{len(loader)}  ({time.time() - t_start:.0f}s)", flush=True)

mi_pre = [a.compute() for a in acc_pretrain]
mi_pap = [a.compute() for a in acc_paper]
ce = (ce_total / count).tolist()

print(f"\nimages={count}  wall={time.time() - t_start:.0f}s")
print(f"{'t':>3} {'CE':>9} {'mIoU pretrain':>15} {'mIoU paper':>12} {'delta':>8}")
for t in range(H):
    print(f"t{t + 1:<2} {ce[t]:9.4f} {mi_pre[t] * 100:15.2f} {mi_pap[t] * 100:12.2f} "
          f"{(mi_pap[t] - mi_pre[t]) * 100:+8.2f}")
print(f"\nmean(t1..t{H}) CE = {sum(ce) / H:.4f}   (band 0.6853 +- 0.0007)")
print(f"mIoU t{H}: pretrain {mi_pre[-1] * 100:.2f}  paper {mi_pap[-1] * 100:.2f}  "
      f"(band {44.97:.2f} +- 0.10)")
