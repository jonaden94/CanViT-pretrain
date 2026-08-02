"""Evaluate a trained ADE20K probe/finetune checkpoint under a chosen viewpoint policy.

A training run evaluates under ONE policy -- whatever `eval_policy` resolved to for that
run -- and writes that curve to its log. This re-evaluates a finished checkpoint under a
DIFFERENT policy without retraining, which is how you answer "what would this model score
under coarse-to-fine?".

Why not CanViT-eval: its ADE20K task builds the model with
`from_pretrained_with_probe(pretrained_repo, probe_repo)`, i.e. it needs the probe as a
published HF repo. A harness checkpoint carries backbone+head in one .pt and
`canvit_train.checkpoint.to_hf` refuses ADE20K payloads ("only a distill checkpoint can be
converted to the pretraining HF layout"). This driver therefore rebuilds the same model
through the ADE20K task and loads the checkpoint's weights into it. The metric code is
shared: `preds_from_logits` upsamples logits bilinearly and argmaxes at full resolution,
the same reduction canvit_eval/tasks/ade20k_seg.py uses (fixed in 68b635f).

    python scripts/eval_ade20k_checkpoint.py \
        --ckpt logs/exp34_ade20k_probe/<run>/checkpoints/best.pt \
        --model-repo <the pretrained backbone dir the probe was trained on> \
        --eval-policy coarse_to_fine --n-timesteps 21 --out results.json

FOVEATED backbones need `--override-scale`. `coarse_to_fine` is a quadtree over scales
{1.0, 0.5, 0.25}; a fixed-scale foveated model derives its fixation window as
`fix_size = scale * H`, so a glimpse at a scale it never trained on is out of distribution
and mIoU decays as glimpses accumulate. Passing `--override-scale <pretrain scale>` keeps
C2F's CENTERS and pins the scale, which is what canvit_eval's `override_scale` does and
the only way the number describes the model rather than the mismatch.
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from canvit_train.ade20k.config import Ade20kConfig
from canvit_train.ade20k.task import Ade20kRunTask


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--model-repo", required=True,
                   help="Pretrained backbone the probe was trained on (HF dir or hub id). "
                        "Taken from the run's launcher; the checkpoint records it too, but "
                        "under whichever path spelling the submitting user had.")
    p.add_argument("--eval-policy", default="coarse_to_fine")
    p.add_argument("--n-timesteps", type=int, default=21)
    p.add_argument("--fixed-scale", type=float, default=None,
                   help="Foveated pretraining view scale. Required for a foveated backbone "
                        "under a scale-aware policy (random / fixation_grid).")
    p.add_argument("--override-scale", type=float, default=None,
                   help="Pin every glimpse to this scale, keeping the policy's CENTERS "
                        "(canvit_eval's override_scale). Use the pretraining view scale "
                        "when deploying a fixed-scale foveated backbone under C2F.")
    p.add_argument("--resize-mode", default="squish", choices=["squish", "center_crop"])
    p.add_argument("--canvas-grid", type=int, default=32)
    p.add_argument("--scene-size", type=int, default=512)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit-val-batches", type=int, default=None, help="smoke-test knob")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = Ade20kConfig(
        model_repo=args.model_repo, scene_size=args.scene_size, canvas_grid=args.canvas_grid,
        n_timesteps=args.n_timesteps, eval_policy=args.eval_policy,
        resize_mode=args.resize_mode, eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers, limit_val_batches=args.limit_val_batches,
        eval_override_scale=args.override_scale, tracker="none",
    )
    if args.fixed_scale is not None:
        cfg.foveated_scale.mode = "fixed"
        cfg.foveated_scale.fixed_scale = args.fixed_scale

    task = Ade20kRunTask(cfg)
    model, _ = task.build_model(device)

    payload = torch.load(args.ckpt, weights_only=False, map_location=device)
    # Strict: a silently-missing head would still produce plausible-looking mIoU, just
    # from the freshly-built probe rather than the trained one.
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()

    _, val_loader = task.build_loaders(world_size=1, rank=0)
    metrics = task.evaluate(model=model, head=model.head, val_loader=val_loader,
                            device=device, step=int(payload.get("step", 0)))

    record = {
        "ckpt": str(args.ckpt), "step": payload.get("step"),
        "model_repo": args.model_repo, "eval_policy": args.eval_policy,
        "n_timesteps": args.n_timesteps, "fixed_scale": args.fixed_scale,
        "override_scale": args.override_scale,
        "resize_mode": args.resize_mode, "canvas_grid": args.canvas_grid,
        "limit_val_batches": args.limit_val_batches,
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    print(json.dumps(record, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
