#!/usr/bin/env python3
"""Multi-grid inference: run model at various scene grid sizes."""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from PIL import Image
from torchvision import transforms

from avp_vit import AVPConfig, AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone
from avp_vit.checkpoint import load as load_checkpoint
from avp_vit.glimpse import Viewpoint
from avp_vit.train.data import imagenet_normalize
from scripts.train_scene_match.model import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass
class Args:
    checkpoint: Path
    image: Path
    grid_sizes: tuple[int, ...] = (8, 16, 32, 64, 128)
    n_glimpses: int = 4
    backbone_weights: Path | None = None  # If None, use pretrained
    device: str = "mps"
    seed: int = 42


def load_model(ckpt_path: Path, backbone_weights: Path | None, device: torch.device) -> AVPViT:
    """Load AVP model from checkpoint."""
    ckpt = load_checkpoint(ckpt_path, device)
    cfg = AVPConfig(**ckpt["avp_config"])
    backbone_slug = ckpt["backbone"]
    teacher_dim = ckpt["teacher_dim"]

    # Load backbone
    factory = MODEL_REGISTRY[backbone_slug]
    if backbone_weights is not None:
        log.info(f"Loading backbone {backbone_slug} from {backbone_weights}")
        raw_backbone = factory(pretrained=True, weights=str(backbone_weights))
    else:
        log.info(f"Loading backbone {backbone_slug} with default pretrained weights")
        raw_backbone = factory(pretrained=True)

    backbone = DINOv3Backbone(raw_backbone.to(device).eval())
    for p in backbone.parameters():
        p.requires_grad = False

    avp = AVPViT(backbone, cfg, teacher_dim).to(device)
    avp.load_state_dict(ckpt["state_dict"])
    avp.eval()

    log.info(f"Model loaded: {backbone_slug}, teacher_dim={teacher_dim}")
    log.info(f"  glimpse_grid_size={cfg.glimpse_grid_size}, registers={cfg.n_scene_registers}")
    return avp


def load_image(path: Path, size: int, device: torch.device) -> torch.Tensor:
    """Load and preprocess image."""
    transform = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        imagenet_normalize(),
    ])
    pil_img = Image.open(path).convert("RGB")
    tensor = transform(pil_img)
    assert isinstance(tensor, torch.Tensor)
    return tensor.unsqueeze(0).to(device)


def random_viewpoints(n: int, batch_size: int, device: torch.device, seed: int) -> list[Viewpoint]:
    """Generate reproducible random viewpoints."""
    gen = torch.Generator(device=device).manual_seed(seed)
    viewpoints = []
    for _ in range(n):
        centers = torch.rand(batch_size, 2, generator=gen, device=device)
        scales = 0.3 + 0.7 * torch.rand(batch_size, 1, generator=gen, device=device)
        viewpoints.append(Viewpoint(name=f"random_{len(viewpoints)}", centers=centers, scales=scales))
    return viewpoints


def main(args: Args) -> None:
    device = torch.device(args.device)
    log.info(f"Device: {device}")

    avp = load_model(args.checkpoint, args.backbone_weights, device)
    patch_size = avp.backbone.patch_size

    # Determine image size: must be large enough for largest grid
    max_grid = max(args.grid_sizes)
    img_size = max_grid * patch_size
    log.info(f"Image size: {img_size}px (for max grid {max_grid}, patch_size={patch_size})")

    image = load_image(args.image, img_size, device)
    log.info(f"Loaded image: {args.image} -> {image.shape}")

    # Same viewpoints for all grid sizes (reproducible)
    viewpoints = random_viewpoints(args.n_glimpses, 1, device, args.seed)
    vp_info = [(vp.centers[0].tolist(), vp.scales[0].item()) for vp in viewpoints]
    log.info(f"Viewpoints ({args.n_glimpses}): {vp_info}")

    log.info(f"\n{'='*60}")
    log.info(f"Running inference at grid sizes: {args.grid_sizes}")
    log.info(f"{'='*60}\n")

    results: dict[int, dict] = {}

    with torch.no_grad():
        for G in args.grid_sizes:
            log.info(f"Grid size {G}x{G} ({G*G} spatial tokens)...")

            hidden = avp.init_hidden(1, G)
            outputs, final_hidden = avp.forward_trajectory_full(image, viewpoints, hidden)

            # Collect stats
            scenes = [out.scene for out in outputs]
            scene_norms = [s.norm().item() for s in scenes]
            final_scene = scenes[-1]

            results[G] = {
                "scene_shape": final_scene.shape,
                "scene_norms": scene_norms,
                "hidden_norm": final_hidden.norm().item(),
                "final_scene": final_scene.cpu(),
            }

            log.info(f"  scene shape: {final_scene.shape}")
            log.info(f"  scene norms per step: {[f'{n:.2f}' for n in scene_norms]}")
            log.info(f"  final hidden norm: {final_hidden.norm().item():.2f}")
            log.info("")

    # Summary
    log.info(f"{'='*60}")
    log.info("Summary:")
    log.info(f"{'='*60}")
    for G, r in results.items():
        log.info(f"  G={G:3d}: scene {r['scene_shape']}, final_norm={r['scene_norms'][-1]:.2f}")


if __name__ == "__main__":
    main(tyro.cli(Args))
