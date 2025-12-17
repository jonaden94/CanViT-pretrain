"""Minimal script to visualize AVP output on a single image."""

import copy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dinov3.hub.backbones import dinov3_vits16
from matplotlib.patches import Rectangle
from PIL import Image
from PIL.Image import Resampling
from sklearn.decomposition import PCA
from ytch.device import get_sensible_device

from avp_vit import AVPConfig, AVPViT
from avp_vit.backbone.dinov3 import DINOv3Backbone
from avp_vit.glimpse import Viewpoint
from avp_vit.train.data import IMAGENET_MEAN, IMAGENET_STD


@dataclass
class Args:
    ckpt: Path
    image: Path
    teacher_ckpt: Path = Path("dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    n_glimpses: int = 5


def load_image(path: Path, size: int, device: torch.device) -> torch.Tensor:
    """Load and preprocess image to [1, 3, H, W]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Resampling.BILINEAR)
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0).to(device)


def make_viewpoints(n: int, device: torch.device) -> list[Viewpoint]:
    """Full scene + quadrants."""
    vps = [Viewpoint.full_scene(1, device)]
    quads = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for qx, qy in quads[: n - 1]:
        vps.append(Viewpoint.quadrant(1, device, qx, qy))
    return vps[:n]


def pca_rgb_sigmoid(features: np.ndarray, H: int, W: int) -> np.ndarray:
    """PCA -> RGB via sigmoid."""
    pca = PCA(n_components=3, whiten=True)
    proj = pca.fit_transform(features)
    return 1.0 / (1.0 + np.exp(-proj.reshape(H, W, 3) * 2.0))


def pca_rgb_normalized(features: np.ndarray, H: int, W: int) -> np.ndarray:
    """PCA -> RGB, centered and normalized to [0, 1]."""
    pca = PCA(n_components=3, whiten=True)
    proj = pca.fit_transform(features).reshape(H, W, 3)
    # Per-channel min-max normalization
    for c in range(3):
        cmin, cmax = proj[:, :, c].min(), proj[:, :, c].max()
        if cmax > cmin:
            proj[:, :, c] = (proj[:, :, c] - cmin) / (cmax - cmin)
        else:
            proj[:, :, c] = 0.5
    return proj


def spatial_std(features: np.ndarray, H: int, W: int) -> np.ndarray:
    """Per-token std across feature dimension, normalized for viz."""
    std = features.std(axis=1).reshape(H, W)
    std = (std - std.min()) / (std.max() - std.min() + 1e-8)
    return std


def main(args: Args) -> None:
    device = get_sensible_device()

    # Load backbone for architecture
    backbone = DINOv3Backbone(
        dinov3_vits16(weights=str(args.teacher_ckpt), pretrained=True).eval()
    )

    # Load checkpoint to get grid size
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    G = ckpt.get("current_grid_size", 16)

    # Create AVP and load weights
    cfg = AVPConfig(
        glimpse_grid_size=7,
        n_scene_registers=32,
        gating="full",
    )
    avp = AVPViT(copy.deepcopy(backbone), cfg, teacher_dim=backbone.embed_dim).to(device)
    avp.load_state_dict(ckpt["avp"])
    avp.eval()

    # Load image
    scene_px = G * backbone.patch_size
    img = load_image(args.image, scene_px, device)

    # Run forward
    vps = make_viewpoints(args.n_glimpses, device)
    with torch.inference_mode():
        hidden = avp.init_hidden(1, G)
        outputs, _ = avp.forward_trajectory_full(img, vps, hidden)

    # Visualize: image + trajectory + PCA variants + std
    final_scene = outputs[-1].scene[0].cpu().numpy()  # [G*G, D]
    scene_sigmoid = pca_rgb_sigmoid(final_scene, G, G)
    scene_norm = pca_rgb_normalized(final_scene, G, G)
    scene_std = spatial_std(final_scene, G, G)

    # Denormalize image for display
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img_np = ((img[0].cpu() * std + mean).clamp(0, 1)).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Top-left: image with boxes
    ax = axes[0, 0]
    ax.imshow(img_np)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, len(vps) - 1)) for i in range(len(vps))]
    for i, vp in enumerate(vps):
        box = vp.to_pixel_box(0, scene_px, scene_px)
        rect = Rectangle(
            (box.left, box.top), box.width, box.height,
            linewidth=2, edgecolor=colors[i], facecolor="none",
        )
        ax.add_patch(rect)
        ax.plot(box.center_x, box.center_y, "o", color=colors[i], markersize=6)
    ax.set_title("Glimpse trajectory")
    ax.axis("off")

    # Top-right: PCA sigmoid
    axes[0, 1].imshow(scene_sigmoid)
    axes[0, 1].set_title(f"PCA sigmoid (G={G})")
    axes[0, 1].axis("off")

    # Bottom-left: PCA normalized
    axes[1, 0].imshow(scene_norm)
    axes[1, 0].set_title(f"PCA normalized (G={G})")
    axes[1, 0].axis("off")

    # Bottom-right: spatial std
    axes[1, 1].imshow(scene_std, cmap="inferno")
    axes[1, 1].set_title(f"Spatial std (G={G})")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import tyro
    main(tyro.cli(Args))
