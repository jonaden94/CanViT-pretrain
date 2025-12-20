#!/usr/bin/env python3
"""Multi-grid inference: run model at various scene grid sizes with PCA viz."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from PIL import Image
from torchvision import transforms

from avp_vit import ActiveCanViT, ActiveCanViTConfig
from avp_vit.checkpoint import load as load_checkpoint
from canvit.backbone.dinov3 import DINOv3Backbone
from avp_vit.train.data import imagenet_normalize
from avp_vit.glimpse import Viewpoint
from avp_vit.train.loss import cos_dissim
from avp_vit.train.viewpoint import make_eval_viewpoints, random_viewpoint
from avp_vit.train.viz import fit_pca, pca_rgb, imagenet_denormalize, timestep_colors
from scripts.train_scene_match.model import MODEL_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


@dataclass
class Args:
    checkpoint: Path
    image: Path
    output: Path = Path("outputs/multi_grid_pca.png")
    grid_sizes: tuple[int, ...] = (8, 16, 32, 64, 128)
    viewpoint_mode: Literal["quadrants", "sixteenths", "random"] = "quadrants"
    n_viewpoints: int = 8
    hidden_ln: bool = False  # Apply LayerNorm to hidden spatial before viz
    backbone_weights: Path | None = None
    device: str = "mps"
    seed: int | None = None


def load_model(
    ckpt_path: Path, backbone_weights: Path | None, device: torch.device
) -> tuple[ActiveCanViT, DINOv3Backbone]:
    """Load model and teacher backbone from checkpoint."""
    ckpt = load_checkpoint(ckpt_path, device)
    cfg = ActiveCanViTConfig(**ckpt["model_config"])
    backbone_slug = ckpt["backbone"]
    teacher_dim = ckpt["teacher_dim"]

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

    model = ActiveCanViT(backbone, cfg, teacher_dim).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Load TEACHER backbone separately - must use same weights as training!
    # The teacher during training uses explicit checkpoint path, not default hub weights
    teacher_ckpt = backbone_weights if backbone_weights is not None else None
    if teacher_ckpt is not None:
        log.info(f"Loading TEACHER backbone {backbone_slug} from {teacher_ckpt}")
        raw_teacher = factory(pretrained=True, weights=str(teacher_ckpt))
    else:
        log.info(f"Loading TEACHER backbone {backbone_slug} with default pretrained weights")
        raw_teacher = factory(pretrained=True)
    teacher = DINOv3Backbone(raw_teacher.to(device).eval())
    for p in teacher.parameters():
        p.requires_grad = False

    log.info(f"Model loaded: {backbone_slug}, teacher_dim={teacher_dim}")
    log.info(f"  glimpse_grid_size={cfg.glimpse_grid_size}, registers={cfg.canvit.n_canvas_registers}")
    return model, teacher


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


def make_viewpoints(
    mode: Literal["quadrants", "sixteenths", "random"],
    n: int,
    B: int,
    device: torch.device,
) -> list[Viewpoint]:
    """Generate viewpoints based on mode."""
    if mode == "quadrants":
        return make_eval_viewpoints(B, device)
    elif mode == "sixteenths":
        # 4x4 grid at scale=0.125 (4x smaller than quadrants)
        vps = []
        positions = [-0.375, -0.125, 0.125, 0.375]
        for cy in positions:
            for cx in positions:
                centers = torch.tensor([[cy, cx]], device=device).expand(B, -1)
                scales = torch.full((B,), 0.125, device=device)
                vps.append(Viewpoint(name=f"({cy:.2f},{cx:.2f})", centers=centers, scales=scales))
        return vps[:n]
    else:  # random
        vps = []
        for i in range(n):
            vp = random_viewpoint(B, device, min_scale=0.125, max_scale=0.25)
            vps.append(Viewpoint(name=f"r{i}", centers=vp.centers, scales=vp.scales))
        return vps


def draw_trajectory(
    ax: plt.Axes,
    viewpoints: list[Viewpoint],
    colors: list[tuple[float, ...]],
    H: int,
    W: int,
    up_to: int | None = None,
    alpha: float = 1.0,
) -> None:
    """Draw saccade trajectory: boxes, connecting lines, and numbers."""
    from matplotlib.patches import Rectangle

    n = len(viewpoints) if up_to is None else up_to + 1
    boxes = [vp.to_pixel_box(0, H, W) for vp in viewpoints[:n]]

    # Draw boxes
    for t, (box, color) in enumerate(zip(boxes, colors[:n], strict=True)):
        rect = Rectangle(
            (box.left, box.top), box.width, box.height,
            linewidth=1.5 if alpha < 1 else 2, edgecolor=color, facecolor="none", alpha=alpha,
        )
        ax.add_patch(rect)
        # Number label
        ax.text(
            box.center_x, box.center_y, str(t),
            color="white", fontsize=7 if alpha < 1 else 8, fontweight="bold",
            ha="center", va="center", alpha=alpha,
            bbox={"boxstyle": "circle,pad=0.15", "facecolor": color, "edgecolor": "none", "alpha": alpha * 0.8},
        )

    # Draw trajectory lines between consecutive centers
    for t in range(1, n):
        prev, curr = boxes[t - 1], boxes[t]
        ax.plot(
            [prev.center_x, curr.center_x],
            [prev.center_y, curr.center_y],
            color=colors[t], linewidth=1, linestyle="--", alpha=alpha * 0.5,
        )


def main(args: Args) -> None:
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = torch.device(args.device)
    log.info(f"Device: {device}")

    model, teacher = load_model(args.checkpoint, args.backbone_weights, device)
    patch_size = teacher.patch_size_px

    max_grid = max(args.grid_sizes)
    img_size = max_grid * patch_size
    log.info(f"Image size: {img_size}px (for max grid {max_grid}, patch_size={patch_size})")

    image = load_image(args.image, img_size, device)
    log.info(f"Loaded image: {args.image} -> {image.shape}")

    viewpoints = make_viewpoints(args.viewpoint_mode, args.n_viewpoints, 1, device)
    log.info(f"Viewpoints ({args.viewpoint_mode}): {[vp.name for vp in viewpoints]}")

    # Collect per grid size: hidden spatial, projected scene, teacher target, losses
    all_hidden: dict[int, list[np.ndarray]] = {}  # raw canvas spatial tokens
    all_projected: dict[int, list[np.ndarray]] = {}  # after scene_proj (LayerNorm + Linear)
    all_unique: dict[int, list[np.ndarray]] = {}  # hidden - teacher @ lstsq (teacher-orthogonal)
    all_losses: dict[int, list[float]] = {}  # MSE loss vs teacher per timestep
    teacher_upsampled: dict[int, np.ndarray] = {}
    teacher_upsampled_torch: dict[int, torch.Tensor] = {}

    with torch.no_grad():
        # Teacher at max resolution (ground truth + PCA fitting)
        teacher_patches = teacher.forward_norm_features(image).patches
        assert teacher_patches.shape == (1, max_grid * max_grid, teacher_patches.shape[-1])
        teacher_dim = teacher_patches.shape[-1]
        embed_dim = model.backbone.embed_dim
        log.info(f"Teacher at full res: {max_grid}×{max_grid}, dim={teacher_dim}")
        log.info(f"Model embed_dim={embed_dim}")

        # Teacher at FIXED 16×16 resolution (256px input)
        teacher_base_grid = 16
        teacher_base_px = teacher_base_grid * patch_size  # 256px
        img_at_base = F.interpolate(image, size=(teacher_base_px, teacher_base_px), mode="bilinear", align_corners=False)
        teacher_base = teacher.forward_norm_features(img_at_base).patches  # [1, 16*16, D]
        assert teacher_base.shape == (1, teacher_base_grid * teacher_base_grid, teacher_dim)
        teacher_base_spatial = teacher_base.view(1, teacher_base_grid, teacher_base_grid, teacher_dim).permute(0, 3, 1, 2)  # [1, D, 16, 16]
        log.info(f"Teacher baseline: {teacher_base_px}px → {teacher_base_grid}×{teacher_base_grid} tokens")

        for G in args.grid_sizes:
            # Upsample the SAME 16×16 teacher tokens to G×G
            teacher_at_G = F.interpolate(teacher_base_spatial, size=(G, G), mode="bilinear", align_corners=False)
            teacher_upsampled_torch[G] = teacher_at_G[0].permute(1, 2, 0).reshape(-1, teacher_dim)  # [G*G, D]
            teacher_upsampled[G] = teacher_upsampled_torch[G].cpu().numpy()
            log.info(f"G={G}: teacher 16×16 → upsample to {G}×{G}")

            # Model at this grid size
            canvas = model.init_canvas(1, G)
            outputs, _ = model.forward_trajectory_full(image, viewpoints, canvas)

            # Collect hidden spatial, projected scene, and losses
            hidden_list, projected_list, loss_list = [], [], []
            target = teacher_upsampled_torch[G]
            teacher_np = teacher_upsampled[G]  # [G*G, teacher_dim]
            scene_ln = model.scene_proj[0]  # LayerNorm
            for out in outputs:
                spatial = model.get_spatial(out.canvas)[0]  # [G*G, embed_dim]
                hidden = scene_ln(spatial).cpu().numpy() if args.hidden_ln else spatial.cpu().numpy()
                projected = out.scene[0].cpu().numpy()  # [G*G, teacher_dim]
                loss = cos_dissim(out.scene[0], target).item()
                hidden_list.append(hidden)
                projected_list.append(projected)
                loss_list.append(loss)
            all_hidden[G] = hidden_list
            all_projected[G] = projected_list
            all_losses[G] = loss_list

            # Compute hidden_unique: part of hidden not linearly predictable from teacher
            # hidden_unique = hidden - teacher @ A where A = lstsq(teacher, hidden)
            unique_list = []
            for hidden in hidden_list:
                A, _, _, _ = np.linalg.lstsq(teacher_np, hidden, rcond=None)
                hidden_predicted = teacher_np @ A
                hidden_unique = hidden - hidden_predicted
                unique_list.append(hidden_unique)
            all_unique[G] = unique_list

    # Create figure: loss curve row + 4 rows per grid size (hidden, unique, projected, teacher)
    n_viewpoints = len(viewpoints)
    n_rows = 1 + 4 * len(args.grid_sizes)  # 1 for loss + 4 per G (hidden, unique, proj, teacher)
    n_cols = n_viewpoints + 1  # +1 for input/loss column
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))

    # Input image (denormalized)
    img_np = imagenet_denormalize(image[0].cpu()).numpy()
    H, W = img_np.shape[:2]
    colors = timestep_colors(n_viewpoints)

    # Fit PCA: hidden on last timestep, projected/teacher on teacher target, unique on last timestep
    first_G = args.grid_sizes[0]
    pca_hidden = fit_pca(all_hidden[first_G][-1])  # last timestep hidden
    pca_unique = fit_pca(all_unique[first_G][-1])  # last timestep unique (teacher-orthogonal)
    pca_scene = fit_pca(teacher_upsampled[first_G])  # teacher target (proj tries to match this)

    # Row 0: Loss curves
    ax = axes[0, 0]
    ax.imshow(img_np)
    draw_trajectory(ax, viewpoints, colors, H, W)
    ax.set_ylabel("Input")
    ax.set_title("Trajectory")
    ax.set_xticks([])
    ax.set_yticks([])

    # Merge remaining columns in row 0 for loss curve
    # Plot in the middle column area
    ax_loss = axes[0, n_cols // 2]
    for G in args.grid_sizes:
        losses = all_losses[G]
        ax_loss.plot(range(len(losses)), losses, marker="o", markersize=3, label=f"G={G}")
    ax_loss.set_xlim(-0.5, n_viewpoints - 0.5)
    ax_loss.set_xlabel("Timestep")
    ax_loss.set_ylabel("Cosine dissimilarity")
    ax_loss.set_title("1 - cos_sim vs timestep")
    ax_loss.legend(fontsize=7)
    ax_loss.grid(True, alpha=0.3)

    # Hide unused columns in row 0
    for c in range(1, n_cols):
        if c != n_cols // 2:
            axes[0, c].axis("off")

    # Per grid size: 4 rows (hidden spatial, unique, projected scene, teacher)
    for g_idx, G in enumerate(args.grid_sizes):
        hidden_list = all_hidden[G]
        unique_list = all_unique[G]
        projected_list = all_projected[G]
        teacher_rgb = pca_rgb(pca_scene, teacher_upsampled[G], G, G)
        g_H, g_W = G, G

        # Row: Hidden spatial (raw canvas tokens, own PCA)
        row_hidden = 1 + g_idx * 4
        ax = axes[row_hidden, 0]
        ax.imshow(img_np)
        draw_trajectory(ax, viewpoints, colors, H, W)
        hidden_label = "hidden+LN" if args.hidden_ln else "hidden"
        ax.set_ylabel(f"G={G} {hidden_label}")
        ax.set_xticks([])
        ax.set_yticks([])

        for t, hidden in enumerate(hidden_list):
            ax = axes[row_hidden, t + 1]
            rgb = pca_rgb(pca_hidden, hidden, G, G)
            ax.imshow(rgb)
            draw_trajectory(ax, viewpoints, colors, g_H, g_W, up_to=t, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])

        # Row: Unique (teacher-orthogonal part of hidden, own PCA)
        row_unique = 1 + g_idx * 4 + 1
        ax = axes[row_unique, 0]
        ax.imshow(img_np)
        draw_trajectory(ax, viewpoints, colors, H, W)
        ax.set_ylabel(f"G={G} unique")
        ax.set_xticks([])
        ax.set_yticks([])

        for t, unique in enumerate(unique_list):
            ax = axes[row_unique, t + 1]
            rgb = pca_rgb(pca_unique, unique, G, G)
            ax.imshow(rgb)
            draw_trajectory(ax, viewpoints, colors, g_H, g_W, up_to=t, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])

        # Row: Projected scene (after LayerNorm + Linear, teacher PCA)
        row_proj = 1 + g_idx * 4 + 2
        ax = axes[row_proj, 0]
        ax.imshow(img_np)
        draw_trajectory(ax, viewpoints, colors, H, W)
        ax.set_ylabel(f"G={G} proj")
        ax.set_xticks([])
        ax.set_yticks([])

        for t, proj in enumerate(projected_list):
            ax = axes[row_proj, t + 1]
            rgb = pca_rgb(pca_scene, proj, G, G)
            ax.imshow(rgb)
            draw_trajectory(ax, viewpoints, colors, g_H, g_W, up_to=t, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])

        # Row: Teacher upsampled (static reference)
        row_teacher = 1 + g_idx * 4 + 3
        ax = axes[row_teacher, 0]
        ax.imshow(img_np)
        draw_trajectory(ax, viewpoints, colors, H, W)
        ax.set_ylabel(f"G={G} teacher")
        ax.set_xticks([])
        ax.set_yticks([])

        for t in range(n_viewpoints):
            ax = axes[row_teacher, t + 1]
            ax.imshow(teacher_rgb)
            draw_trajectory(ax, viewpoints, colors, g_H, g_W, up_to=t, alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {args.output}")


if __name__ == "__main__":
    main(tyro.cli(Args))
