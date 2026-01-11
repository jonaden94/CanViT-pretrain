"""Validate exported features against fresh inference.

Usage (in interactive session):
    source slurm/env.sh
    uv run python scripts/validate_features.py \
        --shard $FEATURES_DIR/dinov3_vitb16/512/shards/00000.pt \
        --image-root $IN21K_DIR \
        --teacher-ckpt $DINOV3_VITB16_CKPT \
        --idx 42

Validates:
    1. mmap loading works (key for training speed)
    2. Stored features match fresh inference
    3. PCA visualization looks sensible
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from canvit.hub import create_backbone
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_shard_mmap(path: Path) -> dict:
    """Load shard with memory-mapping (no full load into RAM)."""
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def load_and_transform(path: Path, size: int) -> torch.Tensor:
    """Load image and apply teacher preprocessing."""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img)


def pca_rgb(patches: torch.Tensor, n_components: int = 3) -> torch.Tensor:
    """Project patch features to RGB via PCA. Input: [n_patches, dim]."""
    patches_np = patches.float().numpy()
    pca = PCA(n_components=n_components)
    rgb = pca.fit_transform(patches_np)
    # Normalize to [0, 1]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    return torch.from_numpy(rgb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--teacher-ckpt", type=Path, required=True)
    parser.add_argument("--idx", type=int, default=0, help="Index within shard")
    parser.add_argument("--teacher-model", type=str, default="dinov3_vitb16")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load shard (mmap) ---
    print(f"Loading shard (mmap): {args.shard}")
    shard = load_shard_mmap(args.shard)

    print(f"  Shard metadata:")
    print(f"    teacher_model: {shard['teacher_model']}")
    print(f"    image_size: {shard['image_size']}")
    print(f"    n_patches: {shard['n_patches']}")
    print(f"    embed_dim: {shard['embed_dim']}")
    print(f"    dtype: {shard['dtype']}")
    print(f"    images: {len(shard['paths'])}")
    print(f"    git_commit: {shard['git_commit'][:12]}")

    n_images = len(shard["paths"])
    assert 0 <= args.idx < n_images, f"idx must be in [0, {n_images})"

    # --- Get stored features ---
    stored_patches = shard["patches"][args.idx]  # [n_patches, embed_dim]
    stored_cls = shard["cls"][args.idx]  # [embed_dim]
    rel_path = shard["paths"][args.idx]
    stored_hash = shard["image_hashes"][args.idx]

    print(f"\n  Image {args.idx}: {rel_path}")
    print(f"    stored_hash: {stored_hash}")
    print(f"    patches shape: {stored_patches.shape}")
    print(f"    cls shape: {stored_cls.shape}")

    # --- Load image ---
    image_path = args.image_root / rel_path
    print(f"\nLoading image: {image_path}")

    img_pil = Image.open(image_path).convert("RGB")
    img_tensor = load_and_transform(image_path, shard["image_size"])

    # Verify hash
    import xxhash
    fresh_hash = xxhash.xxh64(img_pil.tobytes()).hexdigest()
    hash_match = fresh_hash == stored_hash
    print(f"  Hash match: {hash_match} (stored={stored_hash}, fresh={fresh_hash})")

    # --- Fresh inference ---
    print(f"\nRunning fresh inference with {args.teacher_model}...")
    teacher = create_backbone(args.teacher_model, weights=str(args.teacher_ckpt))
    teacher = teacher.to(device).eval()

    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        img_batch = img_tensor.unsqueeze(0).to(device)
        feats = teacher.forward_norm_features(img_batch)
        fresh_patches = feats.patches[0].to(torch.bfloat16).cpu()  # [n_patches, embed_dim]
        fresh_cls = feats.cls[0].to(torch.bfloat16).cpu()  # [embed_dim]

    # --- Compare ---
    patches_diff = (stored_patches.float() - fresh_patches.float()).abs()
    cls_diff = (stored_cls.float() - fresh_cls.float()).abs()

    print(f"\nFeature comparison:")
    print(f"  Patches max diff: {patches_diff.max().item():.6f}")
    print(f"  Patches mean diff: {patches_diff.mean().item():.6f}")
    print(f"  CLS max diff: {cls_diff.max().item():.6f}")
    print(f"  CLS mean diff: {cls_diff.mean().item():.6f}")

    match = patches_diff.max().item() < 1e-3 and cls_diff.max().item() < 1e-3
    print(f"  Features match: {'✓ YES' if match else '✗ NO (MISMATCH!)'}")

    # --- PCA visualization ---
    print("\nGenerating PCA visualization...")

    grid_size = int(shard["n_patches"] ** 0.5)

    stored_pca = pca_rgb(stored_patches).reshape(grid_size, grid_size, 3)
    fresh_pca = pca_rgb(fresh_patches).reshape(grid_size, grid_size, 3)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img_pil)
    axes[0].set_title(f"Original ({shard['image_size']}px)")
    axes[0].axis("off")

    axes[1].imshow(stored_pca)
    axes[1].set_title("Stored features (PCA)")
    axes[1].axis("off")

    axes[2].imshow(fresh_pca)
    axes[2].set_title("Fresh inference (PCA)")
    axes[2].axis("off")

    diff_img = (stored_pca - fresh_pca).abs()
    axes[3].imshow(diff_img)
    axes[3].set_title(f"Difference (max={diff_img.max():.4f})")
    axes[3].axis("off")

    plt.tight_layout()
    out_path = Path("validate_features.png")
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")

    # --- mmap random access demo ---
    print("\n--- mmap random access demo ---")
    import time

    # Access 100 random indices
    indices = torch.randint(0, n_images, (100,))
    t0 = time.perf_counter()
    for i in indices:
        _ = shard["patches"][i]
    elapsed = time.perf_counter() - t0
    print(f"100 random patch accesses: {elapsed*1000:.1f}ms ({elapsed*10:.2f}ms/access)")

    print("\n✓ Validation complete")


if __name__ == "__main__":
    main()
