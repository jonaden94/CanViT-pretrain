"""Compare fp16 vs bf16 storage error (bf16 autocast for inference)."""

import argparse
from pathlib import Path

import torch
from canvit.hub import create_backbone
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda")

    # Load and transform
    transform = transforms.Compose([
        transforms.Resize(args.size),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    img = Image.open(args.image).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    if args.batch_size > 1:
        tensor = tensor.repeat(args.batch_size, 1, 1, 1)

    # Load model
    teacher = create_backbone("dinov3_vitb16", weights=str(args.ckpt))
    teacher = teacher.to(device).eval()

    # Inference (bf16 autocast)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        feats = teacher.forward_norm_features(tensor)
        patches = feats.patches[0].cpu()  # f32 from autocast

    print(f"Raw output: {patches.shape}, dtype={patches.dtype}")
    print(f"Range: [{patches.min():.4f}, {patches.max():.4f}]")

    # Compare storage dtypes
    to_f16 = patches.to(torch.float16)
    to_bf16 = patches.to(torch.bfloat16)

    f16_err = (patches - to_f16.float()).abs()
    bf16_err = (patches - to_bf16.float()).abs()

    print(f"\nStorage quantization error:")
    print(f"  fp16:  max={f16_err.max():.6f}, mean={f16_err.mean():.6f}")
    print(f"  bf16:  max={bf16_err.max():.6f}, mean={bf16_err.mean():.6f}")
    print(f"  ratio: bf16/fp16 = {bf16_err.max() / f16_err.max():.1f}x worse")


if __name__ == "__main__":
    main()
