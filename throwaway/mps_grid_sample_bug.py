#!/usr/bin/env python3
"""
MPS grid_sample bug reproduction.

PyTorch version: 2.9.1
Platform: macOS (Apple Silicon M4 Pro)
"""
import torch
from PIL import Image
from torchvision import transforms
from avp_vit.backbone.dinov3 import DINOv3Backbone
from avp_vit.glimpse import sample_at_viewpoint, Viewpoint
from avp_vit.train.data import imagenet_normalize
from scripts.train_scene_match.model import MODEL_REGISTRY


def main():
    print(f"PyTorch version: {torch.__version__}")
    device = torch.device("mps")
    N = 128
    D = 384

    # Load and run DINOv3 teacher
    factory = MODEL_REGISTRY["dinov3_vits16"]
    raw = factory(pretrained=True, weights="dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
    teacher = DINOv3Backbone(raw.to(device).eval())

    img_size = N * 16
    img = Image.open("/Users/yberreby/code/ava/code/test_data/yohai.png").convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        imagenet_normalize(),
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        patches = teacher.forward_norm_features(img_tensor).patches
        # This is EXACTLY how the failing code path worked
        bdhw = patches.view(1, N, N, D).permute(0, 3, 1, 2)

        # Use sample_at_viewpoint with full scene (should be identity)
        vp = Viewpoint.full_scene(1, device)
        out = sample_at_viewpoint(bdhw, vp, N)

        diff = (bdhw - out).abs().max().item()
        status = "✓ OK" if diff < 1e-4 else "✗ BUG"
        print(f"MPS direct: max_diff = {diff:.6f} {status}")

        # CPU roundtrip comparison
        bdhw_cpu = bdhw.cpu()
        vp_cpu = Viewpoint.full_scene(1, "cpu")
        out_cpu = sample_at_viewpoint(bdhw_cpu, vp_cpu, N)
        diff_cpu = (bdhw_cpu - out_cpu).abs().max().item()
        status_cpu = "✓ OK" if diff_cpu < 1e-4 else "✗ BUG"
        print(f"CPU:        max_diff = {diff_cpu:.6f} {status_cpu}")


if __name__ == "__main__":
    main()
