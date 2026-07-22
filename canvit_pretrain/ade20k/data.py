"""ADE20K dataset + loaders + optimizer/scheduler/amp helpers.

Faithful port of canvit_specialize's datasets/ade20k.py + training/ade20k/common.py
(the P2 gate is reproducing specialize's probe numbers, so augmentation and
optimization are kept identical). This is the ONE train-time ADE20K pipeline of
the unified repo (master plan §3 — the specialize/RL duplicates retire with
their repos); validation-protocol comparability is anchored by the same squish
resize canvit_eval uses.
"""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from dinov3.eval.segmentation.schedulers import WarmupOneCycleLR
from dinov3.eval.segmentation.transforms import make_segmentation_train_transforms
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision import transforms as T

from .config import Ade20kConfig, ResizeMode

NUM_CLASSES = 150
IGNORE_LABEL = 255


class ADE20kDataset(torch.utils.data.Dataset):
    """ADE20K-SceneParse150.

    Two modes:
    - Separate transforms (eval): img_transform + mask_transform applied independently.
    - Joint transform (training): single callable (img, mask) -> (img_t, mask_t).
    """

    def __init__(
        self,
        root: Path,
        split: str,
        img_transform: T.Compose | None = None,
        mask_transform: T.Compose | None = None,
        joint_transform: Callable[[Image.Image, Image.Image], tuple[Tensor, Tensor]] | None = None,
    ) -> None:
        assert (img_transform is not None and mask_transform is not None) or joint_transform is not None, \
            "Provide either (img_transform + mask_transform) or joint_transform"

        img_dir = root / "images" / split
        ann_dir = root / "annotations" / split
        assert img_dir.is_dir(), f"Image dir not found: {img_dir}"
        assert ann_dir.is_dir(), f"Annotation dir not found: {ann_dir}"

        self.images = sorted(img_dir.glob("*.jpg"))
        self.masks = [ann_dir / f"{p.stem}.png" for p in self.images]
        assert len(self.images) > 0, f"No images found in {img_dir}"
        assert all(m.exists() for m in self.masks), "Missing mask files"

        self._img_transform = img_transform
        self._mask_transform = mask_transform
        self._joint_transform = joint_transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

        if self._joint_transform is not None:
            img_t, mask_t = self._joint_transform(img, mask)
        else:
            assert self._img_transform is not None and self._mask_transform is not None
            img_t = cast(Tensor, self._img_transform(img))
            # Masks: subtract 1 (ADE20K uses 1-indexed classes, 0 = ignore)
            mask_t = cast(Tensor, self._mask_transform(mask)).squeeze(0).long() - 1
            mask_t[mask_t < 0] = IGNORE_LABEL
        return img_t, mask_t


def make_val_transforms(size: int, mode: ResizeMode) -> tuple[T.Compose, T.Compose]:
    """Image and mask transforms for ADE20K validation."""
    if mode == "center_crop":
        img_transform = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(),
                                   T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
        mask_transform = T.Compose([T.Resize(size, T.InterpolationMode.NEAREST), T.CenterCrop(size),
                                    T.PILToTensor()])
    else:
        img_transform = T.Compose([T.Resize((size, size)), T.ToTensor(),
                                   T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])
        mask_transform = T.Compose([T.Resize((size, size), T.InterpolationMode.NEAREST), T.PILToTensor()])
    return img_transform, mask_transform


def make_ade20k_loaders(cfg: Ade20kConfig) -> tuple[DataLoader, DataLoader]:
    """Build ADE20K train/val data loaders (dinov3 train augmentation, squish val)."""
    if not cfg.ade20k_root.exists():
        raise FileNotFoundError(
            f"ADE20K root not found: {cfg.ade20k_root}. Set ADE20K_ROOT or pass --ade20k-root."
        )

    _train_aug = make_segmentation_train_transforms(
        img_size=cfg.scene_size,
        random_img_size_ratio_range=list(cfg.aug_scale_range),
        # Upstream annotation is Tuple[int] but implementation expects (H, W).
        crop_size=(cfg.scene_size, cfg.scene_size),  # pyright: ignore[reportArgumentType]
        flip_prob=cfg.aug_flip_prob,
        reduce_zero_label=True,
    )

    def train_transform(img: Image.Image, mask: Image.Image) -> tuple[Tensor, Tensor]:
        img_t, mask_t = _train_aug(img, mask)
        return img_t, mask_t.squeeze(0)

    train_ds = ADE20kDataset(root=cfg.ade20k_root, split="training", joint_transform=train_transform)
    val_img_tf, val_mask_tf = make_val_transforms(cfg.scene_size, cfg.resize_mode)
    val_ds = ADE20kDataset(root=cfg.ade20k_root, split="validation",
                           img_transform=val_img_tf, mask_transform=val_mask_tf)

    train_loader = DataLoader(
        train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, cfg.eval_batch_size, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader


def make_optimizer_and_scheduler(
    params, *, lr: float, weight_decay: float, max_steps: int,
    warmup_steps: int, warmup_lr_ratio: float,
) -> tuple[AdamW, LRScheduler]:
    """AdamW + WarmupOneCycleLR (identical to specialize's probe recipe)."""
    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = WarmupOneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=max_steps,
        warmup_iters=warmup_steps,
        warmup_ratio=warmup_lr_ratio,
        pct_start=0,
        anneal_strategy="cos",
        final_div_factor=float("inf"),
        use_beta1=False,
        update_momentum=False,
    )
    return optimizer, scheduler


def make_amp_ctx(amp: bool, device: torch.device) -> torch.autocast:
    amp_dtype = torch.bfloat16 if amp else torch.float32
    return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp)
