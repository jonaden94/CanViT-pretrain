"""Is the DATA the two trainers see identical? Transforms, ordering, and epoch reshuffling.

`diff_training_multistep.py` proves the training STEP agrees given identical inputs — it fed
both paths batches from ONE loader. That leaves the data pipeline untested, and the harness
builds its loader through `make_ade20k_loaders` while `rl_train.train()` constructs one inline.
A difference here would not show up in any gradient comparison, and it is the most plausible
remaining source of a run-to-run VARIANCE difference (doc 15 §A5.6).

Three checks:
  1. PIXELS — decoded batches, bit-exact, from identically-seeded loaders.
  2. ORDER  — the sampler permutation over several epochs, including the reshuffle at the
              epoch boundary (rl_train re-`iter()`s on StopIteration; the harness's
              `run._infinite` does `while True: for batch in loader`).
  3. WORKERS — whether num_workers changes the data at all (rl_train defaults to a different
              count than the exp27 launcher passes).
"""
import torch
from torch.utils.data import DataLoader

from canvit_pretrain.ade20k.config import Ade20kConfig
from canvit_pretrain.ade20k.data import ADE20kDataset, make_ade20k_loaders, make_val_transforms
from canvit_pretrain.ade20k.rl_train import PolicyTrainConfig

SCENE, BS, NB = 512, 16, 3
rl_cfg = PolicyTrainConfig(scene_size=SCENE, batch_size=BS, resize_mode="squish")
ha_cfg = Ade20kConfig(scene_size=SCENE, batch_size=BS, resize_mode="squish", augment=False,
                      num_workers=0)
print(f"rl_train num_workers={rl_cfg.num_workers}  harness(launcher)=4  "
      f"resize={rl_cfg.resize_mode}/{ha_cfg.resize_mode}  augment={ha_cfg.augment}")


def rl_train_loader(num_workers: int) -> DataLoader:
    """Exactly what rl_train.train() builds (rl_train.py: make_val_transforms on BOTH splits)."""
    img_tf, mask_tf = make_val_transforms(rl_cfg.scene_size, rl_cfg.resize_mode)
    ds = ADE20kDataset(root=rl_cfg.ade20k_root, split="training",
                       img_transform=img_tf, mask_transform=mask_tf)
    return DataLoader(ds, rl_cfg.batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=True, drop_last=True)


def batches(loader: DataLoader, n: int, seed: int):
    torch.manual_seed(seed)
    out = []
    for i, (im, mk) in enumerate(loader):
        if i >= n:
            break
        out.append((im.clone(), mk.clone()))
    return out


# --- 1. PIXELS -------------------------------------------------------------------
A = batches(rl_train_loader(0), NB, 1234)
B = batches(make_ade20k_loaders(ha_cfg)[0], NB, 1234)
print(f"\n1. PIXELS ({NB} batches, both num_workers=0, same seed)")
worst_i = worst_m = 0.0
for k, ((ia, ma), (ib, mb)) in enumerate(zip(A, B)):
    di = (ia - ib).abs().max().item()
    dm = (ma.float() - mb.float()).abs().max().item()
    worst_i, worst_m = max(worst_i, di), max(worst_m, dm)
    print(f"   batch {k}: img shape {tuple(ia.shape)} dtype {ia.dtype}  max|dimg|={di:.3e}  "
          f"max|dmask|={dm:.3e}")
print(f"   -> {'IDENTICAL' if worst_i == 0 and worst_m == 0 else 'DIFFERENT'}")

# --- 2. ORDER over epochs ---------------------------------------------------------
def order(loader: DataLoader, epochs: int, seed: int) -> list[int]:
    """Sampler indices across `epochs` fresh iterations — no image decoding."""
    torch.manual_seed(seed)
    idx: list[int] = []
    for _ in range(epochs):
        idx.extend(list(iter(loader.sampler)))  # type: ignore[arg-type]
    return idx


oa = order(rl_train_loader(0), 3, 99)
ob = order(make_ade20k_loaders(ha_cfg)[0], 3, 99)
same = oa == ob
print(f"\n2. ORDER (3 epochs of sampler indices, same seed)")
print(f"   len {len(oa)} vs {len(ob)}   first 8: {oa[:8]} vs {ob[:8]}")
print(f"   epoch-2 differs from epoch-1 (i.e. it DOES reshuffle): "
      f"{oa[:len(oa)//3] != oa[len(oa)//3:2*len(oa)//3]}")
print(f"   -> {'IDENTICAL' if same else 'DIFFERENT'}")

# --- 3. WORKERS -------------------------------------------------------------------
W = batches(rl_train_loader(4), NB, 1234)
dw = max((a[0] - w[0]).abs().max().item() for a, w in zip(A, W))
dwm = max((a[1].float() - w[1].float()).abs().max().item() for a, w in zip(A, W))
print(f"\n3. WORKERS (num_workers 0 vs 4, same seed)  max|dimg|={dw:.3e} max|dmask|={dwm:.3e}"
      f"  -> {'no effect' if dw == 0 and dwm == 0 else 'CHANGES THE DATA'}")
