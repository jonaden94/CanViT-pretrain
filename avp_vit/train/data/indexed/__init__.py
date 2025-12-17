"""Indexed ImageFolder: fast loading for large datasets without directory-based splits."""

import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IndexMetadata:
    schema_version: int
    root_name: str
    split: Literal["train", "val"]
    val_ratio: float
    seed: int
    n_samples: int
    n_classes: int
    generated_at: str


@dataclass(frozen=True)
class SplitIndex:
    train: Path
    val: Path


def parse_index_metadata(raw: dict[bytes, bytes]) -> IndexMetadata:
    """Parse raw parquet metadata. Raises ValueError on schema mismatch."""
    schema_version = int(raw.get(b"schema_version", b"-1").decode())
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"Schema version mismatch: got {schema_version}, expected {SCHEMA_VERSION}")

    def get(key: str) -> str:
        return raw.get(key.encode(), b"").decode()

    split = get("split")
    assert split in ("train", "val"), f"Invalid split: {split}"

    return IndexMetadata(
        schema_version=schema_version,
        root_name=get("root_name"),
        split=split,  # type: ignore[arg-type]
        val_ratio=float(get("val_ratio") or 0),
        seed=int(get("seed") or 0),
        n_samples=int(get("n_samples") or 0),
        n_classes=int(get("n_classes") or 0),
        generated_at=get("generated_at"),
    )


def split_by_class(
    class_to_images: dict[str, list[str]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Split images into train/val. Returns (train, val) as [(path, class_name), ...]."""
    assert 0 < val_ratio < 1
    rng = random.Random(seed)
    train, val = [], []

    for cls in sorted(class_to_images):
        imgs = list(class_to_images[cls])
        rng.shuffle(imgs)
        n_val = max(1, int(len(imgs) * val_ratio))
        for img in imgs[:n_val]:
            val.append((f"{cls}/{img}", cls))
        for img in imgs[n_val:]:
            train.append((f"{cls}/{img}", cls))

    return train, val


# ============================================================================
# I/O Functions
# ============================================================================


def generate_split_index(
    root: Path,
    output_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> SplitIndex:
    """Generate train/val parquet index files. Uses tqdm for progress."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tqdm import tqdm

    log.info(f"Generating split index for: {root}")
    log.info(f"  output_dir={output_dir}, val_ratio={val_ratio}, seed={seed}")

    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # Scan directory structure
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir() and d.name != "tars")
    log.info(f"  Found {len(class_dirs):,} class directories")

    class_to_images: dict[str, list[str]] = {}
    for class_dir in tqdm(class_dirs, desc="Scanning classes", unit="class"):
        images = [f.name for f in class_dir.iterdir() if f.is_file()]
        if images:
            class_to_images[class_dir.name] = images

    total_images = sum(len(imgs) for imgs in class_to_images.values())
    log.info(f"  Found {total_images:,} images in {len(class_to_images):,} classes")

    # Split
    train_records, val_records = split_by_class(class_to_images, val_ratio, seed)
    log.info(f"  Split: {len(train_records):,} train, {len(val_records):,} val")

    # Compute class_to_idx
    all_classes = sorted(class_to_images.keys())
    class_to_idx = {c: i for i, c in enumerate(all_classes)}

    def write_parquet(records: list[tuple[str, str]], split: Literal["train", "val"]) -> Path:
        path = output_dir / f"{root.name}_{split}.parquet"
        table = pa.table({
            "path": [r[0] for r in records],
            "class_name": [r[1] for r in records],
            "class_idx": [class_to_idx[r[1]] for r in records],
        })
        metadata = {
            b"schema_version": str(SCHEMA_VERSION).encode(),
            b"root_name": root.name.encode(),
            b"split": split.encode(),
            b"val_ratio": str(val_ratio).encode(),
            b"seed": str(seed).encode(),
            b"n_samples": str(len(records)).encode(),
            b"n_classes": str(len(all_classes)).encode(),
            b"generated_at": datetime.now(timezone.utc).isoformat().encode(),
        }
        table = table.replace_schema_metadata(metadata)
        pq.write_table(table, path)
        size_mb = path.stat().st_size / (1024 * 1024)
        log.info(f"  Wrote {path.name}: {len(records):,} samples, {size_mb:.1f} MB")
        return path

    train_path = write_parquet(train_records, "train")
    val_path = write_parquet(val_records, "val")

    elapsed = time.perf_counter() - t0
    log.info(f"Index generation complete ({elapsed:.1f}s)")

    return SplitIndex(train=train_path, val=val_path)


def ensure_split_index(
    root: Path,
    output_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> SplitIndex:
    """Return index paths, generating if missing."""
    train_path = output_dir / f"{root.name}_train.parquet"
    val_path = output_dir / f"{root.name}_val.parquet"

    if train_path.exists() and val_path.exists():
        log.info(f"Found existing indices: {train_path}, {val_path}")
        return SplitIndex(train=train_path, val=val_path)

    log.info("Index files not found, generating...")
    return generate_split_index(root, output_dir, val_ratio, seed)


class IndexedImageFolder(ImageFolder):
    """ImageFolder that loads from parquet index instead of walking filesystem."""

    def __init__(
        self,
        root: str,
        index_file: Path,
        transform: Compose | None = None,
        target_transform: Compose | None = None,
    ):
        import pyarrow.parquet as pq

        # Don't call super().__init__ - we bypass filesystem walk
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader

        log.info(f"Loading index: {index_file}")
        t0 = time.perf_counter()

        table = pq.read_table(index_file)
        raw_meta = table.schema.metadata or {}
        meta = parse_index_metadata(raw_meta)

        # Log metadata
        log.info(f"  root_name={meta.root_name}  split={meta.split}  schema_version={meta.schema_version}")
        log.info(f"  n_samples={meta.n_samples:,}  n_classes={meta.n_classes:,}")
        log.info(f"  val_ratio={meta.val_ratio}  seed={meta.seed}")
        log.info(f"  generated_at={meta.generated_at}")

        # Validate root name
        actual_root_name = Path(root).name
        assert meta.root_name == actual_root_name, (
            f"Root name mismatch: index has '{meta.root_name}', actual is '{actual_root_name}'"
        )

        # Build samples from parquet
        df = table.to_pandas()
        self.classes = sorted(df["class_name"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = [(f"{root}/{row.path}", row.class_idx) for row in df.itertuples()]
        self.targets = [s[1] for s in self.samples]
        self.imgs = self.samples  # ImageFolder compat

        elapsed = time.perf_counter() - t0
        log.info(f"IndexedImageFolder ready: {len(self.samples):,} samples ({elapsed:.2f}s)")
