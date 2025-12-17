# ImageNet-21k Support for SLURM/Compute Canada

## Context

We're adapting `scripts/train_scene_match/` to run on Alliance Canada (Compute Canada) with ImageNet-21k (IN21k), which has **no train/val split** (unlike ILSVRC which has separate directories).

**Problem**: IN21k on Alliance Canada is at `/project/.../winter21_whole/` with 13M+ images in 19k class folders. Read-only filesystem, limited inodes (can't create symlinks). Need to split into train/val programmatically.

**Solution**: Parquet index files that list which images belong to train vs val. Fast loading (~0.5s for 13M entries) vs walking filesystem (~20-30 min).

---

## What Was Implemented

### Location: `avp_vit/train/data/indexed/`

```
avp_vit/train/data/indexed/
├── __init__.py   # Core pure functions (72 lines)
└── test.py       # 5 tests for important properties
```

### Exports

```python
from avp_vit.train.data.indexed import (
    SCHEMA_VERSION,      # int = 1
    IndexMetadata,       # Dataclass for parsed parquet metadata
    SplitIndex,          # Dataclass holding train/val Path pair
    parse_index_metadata,  # Parse raw parquet metadata bytes
    split_by_class,      # Core splitting logic
)
```

### `IndexMetadata` (dataclass)

```python
@dataclass(frozen=True)
class IndexMetadata:
    schema_version: int
    root_name: str                    # e.g., "winter21_whole"
    split: Literal["train", "val"]
    val_ratio: float
    seed: int
    n_samples: int
    n_classes: int
    generated_at: str                 # ISO timestamp
```

### `SplitIndex` (dataclass)

```python
@dataclass(frozen=True)
class SplitIndex:
    train: Path
    val: Path
```

### `parse_index_metadata(raw: dict[bytes, bytes]) -> IndexMetadata`

Parses raw parquet file metadata. Raises `ValueError` if schema version doesn't match `SCHEMA_VERSION`.

### `split_by_class(class_to_images, val_ratio, seed) -> (train, val)`

- Input: `{class_name: [image_filenames]}`
- Output: `(train_records, val_records)` where each record is `(relative_path, class_name)`
- Guarantees:
  - Deterministic (same seed = same split)
  - At least 1 image per class goes to val
  - No data loss (all images accounted for)
  - Classes processed in sorted order

---

## What Remains To Be Done

### 1. `generate_split_index(root, output_dir, val_ratio, seed)` function

Walks filesystem, calls `split_by_class`, writes parquet files with metadata.

```python
def generate_split_index(
    root: Path,
    output_dir: Path,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> SplitIndex:
    """Generate train/val parquet indices. Uses tqdm, logs progress."""
    # Walk root, build class_to_images dict
    # Call split_by_class(...)
    # Write parquet with columns: path, class_name, class_idx
    # Include metadata: schema_version, root_name, split, val_ratio, seed, n_samples, n_classes, generated_at
    # Return SplitIndex(train=..., val=...)
```

**Parquet schema**:
- Columns: `path` (str), `class_name` (str), `class_idx` (int32)
- File metadata: all fields from `IndexMetadata`

### 2. `IndexedImageFolder` class

Drop-in replacement for `torchvision.datasets.ImageFolder`, loads from parquet instead of walking filesystem.

```python
class IndexedImageFolder(ImageFolder):
    def __init__(self, root: str, index_file: Path, transform=None):
        # Read parquet
        # Parse and validate metadata via parse_index_metadata()
        # Assert root_name matches Path(root).name (CRASH on mismatch, not warn)
        # Build self.samples, self.targets, self.classes, self.class_to_idx
        # Log: index path, n_samples, n_classes, val_ratio, seed, generated_at
```

### 3. `ensure_split_index(root, output_dir, val_ratio, seed) -> SplitIndex`

Auto-generates if missing, returns paths.

```python
def ensure_split_index(...) -> SplitIndex:
    train_path = output_dir / f"{root.name}_train.parquet"
    val_path = output_dir / f"{root.name}_val.parquet"
    if not train_path.exists() or not val_path.exists():
        log.info("Index not found, generating...")
        return generate_split_index(root, output_dir, val_ratio, seed)
    log.info(f"Found existing indices: {train_path}, {val_path}")
    return SplitIndex(train=train_path, val=val_path)
```

### 4. Update `avp_vit/train/data/__init__.py`

Modify `make_loader` to accept optional `index_file` parameter:

```python
def make_loader(
    root: Path,
    transform: transforms.Compose,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    index_file: Path | None = None,  # NEW
) -> DataLoader[ImageBatch]:
    if index_file is not None:
        log.info(f"IndexedImageFolder: root={root}, index={index_file}")
        dataset = IndexedImageFolder(str(root), index_file, transform=transform)
    else:
        log.info(f"ImageFolder: root={root}")
        dataset = ImageFolder(str(root), transform=transform)

    log.info(f"  {len(dataset):,} samples, batch_size={batch_size}")
    return DataLoader(...)
```

### 5. Update `scripts/train_scene_match/config.py`

```python
@dataclass
class Config:
    # ... existing fields ...
    index_dir: Path = Path("indices")  # Where to store/load indices
    val_ratio: float = 0.1             # For index generation
    split_seed: int = 42               # For index generation
```

### 6. Update `scripts/train_scene_match/data.py`

Auto-detect when indexing is needed:

```python
def create_loaders_for_resolution(cfg, stages):
    needs_index = cfg.train_dir.resolve() == cfg.val_dir.resolve()

    if needs_index:
        log.info(f"train_dir == val_dir → indexed split mode")
        log.info(f"  val_ratio={cfg.val_ratio}, seed={cfg.split_seed}")
        index = ensure_split_index(cfg.train_dir, cfg.index_dir, cfg.val_ratio, cfg.split_seed)
        train_index, val_index = index.train, index.val
    else:
        log.info(f"Separate train/val directories → ImageFolder mode")
        train_index, val_index = None, None

    # Pass index files through to make_loader(...)
```

---

## Design Decisions

1. **Parquet format**: ~60MB for 13M entries, loads in ~0.5s. pyarrow dependency is fine.

2. **Auto-generate if missing**: First run is slow (~20 min to walk 13M files), subsequent runs fast. Uses tqdm.

3. **Crash on root_name mismatch**: Silent mismatch = silent data corruption. Fail fast.

4. **Schema versioning**: `SCHEMA_VERSION = 1`. Hard fail if index has different version.

5. **At least 1 val per class**: Even with tiny val_ratio, `max(1, int(n * ratio))` ensures stratification.

6. **Deterministic splits**: Same seed = same split. Reproducibility.

7. **No `validate_index_metadata` function**: Just an inline assert at call site. YAGNI.

8. **No `compute_class_to_idx` / `build_samples_list` functions**: Trivial one-liners, inline at call site.

---

## Usage (Once Complete)

**ILSVRC** (unchanged):
```bash
uv run python -m scripts.train_scene_match \
    --train-dir /datasets/ILSVRC/.../train \
    --val-dir /datasets/ILSVRC/.../val
```

**IN21k** (auto-generates index on first run):
```bash
uv run python -m scripts.train_scene_match \
    --train-dir /project/.../winter21_whole \
    --val-dir /project/.../winter21_whole \
    --index-dir ./indices
```

---

## Performance

| Operation | Time |
|-----------|------|
| Index generation (one-time, 13M files) | ~15-30 min |
| Index file size (parquet) | ~60 MB each |
| `IndexedImageFolder.__init__` | ~0.5 sec |
| `ImageFolder.__init__` on 13M files | ~20-30 min |

---

## Tests

5 tests in `avp_vit/train/data/indexed/test.py`:
1. `test_parse_metadata_valid` - basic parsing works
2. `test_parse_metadata_wrong_schema_version` - raises on mismatch
3. `test_split_deterministic` - same seed = same split
4. `test_split_no_data_loss` - all images accounted for
5. `test_split_at_least_one_val_per_class` - stratification guarantee

Run: `uv run pytest avp_vit/train/data/indexed/test.py -v`

---

## SLURM Notes

See conversation for full SLURM job script. Key points:
- ImageNet on Alliance Canada: copy to `$SLURM_TMPDIR` for fast I/O
- `COMET_OFFLINE_DIRECTORY=$SCRATCH/comet_offline` for offline logging
- `TORCH_COMPILE_CACHE_DIR=$SCRATCH/torch_compile_cache`
- All config overridable via CLI (tyro)
