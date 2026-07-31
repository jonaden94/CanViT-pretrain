"""Read images directly from mmap'd tar files. No extraction needed.

Scan tar headers to build a {name: (offset, size)} index, then read images
via mmap slicing. Forked DataLoader workers share mmap'd pages (copy-on-write).

scan_tar_headers(tar_path) — slow header scan; for export/bench.
load_tar_index(tar_path)   — instant load from .idx file; for training.

.idx files carry SHA256 + size. They used to be produced by `sa1b/build_tar_indexes.py`,
deleted 2026-07-31 along with the rest of the SA-1B pipeline: no .idx data exists on the
cluster any more and `slurm/harness_train.sbatch` REQUIRES `WEBDATASET_DIR`, so the
sharded-tar path this module serves is unreachable from any production launcher. The
module is kept because `train/data/__init__.py` still exposes the path (`create_loaders`
falls back to it when `cfg.webdataset_dir` is unset) — but building a fresh index now
means restoring that script from git history (see commit deleting sa1b/).
"""

import io
import logging
import mmap
import pickle
import tarfile
import time
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)

# {stripped_name: (data_offset, data_size)}
TarIndex = dict[str, tuple[int, int]]


def scan_tar_headers(tar_path: Path) -> TarIndex:
    """Scan tar headers → {stripped_name: (data_offset, data_size)}.

    Slow on large tars. Use for export scripts and benchmarks.
    For training, use load_tar_index() instead.
    """
    t0 = time.perf_counter()
    index: TarIndex = {}
    with tarfile.open(tar_path, "r") as tf:
        for member in tf:
            if not member.name.endswith(".jpg"):
                continue
            stripped = member.name.split("/", 1)[-1] if "/" in member.name else member.name
            index[stripped] = (member.offset_data, member.size)
    elapsed = time.perf_counter() - t0
    log.info(f"Scanned tar: {tar_path.name}, {len(index)} JPEGs in {elapsed:.1f}s")
    return index


def load_tar_index(tar_path: Path) -> TarIndex:
    """Load pre-built .idx file for a tar. Crashes if missing or stale.

    Verifies the tar file size matches (instant stat() check, no full read). The builder
    was `sa1b/build_tar_indexes.py`, deleted with the SA-1B pipeline — see the module
    docstring.
    """
    idx_path = tar_path.parent / f"{tar_path.name}.idx"
    assert idx_path.exists(), (
        f"No .idx for {tar_path.name}. The index builder (sa1b/build_tar_indexes.py) was "
        f"deleted 2026-07-31 with the SA-1B pipeline; restore it from git history to build "
        f"one, or use the webdataset path (--cfg.webdataset-dir) instead."
    )

    t0 = time.perf_counter()
    with open(idx_path, "rb") as f:
        data = pickle.load(f)

    actual_size = tar_path.stat().st_size
    assert data["tar_size"] == actual_size, (
        f"Tar size mismatch: {tar_path.name} "
        f"(index={data['tar_size']}, actual={actual_size}). The index is stale and its "
        f"builder (sa1b/build_tar_indexes.py) was deleted 2026-07-31 — restore it from git "
        f"history to rebuild, or use the webdataset path instead."
    )

    index = data["index"]
    elapsed = time.perf_counter() - t0
    log.info(
        f"Loaded tar index: {tar_path.name}, {len(index)} JPEGs "
        f"(sha256={data['sha256'][:12]}..., {elapsed:.3f}s)"
    )
    return index


class TarImageReader:
    """Read images from an mmap'd tar file by name."""

    def __init__(self, tar_path: Path, *, index: TarIndex) -> None:
        self._fd = open(tar_path, "rb")
        self._mm = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)
        self.index = index

    def read_image(self, name: str) -> Image.Image:
        assert self._mm is not None
        data_offset, size = self.index[name]
        return Image.open(io.BytesIO(self._mm[data_offset : data_offset + size])).convert("RGB")

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None  # type: ignore[assignment]
        if self._fd is not None:
            self._fd.close()
            self._fd = None  # type: ignore[assignment]

    def __del__(self) -> None:
        self.close()
