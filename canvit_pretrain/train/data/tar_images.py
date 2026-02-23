"""Read images directly from mmap'd SA-1B tar files. No extraction needed.

Each SA-1B tar (~70 GB) contains ~11k JPEGs + ~11k JSONs. We use `tarfile`
to scan headers once and build a {name: (offset, size)} index, then serve
images via mmap slicing.  All forked DataLoader workers share the same
mmap'd pages (copy-on-write).

Verified on sa_000020.tar: index build ~2s, image read ~18ms/img, 0 missing
vs shard paths.
"""

import io
import logging
import mmap
import tarfile
import time
from pathlib import Path

from PIL import Image

log = logging.getLogger(__name__)


class TarImageReader:
    """Read images from an mmap'd tar file by name.

    Index build: iterate tarfile members (~2s for 70 GB tar).
    Image read: mmap slice → BytesIO → PIL decode (~18ms/img).
    """

    def __init__(self, tar_path: Path) -> None:
        self._fd = open(tar_path, "rb")
        self._mm = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)
        t0 = time.perf_counter()
        self.index = self._build_index(tar_path)
        elapsed = time.perf_counter() - t0
        log.info(f"TarImageReader: {tar_path.name}, {len(self.index)} JPEGs indexed in {elapsed:.1f}s")

    def _build_index(self, tar_path: Path) -> dict[str, tuple[int, int]]:
        """Scan tar headers → {stripped_name: (data_offset, data_size)}."""
        index: dict[str, tuple[int, int]] = {}
        with tarfile.open(tar_path, "r") as tf:
            for member in tf:
                if not member.name.endswith(".jpg"):
                    continue
                # Strip leading directory (like tar --strip-components=1)
                stripped = member.name.split("/", 1)[-1] if "/" in member.name else member.name
                index[stripped] = (member.offset_data, member.size)
        return index

    def read_image(self, name: str) -> Image.Image:
        data_offset, size = self.index[name]
        return Image.open(io.BytesIO(self._mm[data_offset : data_offset + size])).convert("RGB")

    def close(self) -> None:
        self._mm.close()
        self._fd.close()
