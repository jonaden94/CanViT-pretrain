"""Normalizer stats pooled over several shards must equal the one-shot computation.

The pooled path streams moments instead of materialising every sample (4 shards of
4096x1024x768 would be ~51 GB), so the thing worth pinning is that the streaming
reconstruction lands on the SAME buffers `set_stats` would have produced from the full
tensor — and that pooling n shards equals computing over their concatenation.
"""

from __future__ import annotations

import io
import tarfile

import numpy as np
import torch
from canvit_pytorch import CLSStandardizer, PatchStandardizer

from canvit_train.distill.data.webdataset import (
    _freeze_stats,
    _MomentAcc,
    init_normalizer_stats_from_tar,
)

G, D = 2, 8            # 2x2 token grid, 8 channels — tiny but the real shapes
N_PER_SHARD = 12
DEV = torch.device("cpu")


def _norms():
    return PatchStandardizer(grid_size=G, embed_dim=D), CLSStandardizer(embed_dim=D)


def _write_shard(path, rng, n=N_PER_SHARD):
    """A shard in the real layout: <key>.cls.npy + <key>.ptch.npy per sample."""
    cls_all, ptch_all = [], []
    with tarfile.open(path, "w") as tf:
        for i in range(n):
            cls = rng.normal(size=(D,)).astype(np.float32)
            ptch = rng.normal(loc=2.0, scale=3.0, size=(G * G, D)).astype(np.float32)
            cls_all.append(cls)
            ptch_all.append(ptch)
            for kind, arr in (("cls", cls), ("ptch", ptch)):
                buf = io.BytesIO()
                np.save(buf, arr)
                info = tarfile.TarInfo(f"{i:05d}.{kind}.npy")
                info.size = buf.tell()
                buf.seek(0)
                tf.addfile(info, buf)
    return np.stack(cls_all), np.stack(ptch_all)


def test_freeze_stats_reproduces_set_stats_exactly():
    """The two-row surrogate must give the same buffers as the full tensor."""
    rng = np.random.default_rng(0)
    data = torch.from_numpy(rng.normal(size=(97, G * G, D)).astype(np.float32))

    direct, _ = _norms()
    direct.set_stats(data)

    streamed, _ = _norms()
    acc = _MomentAcc(DEV)
    for chunk in data.split(16):          # arbitrary chunking must not matter
        acc.add(chunk)
    _freeze_stats(streamed, acc)

    torch.testing.assert_close(streamed.mean, direct.mean, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(streamed.var, direct.var, rtol=1e-5, atol=1e-6)


def test_pooling_two_shards_equals_their_concatenation(tmp_path):
    rng = np.random.default_rng(1)
    a, b = tmp_path / "shard-000000.tar", tmp_path / "shard-000001.tar"
    cls_a, ptch_a = _write_shard(a, rng)
    cls_b, ptch_b = _write_shard(b, rng)

    pooled_scene, pooled_cls = _norms()
    init_normalizer_stats_from_tar([a, b], pooled_scene, pooled_cls, DEV, 0)

    ref_scene, ref_cls = _norms()
    ref_scene.set_stats(torch.from_numpy(np.concatenate([ptch_a, ptch_b])))
    ref_cls.set_stats(torch.from_numpy(np.concatenate([cls_a, cls_b])).unsqueeze(1))

    torch.testing.assert_close(pooled_scene.mean, ref_scene.mean, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(pooled_scene.var, ref_scene.var, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(pooled_cls.mean, ref_cls.mean, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(pooled_cls.var, ref_cls.var, rtol=1e-5, atol=1e-6)


def test_single_shard_matches_the_old_one_shot_behaviour(tmp_path):
    """n_shards=1 must be unchanged from before pooling existed."""
    rng = np.random.default_rng(2)
    a = tmp_path / "shard-000000.tar"
    cls_a, ptch_a = _write_shard(a, rng)

    got_scene, got_cls = _norms()
    init_normalizer_stats_from_tar([a], got_scene, got_cls, DEV, 0)

    ref_scene, ref_cls = _norms()
    ref_scene.set_stats(torch.from_numpy(ptch_a))
    ref_cls.set_stats(torch.from_numpy(cls_a).unsqueeze(1))

    torch.testing.assert_close(got_scene.mean, ref_scene.mean, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(got_scene.var, ref_scene.var, rtol=1e-5, atol=1e-6)


def test_max_samples_is_per_shard(tmp_path):
    rng = np.random.default_rng(3)
    a, b = tmp_path / "shard-000000.tar", tmp_path / "shard-000001.tar"
    cls_a, ptch_a = _write_shard(a, rng)
    cls_b, ptch_b = _write_shard(b, rng)

    got_scene, got_cls = _norms()
    init_normalizer_stats_from_tar([a, b], got_scene, got_cls, DEV, 5)

    ref_scene, ref_cls = _norms()
    ref_scene.set_stats(torch.from_numpy(np.concatenate([ptch_a[:5], ptch_b[:5]])))

    torch.testing.assert_close(got_scene.mean, ref_scene.mean, rtol=1e-5, atol=1e-6)


def test_normalizer_shard_paths_is_seed_and_job_independent(tmp_path):
    """The whole point of the fixed selection: same shards for every run and rank."""
    from canvit_train.distill.data.webdataset import WebDatasetTrainLoader

    for i in range(5):
        _write_shard(tmp_path / f"shard-{i:06d}.tar", np.random.default_rng(i))
    (tmp_path / "info.json").write_text(
        '{"keys": ["jpg", "json", "cls.npy", "ptch.npy"], "images_per_shard": 12}'
    )

    def paths(seed, job_index, n):
        loader = WebDatasetTrainLoader(
            train_dir=tmp_path, seed=seed, job_index=job_index, steps_per_job=2,
            batch_size_per_gpu=6, image_size=32, world_size=1, rank=0, num_workers=1,
        )
        return loader.normalizer_shard_paths(n)

    base = paths(seed=0, job_index=0, n=2)
    assert [p.name for p in base] == ["shard-000000.tar", "shard-000001.tar"]
    assert paths(seed=7, job_index=0, n=2) == base    # seed must not matter
    assert paths(seed=0, job_index=3, n=2) == base    # nor the array position
    # the last shard is treated as partial by compute_schedule_slice, so it is excluded
    assert len(paths(seed=0, job_index=0, n=4)) == 4
