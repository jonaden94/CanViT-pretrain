"""CPU tests for distill's two shard flavours and the normalizer-init rules.

WebDataset shards come in two kinds and the task must handle both (train/loop.py
546-575, 639-643):
  - **with features** (`cls.npy`/`ptch.npy`): teacher targets are precomputed.
  - **raw** (jpg+json only): the frozen teacher produces targets ON THE FLY, both to
    seed the standardizers and for every training batch. This is the exp21 path.
Plus `cfg.reset_normalizer`, which must re-init even when the checkpoint carried stats.

The loader and the teacher are stubbed (a real one needs tar shards + a GPU); the
dispatch, the reset rule and the on-the-fly bind are the real code. The raw path is
exercised against real shards by `unification_docs/harness_run_raw_shards.py`.
"""

from types import SimpleNamespace

import torch

from canvit_train.tasks.distill.task import DistillRunTask
from canvit_train.train.config import Config
from canvit_train.train.data.webdataset import WebDatasetTrainLoader

_G, _D, _BS, _PATCH = 8, 16, 2, 16


class _IdentityNorm:
    """Standardizer stub: bind() both standardizes the targets and hands the engine the
    destandardizer for the raw-space cosine metrics."""

    def __call__(self, x):
        return x

    def destandardize(self, x):
        return x


def _task(tmp_path, *, initialized, reset=False, has_features=True, normalizer_shards=4):
    cfg = Config(webdataset_dir="/nonexistent", batch_size_per_gpu=_BS, steps_per_job=64,
                 canvas_patch_grid_size=_G, reset_normalizer=reset,
                 normalizer_shards=normalizer_shards)
    t = DistillRunTask(cfg)
    t.scene_norm = SimpleNamespace(initialized=initialized)
    t.cls_norm = SimpleNamespace()
    t._device = torch.device("cpu")
    t._model = SimpleNamespace(backbone=SimpleNamespace(patch_size_px=_PATCH))
    loader = object.__new__(WebDatasetTrainLoader)
    loader.samples_per_shard, loader.has_features = 512, has_features
    # `normalizer_shard_paths` globs train_dir rather than reading shard_files (the
    # schedule slice), so the stub needs a real directory of shard-shaped names. 8 of
    # them: the default is 4 and the last is excluded as partial, so this leaves room
    # to raise the default without the fixture silently becoming the binding constraint.
    for i in range(8):
        (tmp_path / f"shard-{i:06d}.tar").touch()
    loader.train_dir = tmp_path
    loader.shard_files = [tmp_path / "shard-000000.tar"]
    return t, loader


def _stub_init(monkeypatch, task, loader):
    """Patch create_loaders + both normalizer initialisers; return the call records."""
    seen: dict = {"tar": [], "raw": []}
    monkeypatch.setattr("canvit_train.train.data.create_loaders",
                        lambda cfg, start_step, **kw: SimpleNamespace(train=loader, val=None))
    monkeypatch.setattr("canvit_train.train.data.webdataset.init_normalizer_stats_from_tar",
                        lambda *a, **k: seen["tar"].append((a, k)))
    monkeypatch.setattr("canvit_train.train.data.webdataset.init_normalizer_stats_from_tar_raw",
                        lambda *a, **k: seen["raw"].append((a, k)))
    monkeypatch.setattr(task, "_teacher_targets", lambda imgs, sz: SimpleNamespace(
        patches=torch.zeros(imgs.shape[0], _G * _G, _D), cls=torch.zeros(imgs.shape[0], _D)))
    return seen


# --- which initialiser, and when -------------------------------------------
def test_feature_shards_use_the_precomputed_initializer(monkeypatch, tmp_path):
    t, loader = _task(tmp_path, initialized=False, has_features=True)
    seen = _stub_init(monkeypatch, t, loader)
    t.build_loaders(world_size=1, rank=0)
    assert len(seen["tar"]) == 1 and not seen["raw"]


def test_raw_shards_use_the_on_the_fly_initializer(monkeypatch, tmp_path):
    """Raw shards have no cls.npy/ptch.npy — calling the precomputed initialiser on them
    would read keys that aren't there."""
    t, loader = _task(tmp_path, initialized=False, has_features=False)
    seen = _stub_init(monkeypatch, t, loader)
    t.build_loaders(world_size=1, rank=0)
    assert len(seen["raw"]) == 1 and not seen["tar"]
    assert seen["raw"][0][1]["image_size"] == _G * _PATCH  # decoded at the scene resolution


def test_normalizer_shards_reaches_the_initializer(monkeypatch, tmp_path):
    """cfg.normalizer_shards must be threaded through, and the selection must be the
    sorted head of train_dir — NOT loader.shard_files, which is the seed-dependent
    schedule slice and would make the stats a function of the seed."""
    t, loader = _task(tmp_path, initialized=False, normalizer_shards=3)
    seen = _stub_init(monkeypatch, t, loader)
    t.build_loaders(world_size=1, rank=0)
    paths = seen["tar"][0][0][0]
    assert [p.name for p in paths] == [
        "shard-000000.tar", "shard-000001.tar", "shard-000002.tar"
    ]


def test_initialized_normalizer_is_not_reinitialized(monkeypatch, tmp_path):
    t, loader = _task(tmp_path, initialized=True)
    seen = _stub_init(monkeypatch, t, loader)
    t.build_loaders(world_size=1, rank=0)
    assert not seen["tar"] and not seen["raw"]


def test_reset_normalizer_forces_reinit(monkeypatch, tmp_path):
    """cfg.reset_normalizer must re-init even though the checkpoint carried stats —
    otherwise the flag silently does nothing on resume."""
    t, loader = _task(tmp_path, initialized=True, reset=True)
    seen = _stub_init(monkeypatch, t, loader)
    t.build_loaders(world_size=1, rank=0)
    assert len(seen["tar"]) == 1


# --- per-batch targets ------------------------------------------------------
def test_bind_computes_teacher_targets_when_the_batch_has_none(monkeypatch, tmp_path):
    """Raw shards yield (images, None, None, labels); bind must produce the targets
    rather than call .to() on a None."""
    t, loader = _task(tmp_path, initialized=True, has_features=False)
    _stub_init(monkeypatch, t, loader)
    t.scene_norm = t.cls_norm = _IdentityNorm()
    images = torch.randn(_BS, 3, _G * _PATCH, _G * _PATCH)
    bound = t.bind((images, None, None, None), torch.device("cpu"), model=None, head=None)
    assert bound.distill.scene_target.shape == (_BS, _G * _G, _D)
    assert bound.distill.cls_target.shape == (_BS, _D)


def test_checkpoint_model_config_wins_over_cli_defaults():
    """On RESUME the checkpoint's arch must override this run's config, or the strict
    weight load fails on missing/unexpected keys (train/loop.py 254-261). The full arch
    rides in model_config["canvit"], so the round-trip has to be lossless."""
    from unittest.mock import patch

    cfg = Config(webdataset_dir="/nonexistent", canvas_patch_grid_size=_G)
    cfg.model.canvas_update_mode = "additive"
    saved = DistillRunTask(cfg).model_config(None)          # what run() writes
    assert saved["canvit"]["canvas_update_mode"] == "additive"

    # a NEW run whose CLI default disagrees with the checkpoint
    cfg2 = Config(webdataset_dir="/nonexistent", canvas_patch_grid_size=_G)
    cfg2.model.canvas_update_mode = "convex"
    t2 = DistillRunTask(cfg2)
    # stop before any real model construction — we only pin the config resolution
    with patch("canvit_train.train.model.load_student_backbone", side_effect=RuntimeError("stop")):
        try:
            t2.build_model(torch.device("cpu"), prior_model_config=saved)
        except RuntimeError as e:
            assert "stop" in str(e)
    assert t2.cfg.model.canvas_update_mode == "additive", "checkpoint arch must win"

    # no prior config (FRESH/SEED) => the run's own config stands
    cfg3 = Config(webdataset_dir="/nonexistent", canvas_patch_grid_size=_G)
    cfg3.model.canvas_update_mode = "convex"
    t3 = DistillRunTask(cfg3)
    with patch("canvit_train.train.model.load_student_backbone", side_effect=RuntimeError("stop")):
        try:
            t3.build_model(torch.device("cpu"), prior_model_config=None)
        except RuntimeError:
            pass
    assert t3.cfg.model.canvas_update_mode == "convex"


def test_bind_uses_precomputed_targets_when_present(monkeypatch, tmp_path):
    t, loader = _task(tmp_path, initialized=True, has_features=True)
    _stub_init(monkeypatch, t, loader)
    t.scene_norm = t.cls_norm = _IdentityNorm()
    patches, cls = torch.randn(_BS, _G * _G, _D), torch.randn(_BS, _D)
    bound = t.bind((torch.randn(_BS, 3, 8, 8), patches, cls, None), torch.device("cpu"),
                   model=None, head=None)
    assert torch.allclose(bound.distill.scene_target, patches)  # NOT recomputed
