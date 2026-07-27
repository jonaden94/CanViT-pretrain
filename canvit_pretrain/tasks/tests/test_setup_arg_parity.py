"""The old loop and the harness must call shared setup helpers with the same arguments.

Guards the exp23 bug class: both stacks called the identical
`init_normalizer_stats_from_tar`, but the harness passed
`self.cfg.normalizer_max_samples or 512`, clobbering the documented `0` = "whole shard"
sentinel. Step-level parity could not see it (it compares rollouts from an ALREADY-built
state), so it took ~24k steps of a production run to surface.
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "unification_docs" / "setup_arg_parity.py"


def _mod():
    spec = importlib.util.spec_from_file_location("setup_arg_parity", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.skipif(not _SCRIPT.exists(), reason="setup_arg_parity.py not present")
def test_shared_setup_helpers_receive_identical_args():
    problems = _mod().compare()
    assert not problems, "setup-arg parity broken:\n" + "\n".join(problems)


@pytest.mark.skipif(not _SCRIPT.exists(), reason="setup_arg_parity.py not present")
def test_the_check_actually_catches_a_divergent_arg(tmp_path):
    """A check that cannot fail is worse than no check. Reproduce the exp23 shape —
    same callee, one side wrapping the value in `or 512` — and require a report."""
    m = _mod()
    old = tmp_path / "loop.py"
    new = tmp_path / "task.py"
    old.write_text(
        "def go(cfg, scene_norm, cls_norm, train_loader):\n"
        "    init_normalizer_stats_from_tar(\n"
        "        train_loader.first_shard_path(), scene_norm, cls_norm,\n"
        "        cfg.device, cfg.normalizer_max_samples)\n"
    )
    new.write_text(
        "class T:\n"
        "    def go(self, train):\n"
        "        init_normalizer_stats_from_tar(\n"
        "            train.first_shard_path(), self.scene_norm, self.cls_norm,\n"
        "            self._device, self.cfg.normalizer_max_samples or 512)\n"
    )
    problems = m.compare(old, new)
    assert problems, "the checker missed a known-divergent argument"
    assert "or 512" in problems[0]
