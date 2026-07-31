"""EG-C2F: the paper's strongest heuristic baseline, ported from canvit_eval.

Validated end to end on full ADE20K val (see PAPER_TABLE4_C64): ours
39.58/42.22/43.31/44.05/44.67 vs the paper's 39.6/42.2/43.3/44.1/44.7, max delta 0.05 —
and EG-C2F is deterministic there, so that is a real check of the port. These tests pin
the SCHEDULE and the pick RULE on CPU so a regression is caught without a GPU run.
"""
import torch

from canvit_train.harness.eval_viewpoints import (
    CLOSED_LOOP,
    PAPER_TABLE4_C64,
    EntropyGuidedC2F,
    closed_loop_rollout,
    open_loop_viewpoints,
)

G, B = 8, 3


class _Seg:
    """Minimal stand-in: entropy is read straight from a canvas we control."""

    def __init__(self, per_cell):
        self._per_cell = per_cell   # [B, G, G] desired entropy

    class _Canvit:
        pass

    canvit = _Canvit()


def _chooser(per_cell):
    ch = EntropyGuidedC2F(seg=_Seg(per_cell), batch_size=B, device=torch.device("cpu"),
                          canvas_grid=G)
    ch._entropy = lambda state: per_cell        # bypass the probe; test the RULE
    return ch


def _state():
    return type("S", (), {"canvas": torch.zeros(B, G * G, 4)})()


def test_it_is_registered_as_closed_loop_and_refused_by_the_open_loop_path():
    assert "entropy_coarse_to_fine" in CLOSED_LOOP
    try:
        open_loop_viewpoints("entropy_coarse_to_fine", batch_size=B, device=torch.device("cpu"),
                             n=5, is_foveated=False, foveated_scale=None)
    except ValueError as e:
        assert "closed-loop" in str(e)
        return
    raise AssertionError("a closed-loop policy must not be precomputable")


def test_t0_is_the_full_scene_anchor():
    ch = _chooser(torch.rand(B, G, G))
    vp = ch(0, None)
    assert torch.allclose(vp.centers, torch.zeros(B, 2))
    assert torch.allclose(vp.scales, torch.ones(B))


def test_it_picks_the_highest_entropy_quadrant_first():
    """Entropy concentrated in the bottom-right => that quadrant must be chosen at t1."""
    per_cell = torch.zeros(B, G, G)
    per_cell[:, G // 2:, G // 2:] = 10.0
    ch = _chooser(per_cell)
    ch(0, None)
    vp = ch(1, _state())
    assert (vp.centers[:, 0] > 0).all() and (vp.centers[:, 1] > 0).all(), vp.centers
    assert torch.allclose(vp.scales, torch.full((B,), 0.5))


def test_no_quadrant_is_revisited_within_the_level():
    """The `visited` mask is the whole point: a constant entropy map would otherwise pick
    the same argmax four times and the trajectory would stall on one quadrant."""
    ch = _chooser(torch.ones(B, G, G))          # perfectly flat -> argmax ties
    ch(0, None)
    seen = [tuple(ch(t, _state()).centers[0].tolist()) for t in range(1, 5)]
    assert len(set(seen)) == 4, f"revisited a quadrant: {seen}"


def test_each_rollout_needs_a_fresh_chooser():
    """Reusing one across batches would carry `visited` over and exclude real candidates."""
    ch = _chooser(torch.ones(B, G, G))
    ch(0, None)
    for t in range(1, 5):
        ch(t, _state())
    # a 5th pick at this level has nothing left -> all -inf; proves state is per-rollout
    assert bool(ch.visited[1].all()), "all 4 quadrants should be marked visited"


def test_closed_loop_rollout_calls_chooser_then_advance_in_order():
    calls = []

    def chooser(t, state):
        calls.append(("choose", t, state is None))
        return type("V", (), {"centers": None, "scales": None})()

    def advance(vp, state, t):
        calls.append(("advance", t, state is None))
        return f"state{t}"

    taken = closed_loop_rollout(chooser=chooser, advance=advance, n=3)
    assert len(taken) == 3
    assert calls[0] == ("choose", 0, True), "t0 chooser must see state=None"
    assert calls[1] == ("advance", 0, True), "t0 advance owns state init"
    assert calls[2] == ("choose", 1, False)


def test_paper_targets_are_recorded_for_the_reproducible_rows_only():
    assert set(PAPER_TABLE4_C64) == {"entropy_coarse_to_fine", "coarse_to_fine"}
    assert PAPER_TABLE4_C64["entropy_coarse_to_fine"][0] == 39.6
    assert "random" not in PAPER_TABLE4_C64, "our `random` is NOT F-IID — measured +0.2..+0.4 above"
