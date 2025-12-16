"""Tests for fresh ratio survival state management."""

import torch

from avp_vit.train.state import SurvivalBatch


class TestSurvivalBatch:
    def test_init(self) -> None:
        B, C, H, W, D, G = 4, 3, 64, 64, 128, 16
        images = torch.randn(B, C, H, W)
        targets = torch.randn(B, G * G, D)
        hidden = torch.randn(B, G * G, D)
        state = SurvivalBatch.init(images, targets, hidden)
        assert state.images is images
        assert state.targets is targets
        assert state.hidden is hidden

    def test_step_shapes(self) -> None:
        B, K, C, H, W, D, G = 4, 2, 3, 64, 64, 128, 16
        state = SurvivalBatch.init(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
            torch.randn(B, G * G, D),
        )

        new_state = state.step(
            fresh_images=torch.randn(K, C, H, W),
            fresh_targets=torch.randn(K, G * G, D),
            next_hidden=torch.randn(B, G * G, D),
            hidden_init=torch.randn(K, G * G, D),
        )
        assert new_state.images.shape == (B, C, H, W)
        assert new_state.targets.shape == (B, G * G, D)
        assert new_state.hidden.shape == (B, G * G, D)

    def test_fresh_count_equals_batch_resets_all(self) -> None:
        """K=B means all items are replaced."""
        B, C, H, W, D, G = 4, 3, 64, 64, 128, 16
        old_images = torch.randn(B, C, H, W)
        old_targets = torch.randn(B, G * G, D)
        state = SurvivalBatch.init(old_images, old_targets, torch.randn(B, G * G, D))

        fresh_images = torch.randn(B, C, H, W)
        fresh_targets = torch.randn(B, G * G, D)
        hidden_init = torch.randn(B, G * G, D)

        new_state = state.step(
            fresh_images=fresh_images,
            fresh_targets=fresh_targets,
            next_hidden=torch.randn(B, G * G, D),
            hidden_init=hidden_init,
        )
        # All items replaced (though permuted)
        assert torch.equal(new_state.images, fresh_images)
        assert torch.equal(new_state.targets, fresh_targets)
        assert torch.equal(new_state.hidden, hidden_init)

    def test_hidden_detached(self) -> None:
        """Surviving hidden states are detached to cut BPTT across optimizer steps."""
        B, K, C, H, W, D, G = 4, 1, 3, 64, 64, 128, 16
        state = SurvivalBatch.init(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
            torch.randn(B, G * G, D),
        )
        next_hidden = torch.randn(B, G * G, D, requires_grad=True)
        new_state = state.step(
            fresh_images=torch.randn(K, C, H, W),
            fresh_targets=torch.randn(K, G * G, D),
            next_hidden=next_hidden,
            hidden_init=torch.randn(K, G * G, D),
        )
        assert not new_state.hidden.requires_grad

    def test_permutation_is_random(self) -> None:
        """Different calls produce different permutations."""
        B, K, C, H, W, D, G = 8, 2, 3, 64, 64, 128, 16
        images = torch.arange(B).view(B, 1, 1, 1).expand(B, C, H, W).float()
        state = SurvivalBatch.init(
            images,
            torch.randn(B, G * G, D),
            torch.randn(B, G * G, D),
        )

        results = []
        for _ in range(5):
            new_state = state.step(
                fresh_images=torch.zeros(K, C, H, W),
                fresh_targets=torch.randn(K, G * G, D),
                next_hidden=torch.randn(B, G * G, D),
                hidden_init=torch.randn(K, G * G, D),
            )
            # Fresh images are zeros, survivors have original index values
            survivor_order = new_state.images[K:, 0, 0, 0].tolist()
            results.append(tuple(survivor_order))

        # Should have some variation (with overwhelming probability)
        assert len(set(results)) > 1

    def test_shape_mismatch_hidden_raises(self) -> None:
        """Catch shape mismatch between next_hidden and hidden_init."""
        B, K, C, H, W, D, G = 4, 2, 3, 64, 64, 128, 16
        N_REGISTERS = 42
        state = SurvivalBatch.init(
            torch.randn(B, C, H, W),
            torch.randn(B, G * G, D),
            torch.randn(B, N_REGISTERS + G * G, D),
        )

        # Matching shapes - should work
        new_state = state.step(
            fresh_images=torch.randn(K, C, H, W),
            fresh_targets=torch.randn(K, G * G, D),
            next_hidden=torch.randn(B, N_REGISTERS + G * G, D),
            hidden_init=torch.randn(K, N_REGISTERS + G * G, D),
        )
        assert new_state.hidden.shape == (B, N_REGISTERS + G * G, D)

        # Mismatched - should fail
        try:
            state.step(
                fresh_images=torch.randn(K, C, H, W),
                fresh_targets=torch.randn(K, G * G, D),
                next_hidden=torch.randn(B, N_REGISTERS + G * G, D),
                hidden_init=torch.randn(K, G * G, D),  # Wrong shape!
            )
            assert False, "Should have raised"
        except RuntimeError:
            pass
