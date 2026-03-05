"""Tests for memory.trainer — Training Coordinator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from config.settings import settings

_device = settings.resolved_device


class TestTrainingCoordinator:
    """Tests for the multi-component training coordinator."""

    def _make_coordinator(self):
        from memory.trainer import TrainingCoordinator
        coord = TrainingCoordinator()
        coord.initialise()
        return coord

    def test_initialise_all_disabled(self) -> None:
        """With default settings (all neural flags off), no components are enabled
        except gru_predictor which is always on when transformer WM is off (default)."""
        coord = self._make_coordinator()
        for name, comp in coord.components.items():
            if name == "gru_predictor":
                # GRU predictor is enabled by default since transformer WM is off
                assert comp.enabled, "gru_predictor should be enabled when transformer WM is off"
            else:
                assert not comp.enabled, f"{name} should be disabled by default"

    def test_step_with_no_enabled(self) -> None:
        """Step should return just the global step when nothing is enabled."""
        coord = self._make_coordinator()
        result = coord.step()
        assert result["global_step"] == 1
        assert len(result) == 1  # only global_step

    def test_summary(self) -> None:
        coord = self._make_coordinator()
        summary = coord.summary()
        assert "global_step" in summary
        assert "gat" in summary
        assert "vae" in summary
        assert summary["gat"]["enabled"] is False

    def test_checkpoint_empty(self) -> None:
        """Checkpointing with no enabled components should work."""
        coord = self._make_coordinator()
        with tempfile.TemporaryDirectory() as tmpdir:
            coord.save_checkpoint(tmpdir)
            meta_path = Path(tmpdir) / "coordinator_meta.json"
            assert meta_path.exists()

            coord2 = self._make_coordinator()
            coord2.load_checkpoint(tmpdir)
            assert coord2.global_step == 0


class TestCoordinatorWithComponents:
    """Test push/step with manually enabled components."""

    def test_push_gate_experience(self) -> None:
        from memory.trainer import TrainingCoordinator

        coord = TrainingCoordinator()
        # Manually enable gate
        coord.initialise()

        # Even if not enabled, push should not crash
        coord.push_gate_experience(
            torch.randn(settings.embedding_dim),
            torch.randn(settings.gru_hidden_dim),
            torch.rand(4),
            reward=1.0,
        )

    def test_push_pattern_sep_experience(self) -> None:
        from memory.trainer import TrainingCoordinator

        coord = TrainingCoordinator()
        coord.initialise()

        coord.push_pattern_sep_experience(
            torch.randn(settings.embedding_dim),
        )

    def test_loss_history(self) -> None:
        from memory.trainer import TrainingCoordinator

        coord = TrainingCoordinator()
        coord.initialise()
        coord.step()
        history = coord.get_loss_history()
        assert isinstance(history, dict)
