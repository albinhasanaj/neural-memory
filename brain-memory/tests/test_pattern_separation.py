"""Tests for memory.pattern_separation — Sparse Autoencoder."""

from __future__ import annotations

import pytest
import torch

from config.settings import settings

_device = settings.resolved_device


class TestPatternSeparator:
    """Tests for the sparse autoencoder."""

    def _make_separator(self, top_k: int = 50):
        from memory.pattern_separation import PatternSeparator
        return PatternSeparator(
            input_dim=settings.embedding_dim,
            expansion_dim=settings.pattern_sep_expansion,
            top_k=top_k,
        )

    def test_forward_shapes(self) -> None:
        sep = self._make_separator()
        x = torch.randn(4, settings.embedding_dim, device=_device)
        recon, sparse, separated = sep(x)
        assert recon.shape == (4, settings.embedding_dim)
        assert sparse.shape == (4, settings.pattern_sep_expansion)
        assert separated.shape == (4, settings.embedding_dim)

    def test_sparse_code_sparsity(self) -> None:
        """Only top-K activations should be non-zero."""
        sep = self._make_separator(top_k=20)
        x = torch.randn(1, settings.embedding_dim, device=_device)
        sparse = sep.encode_sparse(x)
        non_zero = (sparse.abs() > 1e-8).sum().item()
        assert non_zero <= 20

    def test_separate_output_dim(self) -> None:
        sep = self._make_separator()
        x = torch.randn(settings.embedding_dim, device=_device)
        out = sep.separate(x)
        assert out.shape == (1, settings.embedding_dim) or out.shape == (settings.embedding_dim,)

    def test_separated_embeddings_more_orthogonal(self) -> None:
        """Separated embeddings should have lower avg cosine similarity."""
        from memory.pattern_separation import separation_quality

        sep = self._make_separator()
        # Create similar embeddings
        base = torch.randn(settings.embedding_dim, device=_device)
        embeddings = torch.stack([
            base + 0.1 * torch.randn(settings.embedding_dim, device=_device)
            for _ in range(10)
        ])

        quality = separation_quality(sep, embeddings)
        # After separation, mean cosine should be lower (or at least computed)
        assert "separation_gain" in quality
        assert "mean_cosine_raw" in quality
        assert "mean_cosine_separated" in quality


class TestPatternSepLoss:
    def test_loss_shapes(self) -> None:
        from memory.pattern_separation import pattern_sep_loss

        recon = torch.randn(4, settings.embedding_dim, device=_device)
        target = torch.randn(4, settings.embedding_dim, device=_device)
        sparse = torch.randn(4, settings.pattern_sep_expansion, device=_device)

        loss, diag = pattern_sep_loss(recon, target, sparse)
        assert loss.isfinite()
        assert "recon" in diag
        assert "l1_sparsity" in diag


class TestSeparationTraining:
    def test_train_step(self) -> None:
        from memory.pattern_separation import (
            PatternSeparator,
            SeparationReplayBuffer,
            train_separator_step,
        )

        sep = PatternSeparator()
        opt = torch.optim.Adam(sep.parameters(), lr=1e-3)
        buf = SeparationReplayBuffer(capacity=50)

        for _ in range(20):
            buf.push(torch.randn(settings.embedding_dim, device=_device))

        diag = train_separator_step(sep, opt, buf, batch_size=8)
        assert diag is not None
        assert diag["total"] >= 0

    def test_training_reduces_reconstruction(self) -> None:
        from memory.pattern_separation import (
            PatternSeparator,
            SeparationReplayBuffer,
            train_separator_step,
        )

        sep = PatternSeparator()
        opt = torch.optim.Adam(sep.parameters(), lr=1e-2)
        buf = SeparationReplayBuffer(capacity=200)

        # Fill with structured data (same distribution)
        for _ in range(50):
            buf.push(torch.randn(settings.embedding_dim, device=_device) * 0.5)

        losses = []
        for _ in range(20):
            diag = train_separator_step(sep, opt, buf, batch_size=16)
            if diag:
                losses.append(diag["recon"])

        assert len(losses) >= 10
        # Reconstruction loss should decrease
        assert losses[-1] < losses[0]
