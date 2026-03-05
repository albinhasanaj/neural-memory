"""Tests for memory.neural_working_memory — Transformer WM."""

from __future__ import annotations

import pytest
import torch

from config.settings import settings

_device = settings.resolved_device


class TestTransformerContextEncoder:
    """Tests for the Transformer encoder module."""

    def _make_encoder(self):
        from memory.neural_working_memory import TransformerContextEncoder
        return TransformerContextEncoder(
            input_dim=settings.embedding_dim,
            hidden_dim=settings.gru_hidden_dim,
            num_layers=2,
            num_heads=2,  # smaller for tests
            ff_dim=128,
        )

    def test_output_shapes(self) -> None:
        enc = self._make_encoder()
        seq = torch.randn(1, 5, settings.embedding_dim, device=_device)
        ctx, pred = enc(seq)
        assert ctx.shape == (settings.gru_hidden_dim,)
        assert pred.shape == (settings.embedding_dim,)

    def test_single_turn(self) -> None:
        enc = self._make_encoder()
        seq = torch.randn(1, 1, settings.embedding_dim, device=_device)
        ctx, pred = enc(seq)
        assert ctx.shape == (settings.gru_hidden_dim,)

    def test_different_lengths_produce_different_contexts(self) -> None:
        enc = self._make_encoder()
        short = torch.randn(1, 2, settings.embedding_dim, device=_device)
        long = torch.randn(1, 6, settings.embedding_dim, device=_device)

        ctx_short, _ = enc(short)
        ctx_long, _ = enc(long)
        # Different length inputs should give different context vectors
        assert not torch.allclose(ctx_short, ctx_long, atol=1e-3)


class TestTransformerWorkingMemory:
    """Tests for the full TransformerWorkingMemory (drop-in replacement)."""

    def _make_wm(self):
        from memory.neural_working_memory import TransformerWorkingMemory
        return TransformerWorkingMemory(
            capacity=8,
            embedding_dim=settings.embedding_dim,
            hidden_dim=settings.gru_hidden_dim,
        )

    def test_update(self) -> None:
        wm = self._make_wm()
        emb = torch.randn(settings.embedding_dim, device=_device)
        ctx, pred = wm.update(emb)
        assert ctx.shape == (settings.gru_hidden_dim,)
        assert pred.shape == (settings.embedding_dim,)

    def test_context_vector_property(self) -> None:
        wm = self._make_wm()
        assert wm.context_vector is None
        wm.update(torch.randn(settings.embedding_dim, device=_device))
        assert wm.context_vector is not None
        assert wm.context_vector.shape == (settings.gru_hidden_dim,)

    def test_predict_next_embedding(self) -> None:
        wm = self._make_wm()
        wm.update(torch.randn(settings.embedding_dim, device=_device))
        pred = wm.predict_next_embedding()
        assert pred.shape == (settings.embedding_dim,)

    def test_predict_empty_raises(self) -> None:
        wm = self._make_wm()
        with pytest.raises(RuntimeError):
            wm.predict_next_embedding()

    def test_clear(self) -> None:
        wm = self._make_wm()
        wm.update(torch.randn(settings.embedding_dim, device=_device))
        wm.clear()
        assert wm.context_vector is None
        assert wm.buffer.size == 0

    def test_multiple_updates(self) -> None:
        """Context should change with each new turn."""
        wm = self._make_wm()
        contexts = []
        for i in range(5):
            torch.manual_seed(i)
            ctx, _ = wm.update(torch.randn(settings.embedding_dim, device=_device))
            contexts.append(ctx.clone())

        # Each context should be different
        for i in range(1, len(contexts)):
            assert not torch.allclose(contexts[i], contexts[i - 1], atol=1e-4)
