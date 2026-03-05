"""Tests for working memory: ring buffer, GRU encoding, context vector shape."""

from __future__ import annotations

import torch
import pytest

from config.settings import settings
from memory.working_memory import GRUContextEncoder, RingBuffer, WorkingMemory


class TestRingBuffer:
    """Ring buffer behaviour."""

    def test_append_and_size(self) -> None:
        buf = RingBuffer(capacity=3)
        assert buf.size == 0
        buf.append(torch.randn(settings.embedding_dim))
        assert buf.size == 1

    def test_eviction_at_capacity(self) -> None:
        buf = RingBuffer(capacity=2)
        v1 = torch.ones(settings.embedding_dim)
        v2 = torch.ones(settings.embedding_dim) * 2
        v3 = torch.ones(settings.embedding_dim) * 3
        buf.append(v1)
        buf.append(v2)
        buf.append(v3)
        assert buf.size == 2
        t = buf.as_tensor()
        # v1 should have been evicted; first item should be v2
        assert torch.allclose(t[0].cpu(), v2)
        assert torch.allclose(t[1].cpu(), v3)

    def test_as_tensor_shape(self) -> None:
        buf = RingBuffer(capacity=5)
        for _ in range(3):
            buf.append(torch.randn(settings.embedding_dim))
        t = buf.as_tensor()
        assert t.shape == (3, settings.embedding_dim)

    def test_empty_tensor(self) -> None:
        buf = RingBuffer(capacity=5)
        t = buf.as_tensor()
        assert t.shape == (0, settings.embedding_dim)

    def test_clear(self) -> None:
        buf = RingBuffer(capacity=3)
        buf.append(torch.randn(settings.embedding_dim))
        buf.clear()
        assert buf.size == 0


class TestGRUContextEncoder:
    """GRU encoder produces correctly shaped outputs."""

    def test_output_shapes(self) -> None:
        enc = GRUContextEncoder(input_dim=settings.embedding_dim, hidden_dim=128)
        seq = torch.randn(1, 4, settings.embedding_dim)  # batch=1, T=4
        ctx, pred = enc(seq)
        assert ctx.shape == (128,)
        assert pred.shape == (settings.embedding_dim,)

    def test_single_step(self) -> None:
        enc = GRUContextEncoder(input_dim=settings.embedding_dim, hidden_dim=64)
        seq = torch.randn(1, 1, settings.embedding_dim)
        ctx, pred = enc(seq)
        assert ctx.shape == (64,)


class TestWorkingMemory:
    """End-to-end working memory update cycle."""

    def test_update_returns_context_and_prediction(self) -> None:
        wm = WorkingMemory(capacity=4, embedding_dim=settings.embedding_dim, hidden_dim=64)
        emb = torch.randn(settings.embedding_dim)
        ctx, pred = wm.update(emb)
        assert ctx.shape == (64,)
        assert pred.shape == (settings.embedding_dim,)
        assert wm.context_vector is not None

    def test_predict_next_embedding(self) -> None:
        wm = WorkingMemory(capacity=4, embedding_dim=settings.embedding_dim, hidden_dim=64)
        wm.update(torch.randn(settings.embedding_dim))
        pred = wm.predict_next_embedding()
        assert pred.shape == (settings.embedding_dim,)

    def test_predict_empty_raises(self) -> None:
        wm = WorkingMemory()
        with pytest.raises(RuntimeError):
            wm.predict_next_embedding()

    def test_clear(self) -> None:
        wm = WorkingMemory()
        wm.update(torch.randn(settings.embedding_dim))
        wm.clear()
        assert wm.buffer.size == 0
        assert wm.context_vector is None
