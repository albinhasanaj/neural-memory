"""Tests for salience scoring: individual signals and combined scoring."""

from __future__ import annotations

import torch
import pytest

from config.settings import settings
from memory.salience import (
    SalienceScorer,
    compute_entity_density,
    compute_novelty,
    compute_prediction_error,
    detect_emphasis,
)
from memory.semantic import SemanticGraph, SemanticNode


class TestNovelty:
    """Novelty signal computation."""

    def test_empty_graph_is_fully_novel(self) -> None:
        g = SemanticGraph()
        emb = torch.randn(settings.embedding_dim)
        assert compute_novelty(emb, g) == 1.0

    def test_identical_node_is_zero_novelty(self) -> None:
        g = SemanticGraph()
        emb = torch.randn(settings.embedding_dim)
        g.upsert_node(SemanticNode(id="x", label="x", embedding=emb.tolist()))
        novelty = compute_novelty(emb, g)
        assert novelty < 0.05  # near zero

    def test_distant_embedding_is_novel(self) -> None:
        g = SemanticGraph()
        g.upsert_node(
            SemanticNode(id="x", label="x", embedding=torch.ones(settings.embedding_dim).tolist())
        )
        emb = -torch.ones(settings.embedding_dim)  # opposite direction
        novelty = compute_novelty(emb, g)
        assert novelty > 0.5


class TestPredictionError:
    """Prediction error signal."""

    def test_no_prediction_returns_neutral(self) -> None:
        assert compute_prediction_error(torch.randn(settings.embedding_dim), None) == 0.5

    def test_identical_prediction_is_zero(self) -> None:
        emb = torch.randn(settings.embedding_dim)
        err = compute_prediction_error(emb, emb)
        assert err < 0.05

    def test_opposite_prediction_is_high(self) -> None:
        emb = torch.ones(settings.embedding_dim)
        pred = -torch.ones(settings.embedding_dim)
        err = compute_prediction_error(emb, pred)
        assert err > 1.0  # cosine distance > 1 when anti-correlated


class TestEmphasis:
    """Emphasis heuristic detection."""

    def test_no_emphasis(self) -> None:
        assert detect_emphasis("hello world") == 0.0

    def test_all_caps(self) -> None:
        score = detect_emphasis("This is VERY IMPORTANT")
        assert score > 0.0

    def test_exclamation(self) -> None:
        score = detect_emphasis("Wow!!")
        assert score > 0.0

    def test_explicit_memory_request(self) -> None:
        score = detect_emphasis("Remember this: my birthday is March 15")
        assert score >= 0.5

    def test_correction(self) -> None:
        score = detect_emphasis("Actually, I meant Python not Java")
        assert score > 0.0

    def test_combined_signals(self) -> None:
        score = detect_emphasis("REMEMBER THIS!! It's VERY important!")
        assert score > 0.5


class TestEntityDensity:
    """Entity density computation."""

    def test_no_entities(self) -> None:
        assert compute_entity_density([], "hello world") == 0.0

    def test_some_entities(self) -> None:
        density = compute_entity_density(["Python", "FastAPI"], "I use Python and FastAPI daily")
        assert 0 < density < 1.0

    def test_empty_text(self) -> None:
        assert compute_entity_density(["x"], "") == 0.0

    def test_capped_at_one(self) -> None:
        assert compute_entity_density(["a", "b", "c"], "one") == 1.0


class TestSalienceScorer:
    """Combined salience scoring."""

    def test_weighted_sum_range(self) -> None:
        scorer = SalienceScorer(use_mlp=False)
        signals = torch.tensor([0.5, 0.5, 0.5, 0.5])
        score = scorer(signals)
        assert 0 <= score.item() <= 1.0

    def test_zero_signals(self) -> None:
        scorer = SalienceScorer(use_mlp=False)
        signals = torch.tensor([0.0, 0.0, 0.0, 0.0])
        assert scorer(signals).item() == 0.0

    def test_max_signals(self) -> None:
        scorer = SalienceScorer(use_mlp=False)
        signals = torch.tensor([1.0, 1.0, 1.0, 1.0])
        assert scorer(signals).item() == 1.0

    def test_mlp_mode(self) -> None:
        scorer = SalienceScorer(use_mlp=True)
        signals = torch.tensor([0.5, 0.5, 0.5, 0.5])
        score = scorer(signals)
        assert 0 <= score.item() <= 1.0

    def test_score_method(self, sample_graph: SemanticGraph) -> None:
        scorer = SalienceScorer()
        emb = torch.randn(settings.embedding_dim)
        score = scorer.score(
            embedding=emb,
            predicted=None,
            entities=["Python"],
            text="I really love Python programming",
            graph=sample_graph,
        )
        assert 0 <= score <= 1.0
