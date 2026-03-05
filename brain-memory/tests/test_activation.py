"""Tests for spreading activation: multi-channel seeding, propagation, lateral inhibition."""

from __future__ import annotations

import torch
import pytest

from config.settings import settings
from memory.activation import SeedHints, SpreadingActivationEngine
from memory.semantic import SemanticGraph, SemanticNode, SemanticEdge


@pytest.fixture
def activation_graph() -> SemanticGraph:
    """Small deterministic graph for activation tests."""
    g = SemanticGraph()
    dim = settings.embedding_dim

    # Create nodes with known embeddings
    torch.manual_seed(42)
    nodes = {
        "A": ("Alice", torch.randn(dim)),
        "B": ("Bob", torch.randn(dim)),
        "C": ("api-key", torch.randn(dim)),
        "D": ("project-x", torch.randn(dim)),
    }
    for nid, (label, emb) in nodes.items():
        g.upsert_node(SemanticNode(id=nid, label=label, embedding=emb.tolist()))

    # A → B → C, A → D
    g.upsert_edge(SemanticEdge(source="A", target="B", weight=0.8))
    g.upsert_edge(SemanticEdge(source="B", target="C", weight=0.6))
    g.upsert_edge(SemanticEdge(source="A", target="D", weight=0.5))

    return g


class TestMultiChannelSeeding:
    """Tests for the four seed channels."""

    def test_entity_seeding_activates_matching_node(self, activation_graph: SemanticGraph) -> None:
        engine = SpreadingActivationEngine(activation_graph, seed_threshold=0.0)
        hints = SeedHints(entities=["Alice"])
        ctx = torch.randn(settings.embedding_dim)
        activation = engine.seed_nodes(ctx, hints=hints)

        a_idx = engine._node_ids.index("A")
        # Entity channel should activate the "Alice" node
        assert activation[a_idx].item() > 0

    def test_entity_seeding_case_insensitive(self, activation_graph: SemanticGraph) -> None:
        engine = SpreadingActivationEngine(activation_graph, seed_threshold=0.0)
        hints = SeedHints(entities=["alice"])
        ctx = torch.randn(settings.embedding_dim)
        activation = engine.seed_nodes(ctx, hints=hints)

        a_idx = engine._node_ids.index("A")
        assert activation[a_idx].item() > 0

    def test_intent_seeding_activates_target(self, activation_graph: SemanticGraph) -> None:
        engine = SpreadingActivationEngine(activation_graph, seed_threshold=0.0)
        # Simulates "what was my api key?"
        hints = SeedHints(
            intent_targets=["api key"],
            intent_confidence=1.0,
        )
        ctx = torch.randn(settings.embedding_dim)
        activation = engine.seed_nodes(ctx, hints=hints)

        c_idx = engine._node_ids.index("C")  # "api-key" node
        assert activation[c_idx].item() > 0

    def test_intent_seeding_zero_when_no_confidence(self, activation_graph: SemanticGraph) -> None:
        engine = SpreadingActivationEngine(activation_graph, seed_threshold=0.0)
        # Test the intent channel directly — zero confidence should produce all zeros
        activation = engine._seed_intent_cues(["api key"], confidence=0.0)
        assert activation.sum().item() == 0.0

    def test_working_memory_seeding_shape(self, activation_graph: SemanticGraph) -> None:
        engine = SpreadingActivationEngine(activation_graph, seed_threshold=0.0)
        dim = settings.embedding_dim
        wm = [torch.randn(dim) for _ in range(3)]
        hints = SeedHints(working_memory_embeddings=wm)
        ctx = torch.randn(dim)
        activation = engine.seed_nodes(ctx, hints=hints)
        assert activation.shape == (4,)

    def test_embedding_fallback_when_no_hints(self, activation_graph: SemanticGraph) -> None:
        """Without hints, seed_nodes should still work (backward compat)."""
        engine = SpreadingActivationEngine(activation_graph, seed_threshold=0.0)
        ctx = torch.randn(settings.embedding_dim)
        activation = engine.seed_nodes(ctx)
        assert activation.shape == (4,)

    def test_combined_channels_higher_than_single(self, activation_graph: SemanticGraph) -> None:
        """Entity + intent together should produce higher activation for matched node."""
        engine = SpreadingActivationEngine(activation_graph, seed_threshold=0.0)

        # Entity-only
        hints_e = SeedHints(entities=["Alice"])
        ctx = torch.randn(settings.embedding_dim)
        act_e = engine.seed_nodes(ctx, hints=hints_e)

        # Entity + intent (both name Alice)
        hints_both = SeedHints(
            entities=["Alice"],
            intent_targets=["alice"],
            intent_confidence=1.0,
        )
        act_both = engine.seed_nodes(ctx, hints=hints_both)

        a_idx = engine._node_ids.index("A")
        assert act_both[a_idx].item() >= act_e[a_idx].item()


class TestPropagationAndInhibition:
    """Propagation and lateral inhibition (unchanged behaviour)."""

    def test_seed_nodes_shape(self, activation_graph: SemanticGraph) -> None:
        engine = SpreadingActivationEngine(activation_graph, seed_threshold=0.0)
        ctx = torch.randn(settings.embedding_dim)
        activation = engine.seed_nodes(ctx)
        assert activation.shape == (4,)

    def test_seed_with_identical_vector(self, activation_graph: SemanticGraph) -> None:
        engine = SpreadingActivationEngine(activation_graph, seed_threshold=0.5)
        # Use node A's own embedding as context → should strongly activate A
        node_a = activation_graph.get_node("A")
        assert node_a is not None
        ctx = torch.tensor(node_a.embedding)
        activation = engine.seed_nodes(ctx)
        # Node A (index 0 in node_ids) should have high activation
        a_idx = engine._node_ids.index("A")
        assert activation[a_idx].item() > 0.9 * settings.seed_weight_embedding

    def test_propagate_spreads_activation(self, activation_graph: SemanticGraph) -> None:
        engine = SpreadingActivationEngine(activation_graph, max_iterations=2, decay_factor=0.5)
        # Manually seed only A
        n = len(engine._node_ids)
        activation = torch.zeros(n)
        a_idx = engine._node_ids.index("A")
        activation[a_idx] = 1.0

        result = engine.propagate(activation)
        # B and D should have received some activation from A
        b_idx = engine._node_ids.index("B")
        d_idx = engine._node_ids.index("D")
        assert result[b_idx].item() > 0
        assert result[d_idx].item() > 0

    def test_lateral_inhibition_top_k(self, activation_graph: SemanticGraph) -> None:
        engine = SpreadingActivationEngine(activation_graph, top_k=2)
        activation = torch.tensor([0.5, 0.9, 0.3, 0.7])
        result = engine.lateral_inhibition(activation)
        # Only top 2 should remain (indices 1 and 3)
        assert (result > 0).sum().item() == 2
        assert result[1].item() == pytest.approx(0.9)
        assert result[3].item() == pytest.approx(0.7)

    def test_activate_returns_sorted_results(self, activation_graph: SemanticGraph) -> None:
        engine = SpreadingActivationEngine(
            activation_graph, seed_threshold=0.0, top_k=4
        )
        ctx = torch.randn(settings.embedding_dim)
        results = engine.activate(ctx)
        # Results should be sorted descending by activation strength
        strengths = [s for _, s in results]
        assert strengths == sorted(strengths, reverse=True)

    def test_activate_with_hints(self, activation_graph: SemanticGraph) -> None:
        """activate() forwards hints correctly."""
        engine = SpreadingActivationEngine(
            activation_graph, seed_threshold=0.0, top_k=4
        )
        hints = SeedHints(entities=["Alice"])
        ctx = torch.randn(settings.embedding_dim)
        results = engine.activate(ctx, hints=hints)
        # A should be among the activated nodes
        activated_ids = {nid for nid, _ in results}
        assert "A" in activated_ids

    def test_empty_graph(self) -> None:
        g = SemanticGraph()
        engine = SpreadingActivationEngine(g)
        results = engine.activate(torch.randn(settings.embedding_dim))
        assert results == []

    def test_rebuild_updates_adjacency(self, activation_graph: SemanticGraph) -> None:
        engine = SpreadingActivationEngine(activation_graph)
        original_node_count = len(engine._node_ids)
        # Add a new node and edge
        activation_graph.upsert_node(
            SemanticNode(id="E", label="Echo", embedding=torch.randn(settings.embedding_dim).tolist())
        )
        activation_graph.upsert_edge(SemanticEdge(source="C", target="E", weight=0.7))
        engine.rebuild()
        assert len(engine._node_ids) == original_node_count + 1
