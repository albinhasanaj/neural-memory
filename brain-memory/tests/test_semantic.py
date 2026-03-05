"""Tests for semantic knowledge graph: CRUD, edge upsert, subgraph extraction."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from memory.semantic import SemanticEdge, SemanticGraph, SemanticNode


class TestSemanticGraph:
    """Graph operations."""

    def test_upsert_and_get_node(self, sample_graph: SemanticGraph) -> None:
        node = sample_graph.get_node("python")
        assert node is not None
        assert node.label == "Python"

    def test_node_count(self, sample_graph: SemanticGraph) -> None:
        assert sample_graph.num_nodes == 5

    def test_edge_count(self, sample_graph: SemanticGraph) -> None:
        assert sample_graph.num_edges == 4

    def test_has_node(self, sample_graph: SemanticGraph) -> None:
        assert sample_graph.has_node("python")
        assert not sample_graph.has_node("nonexistent")

    def test_get_edge(self, sample_graph: SemanticGraph) -> None:
        edge = sample_graph.get_edge("user", "python")
        assert edge is not None
        assert edge.relation == "prefers"
        assert edge.confidence == 0.72

    def test_edge_upsert_merges(self, sample_graph: SemanticGraph) -> None:
        # Upsert with higher weight and new evidence
        sample_graph.upsert_edge(
            SemanticEdge(
                source="user",
                target="python",
                relation="loves",
                weight=0.95,
                confidence=0.9,
                evidence=["ep_new"],
            )
        )
        edge = sample_graph.get_edge("user", "python")
        assert edge is not None
        assert edge.weight == 0.95  # max of 0.9, 0.95
        assert edge.relation == "loves"
        assert "ep_new" in edge.evidence

    def test_get_neighbors(self, sample_graph: SemanticGraph) -> None:
        neighbors = sample_graph.get_neighbors("user", direction="out")
        target_ids = [nid for nid, _ in neighbors]
        assert "python" in target_ids
        assert "wordpress" in target_ids

    def test_get_subgraph(self, sample_graph: SemanticGraph) -> None:
        sub = sample_graph.get_subgraph(["user", "python", "fastapi"])
        assert sub.num_nodes == 3
        # Only the edge user→python and python→fastapi should survive
        # user→fastapi too if it exists
        assert sub.num_edges >= 1

    def test_activation_set_and_reset(self, sample_graph: SemanticGraph) -> None:
        sample_graph.set_activation("python", 0.75)
        assert sample_graph.get_activation("python") == 0.75
        sample_graph.reset_activations()
        assert sample_graph.get_activation("python") == 0.0

    def test_save_and_load_json(self, sample_graph: SemanticGraph) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"
            sample_graph.save_to_json(path)
            loaded = SemanticGraph.load_from_json(path)
            assert loaded.num_nodes == sample_graph.num_nodes
            assert loaded.num_edges == sample_graph.num_edges
            node = loaded.get_node("python")
            assert node is not None
            assert node.label == "Python"

    def test_all_nodes_returns_pydantic_models(self, sample_graph: SemanticGraph) -> None:
        nodes = sample_graph.all_nodes()
        assert len(nodes) == 5
        assert all(isinstance(n, SemanticNode) for n in nodes)

    def test_all_edges_returns_pydantic_models(self, sample_graph: SemanticGraph) -> None:
        edges = sample_graph.all_edges()
        assert len(edges) == 4
        assert all(isinstance(e, SemanticEdge) for e in edges)
