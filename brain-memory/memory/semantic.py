"""
Semantic Knowledge Graph — typed entity-relation graph with embeddings.

Wraps a NetworkX ``DiGraph`` and provides Pydantic models for nodes
(concepts / entities) and edges (typed relations).  Used by the
spreading activation engine and consolidation pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field


# ────────────────────────────────────────────────────────────────────
# Pydantic models
# ────────────────────────────────────────────────────────────────────


class SemanticNode(BaseModel):
    """A node in the semantic knowledge graph."""

    id: str
    label: str = ""
    node_type: str = "entity"  # entity | concept | topic
    embedding: list[float] = Field(default_factory=list)
    activation: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticEdge(BaseModel):
    """A directed edge (relation) between two semantic nodes."""

    source: str
    target: str
    relation: str = "related_to"
    weight: float = 1.0
    confidence: float = 1.0
    evidence: list[str] = Field(
        default_factory=list,
        description="IDs of episodic entries that support this relation.",
    )


# ────────────────────────────────────────────────────────────────────
# Semantic Graph
# ────────────────────────────────────────────────────────────────────


class SemanticGraph:
    """Typed knowledge graph backed by ``networkx.DiGraph``.

    Nodes carry ``SemanticNode`` attributes; edges carry
    ``SemanticEdge`` attributes.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    # ── nodes ───────────────────────────────────────────────────────

    def upsert_node(self, node: SemanticNode) -> None:
        """Insert or update a node.  Existing attributes are merged."""
        if self._graph.has_node(node.id):
            existing = self._graph.nodes[node.id]
            existing.update(node.model_dump(exclude_defaults=False))
        else:
            self._graph.add_node(node.id, **node.model_dump())

    def get_node(self, node_id: str) -> SemanticNode | None:
        """Return a node by ID, or *None*."""
        if node_id not in self._graph:
            return None
        data = dict(self._graph.nodes[node_id])
        data["id"] = node_id  # ensure id survives serialisation round-trips
        return SemanticNode(**data)

    def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    def all_nodes(self) -> list[SemanticNode]:
        """Return every node as a ``SemanticNode``."""
        nodes = []
        for nid, data in self._graph.nodes(data=True):
            d = dict(data)
            d["id"] = nid
            nodes.append(SemanticNode(**d))
        return nodes

    def node_ids(self) -> list[str]:
        return list(self._graph.nodes)

    # ── edges ───────────────────────────────────────────────────────

    def upsert_edge(self, edge: SemanticEdge) -> None:
        """Insert or update an edge.

        If an edge between *source* → *target* already exists, its
        weight is updated to the max of old and new, confidence is
        averaged, and evidence lists are merged.
        """
        if self._graph.has_edge(edge.source, edge.target):
            existing = self._graph.edges[edge.source, edge.target]
            existing["weight"] = max(existing.get("weight", 0), edge.weight)
            existing["confidence"] = (
                existing.get("confidence", 0) + edge.confidence
            ) / 2.0
            old_evidence = set(existing.get("evidence", []))
            old_evidence.update(edge.evidence)
            existing["evidence"] = list(old_evidence)
            existing["relation"] = edge.relation
        else:
            # Ensure both nodes exist (create stubs if needed)
            if not self._graph.has_node(edge.source):
                self.upsert_node(SemanticNode(id=edge.source, label=edge.source))
            if not self._graph.has_node(edge.target):
                self.upsert_node(SemanticNode(id=edge.target, label=edge.target))
            self._graph.add_edge(edge.source, edge.target, **edge.model_dump())

    def get_edge(self, source: str, target: str) -> SemanticEdge | None:
        if not self._graph.has_edge(source, target):
            return None
        data = dict(self._graph.edges[source, target])
        data["source"] = source
        data["target"] = target
        return SemanticEdge(**data)

    def all_edges(self) -> list[SemanticEdge]:
        edges = []
        for src, tgt, data in self._graph.edges(data=True):
            d = dict(data)
            d["source"] = src
            d["target"] = tgt
            edges.append(SemanticEdge(**d))
        return edges

    # ── traversal ───────────────────────────────────────────────────

    def get_neighbors(
        self, node_id: str, direction: str = "both"
    ) -> list[tuple[str, SemanticEdge]]:
        """Return neighbors and connecting edges.

        Parameters
        ----------
        direction:
            ``"out"`` for successors, ``"in"`` for predecessors,
            ``"both"`` for the union.
        """
        results: list[tuple[str, SemanticEdge]] = []
        if direction in ("out", "both"):
            for _, target, data in self._graph.out_edges(node_id, data=True):
                results.append((target, SemanticEdge(**data)))
        if direction in ("in", "both"):
            for source, _, data in self._graph.in_edges(node_id, data=True):
                results.append((source, SemanticEdge(**data)))
        return results

    def get_subgraph(self, node_ids: Sequence[str]) -> SemanticGraph:
        """Return a new ``SemanticGraph`` induced by the given node IDs."""
        sub = SemanticGraph()
        sub._graph = self._graph.subgraph(node_ids).copy()
        return sub

    # ── activation helpers ──────────────────────────────────────────

    def set_activation(self, node_id: str, value: float) -> None:
        if self._graph.has_node(node_id):
            self._graph.nodes[node_id]["activation"] = value

    def get_activation(self, node_id: str) -> float:
        if self._graph.has_node(node_id):
            return float(self._graph.nodes[node_id].get("activation", 0.0))
        return 0.0

    def reset_activations(self) -> None:
        """Set all node activations to zero."""
        for n in self._graph.nodes:
            self._graph.nodes[n]["activation"] = 0.0

    # ── metrics ─────────────────────────────────────────────────────

    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    @property
    def nx_graph(self) -> nx.DiGraph:
        """Direct access to the underlying NetworkX graph."""
        return self._graph

    # ── serialisation ───────────────────────────────────────────────

    def save_to_json(self, path: str | Path) -> None:
        """Persist the graph as a JSON file."""
        data = nx.node_link_data(self._graph)
        Path(path).write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load_from_json(cls, path: str | Path) -> SemanticGraph:
        """Load a graph from a JSON file produced by ``save_to_json``."""
        raw = json.loads(Path(path).read_text())
        g = cls()
        g._graph = nx.node_link_graph(raw, directed=True)
        return g
