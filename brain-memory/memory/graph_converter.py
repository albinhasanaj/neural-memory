"""
Graph Converter — transform NetworkX SemanticGraph + EpisodicStore into PyG Data.

Handles dynamic graph size (nodes added/removed between turns) and provides
efficient incremental updates so the full graph is not rebuilt every turn.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch
from torch import Tensor

from config.settings import settings
from memory.episodic import EpisodicStore
from memory.semantic import SemanticGraph

logger = logging.getLogger(__name__)
_device = settings.resolved_device

# ── Relation / node-type vocabularies ────────────────────────────────

NODE_TYPES = ["entity", "concept", "topic", "episode", "other"]
NODE_TYPE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(NODE_TYPES)}
NUM_NODE_TYPES = len(NODE_TYPES)

RELATION_TYPES = [
    "related_to", "prefers", "dislikes", "uses", "has_framework",
    "alternative_to", "part_of", "instance_of", "causes", "other",
]
RELATION_TO_IDX: dict[str, int] = {r: i for i, r in enumerate(RELATION_TYPES)}
NUM_RELATION_TYPES = len(RELATION_TYPES)


def _onehot(idx: int, size: int) -> list[float]:
    v = [0.0] * size
    if 0 <= idx < size:
        v[idx] = 1.0
    return v


class PyGData:
    """Lightweight PyG-compatible Data container (no torch_geometric dependency).

    Holds the same fields as ``torch_geometric.data.Data`` so either can
    be used.  When ``torch_geometric`` is available, ``to_pyg()`` returns
    a proper ``Data`` object.
    """

    def __init__(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        node_ids: list[str] | None = None,
    ) -> None:
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.node_ids = node_ids or []
        self.num_nodes = x.shape[0]

    def to_pyg(self) -> Any:
        """Convert to a ``torch_geometric.data.Data`` if the library is available."""
        try:
            from torch_geometric.data import Data
            return Data(
                x=self.x,
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                num_nodes=self.num_nodes,
            )
        except ImportError:
            return self

    def to(self, device: str) -> "PyGData":
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(device)
        return self


class GraphConverter:
    """Incrementally converts a SemanticGraph into a PyG-style Data object.

    Maintains a cache of the last conversion and applies incremental
    updates when possible.
    """

    def __init__(self, embedding_dim: int = settings.embedding_dim) -> None:
        self._embedding_dim = embedding_dim
        self._cached_data: PyGData | None = None
        self._cached_node_set: set[str] = set()
        self._cached_edge_set: set[tuple[str, str]] = set()
        self._version: int = 0

    def convert(
        self,
        graph: SemanticGraph,
        episodic_store: EpisodicStore | None = None,
    ) -> PyGData:
        """Full conversion of the graph into a PyG Data object.

        Node features: [embedding(384), node_type_onehot(5), activation(1),
                        time_since_last_access(1), access_count(1)] = 392d

        Edge features: [relation_type_onehot(10), weight(1),
                        evidence_count(1), edge_age(1)] = 13d
        """
        now = time.time()

        nodes = graph.all_nodes()
        node_ids = [n.id for n in nodes]
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        n = len(nodes)

        # ── Build node features [N, 392] ─────────────────────────────
        node_feats: list[list[float]] = []
        for node in nodes:
            emb = node.embedding if node.embedding else [0.0] * self._embedding_dim
            if len(emb) < self._embedding_dim:
                emb = emb + [0.0] * (self._embedding_dim - len(emb))
            type_oh = _onehot(NODE_TYPE_TO_IDX.get(node.node_type, 4), NUM_NODE_TYPES)
            activation = [node.activation]
            # Time features default to 0 if not available from episodic store
            time_feat = [0.0]
            access_feat = [0.0]
            node_feats.append(emb[:self._embedding_dim] + type_oh + activation + time_feat + access_feat)

        if n == 0:
            x = torch.zeros(0, self._embedding_dim + NUM_NODE_TYPES + 3, device=_device)
        else:
            x = torch.tensor(node_feats, dtype=torch.float32, device=_device)

        # ── Build edge index [2, E] and edge attr [E, 13] ────────────
        edges = graph.all_edges()
        src_list: list[int] = []
        tgt_list: list[int] = []
        edge_feats: list[list[float]] = []

        for edge in edges:
            si = id_to_idx.get(edge.source)
            ti = id_to_idx.get(edge.target)
            if si is not None and ti is not None:
                src_list.append(si)
                tgt_list.append(ti)
                rel_oh = _onehot(RELATION_TO_IDX.get(edge.relation, 9), NUM_RELATION_TYPES)
                feat = rel_oh + [edge.weight, float(len(edge.evidence)), 0.0]
                edge_feats.append(feat)

        if not src_list:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=_device)
            edge_attr = torch.empty((0, NUM_RELATION_TYPES + 3), dtype=torch.float32, device=_device)
        else:
            edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long, device=_device)
            edge_attr = torch.tensor(edge_feats, dtype=torch.float32, device=_device)

        data = PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, node_ids=node_ids)

        # Update cache
        self._cached_data = data
        self._cached_node_set = set(node_ids)
        self._cached_edge_set = {(e.source, e.target) for e in edges}
        self._version += 1

        return data

    def needs_rebuild(self, graph: SemanticGraph) -> bool:
        """Check whether the cached conversion is out of date."""
        if self._cached_data is None:
            return True
        current_nodes = set(graph.node_ids())
        if current_nodes != self._cached_node_set:
            return True
        current_edges = {(e.source, e.target) for e in graph.all_edges()}
        if current_edges != self._cached_edge_set:
            return True
        return False

    def get_or_convert(
        self,
        graph: SemanticGraph,
        episodic_store: EpisodicStore | None = None,
    ) -> PyGData:
        """Return cached data or rebuild if the graph changed."""
        if self.needs_rebuild(graph):
            return self.convert(graph, episodic_store)
        assert self._cached_data is not None
        return self._cached_data

    @property
    def node_feature_dim(self) -> int:
        """Dimensionality of node feature vectors."""
        return self._embedding_dim + NUM_NODE_TYPES + 3  # 392 for 384d embeddings
