"""
Graph persistence — save / load the semantic knowledge graph to JSON.

Node embeddings are serialised as base64-encoded numpy float32 arrays
so the JSON remains human-readable for the non-embedding fields while
keeping embedding data compact.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import settings
from memory.semantic import SemanticGraph, SemanticNode, SemanticEdge


# ────────────────────────────────────────────────────────────────────
# Encoding helpers
# ────────────────────────────────────────────────────────────────────


def _encode_embedding(embedding: list[float]) -> str:
    """Encode a float list as a base64 string (compact for JSON)."""
    arr = np.array(embedding, dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _decode_embedding(b64: str) -> list[float]:
    """Decode a base64 string back to a list of floats."""
    raw = base64.b64decode(b64)
    return np.frombuffer(raw, dtype=np.float32).tolist()


# ────────────────────────────────────────────────────────────────────
# Save / load
# ────────────────────────────────────────────────────────────────────


def save_graph(
    graph: SemanticGraph,
    path: str | Path = settings.semantic_graph_path,
) -> None:
    """Serialise a ``SemanticGraph`` to a JSON file.

    Embeddings are stored as base64 to keep file size reasonable.
    """
    data: dict[str, Any] = {"nodes": [], "edges": []}

    for node in graph.all_nodes():
        node_dict = node.model_dump()
        if node_dict.get("embedding"):
            node_dict["embedding_b64"] = _encode_embedding(node_dict.pop("embedding"))
        else:
            node_dict.pop("embedding", None)
        data["nodes"].append(node_dict)

    for edge in graph.all_edges():
        data["edges"].append(edge.model_dump())

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def load_graph(
    path: str | Path = settings.semantic_graph_path,
) -> SemanticGraph:
    """Deserialise a ``SemanticGraph`` from a JSON file.

    Returns an empty graph if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return SemanticGraph()

    raw = json.loads(path.read_text())
    graph = SemanticGraph()

    for node_dict in raw.get("nodes", []):
        if "embedding_b64" in node_dict:
            node_dict["embedding"] = _decode_embedding(node_dict.pop("embedding_b64"))
        graph.upsert_node(SemanticNode(**node_dict))

    for edge_dict in raw.get("edges", []):
        graph.upsert_edge(SemanticEdge(**edge_dict))

    return graph
