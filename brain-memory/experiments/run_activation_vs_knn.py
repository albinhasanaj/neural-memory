"""
Experiment 4: Spreading activation vs k-NN similarity search.

Compares the spreading-activation engine against a simple cosine-similarity
k-nearest-neighbor retrieval.  Both approaches retrieve the same number
of memories; we measure activation_precision and activation_recall.
"""

from __future__ import annotations

import json
import logging
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from config.settings import settings
from experiments.eval_metrics import activation_precision, activation_recall
from memory.activation import SpreadingActivationEngine
from memory.encoder import get_encoder
from memory.semantic import SemanticEdge, SemanticGraph, SemanticNode

logger = logging.getLogger(__name__)


def knn_retrieval(
    context_vector: Tensor,
    graph: SemanticGraph,
    k: int = settings.lateral_inhibition_k,
) -> list[tuple[str, float]]:
    """Simple k-NN retrieval: return the *k* nodes most similar to the context."""
    nodes = graph.all_nodes()
    if not nodes:
        return []

    sims: list[tuple[str, float]] = []
    for node in nodes:
        if node.embedding:
            node_emb = torch.tensor(node.embedding, dtype=torch.float32)
            sim = F.cosine_similarity(
                context_vector.unsqueeze(0).float(),
                node_emb.unsqueeze(0),
                dim=1,
            ).item()
            sims.append((node.id, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:k]


def build_experiment_graph() -> tuple[SemanticGraph, set[str]]:
    """Build a graph with known relevant nodes for evaluation.

    Returns (graph, ground_truth_relevant_node_ids).
    """
    encoder = get_encoder()
    g = SemanticGraph()

    # Create a cluster of related nodes
    concepts = [
        ("python", "Python programming language"),
        ("fastapi", "FastAPI web framework"),
        ("pydantic", "Pydantic data validation"),
        ("uvicorn", "Uvicorn ASGI server"),
        ("javascript", "JavaScript language"),
        ("react", "React frontend framework"),
        ("cooking", "Italian cooking recipes"),
        ("gardening", "Herb gardening tips"),
    ]
    for nid, label in concepts:
        g.upsert_node(SemanticNode(
            id=nid, label=label,
            embedding=encoder.encode(label).tolist(),
        ))

    # Python-ecosystem edges (should be reachable via spreading activation)
    g.upsert_edge(SemanticEdge(source="python", target="fastapi", weight=0.9))
    g.upsert_edge(SemanticEdge(source="python", target="pydantic", weight=0.8))
    g.upsert_edge(SemanticEdge(source="fastapi", target="uvicorn", weight=0.85))
    g.upsert_edge(SemanticEdge(source="fastapi", target="pydantic", weight=0.7))

    # JS edges
    g.upsert_edge(SemanticEdge(source="javascript", target="react", weight=0.8))

    # No edges to cooking/gardening — they're unrelated

    # When the query is about "FastAPI project", relevant nodes are the Python cluster
    relevant = {"python", "fastapi", "pydantic", "uvicorn"}
    return g, relevant


def run_comparison() -> dict[str, dict[str, float]]:
    """Run the activation-vs-kNN comparison and return metrics for both."""
    logger.info("Running activation vs k-NN experiment...")

    graph, relevant = build_experiment_graph()
    encoder = get_encoder()
    context = encoder.encode("I'm building a FastAPI backend with Pydantic models")

    # ── spreading activation ────────────────────────────────────────
    engine = SpreadingActivationEngine(graph, top_k=4, seed_threshold=0.3)
    sa_results = engine.activate(context)
    sa_ids = [nid for nid, _ in sa_results]

    sa_metrics = {
        "precision": activation_precision(sa_ids, relevant),
        "recall": activation_recall(sa_ids, relevant),
    }

    # ── k-NN ────────────────────────────────────────────────────────
    knn_results = knn_retrieval(context, graph, k=4)
    knn_ids = [nid for nid, _ in knn_results]

    knn_metrics = {
        "precision": activation_precision(knn_ids, relevant),
        "recall": activation_recall(knn_ids, relevant),
    }

    logger.info("Spreading activation: %s", sa_metrics)
    logger.info("k-NN retrieval:       %s", knn_metrics)
    logger.info("SA activated: %s", sa_ids)
    logger.info("kNN retrieved: %s", knn_ids)

    return {"spreading_activation": sa_metrics, "knn": knn_metrics}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    results = run_comparison()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
