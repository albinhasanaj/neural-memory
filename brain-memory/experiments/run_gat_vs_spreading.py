"""
Experiment: GAT Spreading Activation vs Algorithmic Spreading Activation.

Ablation study comparing the learned GAT-based activation engine against
the hand-tuned sparse-matrix spreading activation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from config.settings import settings
from memory.activation import SeedHints, SpreadingActivationEngine
from memory.encoder import get_encoder
from memory.semantic import SemanticEdge, SemanticGraph, SemanticNode

_device = settings.resolved_device


def build_test_graph(encoder=None) -> SemanticGraph:
    """Build a moderately sized graph for benchmarking."""
    if encoder is None:
        encoder = get_encoder()

    graph = SemanticGraph()
    entities = [
        "Python", "JavaScript", "TypeScript", "Rust", "Go",
        "FastAPI", "Django", "React", "Vue", "PostgreSQL",
        "Docker", "Kubernetes", "AWS", "Linux", "Git",
        "Machine Learning", "Neural Networks", "Transformers",
        "Memory Systems", "Hippocampus",
    ]

    for name in entities:
        emb = encoder.encode(name).tolist()
        graph.upsert_node(SemanticNode(
            id=name.lower().replace(" ", "_"),
            label=name,
            embedding=emb,
            node_type="entity",
        ))

    # Add edges
    relations = [
        ("python", "fastapi", "has_framework"),
        ("python", "django", "has_framework"),
        ("python", "machine_learning", "used_in"),
        ("javascript", "react", "has_framework"),
        ("javascript", "vue", "has_framework"),
        ("typescript", "javascript", "extends"),
        ("docker", "kubernetes", "orchestrated_by"),
        ("neural_networks", "transformers", "includes"),
        ("memory_systems", "hippocampus", "inspired_by"),
        ("machine_learning", "neural_networks", "includes"),
    ]

    for src, tgt, rel in relations:
        graph.upsert_edge(SemanticEdge(
            source=src, target=tgt, relation=rel,
            weight=0.8, confidence=0.7, evidence=["bench"],
        ))

    return graph


def benchmark_algorithmic(graph: SemanticGraph, queries: list) -> dict:
    """Benchmark algorithmic spreading activation."""
    engine = SpreadingActivationEngine(graph)
    engine.rebuild()

    times = []
    all_activated = []

    for query_emb in queries:
        t0 = time.perf_counter()
        hints = SeedHints(context_vector=query_emb)
        activated = engine.activate(query_emb, hints=hints)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        all_activated.append(activated)

    return {
        "method": "algorithmic",
        "queries": len(queries),
        "avg_time_ms": sum(times) / len(times) * 1000,
        "avg_activated_nodes": sum(len(a) for a in all_activated) / len(all_activated),
    }


def benchmark_gat(graph: SemanticGraph, queries: list) -> dict:
    """Benchmark GAT-based activation (inference only, untrained)."""
    try:
        from memory.graph_converter import GraphConverter
        from memory.neural_activation import MemoryGAT
    except ImportError:
        return {"method": "gat", "error": "Neural modules not available"}

    converter = GraphConverter(graph)
    data = converter.convert()

    gat = MemoryGAT(
        node_feature_dim=converter.node_feature_dim,
        edge_dim=data.edge_attr.shape[1] if data.edge_attr is not None and data.edge_attr.numel() > 0 else 13,
    )
    gat.eval()

    times = []
    all_scores = []

    for query_emb in queries:
        t0 = time.perf_counter()
        with torch.no_grad():
            if data.x.shape[0] > 0:
                scores = gat(
                    data.x.to(_device),
                    data.edge_index.to(_device),
                    data.edge_attr.to(_device) if data.edge_attr is not None else None,
                    query_emb.to(_device),
                )
            else:
                scores = torch.empty(0)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        all_scores.append(scores)

    return {
        "method": "gat",
        "queries": len(queries),
        "avg_time_ms": sum(times) / len(times) * 1000,
        "avg_max_score": float(
            sum(s.max().item() for s in all_scores if s.numel() > 0) /
            max(1, sum(1 for s in all_scores if s.numel() > 0))
        ),
    }


def main() -> None:
    print("=" * 60)
    print(" GAT vs Algorithmic Spreading Activation")
    print("=" * 60)

    encoder = get_encoder()
    graph = build_test_graph(encoder)

    # Generate queries
    query_texts = [
        "What Python framework should I use?",
        "Tell me about machine learning",
        "How does Docker work with Kubernetes?",
        "What is the hippocampus?",
        "I need help with JavaScript and React",
    ]
    queries = [encoder.encode(t) for t in query_texts]

    algo = benchmark_algorithmic(graph, queries)
    print(f"\nAlgorithmic: {json.dumps(algo, indent=2)}")

    gat = benchmark_gat(graph, queries)
    print(f"\nGAT: {json.dumps(gat, indent=2)}")

    output_dir = Path("data/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "gat_vs_spreading.json", "w") as f:
        json.dump({"algorithmic": algo, "gat": gat}, f, indent=2)

    print(f"\nResults saved to {output_dir / 'gat_vs_spreading.json'}")


if __name__ == "__main__":
    main()
