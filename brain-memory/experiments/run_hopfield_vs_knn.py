"""
Experiment: Hopfield Retrieval vs k-NN Episodic Retrieval.

Compares the Modern Hopfield Network's associative completion against
a simple k-NN cosine-similarity retrieval on episodic memories.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from config.settings import settings
from memory.encoder import get_encoder

_device = settings.resolved_device


def generate_memories(encoder, n: int = 50) -> list[tuple[str, torch.Tensor]]:
    """Generate synthetic episode memories."""
    texts = [
        f"The user discussed topic {i} about {'Python' if i % 3 == 0 else 'JavaScript' if i % 3 == 1 else 'Rust'}"
        for i in range(n)
    ]
    embs = [encoder.encode(t) for t in texts]
    return list(zip(texts, embs))


def knn_retrieve(
    query: torch.Tensor,
    memories: list[tuple[str, torch.Tensor]],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Simple k-NN cosine retrieval."""
    embs = torch.stack([m[1] for m in memories]).to(_device)
    query = query.to(_device)
    sims = F.cosine_similarity(query.unsqueeze(0), embs, dim=1)
    vals, idx = sims.topk(min(top_k, len(memories)))
    return [(memories[i][0], v.item()) for v, i in zip(vals, idx)]


def hopfield_retrieve(
    query: torch.Tensor,
    memories: list[tuple[str, torch.Tensor]],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Hopfield network retrieval."""
    from memory.hopfield_memory import HippocampalMemory

    hop = HippocampalMemory()
    for i, (text, emb) in enumerate(memories):
        hop.store(emb, episode_id=f"ep{i}")

    results = hop.retrieve_episode_ids(query, top_k=top_k)
    return [(f"ep{r[0]}" if isinstance(r[0], int) else r[0], r[1]) for r in results]


def main() -> None:
    print("=" * 60)
    print(" Hopfield vs k-NN Episodic Retrieval")
    print("=" * 60)

    encoder = get_encoder()
    memories = generate_memories(encoder, n=30)

    query_texts = [
        "Tell me about Python programming",
        "What do you know about Rust?",
        "JavaScript frameworks discussion",
    ]
    queries = [encoder.encode(t) for t in query_texts]

    results = {"knn": [], "hopfield": []}

    for qt, q in zip(query_texts, queries):
        print(f"\nQuery: {qt}")

        # k-NN
        t0 = time.perf_counter()
        knn_results = knn_retrieve(q, memories, top_k=5)
        knn_time = time.perf_counter() - t0
        print(f"  k-NN ({knn_time*1000:.1f}ms):")
        for text, sim in knn_results[:3]:
            print(f"    {sim:.3f}: {text[:60]}")

        # Hopfield
        t0 = time.perf_counter()
        hop_results = hopfield_retrieve(q, memories, top_k=5)
        hop_time = time.perf_counter() - t0
        print(f"  Hopfield ({hop_time*1000:.1f}ms):")
        for eid, weight in hop_results[:3]:
            print(f"    {weight:.3f}: {eid}")

        results["knn"].append({"query": qt, "time_ms": knn_time * 1000})
        results["hopfield"].append({"query": qt, "time_ms": hop_time * 1000})

    output_dir = Path("data/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "hopfield_vs_knn.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'hopfield_vs_knn.json'}")


if __name__ == "__main__":
    main()
