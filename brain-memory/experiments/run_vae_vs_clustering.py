"""
Experiment: VAE Consolidation vs Clustering + LLM.

Compares the VAE latent-space consolidation (neural) against
the baseline agglomerative clustering + LLM fact extraction.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from config.settings import settings
from memory.encoder import get_encoder

_device = settings.resolved_device


def generate_episodes(encoder, n: int = 30) -> list[tuple[torch.Tensor, dict]]:
    """Generate synthetic episode data."""
    topics = ["Python web development", "Machine learning basics", "Database design"]
    episodes = []
    for i in range(n):
        topic = topics[i % len(topics)]
        text = f"Episode {i}: Discussion about {topic} with detail {i}"
        emb = encoder.encode(text)
        meta = {"text": text, "topic": topic, "salience": 0.5 + 0.02 * i}
        episodes.append((emb, meta))
    return episodes


def run_vae_consolidation(episodes: list) -> dict:
    """Consolidate using VAE latent space."""
    from memory.neural_consolidation import (
        ConsolidationVAE,
        build_metadata_vector,
        latent_cluster_centroids,
    )

    vae = ConsolidationVAE()
    embs = torch.stack([e[0] for e in episodes]).to(_device)
    metas = torch.stack([
        build_metadata_vector(salience=e[1]["salience"], entity_count=1)
        for e in episodes
    ]).to(_device)

    t0 = time.perf_counter()
    centroids = latent_cluster_centroids(vae, embs, metas, n_clusters=5)
    elapsed = time.perf_counter() - t0

    return {
        "method": "vae",
        "n_episodes": len(episodes),
        "n_clusters": centroids.shape[0],
        "latent_dim": centroids.shape[1],
        "time_seconds": elapsed,
    }


def run_algorithmic_clustering(episodes: list) -> dict:
    """Consolidate using agglomerative clustering."""
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage

    embs = torch.stack([e[0] for e in episodes]).cpu().numpy()
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    embs_norm = embs / norms

    t0 = time.perf_counter()
    Z = linkage(embs_norm, method="average", metric="cosine")
    labels = fcluster(Z, t=0.4, criterion="distance")
    elapsed = time.perf_counter() - t0

    n_clusters = len(set(labels))

    return {
        "method": "algorithmic",
        "n_episodes": len(episodes),
        "n_clusters": n_clusters,
        "time_seconds": elapsed,
    }


def main() -> None:
    print("=" * 60)
    print(" VAE Consolidation vs Clustering + LLM")
    print("=" * 60)

    encoder = get_encoder()
    episodes = generate_episodes(encoder, n=30)

    vae_result = run_vae_consolidation(episodes)
    print(f"\nVAE: {json.dumps(vae_result, indent=2)}")

    algo_result = run_algorithmic_clustering(episodes)
    print(f"\nAlgorithmic: {json.dumps(algo_result, indent=2)}")

    output_dir = Path("data/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "vae_vs_clustering.json", "w") as f:
        json.dump({"vae": vae_result, "algorithmic": algo_result}, f, indent=2)

    print(f"\nResults saved to {output_dir / 'vae_vs_clustering.json'}")


if __name__ == "__main__":
    main()
