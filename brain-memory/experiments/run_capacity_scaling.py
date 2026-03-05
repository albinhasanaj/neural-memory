"""
Experiment: Capacity Scaling — how do neural components scale with memory size?

Tests each component's runtime and memory usage as the number of stored
memories increases from 10 to 1000.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch

from config.settings import settings

_device = settings.resolved_device

MEMORY_SIZES = [10, 25, 50, 100, 250, 500, 1000]


def measure_hopfield_scaling() -> list[dict]:
    """Measure Hopfield retrieval time vs memory count."""
    from memory.hopfield_memory import HippocampalMemory

    results = []
    query = torch.randn(settings.embedding_dim, device=_device)

    for n in MEMORY_SIZES:
        hop = HippocampalMemory(max_patterns=n + 100)

        # Store N patterns
        for i in range(n):
            torch.manual_seed(i)
            hop.store(torch.randn(settings.embedding_dim, device=_device), episode_id=f"ep{i}")

        # Time retrieval
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            hop(query, top_k=5)
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        results.append({"n_patterns": n, "avg_ms": avg * 1000})

        del hop
        gc.collect()
        if _device == "cuda":
            torch.cuda.empty_cache()

    return results


def measure_vae_scaling() -> list[dict]:
    """Measure VAE encoding time vs batch size."""
    from memory.neural_consolidation import ConsolidationVAE

    vae = ConsolidationVAE()
    vae.eval()
    results = []

    for n in MEMORY_SIZES:
        embs = torch.randn(n, settings.embedding_dim, device=_device)
        metas = torch.randn(n, settings.vae_metadata_dim, device=_device)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            with torch.no_grad():
                vae.get_latent(embs, metas)
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        results.append({"n_episodes": n, "avg_ms": avg * 1000})

    return results


def measure_pattern_sep_scaling() -> list[dict]:
    """Measure Pattern Separator throughput."""
    from memory.pattern_separation import PatternSeparator

    sep = PatternSeparator()
    sep.eval()
    results = []

    for n in MEMORY_SIZES:
        batch = torch.randn(n, settings.embedding_dim, device=_device)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            with torch.no_grad():
                sep(batch)
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        results.append({"n_embeddings": n, "avg_ms": avg * 1000})

    return results


def measure_forgetting_scaling() -> list[dict]:
    """Measure Forgetting Network throughput."""
    from memory.forgetting import ForgettingNetwork

    net = ForgettingNetwork()
    net.eval()
    results = []

    for n in MEMORY_SIZES:
        embs = torch.randn(n, settings.embedding_dim, device=_device)
        scalars = torch.rand(n, 5, device=_device)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            with torch.no_grad():
                net(embs, scalars)
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        results.append({"n_memories": n, "avg_ms": avg * 1000})

    return results


def main() -> None:
    print("=" * 60)
    print(" Capacity Scaling — Neural Component Performance")
    print("=" * 60)

    results = {}

    print("\nHopfield scaling...")
    results["hopfield"] = measure_hopfield_scaling()
    for r in results["hopfield"]:
        print(f"  {r['n_patterns']:>5} patterns: {r['avg_ms']:.2f} ms")

    print("\nVAE scaling...")
    results["vae"] = measure_vae_scaling()
    for r in results["vae"]:
        print(f"  {r['n_episodes']:>5} episodes: {r['avg_ms']:.2f} ms")

    print("\nPattern Separator scaling...")
    results["pattern_sep"] = measure_pattern_sep_scaling()
    for r in results["pattern_sep"]:
        print(f"  {r['n_embeddings']:>5} embeddings: {r['avg_ms']:.2f} ms")

    print("\nForgetting Network scaling...")
    results["forgetting"] = measure_forgetting_scaling()
    for r in results["forgetting"]:
        print(f"  {r['n_memories']:>5} memories: {r['avg_ms']:.2f} ms")

    output_dir = Path("data/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "capacity_scaling.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'capacity_scaling.json'}")


if __name__ == "__main__":
    main()
