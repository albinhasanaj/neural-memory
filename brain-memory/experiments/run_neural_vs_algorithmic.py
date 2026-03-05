"""
Experiment: Neural vs Algorithmic Memory System.

Compares the full neural pipeline (all flags enabled) against the
baseline algorithmic implementation across multiple metrics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from config.settings import settings
from experiments.eval_metrics import (
    activation_precision,
    consistency_score,
    recall_accuracy,
)
from memory.encoder import get_encoder
from memory.episodic import EpisodicStore
from memory.observer import MemoryObserver
from memory.semantic import SemanticGraph

_device = settings.resolved_device


# ── Test conversation ───────────────────────────────────────────────

CONVERSATION = [
    ("user", "I'm working on a Python project using FastAPI."),
    ("assistant", "Great choice! FastAPI is excellent for building APIs. What kind of project?"),
    ("user", "It's a memory system for LLMs — inspired by the hippocampus."),
    ("assistant", "That's fascinating. Are you implementing spreading activation?"),
    ("user", "Yes, with a semantic knowledge graph and episodic buffer."),
    ("user", "Remember that I prefer dark mode in all editors."),
    ("user", "I really dislike WordPress — it's too bloated."),
    ("assistant", "Noted. You prefer lightweight tools."),
    ("user", "What was I working on again?"),
    ("user", "Do you remember my preference about editors?"),
]


def run_algorithmic() -> dict:
    """Run baseline algorithmic memory system."""
    store = EpisodicStore()
    graph = SemanticGraph()
    observer = MemoryObserver(episodic_store=store, semantic_graph=graph)

    results = []
    t0 = time.perf_counter()

    for speaker, text in CONVERSATION:
        info = observer.observe(text, speaker=speaker)
        results.append(info)

    elapsed = time.perf_counter() - t0

    return {
        "mode": "algorithmic",
        "turns_processed": len(results),
        "episodes_stored": sum(1 for r in results if r["stored"]),
        "elapsed_seconds": elapsed,
        "avg_salience": sum(r["salience"] for r in results) / len(results),
    }


def run_neural() -> dict:
    """Run neural memory system (all components)."""
    try:
        from memory.observer import NeuralMemoryObserver
    except ImportError:
        return {"mode": "neural", "error": "NeuralMemoryObserver not available"}

    store = EpisodicStore()
    graph = SemanticGraph()

    # Temporarily enable all neural flags for this experiment
    # (Note: in production, modify settings via env vars)
    observer = NeuralMemoryObserver(episodic_store=store, semantic_graph=graph)

    results = []
    t0 = time.perf_counter()

    for speaker, text in CONVERSATION:
        info = observer.observe(text, speaker=speaker)
        results.append(info)

    elapsed = time.perf_counter() - t0

    return {
        "mode": "neural",
        "turns_processed": len(results),
        "episodes_stored": sum(1 for r in results if r["stored"]),
        "elapsed_seconds": elapsed,
        "avg_salience": sum(r["salience"] for r in results) / len(results),
    }


def main() -> None:
    print("=" * 60)
    print(" Neural vs Algorithmic Memory System")
    print("=" * 60)

    algo_results = run_algorithmic()
    print(f"\nAlgorithmic: {json.dumps(algo_results, indent=2)}")

    neural_results = run_neural()
    print(f"\nNeural: {json.dumps(neural_results, indent=2)}")

    # Save results
    output_dir = Path("data/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "neural_vs_algorithmic.json", "w") as f:
        json.dump({"algorithmic": algo_results, "neural": neural_results}, f, indent=2)

    print(f"\nResults saved to {output_dir / 'neural_vs_algorithmic.json'}")


if __name__ == "__main__":
    main()
