"""
Experiment: Dopaminergic Gate vs Heuristic Salience Threshold.

Compares the trained gate network's store/don't-store decisions against
the fixed-threshold heuristic salience scorer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from config.settings import settings
from memory.encoder import get_encoder
from memory.salience import (
    SalienceScorer,
    compute_entity_density,
    compute_novelty,
    compute_prediction_error,
    detect_emphasis,
)
from memory.semantic import SemanticGraph

_device = settings.resolved_device


TURNS = [
    ("I'm building a memory system for LLMs.", ["LLM"]),
    ("The weather is nice today.", []),
    ("REMEMBER THIS: my API key is stored in .env", []),
    ("I prefer Python over JavaScript for backends.", ["Python", "JavaScript"]),
    ("ok", []),
    ("Actually, I changed my mind about WordPress.", ["WordPress"]),
    ("What time is it?", []),
    ("This is CRITICAL — the deployment deadline is Friday!", []),
    ("My name is Alex, I work at TechCorp.", ["Alex", "TechCorp"]),
    ("hmm", []),
]


def run_heuristic(encoder, graph: SemanticGraph) -> list[dict]:
    scorer = SalienceScorer()
    results = []

    for text, entities in TURNS:
        emb = encoder.encode(text)
        salience = scorer.score(emb, None, entities, text, graph)
        results.append({
            "text": text[:50],
            "salience": salience,
            "store": salience >= settings.salience_threshold,
        })

    return results


def run_gate(encoder, graph: SemanticGraph) -> list[dict]:
    from memory.gate_network import DopaminergicGate

    gate = DopaminergicGate()
    results = []
    ctx = torch.randn(settings.gru_hidden_dim, device=_device)  # mock context

    for text, entities in TURNS:
        emb = encoder.encode(text)
        novelty = compute_novelty(emb, graph)
        pred_err = 0.5  # no WM prediction available
        emphasis = detect_emphasis(text)
        density = compute_entity_density(entities, text)
        signals = torch.tensor([novelty, pred_err, emphasis, density], device=_device)

        decision, prob = gate.should_store(emb, ctx, signals, epsilon=0.0)
        results.append({
            "text": text[:50],
            "gate_prob": prob,
            "store": decision,
        })

    return results


def main() -> None:
    print("=" * 60)
    print(" Dopaminergic Gate vs Heuristic Salience")
    print("=" * 60)

    encoder = get_encoder()
    graph = SemanticGraph()

    print("\nHeuristic threshold-based gating:")
    heuristic = run_heuristic(encoder, graph)
    for r in heuristic:
        marker = "✓" if r["store"] else "✗"
        print(f"  {marker} [{r['salience']:.3f}] {r['text']}")

    print("\nNeural gate (untrained):")
    neural = run_gate(encoder, graph)
    for r in neural:
        marker = "✓" if r["store"] else "✗"
        print(f"  {marker} [{r['gate_prob']:.3f}] {r['text']}")

    # Agreement
    agree = sum(
        1 for h, n in zip(heuristic, neural) if h["store"] == n["store"]
    )
    print(f"\nAgreement: {agree}/{len(TURNS)} ({100*agree/len(TURNS):.0f}%)")

    output_dir = Path("data/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "gate_vs_heuristic.json", "w") as f:
        json.dump({"heuristic": heuristic, "neural": neural}, f, indent=2)


if __name__ == "__main__":
    main()
