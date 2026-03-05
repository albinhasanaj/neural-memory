"""
Evaluation metrics for the brain-memory system.

Provides quantitative measures for comparing memory configurations
across ablation experiments.  All metrics return floats in [0, 1]
where higher is better (except where noted).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def recall_accuracy(
    expected_facts: list[str],
    retrieved_context: str,
) -> float:
    """Fraction of expected facts that appear (substring match) in the retrieved context.

    Parameters
    ----------
    expected_facts:
        Ground-truth strings the system should remember.
    retrieved_context:
        The memory context block actually injected.

    Returns
    -------
    float ∈ [0, 1]
    """
    if not expected_facts:
        return 1.0
    hits = sum(1 for fact in expected_facts if fact.lower() in retrieved_context.lower())
    return hits / len(expected_facts)


def consistency_score(
    statements: list[str],
    embeddings: list[Tensor],
    contradiction_threshold: float = -0.3,
) -> float:
    """Estimate self-consistency of a set of statements.

    Uses cosine similarity — pairs with sim < *contradiction_threshold*
    are flagged as potential contradictions.

    Returns
    -------
    float ∈ [0, 1] — 1 means no detected contradictions.
    """
    if len(embeddings) < 2:
        return 1.0

    n = len(embeddings)
    contradictions = 0
    total_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            sim = F.cosine_similarity(
                embeddings[i].unsqueeze(0),
                embeddings[j].unsqueeze(0),
                dim=1,
            ).item()
            total_pairs += 1
            if sim < contradiction_threshold:
                contradictions += 1

    return 1.0 - (contradictions / total_pairs) if total_pairs > 0 else 1.0


def personalization_score(
    user_preferences: dict[str, str],
    response_text: str,
) -> float:
    """Measure how well a response references known user preferences.

    Parameters
    ----------
    user_preferences:
        ``{topic: preference}`` map, e.g. ``{"language": "Python"}``.
    response_text:
        The LLM's response text.

    Returns
    -------
    float ∈ [0, 1]
    """
    if not user_preferences:
        return 1.0
    hits = 0
    for topic, pref in user_preferences.items():
        if pref.lower() in response_text.lower():
            hits += 1
    return hits / len(user_preferences)


def forgetting_accuracy(
    should_forget: list[str],
    retrieved_context: str,
) -> float:
    """Fraction of outdated facts correctly *absent* from the context.

    Parameters
    ----------
    should_forget:
        Strings that should have been forgotten / overridden.
    retrieved_context:
        The memory context block.

    Returns
    -------
    float ∈ [0, 1] — 1 means all outdated facts were successfully forgotten.
    """
    if not should_forget:
        return 1.0
    still_present = sum(
        1 for fact in should_forget if fact.lower() in retrieved_context.lower()
    )
    return 1.0 - (still_present / len(should_forget))


def activation_precision(
    activated_node_ids: list[str],
    relevant_node_ids: set[str],
) -> float:
    """Precision of the spreading-activation retrieval.

    Parameters
    ----------
    activated_node_ids:
        Node IDs returned by the activation engine.
    relevant_node_ids:
        Ground-truth set of node IDs that *should* have been activated.

    Returns
    -------
    float ∈ [0, 1]
    """
    if not activated_node_ids:
        return 0.0
    relevant_hits = sum(1 for n in activated_node_ids if n in relevant_node_ids)
    return relevant_hits / len(activated_node_ids)


def activation_recall(
    activated_node_ids: list[str],
    relevant_node_ids: set[str],
) -> float:
    """Recall of spreading activation.

    Returns
    -------
    float ∈ [0, 1]
    """
    if not relevant_node_ids:
        return 1.0
    hits = sum(1 for n in relevant_node_ids if n in activated_node_ids)
    return hits / len(relevant_node_ids)


# ────────────────────────────────────────────────────────────────────
# Aggregate report
# ────────────────────────────────────────────────────────────────────


def compute_all_metrics(
    expected_facts: list[str],
    should_forget: list[str],
    retrieved_context: str,
    activated_nodes: list[str],
    relevant_nodes: set[str],
    user_preferences: dict[str, str],
    response_text: str,
) -> dict[str, float]:
    """Compute all evaluation metrics and return as a dict."""
    return {
        "recall_accuracy": recall_accuracy(expected_facts, retrieved_context),
        "forgetting_accuracy": forgetting_accuracy(should_forget, retrieved_context),
        "activation_precision": activation_precision(activated_nodes, relevant_nodes),
        "activation_recall": activation_recall(activated_nodes, relevant_nodes),
        "personalization_score": personalization_score(user_preferences, response_text),
    }
