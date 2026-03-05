"""
Experiment 1: Baseline — LLM with no memory system.

Sends a multi-turn conversation to the LLM without any memory
injection to establish baseline recall and consistency scores.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx

from config.settings import settings
from experiments.eval_metrics import compute_all_metrics

logger = logging.getLogger(__name__)

# ── sample conversation for testing ──────────────────────────────────

CONVERSATION: list[dict[str, str]] = [
    {"role": "user", "content": "My name is Alex and I'm a Python developer."},
    {"role": "assistant", "content": "Nice to meet you, Alex! Python is a great language."},
    {"role": "user", "content": "I really dislike WordPress. It's caused me too many headaches."},
    {"role": "assistant", "content": "I understand — WordPress can be frustrating."},
    {"role": "user", "content": "Remember that I prefer dark mode in all my editors."},
    {"role": "assistant", "content": "Noted! Dark mode it is."},
    # ... distractor turns ...
    {"role": "user", "content": "What's the weather like today?"},
    {"role": "assistant", "content": "I don't have real-time weather data."},
    {"role": "user", "content": "What do you know about my preferences?"},
]


async def run_baseline() -> dict[str, float]:
    """Run the baseline experiment (no memory) and return metrics."""
    logger.info("Running baseline experiment (no memory system)...")

    base = settings.llm_base_url or "https://api.openai.com/v1"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{base}/chat/completions",
            headers=headers,
            json={
                "model": settings.llm_model,
                "messages": CONVERSATION,
                "temperature": 0.0,
            },
        )
        resp.raise_for_status()
        response_text = resp.json()["choices"][0]["message"]["content"]

    metrics = compute_all_metrics(
        expected_facts=["Python", "dark mode", "WordPress"],
        should_forget=[],
        retrieved_context="",  # no memory context in baseline
        activated_nodes=[],
        relevant_nodes=set(),
        user_preferences={"language": "Python", "editor_theme": "dark mode"},
        response_text=response_text,
    )

    logger.info("Baseline results: %s", json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(run_baseline())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
