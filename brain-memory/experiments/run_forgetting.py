"""
Experiment 6: Forgetting / belief updates.

Introduces contradictions into the conversation to test whether
the memory system can update beliefs and forget outdated information.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx

from config.settings import settings
from experiments.eval_metrics import compute_all_metrics
from memory.observer import MemoryObserver

logger = logging.getLogger(__name__)

# Conversation with a belief change: user first says they like Java,
# then corrects to Python.
FORGETTING_CONVERSATION: list[dict[str, str]] = [
    {"role": "user", "content": "My favourite programming language is Java."},
    {"role": "assistant", "content": "Java is a solid choice!"},
    {"role": "user", "content": "I use Java for everything at work."},
    {"role": "assistant", "content": "It's great for enterprise applications."},
    # ... time passes / many turns ...
    {"role": "user", "content": "What's the best database for web apps?"},
    {"role": "assistant", "content": "PostgreSQL is very popular for web applications."},
    # Correction:
    {"role": "user", "content": "Actually, I've switched to Python. I no longer use Java."},
    {"role": "assistant", "content": "Python is very versatile! Welcome to the Python community."},
    {"role": "user", "content": "What programming language do I prefer now?"},
]


async def run_forgetting() -> dict[str, float]:
    """Test belief update / forgetting capabilities."""
    logger.info("Running forgetting experiment...")

    observer = MemoryObserver()

    for msg in FORGETTING_CONVERSATION[:-1]:
        observer.observe(msg["content"], speaker=msg["role"])

    query = FORGETTING_CONVERSATION[-1]
    modified_messages, info = observer.process_turn(
        text=query["content"],
        speaker=query["role"],
        messages=FORGETTING_CONVERSATION,
    )

    memory_block = ""
    for msg in modified_messages:
        if msg.get("role") == "system" and "[MEMORY CONTEXT" in msg.get("content", ""):
            memory_block = msg["content"]
            break

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
                "messages": modified_messages,
                "temperature": 0.0,
            },
        )
        resp.raise_for_status()
        response_text = resp.json()["choices"][0]["message"]["content"]

    metrics = compute_all_metrics(
        expected_facts=["Python"],  # Should remember the new preference
        should_forget=["Java"],  # Should have updated away from Java
        retrieved_context=memory_block,
        activated_nodes=[nid for nid, _ in info.get("activated_nodes", [])],
        relevant_nodes={"python"},
        user_preferences={"language": "Python"},
        response_text=response_text,
    )

    logger.info("Forgetting results: %s", json.dumps(metrics, indent=2))
    logger.info("Response: %s", response_text[:200])
    return metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    results = asyncio.run(run_forgetting())
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
