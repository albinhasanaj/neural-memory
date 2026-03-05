"""
Memory Context Injector — formats and injects activated memories into the LLM prompt.

The injector assembles a ``[MEMORY CONTEXT]`` block from activated
semantic facts and relevant episodic memories, then prepends it to the
``messages`` list that will be sent to the LLM.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from config.settings import settings
from memory.episodic import EpisodicEntry
from memory.semantic import SemanticGraph


# ────────────────────────────────────────────────────────────────────
# Formatting helpers
# ────────────────────────────────────────────────────────────────────


def format_semantic_memories(
    graph: SemanticGraph,
    activated_node_ids: list[tuple[str, float]],
    max_facts: int = settings.max_semantic_facts,
) -> str:
    """Render activated semantic facts as natural-language bullet points.

    Parameters
    ----------
    activated_node_ids:
        ``[(node_id, activation_strength), ...]`` from spreading-activation.
    max_facts:
        Maximum number of facts to include.

    Returns
    -------
    str — formatted fact block, or empty string if no facts available.
    """
    lines: list[str] = []

    # For each activated node, look at outgoing edges (facts about it)
    seen_edges: set[tuple[str, str]] = set()
    for node_id, strength in activated_node_ids[:max_facts * 2]:
        neighbors = graph.get_neighbors(node_id, direction="out")
        for target_id, edge in neighbors:
            key = (edge.source, edge.target)
            if key in seen_edges:
                continue
            seen_edges.add(key)

            source_node = graph.get_node(edge.source)
            target_node = graph.get_node(edge.target)
            src_label = source_node.label if source_node else edge.source
            tgt_label = target_node.label if target_node else edge.target
            evidence_count = len(edge.evidence) if edge.evidence else 0
            lines.append(
                f"  - {src_label} {edge.relation} {tgt_label}  "
                f"[confidence: {edge.confidence:.2f}, {evidence_count} episode(s)]"
            )

            if len(lines) >= max_facts:
                break
        if len(lines) >= max_facts:
            break

    return "\n".join(lines)


def format_episodic_memories(
    episodes: list[EpisodicEntry],
    max_episodes: int = settings.max_episodic_memories,
) -> str:
    """Render episodic memories with timestamps and speaker.

    Parameters
    ----------
    episodes:
        Episodes to render (should already be sorted by relevance).
    max_episodes:
        Maximum number of episodes to include.

    Returns
    -------
    str — formatted episode block.
    """
    lines: list[str] = []
    for ep in episodes[:max_episodes]:
        date_str = ep.timestamp.strftime("%Y-%m-%d")
        # Truncate very long texts
        text = ep.raw_text if len(ep.raw_text) <= 200 else ep.raw_text[:197] + "..."
        lines.append(f"  - [{date_str}] {ep.speaker}: {text}")
    return "\n".join(lines)


def format_hopfield_memories(
    memories: list[dict],
    max_memories: int = settings.max_episodic_memories,
) -> str:
    """Render Hopfield-decoded memory dicts in the same format as episodic memories.

    Each dict has ``{text, speaker, timestamp, entities, score, ...}``.
    """
    lines: list[str] = []
    for mem in memories[:max_memories]:
        ts = mem.get("timestamp", 0.0)
        if isinstance(ts, (int, float)) and ts > 0:
            date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        else:
            date_str = "unknown"
        text = mem.get("text", "")
        if len(text) > 200:
            text = text[:197] + "..."
        speaker = mem.get("speaker", "unknown")
        lines.append(f"  - [{date_str}] {speaker}: {text}")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────────
# Build the full memory context block
# ────────────────────────────────────────────────────────────────────


def build_memory_context(
    graph: SemanticGraph,
    activated_nodes: list[tuple[str, float]],
    episodes: list[EpisodicEntry],
) -> str:
    """Assemble the ``[MEMORY CONTEXT]`` block.

    Returns an empty string if there is nothing to inject.
    """
    semantic_block = format_semantic_memories(graph, activated_nodes)
    episodic_block = format_episodic_memories(episodes)

    if not semantic_block and not episodic_block:
        return ""

    parts: list[str] = [
        "[MEMORY CONTEXT — Automatically activated based on current conversation]",
        "",
    ]
    if semantic_block:
        parts.append("Semantic facts:")
        parts.append(semantic_block)
        parts.append("")
    if episodic_block:
        parts.append("Episodic memories:")
        parts.append(episodic_block)
        parts.append("")
    parts.append("[END MEMORY CONTEXT]")
    return "\n".join(parts)


def build_hopfield_memory_context(
    graph: SemanticGraph,
    activated_nodes: list[tuple[str, float]],
    hopfield_memories: list[dict],
) -> str:
    """Assemble ``[MEMORY CONTEXT]`` using Hopfield-decoded memories.

    Same output format as ``build_memory_context`` but sources episodic
    data from decoded Hopfield dicts rather than ``EpisodicEntry`` objects.
    """
    semantic_block = format_semantic_memories(graph, activated_nodes)
    episodic_block = format_hopfield_memories(hopfield_memories)

    if not semantic_block and not episodic_block:
        return ""

    parts: list[str] = [
        "[MEMORY CONTEXT — Automatically activated based on current conversation]",
        "",
    ]
    if semantic_block:
        parts.append("Semantic facts:")
        parts.append(semantic_block)
        parts.append("")
    if episodic_block:
        parts.append("Episodic memories:")
        parts.append(episodic_block)
        parts.append("")
    parts.append("[END MEMORY CONTEXT]")
    return "\n".join(parts)


# ────────────────────────────────────────────────────────────────────
# Inject into messages list
# ────────────────────────────────────────────────────────────────────


def inject(
    messages: list[dict[str, Any]],
    graph: SemanticGraph,
    activated_nodes: list[tuple[str, float]],
    episodes: list[EpisodicEntry],
) -> list[dict[str, Any]]:
    """Prepend the memory context as a system message.

    Returns a *new* messages list — the original is not mutated.

    Parameters
    ----------
    messages:
        The OpenAI-format messages list (``[{"role": ..., "content": ...}, ...]``).
    graph:
        The semantic knowledge graph.
    activated_nodes:
        Nodes returned by the spreading activation engine.
    episodes:
        Relevant episodic memories to inject.

    Returns
    -------
    list[dict] — messages with the memory context prepended.
    """
    memory_block = build_memory_context(graph, activated_nodes, episodes)
    if not memory_block:
        return list(messages)

    memory_message: dict[str, Any] = {
        "role": "system",
        "content": memory_block,
    }

    # Insert the memory block right after any existing system messages
    new_messages = list(messages)
    insert_idx = 0
    for i, msg in enumerate(new_messages):
        if msg.get("role") == "system":
            insert_idx = i + 1
        else:
            break

    new_messages.insert(insert_idx, memory_message)
    return new_messages


def inject_hopfield(
    messages: list[dict[str, Any]],
    graph: SemanticGraph,
    activated_nodes: list[tuple[str, float]],
    hopfield_memories: list[dict],
) -> list[dict[str, Any]]:
    """Inject Hopfield-decoded memories into the messages list.

    Same behaviour as ``inject()`` but uses decoded Hopfield dicts.
    """
    memory_block = build_hopfield_memory_context(graph, activated_nodes, hopfield_memories)
    if not memory_block:
        return list(messages)

    memory_message: dict[str, Any] = {
        "role": "system",
        "content": memory_block,
    }

    new_messages = list(messages)
    insert_idx = 0
    for i, msg in enumerate(new_messages):
        if msg.get("role") == "system":
            insert_idx = i + 1
        else:
            break

    new_messages.insert(insert_idx, memory_message)
    return new_messages
