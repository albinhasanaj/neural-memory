"""
Open WebUI Pipeline — Brain Memory integration.

Implements the Pipeline protocol expected by Open WebUI's Pipelines
feature.  The ``BrainMemoryPipeline.pipe()`` method receives the full
message body, runs the memory observation + activation + injection
cycle, and returns the modified body for the downstream LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from config.settings import settings
from memory.encoder import get_encoder
from memory.episodic import EpisodicStore
from memory.observer import MemoryObserver, NeuralMemoryObserver
from memory.semantic import SemanticGraph
from memory.working_memory import WorkingMemory
from storage.graph_store import load_graph, save_graph
from storage.sqlite_store import SQLiteEpisodicStore

logger = logging.getLogger(__name__)


class BrainMemoryPipeline:
    """Open WebUI Pipeline class for the brain-inspired memory system.

    Lifecycle
    ---------
    1. Open WebUI discovers this class via the Pipelines registry.
    2. On each chat request, ``pipe(body)`` is called.
    3. The pipeline observes the latest turn, activates relevant memories,
       injects them into the prompt, and returns the modified body.
    """

    # Open WebUI Pipeline metadata
    class Valves:
        """Pipeline configuration exposed in Open WebUI settings."""
        SALIENCE_THRESHOLD: float = settings.salience_threshold
        ACTIVATION_ITERATIONS: int = settings.spreading_activation_iterations
        LATERAL_INHIBITION_K: int = settings.lateral_inhibition_k
        MAX_SEMANTIC_FACTS: int = settings.max_semantic_facts
        MAX_EPISODIC_MEMORIES: int = settings.max_episodic_memories

    def __init__(self) -> None:
        self.name = "Brain Memory"
        self.valves = self.Valves()

        # Initialise components
        settings.ensure_data_dirs()
        self._db = SQLiteEpisodicStore()
        self._graph = load_graph()
        self._episodic_store = EpisodicStore()
        self._working_memory = WorkingMemory()

        # Use NeuralMemoryObserver if any neural module is enabled
        any_neural = any([
            settings.use_pattern_separation,
            settings.use_dopaminergic_gate,
            settings.use_hopfield_memory,
            settings.use_vae_consolidation,
            settings.use_transformer_wm,
            settings.use_learned_forgetting,
            settings.use_gnn_activation,
        ])

        if any_neural:
            self._observer = NeuralMemoryObserver(
                working_memory=self._working_memory,
                episodic_store=self._episodic_store,
                semantic_graph=self._graph,
            )
            logger.info("Using NeuralMemoryObserver with neural modules enabled")
        else:
            self._observer = MemoryObserver(
                working_memory=self._working_memory,
                episodic_store=self._episodic_store,
                semantic_graph=self._graph,
            )

        # Hydrate in-memory store from DB
        for entry in self._db.load_into_store():
            self._episodic_store.add(entry)

        logger.info(
            "BrainMemoryPipeline initialised: %d episodes, %d graph nodes",
            self._episodic_store.active_count,
            self._graph.num_nodes,
        )

    # ── Pipeline protocol ────────────────────────────────────────────

    def pipe(self, body: dict[str, Any]) -> dict[str, Any]:
        """Process a chat request through the memory system.

        Parameters
        ----------
        body:
            The full OpenAI-format request body (keys: ``model``,
            ``messages``, etc.).

        Returns
        -------
        dict — the *modified* body with memory context injected.
        """
        messages: list[dict[str, Any]] = body.get("messages", [])
        if not messages:
            return body

        # 1. Extract the latest user message
        latest = messages[-1]
        text = latest.get("content", "")
        speaker = latest.get("role", "user")

        if not text:
            return body

        # 2-6. Run the full observation + injection pipeline
        modified_messages, info = self._observer.process_turn(
            text=text,
            speaker=speaker,
            messages=messages,
        )

        logger.info(
            "Pipeline: salience=%.3f stored=%s activated=%d nodes",
            info["salience"],
            info["stored"],
            len(info.get("activated_nodes", [])),
        )

        # 7. Replace messages in the body
        body["messages"] = modified_messages

        # 8. Persist new episodes to disk
        if info["stored"]:
            entry = self._episodic_store.get(info["episode_id"])
            if entry:
                self._db.insert(entry)

        # Persist graph after every turn that stores or when consolidation fires
        if info["stored"] or self._observer._turn_counter % settings.consolidation_interval == 0:
            save_graph(self._graph)

        return body

    def observe_response(self, response_text: str) -> None:
        """Post-processing: observe the LLM's response for memory encoding.

        Call this after receiving the LLM response to let the memory
        system encode it if salient.
        """
        self._observer.observe(response_text, speaker="assistant")

    # ── cleanup ──────────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Persist all in-memory state before shutdown."""
        self._db.flush_store(self._episodic_store.values())
        save_graph(self._graph)
        self._db.close()
        logger.info("BrainMemoryPipeline shut down cleanly.")
