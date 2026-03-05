"""
Memory Observer — entry point for every conversation turn.

The observer is the *front door* of the memory system.  Every incoming
message (user or assistant) passes through here.  The observer:

1. Encodes the turn text into an embedding
2. Extracts named entities and topic tags
3. Updates working memory → gets context vector + prediction
4. Computes a salience score
5. If salient, writes the turn to the episodic buffer
6. Periodically triggers memory consolidation
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import torch

from config.settings import settings
from memory.activation import SeedHints, SpreadingActivationEngine
from memory.consolidation import run_consolidation_background
from memory.encoder import EmbeddingEncoder, get_encoder
from memory.episodic import EpisodicEntry, EpisodicStore
from memory.injector import inject
from memory.salience import SalienceScorer
from memory.semantic import SemanticGraph
from memory.working_memory import WorkingMemory
from nlp.entity_extractor import extract_entities
from nlp.intent_detector import IntentCueResult, detect_intent_cues
from nlp.topic_tagger import extract_topics

logger = logging.getLogger(__name__)


class MemoryObserver:
    """Orchestrates the per-turn memory processing pipeline.

    Parameters
    ----------
    encoder:
        Shared embedding encoder (defaults to the global singleton).
    working_memory:
        Working-memory instance (created if *None*).
    episodic_store:
        Episodic memory buffer (created if *None*).
    semantic_graph:
        Semantic knowledge graph (created if *None*).
    salience_scorer:
        Salience scorer (created if *None*).
    """

    def __init__(
        self,
        encoder: EmbeddingEncoder | None = None,
        working_memory: WorkingMemory | None = None,
        episodic_store: EpisodicStore | None = None,
        semantic_graph: SemanticGraph | None = None,
        salience_scorer: SalienceScorer | None = None,
    ) -> None:
        self.encoder = encoder or get_encoder()
        self.working_memory = working_memory or WorkingMemory()
        self.episodic_store = episodic_store or EpisodicStore()
        self.graph = semantic_graph or SemanticGraph()
        self.scorer = salience_scorer or SalienceScorer(use_mlp=settings.use_salience_mlp)
        self.activation_engine = SpreadingActivationEngine(self.graph)

        self._turn_counter: int = 0
        # Per-turn NLP signals stashed for activate_and_inject
        self._last_entities: list[str] = []
        self._last_intent: IntentCueResult | None = None

    # ── public API ──────────────────────────────────────────────────

    def observe(
        self,
        text: str,
        speaker: str = "user",
    ) -> dict[str, Any]:
        """Process a single conversation turn.

        Returns a dict with diagnostic information::

            {
                "embedding_norm": float,
                "entities": list[str],
                "topics": list[str],
                "salience": float,
                "stored": bool,
                "episode_id": str | None,
            }
        """
        # 1. Encode
        embedding = self.encoder.encode(text)

        # 2. Extract NLP signals
        entities = extract_entities(text)
        topics = extract_topics(text)
        intent = detect_intent_cues(text)

        # 3. Update working memory
        context_vector, predicted_next = self.working_memory.update(embedding)

        # Stash turn signals for activate_and_inject to use later
        self._last_entities = entities
        self._last_intent = intent

        # 4. Salience scoring
        salience = self.scorer.score(
            embedding=embedding,
            predicted=predicted_next,
            entities=entities,
            text=text,
            graph=self.graph,
        )

        stored = False
        episode_id: str | None = None

        # 5. Episodic storage (if salient enough)
        if salience >= settings.salience_threshold:
            entry = EpisodicEntry(
                speaker=speaker,
                raw_text=text,
                embedding=embedding.tolist(),
                entities=entities,
                topics=topics,
                salience=salience,
            )
            self.episodic_store.add(entry)
            stored = True
            episode_id = entry.id
            logger.info(
                "Stored episode %s (salience=%.3f, entities=%s)",
                entry.id[:8],
                salience,
                entities,
            )

        # 6. Periodic consolidation
        self._turn_counter += 1
        if self._turn_counter % settings.consolidation_interval == 0:
            logger.info("Triggering background consolidation (turn %d)", self._turn_counter)
            run_consolidation_background(self.episodic_store, self.graph)

        return {
            "embedding_norm": float(embedding.norm().item()),
            "entities": entities,
            "topics": topics,
            "salience": salience,
            "stored": stored,
            "episode_id": episode_id,
            "is_recall_intent": intent.is_recall,
            "intent_targets": intent.targets,
        }

    def activate_and_inject(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[tuple[str, float]]]:
        """Run spreading activation and inject memory context into messages.

        Builds a ``SeedHints`` packet from the latest observed turn so the
        activation engine uses entity, intent, and WM cues — not just
        embedding similarity.

        Parameters
        ----------
        messages:
            OpenAI-format messages list.

        Returns
        -------
        (modified_messages, activated_nodes)
        """
        # Use the current working-memory context vector
        ctx = self.working_memory.context_vector
        if ctx is None:
            return messages, []

        # Project context vector from GRU hidden space → embedding space
        with torch.no_grad():
            ctx_emb = self.working_memory.encoder.predictor(ctx)

        # ── Build multi-channel seed hints ───────────────────────────
        wm_embeddings = list(self.working_memory.buffer._items)

        entities = getattr(self, "_last_entities", [])
        intent = getattr(self, "_last_intent", None)

        hints = SeedHints(
            entities=entities,
            intent_targets=intent.targets if intent else [],
            intent_confidence=intent.confidence if intent else 0.0,
            working_memory_embeddings=wm_embeddings,
            context_vector=ctx_emb,
        )

        # Rebuild adjacency if graph has new nodes
        self.activation_engine.rebuild()

        # Run spreading activation with multi-channel hints
        activated = self.activation_engine.activate(ctx_emb, hints=hints)

        # Gather relevant episodic memories (linked to activated nodes)
        episode_ids: set[str] = set()
        for node_id, strength in activated:
            node = self.graph.get_node(node_id)
            if node:
                # Find episodes that link to this node
                for ep in self.episodic_store.get_all_active():
                    if node_id in ep.links or node.label.lower() in [
                        e.lower() for e in ep.entities
                    ]:
                        episode_ids.add(ep.id)

        episodes = self.episodic_store.retrieve_by_ids(list(episode_ids))

        # Inject into messages
        modified = inject(messages, self.graph, activated, episodes)
        return modified, activated

    def process_turn(
        self,
        text: str,
        speaker: str,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Full turn pipeline: observe + activate + inject.

        Returns
        -------
        (modified_messages, observation_info)
        """
        info = self.observe(text, speaker=speaker)
        modified_messages, activated = self.activate_and_inject(messages)
        info["activated_nodes"] = activated
        return modified_messages, info


# ────────────────────────────────────────────────────────────────────
# NeuralMemoryObserver — wraps MemoryObserver with neural modules
# ────────────────────────────────────────────────────────────────────


class NeuralMemoryObserver(MemoryObserver):
    """MemoryObserver augmented with optional neural components.

    Routes processing through neural modules based on config flags:
    * ``use_pattern_separation`` → separate embeddings before storage
    * ``use_dopaminergic_gate`` → learned gating instead of threshold
    * ``use_hopfield_memory`` → Hopfield retrieval alongside episodic
    * ``use_vae_consolidation`` → VAE consolidation replay
    * ``use_transformer_wm`` → Transformer instead of GRU
    * ``use_learned_forgetting`` → Neural forgetting gate
    * ``use_gnn_activation`` → GAT spreading activation

    All neural modules are optional and created only when enabled.
    The training coordinator runs a mini-batch step after each turn.
    """

    def __init__(
        self,
        encoder: "EmbeddingEncoder | None" = None,
        **kwargs: Any,
    ) -> None:
        # If transformer WM is enabled, swap the working memory module
        if settings.use_transformer_wm:
            from memory.neural_working_memory import TransformerWorkingMemory
            kwargs.setdefault("working_memory", TransformerWorkingMemory())

        super().__init__(encoder=encoder, **kwargs)

        # Lazy-init neural components
        self._pattern_separator: Any = None
        self._gate: Any = None
        self._hopfield: Any = None
        self._forgetting_net: Any = None
        self._gat: Any = None
        self._graph_converter: Any = None
        self._trainer: Any = None
        # EMA tracker for adaptive gate reward scaling
        self._salience_ema: float = 0.3
        self._salience_ema_alpha: float = 0.05

        self._init_neural_components()

    def _init_neural_components(self) -> None:
        """Create neural modules for enabled flags."""
        if settings.use_pattern_separation:
            from memory.pattern_separation import PatternSeparator
            self._pattern_separator = PatternSeparator()

        if settings.use_dopaminergic_gate:
            from memory.gate_network import DopaminergicGate
            self._gate = DopaminergicGate()

        if settings.use_hopfield_memory:
            from memory.hopfield_memory import HippocampalMemory
            self._hopfield = HippocampalMemory()

        if settings.use_learned_forgetting:
            from memory.forgetting import ForgettingNetwork
            self._forgetting_net = ForgettingNetwork()

        if settings.use_gnn_activation:
            from memory.neural_activation import MemoryGAT
            from memory.graph_converter import GraphConverter
            self._gat = MemoryGAT()
            self._graph_converter = GraphConverter()

        # Build shared model references so the trainer trains the same
        # instances that the observer uses at inference time
        shared: dict[str, Any] = {}
        if self._pattern_separator is not None:
            shared["pattern_sep"] = self._pattern_separator
        if self._gate is not None:
            shared["gate"] = self._gate
        if self._hopfield is not None:
            shared["hopfield"] = self._hopfield
        if self._forgetting_net is not None:
            shared["forgetting"] = self._forgetting_net
        if self._gat is not None:
            shared["gat"] = self._gat
        if settings.use_salience_mlp:
            shared["salience_mlp"] = self.scorer
        # Transformer WM encoder lives inside self.working_memory
        if settings.use_transformer_wm and hasattr(self.working_memory, "encoder"):
            shared["transformer_wm"] = self.working_memory.encoder
        # GRU predictor training (only when transformer WM is not used)
        if not settings.use_transformer_wm and hasattr(self.working_memory, "encoder"):
            shared["gru_predictor"] = self.working_memory.encoder

        # Training coordinator manages all components
        from memory.trainer import TrainingCoordinator
        self._trainer = TrainingCoordinator()
        self._trainer.initialise(shared_models=shared)

    def observe(
        self,
        text: str,
        speaker: str = "user",
    ) -> dict[str, Any]:
        """Process a turn with neural enhancements.

        Extends the base observe() with:
        1. Pattern separation (before storage)
        2. Dopaminergic gating (instead of threshold)
        3. Hopfield storage (alongside episodic)
        4. VAE replay buffer push
        5. Online training step
        """
        # 1. Encode
        embedding = self.encoder.encode(text)

        # 2. Pattern separation (transform embedding before anything else)
        original_embedding = embedding
        if self._pattern_separator is not None:
            with torch.no_grad():
                embedding = self._pattern_separator.separate(embedding)
            # Push to pattern sep replay buffer
            self._trainer.push_pattern_sep_experience(original_embedding)

        # 3. NLP signals
        entities = extract_entities(text)
        topics = extract_topics(text)
        intent = detect_intent_cues(text)

        # 3b. Snapshot working memory buffer BEFORE update (for WM training)
        wm_snapshot = None
        if self.working_memory.buffer.size > 0:
            wm_snapshot = self.working_memory.buffer.as_tensor().clone()

        # 4. Update working memory
        context_vector, predicted_next = self.working_memory.update(embedding)

        # 4b. Push WM training data: (pre-update sequence, actual next embedding)
        if wm_snapshot is not None:
            if settings.use_transformer_wm:
                self._trainer.push_transformer_wm_experience(wm_snapshot, embedding)
            else:
                self._trainer.push_gru_experience(wm_snapshot, embedding)

        self._last_entities = entities
        self._last_intent = intent

        # 5. Salience scoring
        from memory.salience import (
            compute_entity_density,
            compute_novelty,
            compute_prediction_error,
            detect_emphasis,
        )

        novelty = compute_novelty(embedding, self.graph)
        pred_err = compute_prediction_error(embedding, predicted_next)
        emphasis = detect_emphasis(text)
        entity_density = compute_entity_density(entities, text)

        salience_signals = torch.tensor(
            [novelty, pred_err, emphasis, entity_density],
            dtype=torch.float32,
        )

        # 6. Gating decision
        if self._gate is not None and context_vector is not None:
            should_store, gate_prob = self._gate.should_store(
                embedding, context_vector, salience_signals,
            )
            salience = gate_prob
            # Adaptive intrinsic reward: scale by running EMA so the gate
            # learns from the current salience distribution instead of a
            # hard-coded multiplier.
            raw = float(salience_signals.mean().item())
            self._salience_ema = (
                (1 - self._salience_ema_alpha) * self._salience_ema
                + self._salience_ema_alpha * raw
            )
            # Target: normalise raw salience relative to EMA, clamp [0, 1]
            intrinsic_reward = min(1.0, max(0.0, raw / (2.0 * self._salience_ema + 1e-8)))
            self._trainer.push_gate_experience(
                embedding, context_vector, salience_signals,
                reward=intrinsic_reward,
            )
        else:
            salience = self.scorer.score(
                embedding=embedding,
                predicted=predicted_next,
                entities=entities,
                text=text,
                graph=self.graph,
            )
            should_store = salience >= settings.salience_threshold

        stored = False
        episode_id: str | None = None

        # 7. Episodic storage
        if should_store:
            entry = EpisodicEntry(
                speaker=speaker,
                raw_text=text,
                embedding=embedding.tolist(),
                entities=entities,
                topics=topics,
                salience=salience,
            )
            self.episodic_store.add(entry)
            stored = True
            episode_id = entry.id

            # 8. Hopfield storage
            if self._hopfield is not None:
                self._hopfield.store(
                    embedding=embedding,
                    episode_id=entry.id,
                )
                # Push retrieval training pair: query=embedding, positive=embedding
                # (the network should retrieve stored patterns matching the query)
                self._trainer.push_hopfield_experience(embedding, embedding)

            # 9. VAE replay buffer
            if settings.use_vae_consolidation:
                from memory.neural_consolidation import build_metadata_vector
                meta = build_metadata_vector(
                    salience=salience,
                    entity_count=len(entities),
                    speaker_is_user=(speaker == "user"),
                )
                self._trainer.push_vae_experience(embedding, meta)

            # 10. Forgetting replay buffer — generate training targets
            #     from salience and novelty signals
            if settings.use_learned_forgetting:
                from memory.forgetting import build_forgetting_scalars
                # Salient/novel memories → low target decay; bland → higher decay
                target_decay = max(0.0, 1.0 - float(salience))
                # Entity-rich or emphasis → lower interference
                target_interf = max(0.0, 0.5 - float(entity_density) * 0.5)
                f_scalars = build_forgetting_scalars(
                    age_hours=0.0,
                    access_count=1,
                    salience=float(salience),
                    last_activation=0.0,
                    context_similarity=float(novelty),
                )
                self._trainer.push_forgetting_experience(
                    embedding, f_scalars, delta_t=0.0,
                    target_decay=target_decay,
                    target_interference=target_interf,
                )

        # 11. Periodic consolidation
        self._turn_counter += 1
        if self._turn_counter % settings.consolidation_interval == 0:
            vae_model = None
            if settings.use_vae_consolidation and self._trainer is not None:
                vae_comp = self._trainer.components.get("vae")
                if vae_comp and vae_comp.enabled and vae_comp.model is not None:
                    vae_model = vae_comp.model
            run_consolidation_background(self.episodic_store, self.graph, vae=vae_model)

        # 12. Online training step
        train_diag = self._trainer.step()

        return {
            "embedding_norm": float(embedding.norm().item()),
            "entities": entities,
            "topics": topics,
            "salience": salience,
            "stored": stored,
            "episode_id": episode_id,
            "is_recall_intent": intent.is_recall,
            "intent_targets": intent.targets,
            "neural_training": train_diag,
            "pattern_separated": self._pattern_separator is not None,
            "gate_used": self._gate is not None,
        }

    def activate_and_inject(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[tuple[str, float]]]:
        """Run activation (algorithmic or GAT), Hopfield retrieval, forgetting
        re-ranking, and inject a single memory context block.

        This replaces — rather than extends — the base class method so that
        only ONE ``[MEMORY CONTEXT]`` block is ever produced.
        """
        ctx = self.working_memory.context_vector
        if ctx is None:
            return messages, []

        with torch.no_grad():
            ctx_emb = self.working_memory.encoder.predictor(ctx)

        # ── Build seed hints (same as base class) ────────────────────
        wm_embeddings = list(self.working_memory.buffer._items)
        entities = getattr(self, "_last_entities", [])
        intent = getattr(self, "_last_intent", None)

        hints = SeedHints(
            entities=entities,
            intent_targets=intent.targets if intent else [],
            intent_confidence=intent.confidence if intent else 0.0,
            working_memory_embeddings=wm_embeddings,
            context_vector=ctx_emb,
        )

        # ── Spreading activation: GAT or algorithmic ─────────────────
        if settings.use_gnn_activation and self._gat is not None and self._graph_converter is not None:
            activated = self._run_gat_activation(ctx_emb)
        else:
            self.activation_engine.rebuild()
            activated = self.activation_engine.activate(ctx_emb, hints=hints)

        # ── Gather episodes linked to activated nodes ────────────────
        episode_ids: set[str] = set()
        for node_id, strength in activated:
            node = self.graph.get_node(node_id)
            if node:
                for ep in self.episodic_store.get_all_active():
                    if node_id in ep.links or node.label.lower() in [
                        e.lower() for e in ep.entities
                    ]:
                        episode_ids.add(ep.id)

        # ── Hopfield retrieval → merge episode sets ──────────────────
        hopfield_episode_scores: dict[str, float] = {}
        if self._hopfield is not None and ctx_emb is not None:
            hopfield_results = self._hopfield.retrieve_episode_ids(ctx_emb, top_k=5)
            if hopfield_results:
                for eid, weight in hopfield_results:
                    hopfield_episode_scores[eid] = float(weight)
                    episode_ids.add(eid)

        episodes = self.episodic_store.retrieve_by_ids(list(episode_ids))

        # ── Forgetting network re-ranking ────────────────────────────
        if (
            settings.use_learned_forgetting
            and self._forgetting_net is not None
            and episodes
        ):
            episodes = self._apply_forgetting_reranking(episodes, ctx)

        # ── Inject exactly once ──────────────────────────────────────
        modified = inject(messages, self.graph, activated, episodes)

        # ── Gate reward for retrieved memories ───────────────────────
        retrieved_embeddings: list[torch.Tensor] = []
        for ep in episodes:
            retrieved_embeddings.append(
                torch.tensor(ep.embedding, dtype=torch.float32)
            )
        if retrieved_embeddings and self._trainer is not None:
            self._trainer.reward_gate_for_retrieval(retrieved_embeddings, reward=1.0)

        return modified, activated

    def _run_gat_activation(
        self,
        ctx_emb: torch.Tensor,
    ) -> list[tuple[str, float]]:
        """Run GAT-based spreading activation and return sorted (node_id, score) pairs."""
        from memory.graph_converter import GraphConverter

        if self._graph_converter is None:
            self._graph_converter = GraphConverter()

        pyg_data = self._graph_converter.get_or_convert(self.graph, self.episodic_store)
        if pyg_data.num_nodes == 0:
            return []

        with torch.no_grad():
            scores = self._gat(pyg_data, ctx_emb)  # Tensor[N]

        # Map back to (node_id, score) pairs sorted descending
        node_ids = pyg_data.node_ids
        scored = [
            (node_ids[i], float(scores[i].item()))
            for i in range(len(node_ids))
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:settings.lateral_inhibition_k]

    def _apply_forgetting_reranking(
        self,
        episodes: list[EpisodicEntry],
        context_vector: torch.Tensor,
    ) -> list[EpisodicEntry]:
        """Re-rank episodes using the forgetting network, dropping those below threshold."""
        from datetime import datetime, timezone
        from memory.forgetting import build_forgetting_scalars

        now_ts = datetime.now(timezone.utc).timestamp()
        embeddings_list = []
        scalars_list = []
        base_activations = []
        delta_ts = []

        for ep in episodes:
            emb = torch.tensor(ep.embedding, dtype=torch.float32)
            embeddings_list.append(emb)

            age_hours = (now_ts - ep.timestamp.timestamp()) / 3600.0
            last_recall = ep.recall_times[-1] if ep.recall_times else ep.timestamp.timestamp()
            dt = (now_ts - last_recall) / 3600.0

            # Compute context similarity for scalar features
            emb_device = emb.to(settings.resolved_device)
            ctx_device = context_vector.to(settings.resolved_device)
            if ctx_device.shape[-1] != emb_device.shape[-1]:
                # context_vector is GRU hidden dim, project to embedding space
                with torch.no_grad():
                    ctx_proj = self.working_memory.encoder.predictor(ctx_device)
                ctx_sim = float(torch.nn.functional.cosine_similarity(
                    emb_device.unsqueeze(0), ctx_proj.unsqueeze(0), dim=1
                ).item())
            else:
                ctx_sim = float(torch.nn.functional.cosine_similarity(
                    emb_device.unsqueeze(0), ctx_device.unsqueeze(0), dim=1
                ).item())

            scalars = build_forgetting_scalars(
                age_hours=age_hours,
                access_count=len(ep.recall_times),
                salience=ep.salience,
                last_activation=ep.activation,
                context_similarity=ctx_sim,
            )
            scalars_list.append(scalars)
            base_activations.append(ep.activation)
            delta_ts.append(dt)

        batch_emb = torch.stack(embeddings_list).to(settings.resolved_device)
        batch_scalars = torch.stack(scalars_list).to(settings.resolved_device)
        batch_base = torch.tensor(base_activations, dtype=torch.float32, device=settings.resolved_device)
        batch_dt = torch.tensor(delta_ts, dtype=torch.float32, device=settings.resolved_device)

        with torch.no_grad():
            effective = self._forgetting_net.compute_effective_activation(
                batch_base, batch_emb, batch_scalars, batch_dt, context_vector,
            )

        # Filter and re-rank
        result = []
        for i, ep in enumerate(episodes):
            eff_score = float(effective[i].item())
            if eff_score >= settings.forgetting_threshold:
                ep.activation = eff_score
                result.append(ep)

        result.sort(key=lambda e: e.activation, reverse=True)
        return result
