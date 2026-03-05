"""
Training Coordinator — online multi-component trainer.

Manages optimizers, replay buffers, learning-rate schedules, and
checkpointing for all neural components.  Each component trains
with its own optimizer and replay buffer but shares a global step
counter for logging.

Components
~~~~~~~~~~
1. **GAT** (:pymod:`memory.neural_activation`)
2. **Hopfield** (:pymod:`memory.hopfield_memory`)
3. **VAE** (:pymod:`memory.neural_consolidation`)
4. **Dopaminergic Gate** (:pymod:`memory.gate_network`)
5. **Pattern Separator** (:pymod:`memory.pattern_separation`)
6. **Transformer WM** (:pymod:`memory.neural_working_memory`)
7. **Forgetting Network** (:pymod:`memory.forgetting`)

Each component is trained only if its config flag is enabled.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from config.settings import settings

logger = logging.getLogger(__name__)
_device = settings.resolved_device


# ────────────────────────────────────────────────────────────────────
# Dataclass for per-component state
# ────────────────────────────────────────────────────────────────────

@dataclass
class ComponentState:
    """Tracks one trainable component's model, optimizer, and replay buffer."""
    name: str
    model: nn.Module | None = None
    optimizer: torch.optim.Optimizer | None = None
    scheduler: Any = None
    replay_buffer: Any = None
    enabled: bool = False
    total_steps: int = 0
    last_loss: float | None = None


# ────────────────────────────────────────────────────────────────────
# Training Coordinator
# ────────────────────────────────────────────────────────────────────


class TrainingCoordinator:
    """Manages online training of all neural components.

    Usage::

        coordinator = TrainingCoordinator()
        coordinator.initialise()  # creates enabled components

        # After each conversation turn:
        coordinator.step()        # trains each component one mini-batch
    """

    def __init__(self) -> None:
        self.global_step: int = 0
        self.components: dict[str, ComponentState] = {}
        self._loss_history: dict[str, list[float]] = {}

    def initialise(self, *, shared_models: dict[str, Any] | None = None) -> None:
        """Create models, optimizers, and replay buffers for enabled components.

        Parameters
        ----------
        shared_models :
            Optional dict mapping component name → pre-existing model instance.
            Use this when the observer owns the model (e.g. Hopfield) and the
            trainer should train the same instance rather than creating a new one.
        """
        shared = shared_models or {}
        self._init_gat(shared.get("gat"))
        self._init_hopfield(shared.get("hopfield"))
        self._init_vae(shared.get("vae"))
        self._init_gate(shared.get("gate"))
        self._init_pattern_separator(shared.get("pattern_sep"))
        self._init_transformer_wm(shared.get("transformer_wm"))
        self._init_gru_predictor(shared.get("gru_predictor"))
        self._init_forgetting(shared.get("forgetting"))
        self._init_salience_mlp(shared.get("salience_mlp"))

        enabled = [n for n, c in self.components.items() if c.enabled]
        logger.info("TrainingCoordinator initialised. Enabled: %s", enabled)

        # Attach cosine-annealing LR schedulers to each enabled component
        for name, comp in self.components.items():
            if comp.enabled and comp.optimizer is not None:
                comp.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    comp.optimizer, T_max=5000, eta_min=1e-6,
                )

    # ── Component initialisers ──────────────────────────────────────

    def _init_gat(self, external_model: Any = None) -> None:
        state = ComponentState(name="gat", enabled=settings.use_gnn_activation)
        if state.enabled:
            from memory.neural_activation import ActivationReplayBuffer, MemoryGAT
            model = external_model or MemoryGAT()
            state.model = model
            state.optimizer = torch.optim.Adam(model.parameters(), lr=settings.gat_learning_rate)
            state.replay_buffer = ActivationReplayBuffer(capacity=settings.training_replay_buffer_size)
        self.components["gat"] = state

    def _init_hopfield(self, external_model: Any = None) -> None:
        state = ComponentState(name="hopfield", enabled=settings.use_hopfield_memory)
        if state.enabled:
            from memory.hopfield_memory import HippocampalMemory, HopfieldReplayBuffer
            model = external_model or HippocampalMemory()
            state.model = model
            # Train separator, query_proj, and log_beta
            trainable = (
                list(model.separator.parameters())
                + list(model.query_proj.parameters())
                + [model.log_beta]
            )
            state.optimizer = torch.optim.Adam(trainable, lr=settings.gat_learning_rate)
            state.replay_buffer = HopfieldReplayBuffer(capacity=settings.training_replay_buffer_size)
        self.components["hopfield"] = state

    def _init_vae(self, external_model: Any = None) -> None:
        state = ComponentState(name="vae", enabled=settings.use_vae_consolidation)
        if state.enabled:
            from memory.neural_consolidation import ConsolidationReplayBuffer, ConsolidationVAE
            model = external_model or ConsolidationVAE()
            state.model = model
            state.optimizer = torch.optim.Adam(model.parameters(), lr=settings.consolidation_vae_lr)
            state.replay_buffer = ConsolidationReplayBuffer(capacity=settings.training_replay_buffer_size)
        self.components["vae"] = state

    def _init_gate(self, external_model: Any = None) -> None:
        state = ComponentState(name="gate", enabled=settings.use_dopaminergic_gate)
        if state.enabled:
            from memory.gate_network import DopaminergicGate, GateReplayBuffer
            model = external_model or DopaminergicGate()
            state.model = model
            state.optimizer = torch.optim.Adam(model.parameters(), lr=settings.gate_learning_rate)
            state.replay_buffer = GateReplayBuffer(capacity=settings.training_replay_buffer_size)
        self.components["gate"] = state

    def _init_pattern_separator(self, external_model: Any = None) -> None:
        state = ComponentState(name="pattern_sep", enabled=settings.use_pattern_separation)
        if state.enabled:
            from memory.pattern_separation import PatternSeparator, SeparationReplayBuffer
            model = external_model or PatternSeparator()
            state.model = model
            state.optimizer = torch.optim.Adam(model.parameters(), lr=settings.pattern_sep_lr)
            state.replay_buffer = SeparationReplayBuffer(capacity=settings.training_replay_buffer_size)
        self.components["pattern_sep"] = state

    def _init_transformer_wm(self, external_model: Any = None) -> None:
        state = ComponentState(name="transformer_wm", enabled=settings.use_transformer_wm)
        if state.enabled:
            from memory.neural_working_memory import (
                TransformerContextEncoder,
                TransformerWMReplayBuffer,
            )
            model = external_model or TransformerContextEncoder()
            state.model = model
            state.optimizer = torch.optim.Adam(model.parameters(), lr=settings.wm_transformer_lr)
            state.replay_buffer = TransformerWMReplayBuffer(capacity=settings.training_replay_buffer_size)
        self.components["transformer_wm"] = state

    def _init_gru_predictor(self, external_model: Any = None) -> None:
        # Train GRU predictor only when transformer WM is NOT used
        # (the transformer WM already trains its own prediction head)
        state = ComponentState(name="gru_predictor", enabled=not settings.use_transformer_wm)
        if state.enabled:
            from memory.working_memory import GRUContextEncoder, GRUReplayBuffer
            model = external_model or GRUContextEncoder()
            state.model = model
            state.optimizer = torch.optim.Adam(model.parameters(), lr=settings.gru_predictor_lr)
            state.replay_buffer = GRUReplayBuffer(capacity=settings.training_replay_buffer_size)
        self.components["gru_predictor"] = state

    def _init_forgetting(self, external_model: Any = None) -> None:
        state = ComponentState(name="forgetting", enabled=settings.use_learned_forgetting)
        if state.enabled:
            from memory.forgetting import ForgettingNetwork, ForgettingReplayBuffer
            model = external_model or ForgettingNetwork()
            state.model = model
            state.optimizer = torch.optim.Adam(model.parameters(), lr=settings.forgetting_lr)
            state.replay_buffer = ForgettingReplayBuffer(capacity=settings.training_replay_buffer_size)
        self.components["forgetting"] = state

    def _init_salience_mlp(self, external_model: Any = None) -> None:
        state = ComponentState(name="salience_mlp", enabled=settings.use_salience_mlp)
        if state.enabled:
            from memory.salience import SalienceReplayBuffer, SalienceScorer
            model = external_model or SalienceScorer(use_mlp=True)
            state.model = model
            state.optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad],
                lr=settings.gate_learning_rate,  # reuse gate LR as a reasonable default
            )
            state.replay_buffer = SalienceReplayBuffer(capacity=settings.training_replay_buffer_size)
        self.components["salience_mlp"] = state

    # ── Training step ───────────────────────────────────────────────

    def step(self) -> dict[str, Any]:
        """Run one training step for each enabled component.

        Returns a dict of per-component loss diagnostics.
        """
        self.global_step += 1
        results: dict[str, Any] = {"global_step": self.global_step}

        for name, comp in self.components.items():
            if not comp.enabled or comp.model is None:
                continue

            diag = self._step_component(name, comp)
            if diag is not None:
                results[name] = diag
                comp.last_loss = diag.get("total", diag.get("loss", None))
                comp.total_steps += 1
                self._loss_history.setdefault(name, []).append(comp.last_loss or 0.0)
                if comp.scheduler is not None:
                    comp.scheduler.step()

        return results

    def _step_component(
        self, name: str, comp: ComponentState,
    ) -> dict[str, float] | None:
        """Dispatch a single training step to the appropriate function."""
        if name == "gat" and comp.replay_buffer is not None:
            from memory.neural_activation import train_gat_step
            batch = comp.replay_buffer.sample(8)
            if not batch:
                return None
            loss = train_gat_step(comp.model, comp.optimizer, batch)  # type: ignore
            return {"loss": loss} if loss is not None else None

        if name == "vae" and comp.replay_buffer is not None:
            from memory.neural_consolidation import train_vae_step
            return train_vae_step(comp.model, comp.optimizer, comp.replay_buffer)  # type: ignore

        if name == "gate" and comp.replay_buffer is not None:
            from memory.gate_network import train_gate_step
            return train_gate_step(comp.model, comp.optimizer, comp.replay_buffer)  # type: ignore

        if name == "pattern_sep" and comp.replay_buffer is not None:
            from memory.pattern_separation import train_separator_step
            return train_separator_step(comp.model, comp.optimizer, comp.replay_buffer)  # type: ignore

        if name == "forgetting" and comp.replay_buffer is not None:
            from memory.forgetting import train_forgetting_step
            return train_forgetting_step(comp.model, comp.optimizer, comp.replay_buffer)  # type: ignore

        if name == "hopfield" and comp.replay_buffer is not None:
            from memory.hopfield_memory import train_hopfield_step
            return train_hopfield_step(comp.model, comp.optimizer, comp.replay_buffer)  # type: ignore

        if name == "transformer_wm" and comp.replay_buffer is not None:
            from memory.neural_working_memory import train_transformer_wm_step
            return train_transformer_wm_step(comp.model, comp.optimizer, comp.replay_buffer)  # type: ignore

        if name == "gru_predictor" and comp.replay_buffer is not None:
            from memory.working_memory import train_gru_step
            return train_gru_step(comp.model, comp.optimizer, comp.replay_buffer)  # type: ignore

        if name == "salience_mlp" and comp.replay_buffer is not None:
            from memory.salience import train_salience_step
            return train_salience_step(comp.model, comp.optimizer, comp.replay_buffer)  # type: ignore

        return None

    # ── Push data into replay buffers ───────────────────────────────

    def push_gat_experience(self, pyg_data: Any, context: Any, labels: Any) -> None:
        comp = self.components.get("gat")
        if comp and comp.enabled and comp.replay_buffer is not None:
            comp.replay_buffer.push(pyg_data, context, labels)

    def push_vae_experience(self, embedding: Any, metadata: Any) -> None:
        comp = self.components.get("vae")
        if comp and comp.enabled and comp.replay_buffer is not None:
            comp.replay_buffer.push(embedding, metadata)

    def push_gate_experience(
        self, embedding: Any, context: Any, signals: Any, reward: float,
    ) -> None:
        comp = self.components.get("gate")
        if comp and comp.enabled and comp.replay_buffer is not None:
            comp.replay_buffer.push(embedding, context, signals, reward)

    def push_pattern_sep_experience(self, embedding: Any) -> None:
        comp = self.components.get("pattern_sep")
        if comp and comp.enabled and comp.replay_buffer is not None:
            comp.replay_buffer.push(embedding)

    def push_forgetting_experience(
        self,
        embedding: Any,
        scalars: Any,
        delta_t: float,
        target_decay: float,
        target_interference: float,
    ) -> None:
        comp = self.components.get("forgetting")
        if comp and comp.enabled and comp.replay_buffer is not None:
            comp.replay_buffer.push(embedding, scalars, delta_t, target_decay, target_interference)

    def push_hopfield_experience(self, query: Any, positive: Any) -> None:
        """Push a (query, positive_target) pair for Hopfield retrieval training."""
        comp = self.components.get("hopfield")
        if comp and comp.enabled and comp.replay_buffer is not None:
            comp.replay_buffer.push(query, positive)

    def push_transformer_wm_experience(self, sequence: Any, target_next: Any) -> None:
        """Push a (sequence_snapshot, actual_next_embedding) pair for WM training."""
        comp = self.components.get("transformer_wm")
        if comp and comp.enabled and comp.replay_buffer is not None:
            comp.replay_buffer.push(sequence, target_next)

    def push_gru_experience(self, sequence: Any, target_next: Any) -> None:
        """Push a (sequence_snapshot, actual_next_embedding) pair for GRU predictor training."""
        comp = self.components.get("gru_predictor")
        if comp and comp.enabled and comp.replay_buffer is not None:
            comp.replay_buffer.push(sequence, target_next)

    def push_salience_experience(self, signals: Any, target: float) -> None:
        """Push a (signals, target_salience) pair for salience MLP training."""
        comp = self.components.get("salience_mlp")
        if comp and comp.enabled and comp.replay_buffer is not None:
            comp.replay_buffer.push(signals, target)

    def reward_gate_for_retrieval(self, used_embeddings: list[Any], reward: float = 1.0) -> int:
        """Retroactively reward the gate for memories that were retrieved and used.

        Finds gate replay buffer entries whose embeddings match the used
        embeddings (by cosine similarity) and updates their reward.

        Returns the number of entries updated.
        """
        comp = self.components.get("gate")
        if not (comp and comp.enabled and comp.replay_buffer is not None):
            return 0
        return comp.replay_buffer.update_rewards_for_embeddings(used_embeddings, reward)

    # ── Checkpointing ───────────────────────────────────────────────

    def save_checkpoint(self, path: Path | str) -> None:
        """Save all enabled models and optimizers to a directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        state = {
            "global_step": self.global_step,
            "components": {},
        }

        for name, comp in self.components.items():
            if not comp.enabled or comp.model is None:
                continue
            comp_path = path / f"{name}_model.pt"
            torch.save(comp.model.state_dict(), comp_path)
            if comp.optimizer is not None:
                opt_path = path / f"{name}_optimizer.pt"
                torch.save(comp.optimizer.state_dict(), opt_path)
            if comp.scheduler is not None:
                sched_path = path / f"{name}_scheduler.pt"
                torch.save(comp.scheduler.state_dict(), sched_path)
            state["components"][name] = {
                "total_steps": comp.total_steps,
                "last_loss": comp.last_loss,
            }

        meta_path = path / "coordinator_meta.json"
        with open(meta_path, "w") as f:
            json.dump(state, f, indent=2)

        logger.info("Saved checkpoint to %s (step %d)", path, self.global_step)

    def load_checkpoint(self, path: Path | str) -> None:
        """Load models and optimizers from a checkpoint directory."""
        path = Path(path)
        meta_path = path / "coordinator_meta.json"
        if not meta_path.exists():
            logger.warning("No checkpoint found at %s", path)
            return

        with open(meta_path) as f:
            state = json.load(f)

        self.global_step = state.get("global_step", 0)

        for name, comp in self.components.items():
            if not comp.enabled or comp.model is None:
                continue

            model_path = path / f"{name}_model.pt"
            if model_path.exists():
                try:
                    state_dict = torch.load(model_path, map_location=_device, weights_only=True)
                    # Pre-resize any buffers whose shapes differ from checkpoint
                    for key, val in state_dict.items():
                        parts = key.split(".")
                        try:
                            parent = comp.model
                            for p in parts[:-1]:
                                parent = getattr(parent, p)
                            current = getattr(parent, parts[-1])
                            if isinstance(current, torch.Tensor) and current.shape != val.shape:
                                parent.register_buffer(parts[-1], torch.empty_like(val))
                        except AttributeError:
                            pass
                    comp.model.load_state_dict(state_dict, strict=False)
                    logger.info("Loaded %s model from %s", name, model_path)
                except RuntimeError as e:
                    logger.warning("Could not load %s model (shape mismatch?): %s", name, e)

            opt_path = path / f"{name}_optimizer.pt"
            if opt_path.exists() and comp.optimizer is not None:
                try:
                    comp.optimizer.load_state_dict(torch.load(opt_path, map_location=_device, weights_only=True))
                except (RuntimeError, ValueError) as e:
                    logger.warning("Could not load %s optimizer: %s", name, e)

            sched_path = path / f"{name}_scheduler.pt"
            if sched_path.exists() and comp.scheduler is not None:
                try:
                    comp.scheduler.load_state_dict(torch.load(sched_path, map_location=_device, weights_only=True))
                except (RuntimeError, ValueError) as e:
                    logger.warning("Could not load %s scheduler: %s", name, e)

            comp_meta = state.get("components", {}).get(name, {})
            comp.total_steps = comp_meta.get("total_steps", 0)
            comp.last_loss = comp_meta.get("last_loss", None)

        logger.info("Loaded checkpoint from %s (step %d)", path, self.global_step)

    # ── Diagnostics ─────────────────────────────────────────────────

    def get_loss_history(self) -> dict[str, list[float]]:
        """Return per-component loss history."""
        return dict(self._loss_history)

    def summary(self) -> dict[str, Any]:
        """Return a summary of all components."""
        result: dict[str, Any] = {"global_step": self.global_step}
        for name, comp in self.components.items():
            result[name] = {
                "enabled": comp.enabled,
                "total_steps": comp.total_steps,
                "last_loss": comp.last_loss,
                "has_model": comp.model is not None,
                "param_count": sum(p.numel() for p in comp.model.parameters()) if comp.model else 0,
            }
        return result
