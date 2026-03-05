"""
Salience Detection / Importance Gating.

Computes a salience score (0–1) for each conversation turn by combining
four independent signals:

* **novelty** — cosine distance from the nearest known semantic node
* **prediction_error** — deviation from the working-memory prediction
* **emphasis** — behavioural cues: caps, exclamation, explicit requests, etc.
* **entity_density** — named entities per word

An initial implementation uses a simple weighted sum.  The ``SalienceScorer``
MLP wrapper is provided so the weights can later be learned.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config.settings import settings
from memory.semantic import SemanticGraph

_device = settings.resolved_device


# ────────────────────────────────────────────────────────────────────
# Individual signals
# ────────────────────────────────────────────────────────────────────


def compute_novelty(
    embedding: Tensor,
    graph: SemanticGraph,
) -> float:
    """Minimum cosine distance from *embedding* to any known semantic node.

    Returns 1.0 when the embedding is maximally novel (distant from everything).
    Returns 0.0 when an identical node exists.
    """
    nodes = graph.all_nodes()
    if not nodes:
        return 1.0  # everything is novel when the graph is empty

    embs: list[list[float]] = []
    for n in nodes:
        if n.embedding:
            embs.append(n.embedding)
    if not embs:
        return 1.0

    node_matrix = torch.tensor(embs, dtype=torch.float32, device=_device)
    sim = F.cosine_similarity(
        embedding.unsqueeze(0).float().to(_device), node_matrix, dim=1
    )  # [K]
    max_sim = sim.max().item()
    return float(1.0 - max_sim)


def compute_prediction_error(
    embedding: Tensor,
    predicted: Tensor | None,
) -> float:
    """Cosine distance between the actual embedding and the predicted-next embedding.

    Returns 0.5 (neutral) when no prediction is available.
    """
    if predicted is None:
        return 0.5
    sim = F.cosine_similarity(
        embedding.unsqueeze(0).float().to(_device),
        predicted.unsqueeze(0).float().to(_device),
        dim=1,
    )
    return float(1.0 - sim.item())


# ── emphasis patterns ───────────────────────────────────────────────

_EMPHASIS_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\b[A-Z]{3,}\b"), 0.3),                        # ALL CAPS words
    (re.compile(r"!{2,}"), 0.2),                                 # multiple exclamation marks
    (re.compile(r"!"), 0.1),                                     # single exclamation
    (re.compile(r"\?{2,}"), 0.15),                               # multiple question marks
    (re.compile(r"remember this|don'?t forget|keep in mind|note that", re.I), 0.5),
    (re.compile(r"actually|correction|no,?\s*I meant|wait,?\s", re.I), 0.35),
    (re.compile(r"important|crucial|critical|essential", re.I), 0.25),
]


def detect_emphasis(text: str) -> float:
    """Heuristic emphasis score from textual cues.  Returns ∈ [0, 1]."""
    score = 0.0
    for pattern, weight in _EMPHASIS_PATTERNS:
        if pattern.search(text):
            score += weight
    return min(score, 1.0)


def compute_entity_density(entities: list[str], text: str) -> float:
    """Named-entity density: ``len(entities) / word_count``.  Capped at 1.0."""
    words = text.split()
    if not words:
        return 0.0
    density = len(entities) / len(words)
    return min(density, 1.0)


# ────────────────────────────────────────────────────────────────────
# SalienceScorer — weighted sum (upgradeable to MLP)
# ────────────────────────────────────────────────────────────────────


class SalienceScorer(nn.Module):
    """Combines the four salience signals into a single 0–1 score.

    The default mode (``use_mlp=False``) applies a fixed weighted sum.
    When ``use_mlp=True`` a small MLP is used instead, allowing weights
    to be learned from feedback data.
    """

    def __init__(self, use_mlp: bool = False) -> None:
        super().__init__()
        self.use_mlp = use_mlp

        # Fixed weights — match the spec
        self.register_buffer(
            "weights",
            torch.tensor([
                settings.salience_novelty_weight,
                settings.salience_prediction_error_weight,
                settings.salience_emphasis_weight,
                settings.salience_entity_density_weight,
            ]),
        )

        # Optional MLP for learned scoring
        if use_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Sigmoid(),
            )

        # Move entire module to device (buffers + params)
        self.to(_device)

    def forward(self, signals: Tensor) -> Tensor:
        """Score a single or batch of signal vectors.

        Parameters
        ----------
        signals:
            ``Tensor[4]`` or ``Tensor[B, 4]`` — the four signal values.

        Returns
        -------
        Tensor — salience score(s) ∈ [0, 1].
        """
        signals = signals.to(_device)
        if self.use_mlp:
            return self.mlp(signals).squeeze(-1)  # type: ignore[attr-defined]

        # Weighted sum, clamped to [0, 1]
        raw = (signals * self.weights).sum(dim=-1)
        return raw.clamp(0.0, 1.0)

    def score(
        self,
        embedding: Tensor,
        predicted: Tensor | None,
        entities: list[str],
        text: str,
        graph: SemanticGraph,
    ) -> float:
        """High-level convenience: compute all signals and return the salience score."""
        novelty = compute_novelty(embedding, graph)
        pred_err = compute_prediction_error(embedding, predicted)
        emphasis = detect_emphasis(text)
        entity_density = compute_entity_density(entities, text)

        signals = torch.tensor(
            [novelty, pred_err, emphasis, entity_density],
            dtype=torch.float32,
            device=_device,
        )
        with torch.no_grad():
            return float(self.forward(signals).item())


# ────────────────────────────────────────────────────────────────────
# Salience MLP Replay Buffer & Training
# ────────────────────────────────────────────────────────────────────


class SalienceReplayBuffer:
    """Stores ``(signals, target_salience)`` for MLP training.

    Target salience is derived from whether the stored memory was later
    retrieved and used (1.0) or never accessed (0.0).
    """

    def __init__(self, capacity: int = settings.training_replay_buffer_size) -> None:
        from collections import deque
        self._buffer: deque[tuple[Tensor, float]] = deque(maxlen=capacity)

    def push(self, signals: Tensor, target: float) -> None:
        self._buffer.append((signals.detach().to(_device), target))

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor]:
        import random
        k = min(batch_size, len(self._buffer))
        indices = random.sample(range(len(self._buffer)), k)
        sigs = torch.stack([self._buffer[i][0] for i in indices])
        tgts = torch.tensor([self._buffer[i][1] for i in indices], device=_device)
        return sigs, tgts

    def __len__(self) -> int:
        return len(self._buffer)


def train_salience_step(
    scorer: SalienceScorer,
    optimizer: torch.optim.Optimizer,
    replay_buffer: SalienceReplayBuffer,
    batch_size: int = 16,
    grad_clip: float = 1.0,
) -> dict[str, float] | None:
    """One training step for the salience MLP.

    Loss: MSE between predicted salience and target.
    Returns loss diagnostics, or *None* if the buffer is too small.
    """
    if not scorer.use_mlp or len(replay_buffer) < batch_size:
        return None

    signals, targets = replay_buffer.sample(batch_size)

    scorer.train()
    predicted = scorer(signals)
    loss = F.mse_loss(predicted, targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(scorer.parameters(), grad_clip)
    optimizer.step()

    scorer.eval()
    return {"loss": loss.item()}
