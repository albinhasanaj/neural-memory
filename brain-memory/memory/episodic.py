"""
Episodic Memory Buffer — ACT-R activation decay over conversation episodes.

Each turn that passes the salience gate is stored as an ``EpisodicEntry``.
Entries carry an *activation* level that decays following the ACT-R
power-law formula.  Low-activation entries are archived (excluded from
active retrieval) but never deleted.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Sequence

from pydantic import BaseModel, Field
from torch import Tensor

from config.settings import settings


# ────────────────────────────────────────────────────────────────────
# Pydantic models
# ────────────────────────────────────────────────────────────────────


class EpisodicEntry(BaseModel):
    """A single episodic memory entry."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    speaker: str = "user"  # "user" | "assistant" | "system"
    raw_text: str = ""
    embedding: list[float] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    salience: float = 0.0
    activation: float = 0.0
    recall_times: list[float] = Field(
        default_factory=list,
        description="Unix timestamps of every access/recall event.",
    )
    links: list[str] = Field(
        default_factory=list,
        description="IDs of linked semantic-graph nodes.",
    )
    consolidated: bool = False
    archived: bool = False

    model_config = {"arbitrary_types_allowed": True}


# ────────────────────────────────────────────────────────────────────
# ACT-R activation computation
# ────────────────────────────────────────────────────────────────────


def compute_activation(
    recall_times: Sequence[float],
    t_now: float | None = None,
    decay: float = settings.decay_rate,
) -> float:
    """Compute ACT-R base-level activation.

    .. math::

        B_i = \\ln \\left( \\sum_{j} (t_{now} - t_j)^{-d} \\right)

    Parameters
    ----------
    recall_times:
        Unix timestamps of each recall / access event.
    t_now:
        Current time as a Unix timestamp.  Defaults to *now*.
    decay:
        The power-law exponent *d* (default 0.5).

    Returns
    -------
    float
        The base-level activation value.
    """
    if t_now is None:
        t_now = datetime.now(timezone.utc).timestamp()

    if not recall_times:
        return -float("inf")

    total = 0.0
    for t_j in recall_times:
        age = max(t_now - t_j, 1e-6)  # avoid division by zero
        total += age ** (-decay)

    return math.log(total + 1e-9)


# ────────────────────────────────────────────────────────────────────
# Episodic Store (in-memory, delegates persistence to storage module)
# ────────────────────────────────────────────────────────────────────


class EpisodicStore:
    """In-memory episodic buffer with ACT-R decay mechanics.

    This class manages the *active* set of episodes.  Persistence is
    handled by :mod:`storage.sqlite_store` — this store is designed to
    be hydrated from the DB at startup and flushed back on shutdown.
    """

    def __init__(self) -> None:
        self._entries: dict[str, EpisodicEntry] = {}

    # ── core CRUD ───────────────────────────────────────────────────

    def add(self, entry: EpisodicEntry) -> EpisodicEntry:
        """Insert a new episodic entry.

        The entry's creation timestamp is automatically added to its
        ``recall_times`` list (first access).
        """
        if not entry.recall_times:
            entry.recall_times.append(entry.timestamp.timestamp())
        entry.activation = compute_activation(entry.recall_times)
        self._entries[entry.id] = entry
        return entry

    def retrieve_by_ids(self, ids: Sequence[str]) -> list[EpisodicEntry]:
        """Retrieve entries by ID and record an access event."""
        now_ts = datetime.now(timezone.utc).timestamp()
        results: list[EpisodicEntry] = []
        for eid in ids:
            entry = self._entries.get(eid)
            if entry and not entry.archived:
                entry.recall_times.append(now_ts)
                entry.activation = compute_activation(entry.recall_times)
                results.append(entry)
        return results

    def get_all_active(self) -> list[EpisodicEntry]:
        """Return all non-archived entries, sorted by activation descending."""
        active = [e for e in self._entries.values() if not e.archived]
        active.sort(key=lambda e: e.activation, reverse=True)
        return active

    def get_by_salience(self, min_salience: float = 0.0) -> list[EpisodicEntry]:
        """Return non-archived entries above a salience threshold."""
        return [
            e
            for e in self._entries.values()
            if not e.archived and e.salience >= min_salience
        ]

    def get_unconsolidated(self, min_salience: float = 0.0) -> list[EpisodicEntry]:
        """Return non-archived, unconsolidated entries above a salience floor."""
        return [
            e
            for e in self._entries.values()
            if not e.archived and not e.consolidated and e.salience >= min_salience
        ]

    # ── decay & archival ────────────────────────────────────────────

    def apply_decay(self) -> None:
        """Re-compute activation for every active entry."""
        now_ts = datetime.now(timezone.utc).timestamp()
        for entry in self._entries.values():
            if not entry.archived:
                entry.activation = compute_activation(entry.recall_times, t_now=now_ts)

    def archive_below_threshold(
        self, threshold: float = settings.activation_threshold
    ) -> list[str]:
        """Archive entries whose activation has fallen below *threshold*.

        Returns the IDs of newly archived entries.
        """
        archived_ids: list[str] = []
        for entry in self._entries.values():
            if not entry.archived and entry.activation < threshold:
                entry.archived = True
                archived_ids.append(entry.id)
        return archived_ids

    # ── helpers ─────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def active_count(self) -> int:
        return sum(1 for e in self._entries.values() if not e.archived)

    def get(self, entry_id: str) -> EpisodicEntry | None:
        return self._entries.get(entry_id)

    def values(self) -> list[EpisodicEntry]:
        return list(self._entries.values())
