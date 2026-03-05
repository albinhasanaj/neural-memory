"""Tests for memory consolidation: clustering and fact extraction pipeline."""

from __future__ import annotations

import pytest
import numpy as np

from memory.episodic import EpisodicEntry, EpisodicStore
from memory.consolidation import cluster_episodes, select_candidates
from tests.conftest import MockEncoder


class TestSelectCandidates:
    """Candidate selection for consolidation."""

    def test_returns_only_unconsolidated(self, episodic_store: EpisodicStore) -> None:
        # Mark one as consolidated
        entries = episodic_store.get_all_active()
        entries[0].consolidated = True

        candidates = select_candidates(episodic_store, min_salience=0.0)
        ids = {c.id for c in candidates}
        assert entries[0].id not in ids

    def test_respects_salience_floor(self, episodic_store: EpisodicStore) -> None:
        candidates = select_candidates(episodic_store, min_salience=0.5)
        assert all(c.salience >= 0.5 for c in candidates)

    def test_empty_store(self) -> None:
        store = EpisodicStore()
        assert select_candidates(store) == []


class TestClusterEpisodes:
    """Agglomerative clustering of episodes."""

    def test_single_episode_returns_one_cluster(self) -> None:
        enc = MockEncoder()
        ep = EpisodicEntry(
            raw_text="test",
            embedding=enc.encode("test").tolist(),
            salience=0.5,
        )
        clusters = cluster_episodes([ep])
        assert len(clusters) == 1
        assert len(clusters[0]) == 1

    def test_similar_episodes_grouped(self) -> None:
        # Create two pairs with similar embeddings
        enc = MockEncoder()
        ep1 = EpisodicEntry(
            raw_text="Python is great", embedding=enc.encode("Python is great").tolist(), salience=0.7
        )
        ep2 = EpisodicEntry(
            raw_text="Python is great for ML", embedding=enc.encode("Python is great").tolist(), salience=0.7
        )
        # Use the same embedding → distance = 0 → should cluster together
        ep2.embedding = ep1.embedding.copy()

        ep3 = EpisodicEntry(
            raw_text="I dislike JavaScript", embedding=enc.encode("I dislike JavaScript").tolist(), salience=0.6
        )

        clusters = cluster_episodes([ep1, ep2, ep3], distance_threshold=0.5)
        # ep1 and ep2 should be in the same cluster (identical embeddings)
        assert any(len(c) >= 2 for c in clusters)

    def test_empty_input(self) -> None:
        assert cluster_episodes([]) == []

    def test_all_different(self) -> None:
        enc = MockEncoder()
        episodes = []
        for i in range(5):
            # Use very different texts to get different embeddings
            ep = EpisodicEntry(
                raw_text=f"completely unique text number {i * 1000}",
                embedding=enc.encode(f"unique_{i * 1000}").tolist(),
                salience=0.5,
            )
            episodes.append(ep)
        clusters = cluster_episodes(episodes, distance_threshold=0.01)
        # With very low threshold, most should be separate clusters
        assert len(clusters) >= 2
