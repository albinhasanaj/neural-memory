"""Tests for episodic memory: add/retrieve/decay/archive lifecycle."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from memory.episodic import EpisodicEntry, EpisodicStore, compute_activation


class TestActivationDecay:
    """ACT-R activation computation."""

    def test_single_recent_recall(self) -> None:
        now = time.time()
        act = compute_activation([now - 1.0], t_now=now)
        # Recent recall → positive activation
        assert act > 0

    def test_older_recall_lower_activation(self) -> None:
        now = time.time()
        recent = compute_activation([now - 1.0], t_now=now)
        old = compute_activation([now - 10000.0], t_now=now)
        assert recent > old

    def test_multiple_recalls_higher_activation(self) -> None:
        now = time.time()
        single = compute_activation([now - 10.0], t_now=now)
        multiple = compute_activation([now - 10.0, now - 5.0, now - 1.0], t_now=now)
        assert multiple > single

    def test_empty_recall_list(self) -> None:
        act = compute_activation([])
        assert act == -float("inf")


class TestEpisodicStore:
    """In-memory episodic store operations."""

    def test_add_and_size(self, episodic_store: EpisodicStore) -> None:
        assert episodic_store.size == 5

    def test_active_count(self, episodic_store: EpisodicStore) -> None:
        assert episodic_store.active_count == 5

    def test_retrieve_by_ids_records_access(self, episodic_store: EpisodicStore) -> None:
        entries = episodic_store.get_all_active()
        eid = entries[0].id
        original_recalls = len(entries[0].recall_times)
        results = episodic_store.retrieve_by_ids([eid])
        assert len(results) == 1
        assert len(results[0].recall_times) == original_recalls + 1

    def test_get_by_salience(self, episodic_store: EpisodicStore) -> None:
        high = episodic_store.get_by_salience(min_salience=0.5)
        assert all(e.salience >= 0.5 for e in high)
        assert len(high) >= 1

    def test_get_unconsolidated(self, episodic_store: EpisodicStore) -> None:
        entries = episodic_store.get_unconsolidated(min_salience=0.0)
        assert all(not e.consolidated for e in entries)

    def test_apply_decay(self, episodic_store: EpisodicStore) -> None:
        episodic_store.apply_decay()
        for e in episodic_store.get_all_active():
            assert isinstance(e.activation, float)

    def test_archive_below_threshold(self, episodic_store: EpisodicStore) -> None:
        # Set a very high threshold so everything gets archived
        archived = episodic_store.archive_below_threshold(threshold=999.0)
        assert len(archived) == 5
        assert episodic_store.active_count == 0

    def test_archived_excluded_from_retrieve(self, episodic_store: EpisodicStore) -> None:
        entries = episodic_store.get_all_active()
        eid = entries[0].id
        entries[0].archived = True
        results = episodic_store.retrieve_by_ids([eid])
        assert len(results) == 0
