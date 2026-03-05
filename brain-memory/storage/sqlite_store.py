"""
SQLite backend for episodic memory persistence.

Stores ``EpisodicEntry`` objects in a single ``episodes`` table.
Embeddings are stored as BLOBs (numpy ``tobytes`` / ``frombytes``),
and structured fields (entities, topics, recall_times, links) are
stored as JSON text columns.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from config.settings import settings
from memory.episodic import EpisodicEntry


_CREATE_TABLE = """\
CREATE TABLE IF NOT EXISTS episodes (
    id              TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    speaker         TEXT NOT NULL DEFAULT 'user',
    raw_text        TEXT NOT NULL DEFAULT '',
    embedding       BLOB,
    entities        TEXT NOT NULL DEFAULT '[]',
    topics          TEXT NOT NULL DEFAULT '[]',
    salience        REAL NOT NULL DEFAULT 0.0,
    activation      REAL NOT NULL DEFAULT 0.0,
    recall_times    TEXT NOT NULL DEFAULT '[]',
    links           TEXT NOT NULL DEFAULT '[]',
    consolidated    INTEGER NOT NULL DEFAULT 0,
    archived        INTEGER NOT NULL DEFAULT 0
);
"""

_CREATE_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_episodes_salience ON episodes(salience);
CREATE INDEX IF NOT EXISTS idx_episodes_archived ON episodes(archived);
CREATE INDEX IF NOT EXISTS idx_episodes_consolidated ON episodes(consolidated);
"""


class SQLiteEpisodicStore:
    """SQLite-backed persistence for episodic memory entries.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created if it doesn't exist.
    """

    def __init__(self, db_path: str | Path = settings.episodic_db_path) -> None:
        self._db_path = str(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_CREATE_TABLE + _CREATE_INDEX)
        self._conn.commit()

    # ── serialization helpers ────────────────────────────────────────

    @staticmethod
    def _embedding_to_blob(embedding: list[float]) -> bytes:
        return np.array(embedding, dtype=np.float32).tobytes()

    @staticmethod
    def _blob_to_embedding(blob: bytes) -> list[float]:
        return np.frombuffer(blob, dtype=np.float32).tolist()

    def _row_to_entry(self, row: tuple) -> EpisodicEntry:
        """Convert a database row tuple → ``EpisodicEntry``."""
        (
            id_,
            timestamp_str,
            speaker,
            raw_text,
            embedding_blob,
            entities_json,
            topics_json,
            salience,
            activation,
            recall_json,
            links_json,
            consolidated_int,
            archived_int,
        ) = row
        return EpisodicEntry(
            id=id_,
            timestamp=datetime.fromisoformat(timestamp_str),
            speaker=speaker,
            raw_text=raw_text,
            embedding=self._blob_to_embedding(embedding_blob) if embedding_blob else [],
            entities=json.loads(entities_json),
            topics=json.loads(topics_json),
            salience=salience,
            activation=activation,
            recall_times=json.loads(recall_json),
            links=json.loads(links_json),
            consolidated=bool(consolidated_int),
            archived=bool(archived_int),
        )

    # ── CRUD ─────────────────────────────────────────────────────────

    def insert(self, entry: EpisodicEntry) -> None:
        """Insert a new episode into the database."""
        self._conn.execute(
            """INSERT OR REPLACE INTO episodes
               (id, timestamp, speaker, raw_text, embedding,
                entities, topics, salience, activation,
                recall_times, links, consolidated, archived)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id,
                entry.timestamp.isoformat(),
                entry.speaker,
                entry.raw_text,
                self._embedding_to_blob(entry.embedding),
                json.dumps(entry.entities),
                json.dumps(entry.topics),
                entry.salience,
                entry.activation,
                json.dumps(entry.recall_times),
                json.dumps(entry.links),
                int(entry.consolidated),
                int(entry.archived),
            ),
        )
        self._conn.commit()

    def get(self, entry_id: str) -> EpisodicEntry | None:
        """Retrieve a single episode by ID."""
        cur = self._conn.execute("SELECT * FROM episodes WHERE id = ?", (entry_id,))
        row = cur.fetchone()
        return self._row_to_entry(row) if row else None

    def get_all(self, include_archived: bool = False) -> list[EpisodicEntry]:
        """Retrieve all episodes, optionally including archived ones."""
        if include_archived:
            cur = self._conn.execute("SELECT * FROM episodes ORDER BY timestamp DESC")
        else:
            cur = self._conn.execute(
                "SELECT * FROM episodes WHERE archived = 0 ORDER BY timestamp DESC"
            )
        return [self._row_to_entry(row) for row in cur.fetchall()]

    def get_by_salience(
        self, min_salience: float = 0.0, include_archived: bool = False
    ) -> list[EpisodicEntry]:
        """Retrieve episodes above a salience threshold."""
        query = "SELECT * FROM episodes WHERE salience >= ?"
        if not include_archived:
            query += " AND archived = 0"
        query += " ORDER BY salience DESC"
        cur = self._conn.execute(query, (min_salience,))
        return [self._row_to_entry(row) for row in cur.fetchall()]

    def update(self, entry: EpisodicEntry) -> None:
        """Update an existing episode (same as insert with REPLACE)."""
        self.insert(entry)

    def delete(self, entry_id: str) -> None:
        """Hard-delete an episode."""
        self._conn.execute("DELETE FROM episodes WHERE id = ?", (entry_id,))
        self._conn.commit()

    def bulk_insert(self, entries: Sequence[EpisodicEntry]) -> None:
        """Insert multiple episodes in a single transaction."""
        with self._conn:
            for entry in entries:
                self._conn.execute(
                    """INSERT OR REPLACE INTO episodes
                       (id, timestamp, speaker, raw_text, embedding,
                        entities, topics, salience, activation,
                        recall_times, links, consolidated, archived)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entry.id,
                        entry.timestamp.isoformat(),
                        entry.speaker,
                        entry.raw_text,
                        self._embedding_to_blob(entry.embedding),
                        json.dumps(entry.entities),
                        json.dumps(entry.topics),
                        entry.salience,
                        entry.activation,
                        json.dumps(entry.recall_times),
                        json.dumps(entry.links),
                        int(entry.consolidated),
                        int(entry.archived),
                    ),
                )

    # ── hydration ────────────────────────────────────────────────────

    def load_into_store(self) -> list[EpisodicEntry]:
        """Load all active episodes — used to hydrate the in-memory EpisodicStore at startup."""
        return self.get_all(include_archived=False)

    def flush_store(self, entries: Sequence[EpisodicEntry]) -> None:
        """Persist an entire in-memory store to the database."""
        self.bulk_insert(entries)

    # ── partial updates ──────────────────────────────────────────────

    def update_links(self, episode_id: str, links: list[str]) -> None:
        """Update the links JSON column for a specific episode."""
        self._conn.execute(
            "UPDATE episodes SET links = ? WHERE id = ?",
            (json.dumps(links), episode_id),
        )
        self._conn.commit()

    # ── lifecycle ────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM episodes")
        return cur.fetchone()[0]  # type: ignore[index]

    def close(self) -> None:
        self._conn.close()
