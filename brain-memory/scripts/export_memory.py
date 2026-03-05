"""
Export memory — dump the full memory state to JSON for analysis.

Exports both the episodic memory store and the semantic knowledge
graph into a single JSON file that can be loaded into notebooks,
shared with collaborators, or diffed across experiments.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings
from storage.graph_store import load_graph, _encode_embedding
from storage.sqlite_store import SQLiteEpisodicStore

logger = logging.getLogger(__name__)


def export_all(output_path: str | Path = "data/memory_export.json") -> Path:
    """Export episodes + graph to a single JSON file.

    Returns the path to the written file.
    """
    settings.ensure_data_dirs()

    # Load episodic data
    db = SQLiteEpisodicStore()
    episodes = db.get_all(include_archived=True)
    db.close()

    # Load semantic graph
    graph = load_graph()

    # Build export structure
    export_data = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_episodes": len(episodes),
            "active_episodes": sum(1 for e in episodes if not e.archived),
            "archived_episodes": sum(1 for e in episodes if e.archived),
            "graph_nodes": graph.num_nodes,
            "graph_edges": graph.num_edges,
        },
        "episodes": [
            {
                "id": e.id,
                "timestamp": e.timestamp.isoformat(),
                "speaker": e.speaker,
                "raw_text": e.raw_text,
                "entities": e.entities,
                "topics": e.topics,
                "salience": e.salience,
                "activation": e.activation,
                "consolidated": e.consolidated,
                "archived": e.archived,
                "embedding_b64": _encode_embedding(e.embedding) if e.embedding else None,
            }
            for e in episodes
        ],
        "graph": {
            "nodes": [
                {
                    "id": n.id,
                    "label": n.label,
                    "node_type": n.node_type,
                    "activation": n.activation,
                    "embedding_b64": _encode_embedding(n.embedding) if n.embedding else None,
                    "metadata": n.metadata,
                }
                for n in graph.all_nodes()
            ],
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "relation": e.relation,
                    "weight": e.weight,
                    "confidence": e.confidence,
                    "evidence": e.evidence,
                }
                for e in graph.all_edges()
            ],
        },
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(export_data, indent=2, default=str))

    logger.info(
        "Exported %d episodes + %d nodes + %d edges → %s",
        len(episodes),
        graph.num_nodes,
        graph.num_edges,
        output_path,
    )
    return output_path


def main() -> None:
    """CLI entry point for ``brain-export``."""
    logging.basicConfig(level=logging.INFO)

    output = "data/memory_export.json"
    if len(sys.argv) > 1:
        output = sys.argv[1]

    path = export_all(output)
    print(f"Memory exported to {path}")


if __name__ == "__main__":
    main()
