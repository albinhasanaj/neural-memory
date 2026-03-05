"""
Dashboard — Streamlit-based research introspection tool.

Combines all visualization modules into a single interactive dashboard:
  - Live working memory buffer contents
  - Semantic graph explorer
  - Activation trace per conversation turn
  - Episodic memory browser with search
  - Consolidation log

Launch with::

    streamlit run visualization/dashboard.py
    # or
    brain-dashboard
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    """Dashboard entry point — must be run via ``streamlit run``."""
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit is not installed. Install with: pip install streamlit")
        sys.exit(1)

    from config.settings import settings
    from memory.episodic import EpisodicStore
    from memory.semantic import SemanticGraph
    from storage.graph_store import load_graph
    from storage.sqlite_store import SQLiteEpisodicStore
    from visualization.activation_map import render_activation_text
    from visualization.memory_timeline import render_timeline_text

    st.set_page_config(page_title="Brain Memory Dashboard", layout="wide")
    st.title("🧠 Brain Memory — Research Dashboard")

    # ── Sidebar: data loading ────────────────────────────────────────
    st.sidebar.header("Data Sources")
    db_path = st.sidebar.text_input("Episodic DB", str(settings.episodic_db_path))
    graph_path = st.sidebar.text_input("Semantic Graph", str(settings.semantic_graph_path))

    # Load data
    @st.cache_resource
    def load_episodic(path: str) -> SQLiteEpisodicStore:
        return SQLiteEpisodicStore(path)

    @st.cache_resource
    def load_semantic(path: str) -> SemanticGraph:
        return load_graph(path)

    db = load_episodic(db_path)
    graph = load_semantic(graph_path)

    episodes = db.get_all(include_archived=True)

    # ── Tab layout ───────────────────────────────────────────────────
    tab_episodes, tab_graph, tab_activation, tab_settings = st.tabs([
        "📝 Episodes", "🕸️ Graph", "⚡ Activation", "⚙️ Settings",
    ])

    # ── Episodes tab ─────────────────────────────────────────────────
    with tab_episodes:
        st.subheader("Episodic Memory Browser")
        st.metric("Total episodes", len(episodes))
        active = [e for e in episodes if not e.archived]
        archived = [e for e in episodes if e.archived]
        st.metric("Active", len(active))
        st.metric("Archived", len(archived))

        search_query = st.text_input("Search episodes", "")
        filtered = episodes
        if search_query:
            filtered = [
                e for e in episodes
                if search_query.lower() in e.raw_text.lower()
                or search_query.lower() in " ".join(e.entities).lower()
            ]

        st.text("Timeline view:")
        st.code(render_timeline_text(filtered))

        if filtered:
            selected_idx = st.selectbox(
                "Inspect episode",
                range(len(filtered)),
                format_func=lambda i: f"[{filtered[i].timestamp.strftime('%Y-%m-%d %H:%M')}] {filtered[i].raw_text[:50]}...",
            )
            ep = filtered[selected_idx]
            st.json(ep.model_dump(mode="json"))

    # ── Graph tab ─────────────────────────────────────────────────────
    with tab_graph:
        st.subheader("Semantic Knowledge Graph")
        st.metric("Nodes", graph.num_nodes)
        st.metric("Edges", graph.num_edges)

        if graph.num_nodes > 0:
            st.write("**Nodes:**")
            for node in graph.all_nodes():
                with st.expander(f"{node.label} ({node.node_type})"):
                    st.write(f"ID: `{node.id}`")
                    st.write(f"Activation: {node.activation:.3f}")
                    st.write(f"Embedding dim: {len(node.embedding)}")
                    neighbors = graph.get_neighbors(node.id)
                    if neighbors:
                        st.write("**Connections:**")
                        for neighbor_id, edge in neighbors:
                            st.write(
                                f"  → {neighbor_id}: {edge.relation} "
                                f"(w={edge.weight:.2f}, c={edge.confidence:.2f})"
                            )

            # pyvis rendering
            try:
                from visualization.graph_viz import render_graph_html

                html_path = render_graph_html(graph, output_path="data/dashboard_graph.html")
                st.components.v1.html(html_path.read_text(), height=500, scrolling=True)
            except ImportError:
                st.info("Install pyvis for interactive graph: `pip install pyvis`")
        else:
            st.info("No semantic nodes yet. Run some conversations to populate the graph.")

    # ── Activation tab ────────────────────────────────────────────────
    with tab_activation:
        st.subheader("Spreading Activation Trace")
        st.info(
            "This tab will show activation traces once you run conversations "
            "through the system. Use the proxy server or pipeline integration."
        )

        # Demo: manual activation test
        if graph.num_nodes > 0:
            st.write("**Test activation with a query:**")
            test_query = st.text_input("Enter query text", "Python web framework")
            if st.button("Run Activation"):
                try:
                    from memory.encoder import get_encoder
                    from memory.activation import SpreadingActivationEngine

                    encoder = get_encoder()
                    ctx = encoder.encode(test_query)
                    engine = SpreadingActivationEngine(graph)
                    results = engine.activate(ctx)
                    st.code(render_activation_text(results))
                except Exception as e:
                    st.error(f"Activation failed: {e}")

    # ── Settings tab ──────────────────────────────────────────────────
    with tab_settings:
        st.subheader("Current Configuration")
        st.json(settings.model_dump(mode="json"))


if __name__ == "__main__":
    main()
