"""
Semantic graph visualizer — interactive HTML via pyvis.

Renders the ``SemanticGraph`` as a force-directed network graph
that can be explored in a browser.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from memory.semantic import SemanticGraph

logger = logging.getLogger(__name__)


def render_graph_html(
    graph: SemanticGraph,
    output_path: str | Path = "data/graph_viz.html",
    height: str = "750px",
    width: str = "100%",
    highlight_nodes: set[str] | None = None,
) -> Path:
    """Render the semantic graph as an interactive HTML file.

    Parameters
    ----------
    graph:
        The semantic graph to visualize.
    output_path:
        Where to write the HTML file.
    height, width:
        Dimensions of the visualization canvas.
    highlight_nodes:
        Node IDs to highlight (e.g. recently activated nodes).

    Returns
    -------
    Path to the generated HTML file.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        logger.error("pyvis is not installed. Install with: pip install pyvis")
        raise

    net = Network(height=height, width=width, directed=True, notebook=False)
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=150)

    highlight = highlight_nodes or set()

    # Add nodes
    for node in graph.all_nodes():
        color = "#ff6b6b" if node.id in highlight else "#4ecdc4"
        size = 25 + (node.activation * 20)
        net.add_node(
            node.id,
            label=node.label or node.id,
            title=f"type: {node.node_type}\nactivation: {node.activation:.3f}",
            color=color,
            size=size,
        )

    # Add edges
    for edge in graph.all_edges():
        net.add_edge(
            edge.source,
            edge.target,
            title=f"{edge.relation} (w={edge.weight:.2f}, c={edge.confidence:.2f})",
            label=edge.relation,
            width=edge.weight * 3,
            color="#888888",
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output_path))
    logger.info("Graph visualization saved to %s", output_path)
    return output_path


def render_graph_matplotlib(
    graph: SemanticGraph,
    output_path: str | Path = "data/graph_viz.png",
    highlight_nodes: set[str] | None = None,
) -> Path:
    """Render the graph using matplotlib (static image fallback)."""
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        logger.error("matplotlib not installed.")
        raise

    highlight = highlight_nodes or set()
    G = graph.nx_graph

    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    colors = ["#ff6b6b" if n in highlight else "#4ecdc4" for n in G.nodes]

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=600, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#aaaaaa", arrows=True, ax=ax)

    edge_labels = {
        (e.source, e.target): e.relation for e in graph.all_edges()
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    ax.set_title("Semantic Knowledge Graph")
    ax.axis("off")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Static graph image saved to %s", output_path)
    return output_path
