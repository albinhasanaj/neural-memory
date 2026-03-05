"""
Activation map — visualize which semantic nodes activated for a given turn.

Generates a heatmap-style view of node activations, useful for
debugging the spreading activation engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


def render_activation_map(
    activated_nodes: list[tuple[str, float]],
    output_path: str | Path = "data/activation_map.html",
    title: str = "Activation Map",
) -> Path:
    """Render an activation bar chart as an interactive HTML file via plotly.

    Parameters
    ----------
    activated_nodes:
        ``[(node_id, strength), ...]`` sorted by strength descending.
    output_path:
        Where to write the HTML.
    title:
        Chart title.

    Returns
    -------
    Path to the generated HTML file.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.error("plotly is not installed. Install with: pip install plotly")
        raise

    if not activated_nodes:
        logger.warning("No activated nodes to visualize.")
        activated_nodes = [("(none)", 0.0)]

    node_ids = [n for n, _ in activated_nodes]
    strengths = [s for _, s in activated_nodes]

    colors = [
        f"rgba(255, {int(107 + 148 * (1 - s / max(strengths, default=1)))}, 107, 0.8)"
        if max(strengths, default=0) > 0
        else "rgba(200,200,200,0.8)"
        for s in strengths
    ]

    fig = go.Figure(data=[
        go.Bar(
            x=node_ids,
            y=strengths,
            marker_color=colors,
            text=[f"{s:.3f}" for s in strengths],
            textposition="auto",
        )
    ])
    fig.update_layout(
        title=title,
        xaxis_title="Semantic Node",
        yaxis_title="Activation Strength",
        yaxis_range=[0, max(strengths, default=1) * 1.1],
        template="plotly_white",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    logger.info("Activation map saved to %s", output_path)
    return output_path


def render_activation_text(
    activated_nodes: list[tuple[str, float]],
    max_display: int = 20,
) -> str:
    """Return a plain-text activation report (for terminal / logging)."""
    if not activated_nodes:
        return "  (no nodes activated)"

    lines = []
    bar_width = 40
    max_val = max(s for _, s in activated_nodes) or 1.0

    for node_id, strength in activated_nodes[:max_display]:
        bar_len = int((strength / max_val) * bar_width)
        bar = "█" * bar_len + "░" * (bar_width - bar_len)
        lines.append(f"  {node_id:20s} │{bar}│ {strength:.4f}")

    return "\n".join(lines)
