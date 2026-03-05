"""
Memory timeline — plot episodic memories on a timeline with salience coloring.

Generates an interactive timeline (plotly) or a static matplotlib image.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from memory.episodic import EpisodicEntry

logger = logging.getLogger(__name__)


def render_timeline_html(
    episodes: list[EpisodicEntry],
    output_path: str | Path = "data/memory_timeline.html",
    title: str = "Episodic Memory Timeline",
) -> Path:
    """Render episodes as a scatter plot on a timeline (x = time, y = salience).

    Parameters
    ----------
    episodes:
        Episodes to plot.
    output_path:
        Where to save the HTML.

    Returns
    -------
    Path to HTML file.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.error("plotly not installed.")
        raise

    if not episodes:
        logger.warning("No episodes to plot.")
        episodes_sorted: list[EpisodicEntry] = []
    else:
        episodes_sorted = sorted(episodes, key=lambda e: e.timestamp)

    times = [e.timestamp for e in episodes_sorted]
    saliences = [e.salience for e in episodes_sorted]
    activations = [e.activation for e in episodes_sorted]
    texts = [
        f"<b>{e.speaker}</b>: {e.raw_text[:80]}...<br>"
        f"salience: {e.salience:.3f}<br>"
        f"activation: {e.activation:.3f}<br>"
        f"entities: {', '.join(e.entities)}<br>"
        f"consolidated: {e.consolidated}"
        for e in episodes_sorted
    ]

    # Color by salience (green → red)
    colors = [
        f"rgba({int(255 * s)}, {int(255 * (1 - s))}, 80, 0.8)"
        for s in saliences
    ]

    fig = go.Figure(data=[
        go.Scatter(
            x=times,
            y=saliences,
            mode="markers+lines",
            marker=dict(size=12, color=colors, line=dict(width=1, color="gray")),
            text=texts,
            hoverinfo="text",
            line=dict(color="rgba(150,150,150,0.3)"),
        )
    ])
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Salience",
        yaxis_range=[0, 1.05],
        template="plotly_white",
        hovermode="closest",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    logger.info("Timeline saved to %s", output_path)
    return output_path


def render_timeline_text(episodes: list[EpisodicEntry], max_display: int = 20) -> str:
    """Plain-text timeline for terminal display."""
    if not episodes:
        return "  (no episodes)"

    sorted_eps = sorted(episodes, key=lambda e: e.timestamp, reverse=True)
    lines: list[str] = []
    for ep in sorted_eps[:max_display]:
        ts = ep.timestamp.strftime("%Y-%m-%d %H:%M")
        marker = "●" if ep.salience >= 0.5 else "○"
        archived = " [archived]" if ep.archived else ""
        consolidated = " [consolidated]" if ep.consolidated else ""
        lines.append(
            f"  {marker} [{ts}] (s={ep.salience:.2f} a={ep.activation:.2f}) "
            f"{ep.speaker}: {ep.raw_text[:60]}{archived}{consolidated}"
        )
    return "\n".join(lines)
