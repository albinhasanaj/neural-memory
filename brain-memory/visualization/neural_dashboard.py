"""Real-time neural visualization dashboard.

Connects to the NeuralEventBus and displays live brain activity
as a Gradio web app with Plotly charts.

Launch standalone::

    python -m visualization.neural_dashboard

Or via ``scripts/start_brain.py`` which runs chat + dashboard together.
"""
from __future__ import annotations

import math
import time
from typing import Any

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from memory.neural_events import NeuralEvent, NeuralEventBus, event_bus as _default_bus

# ── Colour palette ──────────────────────────────────────────────────

EVENT_COLORS: dict[str, str] = {
    "encode": "#00d4ff",
    "gate_decision": "#ffdd00",
    "write": "#00ff88",
    "retrieve": "#aa44ff",
    "activate": "#ff8800",
    "inject": "#ff3355",
    "consolidate": "#66ffcc",
    "forget": "#ff6666",
}

BG = "#0a0a0a"
MODULE_INACTIVE = "#1a1a2e"
MODULE_WRITE = "#00ff88"
MODULE_RETRIEVE = "#ff6600"
GATE_STORE = "#00ff00"
GATE_SKIP = "#ff0000"
TEXT_COLOR = "#e0e0e0"
ACCENT = "#00d4ff"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=BG,
    plot_bgcolor="#111122",
    font_color=TEXT_COLOR,
    margin=dict(l=40, r=20, t=30, b=30),
)

NUM_MODULES = 32
GRID_ROWS, GRID_COLS = 4, 8

# ── Dashboard state ─────────────────────────────────────────────────

class DashboardState:
    """Mutable state shared across all update callbacks."""

    def __init__(self, bus: NeuralEventBus) -> None:
        self.bus = bus
        self.last_update: float = 0.0
        # Module heatmap: decaying activity per module
        self.module_heat = np.zeros(NUM_MODULES, dtype=np.float64)
        # Last gate decision
        self.gate_store: bool = False
        self.gate_prob: float = 0.0
        self.gate_novelty: float = 0.0
        self.gate_pred_err: float = 0.0
        self.gate_emphasis: float = 0.0
        self.gate_entity_density: float = 0.0
        # Last retrieval results
        self.retrieval_results: list[dict[str, Any]] = []
        # Stats counters
        self.total_writes: int = 0
        self.total_encodes: int = 0
        self.total_stores: int = 0
        self.total_skips: int = 0
        # Recent events for timeline
        self.timeline_events: list[NeuralEvent] = []
        # Active graph nodes
        self.active_nodes: list[tuple[str, float]] = []
        # Last write module indices
        self.last_write_modules: list[int] = []

    def process_new_events(self) -> None:
        """Pull new events from the bus and update state."""
        events = self.bus.events_since(self.last_update)
        if not events:
            # Decay even when no events
            self.module_heat *= 0.9
            return

        self.last_update = events[-1].timestamp

        # Keep last 200 events for timeline
        self.timeline_events.extend(events)
        self.timeline_events = self.timeline_events[-200:]

        for ev in events:
            if ev.event_type == "encode":
                self.total_encodes += 1

            elif ev.event_type == "gate_decision":
                self.gate_store = ev.data.get("stored", False)
                self.gate_prob = ev.data.get("salience", 0.0)
                self.gate_novelty = ev.data.get("novelty", 0.0)
                self.gate_pred_err = ev.data.get("prediction_error", 0.0)
                self.gate_emphasis = ev.data.get("emphasis", 0.0)
                self.gate_entity_density = ev.data.get("entity_density", 0.0)
                if self.gate_store:
                    self.total_stores += 1
                else:
                    self.total_skips += 1

            elif ev.event_type == "write":
                self.total_writes += 1
                # Flash modules if we know which ones were written to
                mods = ev.data.get("module_indices", [])
                self.last_write_modules = mods
                for m in mods:
                    if 0 <= m < NUM_MODULES:
                        self.module_heat[m] = 1.0

            elif ev.event_type == "retrieve":
                results = ev.data.get("results", [])
                self.retrieval_results = results
                for r in results:
                    m = r.get("source_module", -1)
                    if 0 <= m < NUM_MODULES:
                        self.module_heat[m] = max(self.module_heat[m], 0.7)

            elif ev.event_type == "activate":
                self.active_nodes = ev.data.get("top_nodes", [])

            elif ev.event_type == "inject":
                pass  # Covered by retrieve

        # Decay module heat
        self.module_heat *= 0.9


# ── Plotly figure builders ──────────────────────────────────────────

def build_timeline_figure(state: DashboardState) -> go.Figure:
    """Section 1: Brain activity timeline."""
    fig = go.Figure()

    if not state.timeline_events:
        fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Brain Activity Timeline",
            xaxis_title="Time",
            yaxis_title="Event",
            height=200,
        )
        return fig

    now = time.time()
    window = 60  # last 60 seconds
    recent = [e for e in state.timeline_events if e.timestamp > now - window]

    if not recent:
        recent = state.timeline_events[-20:]

    event_types = list(EVENT_COLORS.keys())
    type_to_y = {t: i for i, t in enumerate(event_types)}

    x_vals, y_vals, colors, texts, sizes = [], [], [], [], []
    for ev in recent:
        if ev.event_type not in type_to_y:
            continue
        x_vals.append(ev.timestamp - recent[0].timestamp)
        y_vals.append(type_to_y[ev.event_type])
        colors.append(EVENT_COLORS.get(ev.event_type, "#ffffff"))
        # Build hover text
        summary = ev.event_type
        if ev.event_type == "encode":
            summary = f"encode: {ev.data.get('text', '')[:40]}"
        elif ev.event_type == "gate_decision":
            summary = f"gate: {'STORE' if ev.data.get('stored') else 'SKIP'} (p={ev.data.get('salience', 0):.2f})"
        elif ev.event_type == "write":
            summary = f"write: {ev.data.get('text', '')[:40]}"
        elif ev.event_type == "retrieve":
            summary = f"retrieve: {ev.data.get('num_results', 0)} results"
        elif ev.event_type == "inject":
            summary = f"inject: {ev.data.get('num_memories', 0)} memories"
        texts.append(summary)
        sizes.append(12)

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals,
        mode="markers",
        marker=dict(size=sizes, color=colors, opacity=0.85),
        text=texts,
        hoverinfo="text",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Brain Activity Timeline",
        height=200,
        showlegend=False,
        yaxis=dict(
            tickvals=list(range(len(event_types))),
            ticktext=event_types,
            gridcolor="#222233",
        ),
        xaxis=dict(title="seconds", gridcolor="#222233"),
    )
    return fig


def build_module_heatmap(state: DashboardState) -> go.Figure:
    """Section 2: Module activation heatmap (8x4 grid)."""
    heat = state.module_heat.reshape(GRID_ROWS, GRID_COLS)

    # Build hover text with module index
    hover = [[f"Module {r * GRID_COLS + c}<br>heat={heat[r][c]:.2f}"
              for c in range(GRID_COLS)] for r in range(GRID_ROWS)]

    # Custom colorscale: dark navy → green (write flash)
    colorscale = [
        [0.0, MODULE_INACTIVE],
        [0.3, "#1a3a3e"],
        [0.5, "#2a6a4e"],
        [0.7, "#44aa66"],
        [1.0, MODULE_WRITE],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=heat,
        colorscale=colorscale,
        zmin=0.0, zmax=1.0,
        text=hover,
        hoverinfo="text",
        showscale=False,
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Module Activation (32 modules)",
        height=250,
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed"),
    )
    return fig


def build_gate_figure(state: DashboardState) -> go.Figure:
    """Section 4: Gate decision gauge + signal bars."""
    fig = go.Figure()

    # Gauge for gate probability
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=state.gate_prob,
        title={"text": "Gate Store Probability", "font": {"color": TEXT_COLOR}},
        number={"font": {"color": GATE_STORE if state.gate_store else GATE_SKIP}},
        gauge=dict(
            axis=dict(range=[0, 1], tickcolor=TEXT_COLOR),
            bar=dict(color=GATE_STORE if state.gate_store else GATE_SKIP),
            bgcolor="#1a1a2e",
            bordercolor="#333",
            steps=[
                dict(range=[0, 0.3], color="#1a0a0a"),
                dict(range=[0.3, 0.7], color="#1a1a0a"),
                dict(range=[0.7, 1], color="#0a1a0a"),
            ],
        ),
        domain=dict(x=[0, 1], y=[0.45, 1.0]),
    ))

    # Bar chart for salience signals
    signals = ["novelty", "pred_error", "emphasis", "entity_density"]
    values = [state.gate_novelty, state.gate_pred_err, state.gate_emphasis, state.gate_entity_density]
    bar_colors = [ACCENT, "#ffdd00", "#ff8800", "#aa44ff"]

    fig.add_trace(go.Bar(
        x=signals,
        y=values,
        marker_color=bar_colors,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        textfont=dict(color=TEXT_COLOR, size=10),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Gate Decision",
        height=350,
        showlegend=False,
        yaxis=dict(range=[0, 1.2], gridcolor="#222233", domain=[0, 0.35]),
        xaxis=dict(domain=[0, 1], gridcolor="#222233"),
    )
    return fig


def build_retrieval_figure(state: DashboardState) -> go.Figure:
    """Section 5: Retrieval results horizontal bar chart."""
    fig = go.Figure()

    results = state.retrieval_results[:5]
    if not results:
        fig.update_layout(**PLOTLY_LAYOUT, title="Memory Retrieval", height=220)
        return fig

    texts = [r.get("text", "?")[:50] for r in results]
    scores = [r.get("score", 0) for r in results]
    modules = [r.get("source_module", -1) for r in results]
    bar_colors = []
    for m in modules:
        if m >= 0:
            # Colour based on module index
            hue = (m / NUM_MODULES) * 0.7
            r_c = int(128 + 127 * math.sin(hue * 6.28))
            g_c = int(128 + 127 * math.sin(hue * 6.28 + 2.09))
            b_c = int(128 + 127 * math.sin(hue * 6.28 + 4.19))
            bar_colors.append(f"rgb({r_c},{g_c},{b_c})")
        else:
            bar_colors.append(ACCENT)

    fig.add_trace(go.Bar(
        y=list(range(len(texts))),
        x=scores,
        orientation="h",
        text=[f"[M{m}] {t}" for t, m in zip(texts, modules)],
        textposition="inside",
        textfont=dict(color=TEXT_COLOR, size=10),
        marker_color=bar_colors,
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Retrieved Memories",
        height=220,
        showlegend=False,
        yaxis=dict(showticklabels=False, autorange="reversed"),
        xaxis=dict(title="score", gridcolor="#222233"),
    )
    return fig


def build_graph_figure(state: DashboardState, observer: Any | None = None) -> go.Figure:
    """Section 6: Semantic graph network visualization."""
    fig = go.Figure()

    if observer is None:
        fig.update_layout(**PLOTLY_LAYOUT, title="Semantic Graph", height=280)
        return fig

    graph = observer.graph
    nodes = list(graph._nodes.values()) if hasattr(graph, "_nodes") else []

    if len(nodes) < 2:
        fig.update_layout(**PLOTLY_LAYOUT, title="Semantic Graph", height=280)
        return fig

    # Simple circular layout
    n = len(nodes)
    positions = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        positions[node.id] = (math.cos(angle), math.sin(angle))

    # Build active-node lookup
    active_lookup = {nid: score for nid, score in state.active_nodes}

    # Draw edges first
    edge_x, edge_y = [], []
    if hasattr(graph, "_edges"):
        for edge in graph._edges:
            src = edge.get("source") or edge.get("src")
            tgt = edge.get("target") or edge.get("tgt")
            if src in positions and tgt in positions:
                x0, y0 = positions[src]
                x1, y1 = positions[tgt]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
    elif hasattr(graph, "get_edges"):
        try:
            for src, tgt, _ in graph.get_edges():
                if src in positions and tgt in positions:
                    x0, y0 = positions[src]
                    x1, y1 = positions[tgt]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
        except Exception:
            pass

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#333344", width=0.5),
        hoverinfo="none",
    ))

    # Draw nodes
    node_x = [positions[n.id][0] for n in nodes]
    node_y = [positions[n.id][1] for n in nodes]
    node_labels = [n.label for n in nodes]
    node_sizes = []
    node_colors = []
    for n_obj in nodes:
        score = active_lookup.get(n_obj.id, 0.0)
        node_sizes.append(8 + score * 20)
        if score > 0:
            node_colors.append(MODULE_RETRIEVE)
        else:
            node_colors.append("#445566")

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=node_sizes, color=node_colors, opacity=0.85),
        text=node_labels,
        textposition="top center",
        textfont=dict(size=8, color=TEXT_COLOR),
        hoverinfo="text",
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=f"Semantic Graph ({len(nodes)} nodes)",
        height=280,
        showlegend=False,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    return fig


def build_stats_text(state: DashboardState, observer: Any | None = None) -> str:
    """Section 7: Brain stats panel (markdown text)."""
    store_rate = 0.0
    total_decisions = state.total_stores + state.total_skips
    if total_decisions > 0:
        store_rate = state.total_stores / total_decisions * 100

    active_mod = int(np.sum(state.module_heat > 0.01))
    avg_heat = float(np.mean(state.module_heat))

    lines = [
        f"**Total Encodes:** {state.total_encodes}",
        f"**Total Writes:** {state.total_writes}",
        f"**Gate Store Rate:** {store_rate:.1f}% ({state.total_stores}/{total_decisions})",
        f"**Active Modules:** {active_mod} / {NUM_MODULES}",
        f"**Avg Module Heat:** {avg_heat:.3f}",
    ]

    if observer is not None:
        lines.append(f"**Episodic Entries:** {observer.episodic_store.active_count}")
        lines.append(f"**Graph Nodes:** {observer.graph.num_nodes}")
        if hasattr(observer, "_hopfield") and observer._hopfield is not None:
            if hasattr(observer._hopfield, "total_writes"):
                lines.append(f"**Hopfield Writes:** {observer._hopfield.total_writes()}")

    return "\n\n".join(lines)


# ── Dashboard factory ───────────────────────────────────────────────

def create_dashboard(
    observer: Any | None = None,
    bus: NeuralEventBus | None = None,
) -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    if bus is None:
        bus = _default_bus

    state = DashboardState(bus)

    custom_css = f"""
    .gradio-container {{
        background-color: {BG} !important;
    }}
    .dark {{
        background-color: {BG} !important;
    }}
    """

    _theme = gr.themes.Base(
        primary_hue="cyan",
        neutral_hue="slate",
    )

    with gr.Blocks(
        title="Brain Memory — Neural Dashboard",
    ) as app:
        # Store for launch() — Gradio 6.0+ wants these there
        app._brain_theme = _theme
        app._brain_css = custom_css

        gr.Markdown(
            f"# <span style='color:{ACCENT}'>Brain Memory</span> — Neural Activity Dashboard",
        )

        # Section 1: Timeline (full width)
        timeline_plot = gr.Plot(label="Activity Timeline")

        with gr.Row():
            # Section 2: Module heatmap (left)
            with gr.Column(scale=1):
                module_plot = gr.Plot(label="Module Activation")
                # Section 5: Retrieval chart (below modules)
                retrieval_plot = gr.Plot(label="Memory Retrieval")

            # Section 6: Semantic graph (center)
            with gr.Column(scale=1):
                graph_plot = gr.Plot(label="Semantic Graph")
                # Section 7: Stats
                stats_md = gr.Markdown("*Waiting for brain activity...*")

            # Section 4: Gate (right)
            with gr.Column(scale=1):
                gate_plot = gr.Plot(label="Gate Decision")

        # ── Auto-refresh via Timer ──────────────────────────────────
        timer = gr.Timer(0.5)

        def refresh_all() -> tuple:
            state.process_new_events()
            return (
                build_timeline_figure(state),
                build_module_heatmap(state),
                build_gate_figure(state),
                build_retrieval_figure(state),
                build_graph_figure(state, observer),
                build_stats_text(state, observer),
            )

        timer.tick(
            fn=refresh_all,
            outputs=[timeline_plot, module_plot, gate_plot, retrieval_plot, graph_plot, stats_md],
        )

    return app


# ── Standalone entry point ──────────────────────────────────────────

def main() -> None:
    """Run dashboard standalone (for testing without chat)."""
    app = create_dashboard()
    app.launch(
        server_name="127.0.0.1", server_port=7860, share=False,
        theme=getattr(app, "_brain_theme", None),
        css=getattr(app, "_brain_css", None),
    )


if __name__ == "__main__":
    main()
