"""Plotly helpers shared by the Streamlit dashboard."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

PRIMARY = "#3D5A80"
ACCENT = "#EE6C4D"
SOFT = "#98C1D9"
INK = "#1B2838"

EFFECT_COLORS = {
    "Improve": PRIMARY,
    "No effect": SOFT,
    "Worsen": ACCENT,
}

LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=INK, size=13),
    margin=dict(l=16, r=16, t=56, b=16),
    hoverlabel=dict(bgcolor="white", font_size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["sendDataToCloud"],
}


def style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(**LAYOUT, height=height)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(61,90,128,0.12)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(61,90,128,0.12)", zeroline=False)
    return fig


def draw(fig: go.Figure, *, height: int = 420, key: str | None = None, **kwargs):
    """Render a Plotly figure at full width."""
    style(fig, height=height)
    return st.plotly_chart(
        fig,
        use_container_width=True,
        config=PLOTLY_CONFIG,
        key=key,
        **kwargs,
    )


def empty_state(message: str = "No respondents match the current filters.") -> None:
    st.warning(message)
