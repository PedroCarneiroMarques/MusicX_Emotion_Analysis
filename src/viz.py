"""Plotly helpers shared by the Streamlit dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
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
    margin=dict(l=80, r=56, t=80, b=88),
    hoverlabel=dict(bgcolor="white", font_size=12),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.22,
        x=0,
        xanchor="left",
        bgcolor="rgba(0,0,0,0)",
        tracegroupgap=12,
        itemwidth=80,
    ),
    bargap=0.32,
    bargroupgap=0.18,
    boxgap=0.4,
    boxgroupgap=0.22,
    violinmode="group",
)

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["sendDataToCloud"],
}


def for_plotly(df: pd.DataFrame) -> pd.DataFrame:
    """Replace pandas NAType with JSON-safe nulls so Plotly/orjson can serialise."""
    out = df.copy()
    out = out.replace({pd.NA: np.nan})
    for col in out.columns:
        dtype = str(out[col].dtype)
        if dtype == "string" or dtype.startswith("str"):
            out[col] = out[col].astype(object)
        if out[col].dtype == object:
            out[col] = out[col].where(out[col].notna(), None)
    return out


def why(text: str) -> None:
    """One-line rationale under a chart."""
    st.caption(f"**Why this chart.** {text}")


def height_for_bars(n: int, row_px: int = 32, extra: int = 160) -> int:
    return int(min(920, max(380, n * row_px + extra)))


def style(fig: go.Figure, height: int = 480) -> go.Figure:
    fig.update_layout(**LAYOUT, height=height)
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(61,90,128,0.10)",
        zeroline=False,
        automargin=True,
        title_standoff=18,
        ticks="outside",
        ticklen=6,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(61,90,128,0.10)",
        zeroline=False,
        automargin=True,
        title_standoff=18,
        ticks="outside",
        ticklen=6,
    )
    return fig


def draw(fig: go.Figure, *, height: int = 480, key: str | None = None, **kwargs):
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
