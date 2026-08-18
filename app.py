"""MusicX — interactive Streamlit dashboard for the MxMH survey.

Run from the repo root:

    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from src.data import MENTAL_HEALTH_COLS, frequency_columns, genre_label, load_clean
from src.model import fit_improve_model
from src.viz import (
    ACCENT,
    EFFECT_COLORS,
    PRIMARY,
    draw,
    empty_state,
    height_for_bars,
    why,
    for_plotly,
)

BANNER = Path(__file__).resolve().parent / "assets" / "banner.png"

st.set_page_config(
    page_title="MusicX · Music & Mental Health",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

NUMERIC_EXPLORE = [
    "Age",
    "Hours per day",
    "BPM",
    *MENTAL_HEALTH_COLS,
]
CATEGORICAL_EXPLORE = [
    "Fav genre",
    "Primary streaming service",
    "Music effects",
    "While working",
    "Exploratory",
    "Instrumentalist",
    "Composer",
]


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.1rem; padding-bottom: 2.4rem; max-width: 1400px; }
        h1, h2, h3 { letter-spacing: -0.03em; }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d5e1ec;
            border-radius: 14px;
            padding: 12px 14px;
            box-shadow: 0 1px 0 rgba(61, 90, 128, 0.06);
        }
        [data-testid="stSidebar"] { background: #eef4f8; }
        div[data-testid="stTabs"] button { font-weight: 600; }
        .stPlotlyChart { margin: 0.4rem 0 2.4rem 0; }
        [data-testid="stHorizontalBlock"] { gap: 2rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def get_data() -> pd.DataFrame:
    df = load_clean()
    df["Primary streaming service"] = df["Primary streaming service"].fillna(
        "Not reported"
    )
    return for_plotly(df)


@st.cache_data
def cached_improve_model(df: pd.DataFrame) -> dict:
    return fit_improve_model(df)


def _init_filter_state(df: pd.DataFrame) -> None:
    age = df["Age"].dropna()
    hours = df["Hours per day"].dropna()
    defaults = {
        "flt_age": (int(age.min()), int(age.max())),
        "flt_hours": (0.0, float(hours.max())),
        "flt_services": sorted(df["Primary streaming service"].dropna().unique().tolist()),
        "flt_genres": sorted(df["Fav genre"].dropna().unique().tolist()),
        "flt_effects": ["Improve", "No effect", "Worsen"],
        "flt_working": ["Yes", "No"],
        "flt_explore": ["Yes", "No"],
        "clicked_genre": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    _init_filter_state(df)
    st.sidebar.markdown("### Filters")
    st.sidebar.caption("Every chart below follows these cuts.")

    if st.sidebar.button("Reset filters", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("flt_") or key == "clicked_genre":
                del st.session_state[key]
        st.rerun()

    age = df["Age"].dropna()
    st.sidebar.slider(
        "Age",
        min_value=int(age.min()),
        max_value=int(age.max()),
        key="flt_age",
    )
    st.sidebar.slider(
        "Hours of music / day",
        min_value=0.0,
        max_value=float(df["Hours per day"].max()),
        step=0.5,
        key="flt_hours",
    )
    st.sidebar.multiselect(
        "Streaming service",
        options=sorted(df["Primary streaming service"].dropna().unique().tolist()),
        key="flt_services",
    )
    st.sidebar.multiselect(
        "Favorite genre",
        options=sorted(df["Fav genre"].dropna().unique().tolist()),
        key="flt_genres",
    )
    if st.session_state.get("clicked_genre"):
        st.sidebar.info(f"Chart click: **{st.session_state.clicked_genre}**")
        if st.sidebar.button("Clear chart click"):
            st.session_state.clicked_genre = None
            st.rerun()

    st.sidebar.multiselect(
        "Music effects",
        options=["Improve", "No effect", "Worsen"],
        key="flt_effects",
    )
    with st.sidebar.expander("More cuts"):
        st.multiselect("Listen while working", ["Yes", "No"], key="flt_working")
        st.multiselect("Explores new music", ["Yes", "No"], key="flt_explore")

    out = df.copy()
    age_lo, age_hi = st.session_state.flt_age
    hours_lo, hours_hi = st.session_state.flt_hours
    out = out[out["Age"].between(age_lo, age_hi) | out["Age"].isna()]
    out = out[out["Hours per day"].between(hours_lo, hours_hi)]
    out = out[out["Primary streaming service"].isin(st.session_state.flt_services)]
    genres = list(st.session_state.flt_genres)
    if st.session_state.get("clicked_genre"):
        genres = [st.session_state.clicked_genre]
    out = out[out["Fav genre"].isin(genres)]
    effect_ok = out["Music effects"].isin(st.session_state.flt_effects) | out[
        "Music effects"
    ].isna()
    out = out[effect_ok]
    working_ok = out["While working"].isin(st.session_state.flt_working) | out[
        "While working"
    ].isna()
    out = out[working_ok]
    explore_ok = out["Exploratory"].isin(st.session_state.flt_explore)
    out = out[explore_ok]

    st.sidebar.markdown(
        f"**{len(out):,}** / {len(df):,} respondents "
        f"({len(out) / max(len(df), 1):.0%})"
    )
    return for_plotly(out)


def kpi_row(filtered: pd.DataFrame, full: pd.DataFrame) -> None:
    effects = filtered["Music effects"].dropna()
    improve = (effects == "Improve").mean() if len(effects) else float("nan")
    full_improve = (full["Music effects"].dropna() == "Improve").mean()
    c1, c2, c3, c4 = st.columns(4, gap="large")
    c1.metric("Respondents", f"{len(filtered):,}", delta=len(filtered) - len(full))
    c2.metric(
        "Median age",
        f"{filtered['Age'].median():.0f}" if filtered["Age"].notna().any() else "—",
    )
    c3.metric(
        "Median hours / day",
        f"{filtered['Hours per day'].median():.1f}",
    )
    delta = None
    if pd.notna(improve):
        delta = f"{improve - full_improve:+.0%} vs all"
        c4.metric("Say music improves MH", f"{improve:.0%}", delta=delta)
    else:
        c4.metric("Say music improves MH", "—")


def page_overview(filtered: pd.DataFrame) -> None:
    st.subheader("Snapshot")
    st.markdown(
        "Start with the two questions the survey was built to answer: "
        "do people *say* music helps, and how do the four symptom scores sit "
        "on the 0–10 scale."
    )
    if filtered.empty:
        empty_state()
        return

    left, right = st.columns(2, gap="large")
    with left:
        effects = filtered["Music effects"].dropna()
        if effects.empty:
            empty_state("No music-effect answers in this cut.")
        else:
            pie = effects.value_counts().rename_axis("Music effects").reset_index(name="count")
            fig = px.pie(
                pie,
                names="Music effects",
                values="count",
                title="Self-reported effect of music",
                color="Music effects",
                color_discrete_map=EFFECT_COLORS,
                hole=0.42,
            )
            fig.update_traces(
                textinfo="percent+label",
                textposition="outside",
                pull=[0.02, 0.02, 0.08],
                hovertemplate="%{label}: %{value}<br>%{percent}<extra></extra>",
            )
            draw(fig, height=500, key="overview_pie")
            why(
                "A donut is the right mark for one categorical with three levels — "
                "parts of a whole. Labels sit outside the slices so they do not collide."
            )
    with right:
        mh = filtered[MENTAL_HEALTH_COLS].melt(var_name="Symptom", value_name="Score")
        fig = px.box(
            mh,
            x="Symptom",
            y="Score",
            color="Symptom",
            points="outliers",
            title="Symptom scores (0 = none, 10 = extreme)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False)
        draw(fig, height=500, key="overview_box")
        why(
            "The four scores share a 0–10 scale. A box plot compares medians and "
            "spread without assuming a bell curve. Only outliers are drawn as "
            "points so 700 dots do not pile on top of each other."
        )


def page_listeners(filtered: pd.DataFrame) -> None:
    st.subheader("Who is in this cut")
    st.markdown(
        "This survey is a convenience sample. Age, hours, and platform tell you "
        "**who answered**, which is the limit of every chart that follows."
    )
    if filtered.empty:
        empty_state()
        return

    c1, c2 = st.columns(2, gap="large")
    with c1:
        bins = st.slider("Age bins", 8, 40, 20, key="age_bins")
        fig = px.histogram(filtered, x="Age", nbins=bins, title="Age")
        fig.update_traces(marker_color=PRIMARY, marker_line_width=0)
        draw(fig, height=440, key="hist_age")
        why(
            "Age is continuous and skewed young. A histogram shows the pile-up "
            "around 18–21 that a single median would hide."
        )
    with c2:
        bins_h = st.slider("Hours bins", 5, 30, 14, key="hours_bins")
        fig = px.histogram(
            filtered, x="Hours per day", nbins=bins_h, title="Hours of music per day"
        )
        fig.update_traces(marker_color=ACCENT, marker_line_width=0)
        draw(fig, height=440, key="hist_hours")
        why(
            "Hours have a long right tail (some people report 12–24 h). The "
            "histogram makes that tail visible instead of letting it inflate a mean."
        )

    services = (
        filtered["Primary streaming service"]
        .value_counts()
        .rename_axis("Service")
        .reset_index(name="Respondents")
    )
    fig = px.bar(
        services,
        x="Respondents",
        y="Service",
        orientation="h",
        title="Primary streaming service",
        color="Respondents",
        color_continuous_scale="Blues",
        text="Respondents",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        coloraxis_showscale=False,
    )
    draw(fig, height=height_for_bars(len(services), extra=180), key="bar_service")
    why(
        "Service names are long. A horizontal bar keeps every label horizontal "
        "and readable — a vertical bar would rotate them into each other."
    )

    genre_counts = (
        filtered["Fav genre"]
        .value_counts()
        .rename_axis("Favorite genre")
        .reset_index(name="Respondents")
    )
    fig = px.bar(
        genre_counts,
        x="Respondents",
        y="Favorite genre",
        orientation="h",
        title="Favorite genre — click a bar to lock that genre",
        color_discrete_sequence=[PRIMARY],
        text="Respondents",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    event = draw(
        fig,
        height=height_for_bars(len(genre_counts)),
        key="bar_genre",
        on_select="rerun",
        selection_mode="points",
    )
    _lock_genre_from_event(event)
    why(
        "Sixteen genres, uneven counts. Ranking by bar length is faster to read "
        "than a pie, and the click-to-filter turns this chart into a control."
    )

    fig = px.violin(
        filtered.dropna(subset=["Age", "Fav genre"]),
        x="Fav genre",
        y="Age",
        color="Fav genre",
        box=True,
        points="outliers",
        title="Age by favorite genre",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-40)
    n_g = filtered["Fav genre"].nunique()
    draw(fig, height=height_for_bars(n_g, row_px=18, extra=220), key="violin_age_genre")
    why(
        "A violin shows the full age shape per genre, not just a mean. Classical "
        "and jazz skew older; a bar of averages would flatten that."
    )


def _point_get(point, *keys):
    for key in keys:
        if isinstance(point, dict) and key in point:
            value = point[key]
        else:
            value = getattr(point, key, None)
        if value is not None:
            return value[0] if isinstance(value, list) else value
    return None


def _lock_genre_from_event(event) -> None:
    try:
        points = event.selection.points
    except Exception:
        return
    if not points:
        return
    genre = _point_get(points[0], "y", "legendgroup", "label")
    if genre and st.session_state.get("clicked_genre") != str(genre):
        st.session_state.clicked_genre = str(genre)
        st.rerun()


def page_builder(filtered: pd.DataFrame) -> None:
    st.subheader("Chart builder")
    st.markdown(
        "Pick a mark that matches the question: scatter for two numbers, "
        "histogram for a shape, box/violin for a number by a category, "
        "bar for counts, density heatmap when points would overplot."
    )
    if filtered.empty:
        empty_state()
        return

    numeric = NUMERIC_EXPLORE + frequency_columns(filtered)
    cats = CATEGORICAL_EXPLORE
    c1, c2, c3, c4 = st.columns(4, gap="large")
    mark = c1.selectbox(
        "Chart type",
        ["Scatter", "Histogram", "Box", "Violin", "Bar", "Density heatmap"],
    )
    x_col = c2.selectbox("X", numeric + cats, index=0)
    y_default = numeric.index("Anxiety") if "Anxiety" in numeric else 0
    y_col = c3.selectbox("Y", numeric, index=y_default)
    color_opt = c4.selectbox("Color", ["(none)"] + cats, index=1)
    color = None if color_opt == "(none)" else color_opt

    extra1, extra2, extra3 = st.columns(3, gap="large")
    opacity = extra1.slider("Opacity", 0.2, 1.0, 0.7)
    trend = extra2.toggle("Trendline (OLS)", value=False)
    bins = extra3.slider("Histogram bins", 8, 40, 20)

    work = for_plotly(filtered)
    hover = ["Fav genre", "Music effects", "Age", "Hours per day"]
    hover = [c for c in hover if c in work.columns]

    fig = None
    try:
        if mark == "Scatter":
            fig = px.scatter(
                work,
                x=x_col,
                y=y_col,
                color=color,
                opacity=opacity,
                hover_data=hover,
                trendline="ols"
                if trend and pd.api.types.is_numeric_dtype(work[x_col])
                else None,
                title=f"{y_col} vs {x_col}",
                color_discrete_map=EFFECT_COLORS if color == "Music effects" else None,
            )
            fig.update_traces(marker=dict(size=8))
        elif mark == "Histogram":
            fig = px.histogram(
                work,
                x=x_col,
                color=color,
                nbins=bins,
                barmode="overlay",
                opacity=opacity,
                title=x_col,
                color_discrete_map=EFFECT_COLORS if color == "Music effects" else None,
            )
        elif mark == "Box":
            fig = px.box(
                work,
                x=x_col,
                y=y_col,
                color=color or x_col,
                points="outliers",
                title=f"{y_col} by {x_col}",
            )
        elif mark == "Violin":
            fig = px.violin(
                work,
                x=x_col,
                y=y_col,
                color=color or x_col,
                box=True,
                points="outliers",
                title=f"{y_col} by {x_col}",
            )
        elif mark == "Bar":
            if x_col not in cats:
                st.info("Bar charts work best with a categorical X — using counts of X.")
            counts = (
                work[x_col]
                .value_counts(dropna=False)
                .rename_axis(x_col)
                .reset_index(name="Respondents")
            )
            fig = px.bar(
                counts,
                x="Respondents",
                y=x_col,
                orientation="h",
                title=f"Counts of {x_col}",
                color_discrete_sequence=[PRIMARY],
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
        else:
            fig = px.density_heatmap(
                work,
                x=x_col,
                y=y_col,
                color_continuous_scale="Blues",
                title=f"Density of {y_col} vs {x_col}",
            )
    except Exception as exc:
        st.error(f"Could not draw that combination: {exc}")
        return

    if fig is not None:
        draw(fig, height=560, key="builder_chart")
        why(
            "The mark should follow the data type you picked. Scatter overplots "
            "on 700 rows — use the density heatmap when the cloud turns into a blot."
        )


def page_associations(filtered: pd.DataFrame) -> None:
    st.subheader("Genres and symptom scores")
    st.markdown(
        "Frequencies are encoded Never=0 … Very frequently=3. "
        "Anything below **|r| ≈ 0.20** is a weak link. None of this is causal."
    )
    if filtered.empty or len(filtered) < 8:
        empty_state("Need at least a handful of rows for a correlation matrix.")
        return

    freq = frequency_columns(filtered)
    method = st.radio("Correlation", ["pearson", "spearman"], horizontal=True)
    corr = (
        filtered[freq + MENTAL_HEALTH_COLS]
        .corr(method=method)
        .loc[freq, MENTAL_HEALTH_COLS]
    )
    corr.index = [genre_label(c) for c in corr.index]
    fig = px.imshow(
        corr,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        zmin=-0.3,
        zmax=0.3,
        title=f"{method.title()} r — listening frequency vs. symptom score",
        labels={"color": "r"},
        text_auto=".2f",
    )
    fig.update_traces(textfont_size=11)
    event = draw(fig, height=680, key="heatmap", on_select="rerun", selection_mode="points")
    why(
        "A 16×4 table of correlations is unreadable as numbers. A diverging "
        "heatmap puts the weak field in one glance — and the click opens a "
        "box plot for the cell you care about."
    )
    _show_heatmap_drilldown(filtered, freq, event)

    st.markdown("##### Improve rate by favorite genre")
    min_n = st.slider("Hide genres with fewer than N people", 5, 50, 20, key="min_n")
    work = filtered.dropna(subset=["Music effects", "Fav genre"]).copy()
    work["improves"] = work["Music effects"].eq("Improve")
    summary = (
        work.groupby("Fav genre")
        .agg(n=("improves", "size"), improve_rate=("improves", "mean"))
        .query("n >= @min_n")
        .sort_values("improve_rate")
        .reset_index()
    )
    if summary.empty:
        empty_state("No genre clears that sample-size bar.")
    else:
        fig = px.bar(
            summary,
            x="improve_rate",
            y="Fav genre",
            orientation="h",
            text="n",
            title=f"Share who say music improves their mental health (n ≥ {min_n})",
            color="improve_rate",
            color_continuous_scale="Blues",
            hover_data={"n": True, "improve_rate": ":.0%"},
        )
        fig.update_traces(texttemplate="n=%{text}", textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_tickformat=".0%", coloraxis_showscale=False)
        event_bar = draw(
            fig,
            height=height_for_bars(len(summary)),
            key="improve_bar",
            on_select="rerun",
            selection_mode="points",
        )
        _lock_genre_from_event(event_bar)
        why(
            "The question is a ranking, not a shape. A sorted horizontal bar "
            "with n on the label stops a 6-person genre from looking like a 100% win."
        )

    means = (
        filtered.dropna(subset=["Music effects"])
        .groupby("Music effects")[MENTAL_HEALTH_COLS]
        .mean()
        .reindex(["Improve", "No effect", "Worsen"])
        .dropna(how="all")
        .reset_index()
        .melt(id_vars="Music effects", var_name="Symptom", value_name="Mean score")
    )
    if not means.empty:
        fig = px.bar(
            means,
            x="Symptom",
            y="Mean score",
            color="Music effects",
            barmode="group",
            color_discrete_map=EFFECT_COLORS,
            title="Mean symptom score by reported music effect",
        )
        draw(fig, height=480, key="effect_means")
        why(
            "Four symptoms × three effect groups: a grouped bar puts the same "
            "symptom side by side. The “Worsen” bar is a flag (n = 17 in the "
            "full sample), not a population rate."
        )


def _show_heatmap_drilldown(filtered: pd.DataFrame, freq: list[str], event) -> None:
    try:
        points = event.selection.points
    except Exception:
        return
    if not points:
        return
    genre = _point_get(points[0], "y")
    symptom = _point_get(points[0], "x")
    if not genre or not symptom:
        return
    col = next((c for c in freq if genre_label(c) == genre), None)
    if col is None or symptom not in MENTAL_HEALTH_COLS:
        return
    st.markdown(f"##### Drill-down: **{genre}** frequency vs **{symptom}**")
    fig = px.box(
        filtered,
        x=col,
        y=symptom,
        points="outliers",
        title=f"{symptom} by how often people listen to {genre}",
        labels={col: f"{genre} (0 = Never … 3 = Very frequently)"},
        color_discrete_sequence=[PRIMARY],
    )
    draw(fig, height=440, key="heatmap_drill")
    why(
        "The heatmap is a summary. This box plot is the raw distribution behind "
        "one cell — four ordinal listening levels, same 0–10 score."
    )


def page_compare(filtered: pd.DataFrame) -> None:
    st.subheader("Compare two groups")
    st.markdown(
        "A mean can lie when two groups have the same average and different shapes. "
        "Overlay the full score distribution, then read the table for the rest."
    )
    if filtered.empty:
        empty_state()
        return

    dim = st.selectbox(
        "Split by",
        ["Fav genre", "Primary streaming service", "Music effects", "Instrumentalist"],
    )
    values = sorted(filtered[dim].dropna().unique().tolist())
    if len(values) < 2:
        empty_state("Need two groups in this cut.")
        return
    left, right = st.columns(2, gap="large")
    a = left.selectbox("Group A", values, index=0, key="cmp_a")
    b_default = 1 if len(values) > 1 else 0
    b = right.selectbox("Group B", values, index=b_default, key="cmp_b")
    if a == b:
        st.info("Pick two different groups.")
        return

    fa = filtered[filtered[dim] == a]
    fb = filtered[filtered[dim] == b]
    k1, k2, k3, k4 = st.columns(4, gap="large")
    k1.metric(f"{a} n", f"{len(fa):,}")
    k2.metric(f"{b} n", f"{len(fb):,}")
    k3.metric(f"{a} mean anxiety", f"{fa['Anxiety'].mean():.1f}")
    k4.metric(f"{b} mean anxiety", f"{fb['Anxiety'].mean():.1f}")

    both = pd.concat(
        [fa.assign(group=a), fb.assign(group=b)],
        ignore_index=True,
    )
    symptom = st.selectbox("Symptom to overlay", MENTAL_HEALTH_COLS, key="cmp_symptom")
    fig = px.histogram(
        both,
        x=symptom,
        color="group",
        barmode="overlay",
        opacity=0.65,
        nbins=11,
        title=f"{symptom} — {a} vs {b}",
        color_discrete_sequence=[PRIMARY, ACCENT],
    )
    draw(fig, height=480, key="cmp_hist")
    why(
        "An overlay histogram (not two separate charts) keeps the 0–10 axis "
        "shared, so a shift in the mass is visible. 11 bins match the integer scores."
    )

    summary = (
        both.groupby("group")[MENTAL_HEALTH_COLS + ["Hours per day", "Age"]]
        .mean()
        .round(2)
        .T.rename_axis("Metric")
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)


def page_model(full: pd.DataFrame) -> None:
    st.subheader("Model — does music help?")
    st.markdown(
        """
        The survey's own music × mental-health question is **Music effects**
        (Improve / No effect / Worsen). Worsen has 17 rows, so a three-class
        model would be theatre. The target is binary: **Improve vs not**.

        **Model: L2 logistic regression** on 16 genre frequencies + hours + age,
        after standardising the features. n is small, genres are correlated, and
        we need *signed* coefficients. A random forest would overfit the noise
        and hide whether Rock pushes the log-odds up or down.
        """
    )
    with st.expander("What not to model"):
        st.markdown(
            "Building `TotalMusicFreq` as the sum of four frequency columns and "
            "then predicting that sum recovers coefficients `[1, 1, 1, 1]` and "
            "MSE ≈ 0. That is leakage, not insight."
        )

    result = cached_improve_model(full)
    c1, c2, c3, c4 = st.columns(4, gap="large")
    c1.metric("Test ROC-AUC", f"{result['auc']:.3f}", help="0.5 = coin flip")
    c2.metric("Majority-class AUC", f"{result['dummy_auc']:.3f}")
    c3.metric("Test accuracy", f"{result['accuracy']:.1%}")
    c4.metric("Always-Improve accuracy", f"{result['dummy_accuracy']:.1%}")

    if result["auc"] <= result["dummy_auc"] + 0.03:
        st.info(
            f"Listening habits barely beat a coin flip (AUC {result['auc']:.2f} vs "
            f"{result['dummy_auc']:.2f}). The base rate is already "
            f"{result['base_rate']:.0%} Improve — accuracy near that number is "
            "the base rate, not skill."
        )
    else:
        st.success("The logistic model ranks Improve vs not better than chance.")

    coef = result["coefficients"].copy()
    coef["direction"] = coef["log_odds"].map(lambda v: "↑ Improve" if v >= 0 else "↓ not")
    left, right = st.columns(2, gap="large")
    with left:
        fig = px.bar(
            coef,
            x="log_odds",
            y="feature",
            orientation="h",
            color="direction",
            color_discrete_map={"↑ Improve": PRIMARY, "↓ not": ACCENT},
            title="Log-odds per 1 SD of the feature",
            labels={"log_odds": "Coefficient (standardised)"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        draw(fig, height=height_for_bars(len(coef), row_px=26), key="model_coef")
        why(
            "A signed coefficient bar is why logistic regression belongs here. "
            "Each bar is the change in log-odds of saying “Improve” for a "
            "one-standard-deviation bump in that feature."
        )
    with right:
        roc = pd.DataFrame({"False positive rate": result["fpr"], "True positive rate": result["tpr"]})
        fig = px.line(
            roc,
            x="False positive rate",
            y="True positive rate",
            title="ROC curve on the held-out 20%",
            color_discrete_sequence=[PRIMARY],
        )
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line=dict(color=ACCENT, dash="dash"),
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        draw(fig, height=480, key="model_roc")
        why(
            "ROC asks a ranking question: if you sort people by predicted "
            "probability, how often does an “Improve” land above a “not”? "
            "The dashed line is chance."
        )

    cm = pd.DataFrame(
        result["confusion"],
        index=["True not", "True Improve"],
        columns=["Pred not", "Pred Improve"],
    )
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion matrix (threshold 0.5)",
        aspect="equal",
    )
    fig.update_traces(textfont_size=16)
    draw(fig, height=420, key="model_cm")
    why(
        "With a 75% Improve base rate the model can look “accurate” by always "
        "guessing Improve. The matrix shows whether the minority class is "
        "invisible — which is the usual failure mode here."
    )


def page_table(filtered: pd.DataFrame) -> None:
    st.subheader("Filtered rows")
    st.markdown(
        "The table is the receipt: every chart above is a summary of these rows. "
        "Download the cut if you want to check a number in a spreadsheet."
    )
    if filtered.empty:
        empty_state()
        return
    show_freq = st.toggle("Show encoded genre frequencies", value=False)
    cols = [
        "Age",
        "Hours per day",
        "Primary streaming service",
        "Fav genre",
        "Music effects",
        *MENTAL_HEALTH_COLS,
    ]
    if show_freq:
        cols.extend(frequency_columns(filtered))
    view = filtered[cols].rename(
        columns={c: genre_label(c) for c in frequency_columns(filtered)}
    )
    st.dataframe(view, use_container_width=True, height=480)
    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download this cut as CSV",
        data=csv,
        file_name="musicx_filtered.csv",
        mime="text/csv",
    )


def main() -> None:
    inject_style()
    full = get_data()
    filtered = sidebar_filters(full)

    header_l, header_r = st.columns([1.4, 2.6], gap="large")
    with header_l:
        if BANNER.exists():
            st.image(str(BANNER))
    with header_r:
        st.title("MusicX")
        st.caption("Music habits and self-reported mental health")
        kpi_row(filtered, full)

    tabs = st.tabs(
        [
            "Overview",
            "Listeners",
            "Chart builder",
            "Associations",
            "Compare",
            "Model",
            "Data",
        ]
    )
    with tabs[0]:
        page_overview(filtered)
    with tabs[1]:
        page_listeners(filtered)
    with tabs[2]:
        page_builder(filtered)
    with tabs[3]:
        page_associations(filtered)
    with tabs[4]:
        page_compare(filtered)
    with tabs[5]:
        page_model(full)
    with tabs[6]:
        page_table(filtered)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "[MxMH dataset](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results) · "
        "[Source](https://github.com/PedroCarneiroMarques/MusicX_Emotion_Analysis)"
    )


if __name__ == "__main__":
    main()
