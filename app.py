"""MusicX — interactive Streamlit dashboard for the MxMH survey.

Run from the repo root:

    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data import MENTAL_HEALTH_COLS, frequency_columns, genre_label, load_clean
from src.viz import ACCENT, EFFECT_COLORS, PRIMARY, SOFT, draw, empty_state

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
        .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1400px; }
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
    df["Music effects"] = df["Music effects"].astype("string")
    df["Fav genre"] = df["Fav genre"].astype("string")
    return df


@st.cache_data
def honest_regression(df: pd.DataFrame, target: str) -> dict:
    work = df.dropna(subset=["Age", target]).copy()
    feats = frequency_columns(work) + ["Hours per day", "Age"]
    x_train, x_test, y_train, y_test = train_test_split(
        work[feats], work[target], test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    dummy = DummyRegressor(strategy="mean").fit(x_train, y_train)
    base = dummy.predict(x_test)
    importance = (
        pd.Series(model.feature_importances_, index=feats)
        .sort_values(ascending=False)
        .rename(index=lambda c: genre_label(c) if str(c).startswith("Frequency") else c)
        .head(8)
        .reset_index()
    )
    importance.columns = ["feature", "importance"]
    return {
        "n": len(work),
        "rf_r2": r2_score(y_test, pred),
        "rf_mse": mean_squared_error(y_test, pred),
        "base_r2": r2_score(y_test, base),
        "base_mse": mean_squared_error(y_test, base),
        "importance": importance,
        "y_test": list(y_test),
        "pred": list(pred),
    }


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
    return out


def kpi_row(filtered: pd.DataFrame, full: pd.DataFrame) -> None:
    effects = filtered["Music effects"].dropna()
    improve = (effects == "Improve").mean() if len(effects) else float("nan")
    full_improve = (full["Music effects"].dropna() == "Improve").mean()
    c1, c2, c3, c4 = st.columns(4)
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
    st.caption(
        "MxMH survey · collected Aug–Nov 2022. Filters on the left apply to every tab."
    )
    if filtered.empty:
        empty_state()
        return

    left, right = st.columns(2)
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
                hole=0.45,
            )
            fig.update_traces(textinfo="percent+label", hovertemplate="%{label}: %{value}<br>%{percent}")
            draw(fig, key="overview_pie")
    with right:
        mh = filtered[MENTAL_HEALTH_COLS].melt(var_name="Symptom", value_name="Score")
        fig = px.box(
            mh,
            x="Symptom",
            y="Score",
            color="Symptom",
            points="all",
            title="Symptom scores (hover a point)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(jitter=0.3, marker=dict(size=5, opacity=0.45))
        fig.update_layout(showlegend=False)
        draw(fig, key="overview_box")

    st.markdown(
        "Three in four people in the full sample say music **improves** their "
        "mental health. Genre preferences only weakly track the 0–10 scores. "
        "Use **Chart builder** to ask your own question; use **Model** to see "
        "why a flashy R² is not always a finding."
    )


def page_listeners(filtered: pd.DataFrame) -> None:
    st.subheader("Who is in this cut")
    if filtered.empty:
        empty_state()
        return

    c1, c2 = st.columns(2)
    with c1:
        bins = st.slider("Age bins", 8, 40, 24, key="age_bins")
        fig = px.histogram(filtered, x="Age", nbins=bins, title="Age")
        fig.update_traces(marker_color=PRIMARY)
        draw(fig, key="hist_age")
    with c2:
        bins_h = st.slider("Hours bins", 5, 30, 16, key="hours_bins")
        fig = px.histogram(
            filtered, x="Hours per day", nbins=bins_h, title="Hours of music per day"
        )
        fig.update_traces(marker_color=ACCENT)
        draw(fig, key="hist_hours")

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
        title="Primary streaming service — click a bar in Favorite genre below to focus",
        color="Respondents",
        color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    draw(fig, height=360, key="bar_service")

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
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    event = draw(fig, height=520, key="bar_genre", on_select="rerun", selection_mode="points")
    _lock_genre_from_event(event)

    fig = px.violin(
        filtered.dropna(subset=["Age", "Fav genre"]),
        x="Fav genre",
        y="Age",
        color="Fav genre",
        box=True,
        points="outliers",
        title="Age by favorite genre",
    )
    fig.update_layout(showlegend=False, xaxis_tickangle=-35)
    draw(fig, height=460, key="violin_age_genre")


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
    st.caption("Pick axes, a colour, and a mark. Hover, zoom, and lasso in the toolbar.")
    if filtered.empty:
        empty_state()
        return

    numeric = NUMERIC_EXPLORE + frequency_columns(filtered)
    cats = CATEGORICAL_EXPLORE
    c1, c2, c3, c4 = st.columns(4)
    mark = c1.selectbox(
        "Chart type",
        ["Scatter", "Histogram", "Box", "Violin", "Bar", "Density heatmap"],
    )
    x_col = c2.selectbox("X", numeric + cats, index=0)
    y_default = numeric.index("Anxiety") if "Anxiety" in numeric else 0
    y_col = c3.selectbox("Y", numeric, index=y_default)
    color_opt = c4.selectbox("Color", ["(none)"] + cats, index=1)
    color = None if color_opt == "(none)" else color_opt

    extra1, extra2, extra3 = st.columns(3)
    opacity = extra1.slider("Opacity", 0.2, 1.0, 0.7)
    trend = extra2.toggle("Trendline (OLS)", value=False)
    bins = extra3.slider("Histogram bins", 8, 40, 20)

    work = filtered.copy()
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
                points="all",
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
        draw(fig, height=520, key="builder_chart")


def page_associations(filtered: pd.DataFrame) -> None:
    st.subheader("Genres and symptom scores")
    st.caption(
        "Frequencies are Never=0 … Very frequently=3. |r| below ~0.20 is weak. "
        "Not causal."
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
    event = draw(fig, height=640, key="heatmap", on_select="rerun", selection_mode="points")
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
        fig.update_traces(texttemplate="n=%{text}", textposition="outside")
        fig.update_layout(xaxis_tickformat=".0%", coloraxis_showscale=False)
        event_bar = draw(fig, height=480, key="improve_bar", on_select="rerun", selection_mode="points")
        _lock_genre_from_event(event_bar)

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
        draw(fig, key="effect_means")
        st.caption("The full-sample “Worsen” group has only 17 people. Treat it as a flag.")


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
        points="all",
        title=f"{symptom} by how often people listen to {genre}",
        labels={col: f"{genre} (0=Never … 3=Very frequently)"},
        color_discrete_sequence=[PRIMARY],
    )
    draw(fig, height=380, key="heatmap_drill")


def page_compare(filtered: pd.DataFrame) -> None:
    st.subheader("Compare two groups")
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
    left, right = st.columns(2)
    a = left.selectbox("Group A", values, index=0, key="cmp_a")
    b_default = 1 if len(values) > 1 else 0
    b = right.selectbox("Group B", values, index=b_default, key="cmp_b")
    if a == b:
        st.info("Pick two different groups.")
        return

    fa = filtered[filtered[dim] == a]
    fb = filtered[filtered[dim] == b]
    k1, k2, k3, k4 = st.columns(4)
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
    draw(fig, key="cmp_hist")

    summary = (
        both.groupby("group")[MENTAL_HEALTH_COLS + ["Hours per day", "Age"]]
        .mean()
        .round(2)
        .T.rename_axis("Metric")
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)


def page_model(full: pd.DataFrame) -> None:
    st.subheader("What a model can (and cannot) do")
    st.caption("Trained on the full cleaned sample, not the sidebar filters — the question is the target, not the cut.")
    st.markdown(
        """
        If you build a target `TotalMusicFreq` as the **sum of four genre-frequency
        columns** and then predict that sum with those columns, linear regression
        recovers coefficients `1, 1, 1, 1` and MSE ≈ 0. That is leakage.

        The honest question: **do listening habits help predict a symptom score?**
        """
    )
    target = st.selectbox("Symptom to predict", MENTAL_HEALTH_COLS, key="model_target")
    result = honest_regression(full, target)
    c1, c2, c3 = st.columns(3)
    c1.metric("Random forest R²", f"{result['rf_r2']:.3f}")
    c2.metric("Mean-baseline R²", f"{result['base_r2']:.3f}")
    c3.metric("Train/test n", f"{result['n']}")
    if result["rf_r2"] <= result["base_r2"] + 0.02:
        st.info(
            "The forest does **not** beat predicting the training-set mean. "
            "Genre + hours + age are not a useful predictor of this score here."
        )
    else:
        st.success("The forest beats the mean baseline — still check how small the R² is.")

    left, right = st.columns(2)
    with left:
        fig = px.bar(
            result["importance"],
            x="importance",
            y="feature",
            orientation="h",
            title=f"Feature importance — {target}",
            color_discrete_sequence=[PRIMARY],
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        draw(fig, key="model_imp")
    with right:
        scatter = pd.DataFrame(
            {"actual": result["y_test"], "predicted": result["pred"]}
        )
        fig = px.scatter(
            scatter,
            x="actual",
            y="predicted",
            title="Test set: predicted vs actual",
            opacity=0.7,
            color_discrete_sequence=[SOFT],
        )
        fig.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=10,
            y1=10,
            line=dict(color=ACCENT, dash="dash"),
        )
        draw(fig, key="model_scatter")


def page_table(filtered: pd.DataFrame) -> None:
    st.subheader("Filtered rows")
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

    header_l, header_r = st.columns([1.4, 2.6])
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
