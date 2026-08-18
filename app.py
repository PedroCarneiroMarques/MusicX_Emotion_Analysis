"""MusicX — Streamlit explorer for the MxMH survey.

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

BANNER = Path(__file__).resolve().parent / "assets" / "banner.png"

st.set_page_config(
    page_title="MusicX · Music & Mental Health",
    page_icon="🎵",
    layout="wide",
)


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_clean()


@st.cache_data
def honest_regression(df: pd.DataFrame, target: str) -> dict:
    """Predict a 0–10 symptom score from genres + age + hours vs a mean baseline."""
    work = df.dropna(subset=["Age", target]).copy()
    feats = frequency_columns(work) + ["Hours per day", "Age"]
    x = work[feats]
    y = work[target]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(x_train, y_train)
    base = dummy.predict(x_test)
    importance = (
        pd.Series(model.feature_importances_, index=feats)
        .sort_values(ascending=False)
        .rename(index=lambda c: genre_label(c) if c.startswith("Frequency") else c)
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
    }


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; }
        h1, h2, h3 { letter-spacing: -0.02em; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_overview(df: pd.DataFrame) -> None:
    if BANNER.exists():
        st.image(str(BANNER))

    st.title("Music habits and self-reported mental health")
    st.caption(
        "MxMH survey · 736 respondents · Aug–Nov 2022 · "
        "rewritten from an Ironhack final project (Dec 2023)"
    )

    effects = df["Music effects"].dropna()
    improve = (effects == "Improve").mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Respondents", f"{len(df):,}")
    c2.metric("Median age", f"{df['Age'].median():.0f}")
    c3.metric("Median hours / day", f"{df['Hours per day'].median():.0f} h")
    c4.metric("Say music improves MH", f"{improve:.0%}")

    st.markdown(
        """
        Three in four respondents say music **improves** their mental health.
        Genre preferences, though, only weakly track the 0–10 symptom scores
        (anxiety, depression, insomnia, OCD). A model that *looks* accurate
        in the original notebook was predicting the sum of its own inputs.

        Use the sidebar to explore the sample, the associations, and the
        corrected models. The write-up lives in [`analysis.ipynb`](analysis.ipynb).
        """
    )

    left, right = st.columns(2)
    with left:
        fig = px.pie(
            effects.value_counts().reset_index(),
            names="Music effects",
            values="count",
            title="Self-reported effect of music",
            color="Music effects",
            color_discrete_map={
                "Improve": "#3D5A80",
                "No effect": "#98C1D9",
                "Worsen": "#EE6C4D",
            },
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        mh = df[MENTAL_HEALTH_COLS].melt(var_name="Symptom", value_name="Score")
        fig = px.box(
            mh,
            x="Symptom",
            y="Score",
            color="Symptom",
            title="Symptom scores (0 = none, 10 = extreme)",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def page_listeners(df: pd.DataFrame) -> None:
    st.title("Who answered the survey")
    st.markdown(
        "The sample is **young and Spotify-heavy**. That selection bias "
        "matters: findings do not generalise to older listeners or to "
        "people who do not spend time on music forums."
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Age", nbins=24, title="Age")
        fig.update_traces(marker_color="#3D5A80")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(
            df, x="Hours per day", nbins=20, title="Hours of music per day"
        )
        fig.update_traces(marker_color="#EE6C4D")
        st.plotly_chart(fig, use_container_width=True)

    services = (
        df["Primary streaming service"]
        .value_counts(dropna=False)
        .rename_axis("Service")
        .reset_index(name="Respondents")
    )
    services["Service"] = services["Service"].fillna("Not reported")
    fig = px.bar(
        services,
        x="Respondents",
        y="Service",
        orientation="h",
        title="Primary streaming service",
        color="Respondents",
        color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    genre_counts = (
        df["Fav genre"]
        .value_counts()
        .rename_axis("Favorite genre")
        .reset_index(name="Respondents")
    )
    fig = px.bar(
        genre_counts,
        x="Respondents",
        y="Favorite genre",
        orientation="h",
        title="Favorite genre",
        color_discrete_sequence=["#3D5A80"],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=520)
    st.plotly_chart(fig, use_container_width=True)


def page_associations(df: pd.DataFrame) -> None:
    st.title("Genres and symptom scores")
    st.markdown(
        "Frequencies are encoded as Never=0, Rarely=1, Sometimes=2, "
        "Very frequently=3. Correlations below **|r| ≈ 0.20** are weak. "
        "None of this is causal — people in a low mood may also choose "
        "different music."
    )

    freq = frequency_columns(df)
    corr = df[freq + MENTAL_HEALTH_COLS].corr().loc[freq, MENTAL_HEALTH_COLS]
    corr.index = [genre_label(c) for c in corr.index]
    fig = px.imshow(
        corr,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        zmin=-0.25,
        zmax=0.25,
        title="Pearson r: listening frequency vs. symptom score",
        labels={"color": "r"},
    )
    fig.update_layout(height=620)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Favorite genre vs. “music improves my mental health”")
    st.caption("Genres with fewer than 20 respondents are hidden — small-n rates lie.")
    work = df.dropna(subset=["Music effects", "Fav genre"]).copy()
    work["improves"] = work["Music effects"].eq("Improve")
    summary = (
        work.groupby("Fav genre")
        .agg(n=("improves", "size"), improve_rate=("improves", "mean"))
        .query("n >= 20")
        .sort_values("improve_rate")
        .reset_index()
    )
    fig = px.bar(
        summary,
        x="improve_rate",
        y="Fav genre",
        orientation="h",
        text="n",
        title="Share who say music improves their mental health (n ≥ 20)",
        labels={"improve_rate": "Improve rate", "Fav genre": "Favorite genre"},
        color="improve_rate",
        color_continuous_scale="Blues",
    )
    fig.update_traces(texttemplate="n=%{text}", textposition="outside")
    fig.update_layout(xaxis_tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("People who say music worsens things also report more depression")
    means = (
        df.dropna(subset=["Music effects"])
        .groupby("Music effects")[MENTAL_HEALTH_COLS]
        .mean()
        .reindex(["Improve", "No effect", "Worsen"])
        .reset_index()
        .melt(id_vars="Music effects", var_name="Symptom", value_name="Mean score")
    )
    fig = px.bar(
        means,
        x="Symptom",
        y="Mean score",
        color="Music effects",
        barmode="group",
        color_discrete_map={
            "Improve": "#3D5A80",
            "No effect": "#98C1D9",
            "Worsen": "#EE6C4D",
        },
        title="Mean symptom score by reported music effect",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The “Worsen” group has only 17 people. Treat it as a flag, not a rate."
    )


def page_modeling(df: pd.DataFrame) -> None:
    st.title("What a model can (and cannot) do here")
    st.markdown(
        """
        The original notebook created `TotalMusicFreq` as the **sum of four
        genre-frequency columns**, then used those same columns to predict
        the sum. Linear regression recovered coefficients of `1, 1, 1, 1`
        and an MSE of ~0. That is leakage, not insight.

        The honest question is the one the project title promised:
        **do listening habits help predict a symptom score?**
        """
    )

    target = st.selectbox("Symptom to predict", MENTAL_HEALTH_COLS)
    result = honest_regression(df, target)

    c1, c2, c3 = st.columns(3)
    c1.metric("Random forest R²", f"{result['rf_r2']:.3f}")
    c2.metric("Mean-baseline R²", f"{result['base_r2']:.3f}")
    c3.metric("Train/test n", f"{result['n']}")

    if result["rf_r2"] <= result["base_r2"] + 0.02:
        st.info(
            "The forest does **not** beat predicting the training-set mean. "
            "Genre + hours + age are not a useful predictor of this score "
            "in this sample."
        )
    else:
        st.success("The forest improves on the mean baseline — still check the R² size.")

    fig = px.bar(
        result["importance"],
        x="importance",
        y="feature",
        orientation="h",
        title=f"Feature importance when predicting {target}",
        color_discrete_sequence=["#3D5A80"],
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        **Takeaway for a hiring manager (and for me):** a dashboard that
        flashes a low MSE is worthless if the target is a function of the
        features. Compare every model to a dummy baseline. Weak signal is
        a valid finding — it is what this survey actually contains.
        """
    )


def main() -> None:
    inject_style()
    df = get_data()
    page = st.sidebar.radio(
        "Section",
        ["Overview", "Listeners", "Associations", "Modeling"],
        index=0,
    )
    st.sidebar.markdown(
        """
        **MusicX**  
        Data: [MxMH on Kaggle](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results)  
        Code: `src/data.py` · notes: `analysis.ipynb`
        """
    )
    if page == "Overview":
        page_overview(df)
    elif page == "Listeners":
        page_listeners(df)
    elif page == "Associations":
        page_associations(df)
    else:
        page_modeling(df)


if __name__ == "__main__":
    main()
