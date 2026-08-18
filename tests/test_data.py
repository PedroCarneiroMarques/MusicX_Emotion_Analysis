"""Unit tests for survey cleaning — the bits that used to invent data."""

from __future__ import annotations

import pandas as pd

from src.data import (
    BPM_MAX,
    FREQ_MAP,
    clean_survey,
    encode_frequencies,
    frequency_columns,
    genre_label,
    load_clean,
    load_raw,
)


def test_raw_shape_matches_kaggle_release() -> None:
    raw = load_raw()
    assert raw.shape[0] == 736
    assert len(frequency_columns(raw)) == 16


def test_missing_age_is_not_filled_with_zero() -> None:
    raw = load_raw()
    clean = clean_survey(raw)
    assert raw["Age"].isna().sum() == 1
    assert clean["Age"].isna().sum() == 1
    assert not (clean["Age"] == 0).any()


def test_billion_bpm_is_treated_as_missing() -> None:
    raw = load_raw()
    clean = clean_survey(raw)
    assert raw["BPM"].max() > 1_000_000
    assert clean["BPM"].max() <= BPM_MAX
    assert int(clean["BPM_outlier"].sum()) >= 1


def test_frequency_encoding_is_ordinal_0_to_3() -> None:
    raw = load_raw()
    encoded = encode_frequencies(raw)
    for col in frequency_columns(encoded):
        values = set(encoded[col].dropna().unique())
        assert values <= {0, 1, 2, 3}


def test_genre_label_strips_survey_prefix() -> None:
    assert genre_label("Frequency [Video game music]") == "Video game music"


def test_freq_map_covers_all_labels_in_file() -> None:
    raw = load_raw()
    labels: set[str] = set()
    for col in frequency_columns(raw):
        labels.update(raw[col].dropna().astype(str).unique())
    assert labels == set(FREQ_MAP)


def test_load_clean_keeps_music_effects_missing() -> None:
    df = load_clean()
    assert df["Music effects"].isna().sum() == 8
    assert 0 not in set(df["Music effects"].dropna().unique())


def test_permissions_column_is_dropped() -> None:
    df = load_clean()
    assert "Permissions" not in df.columns
    assert isinstance(df, pd.DataFrame)


def test_missing_values_are_not_pandas_na_type() -> None:
    """Plotly/orjson cannot serialise pandas.NA — missing must be numpy NaN."""
    df = load_clean()
    for col in df.columns:
        assert not df[col].map(lambda x: type(x).__name__ == "NAType").any(), col
