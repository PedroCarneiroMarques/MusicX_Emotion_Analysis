"""Load and clean the Music & Mental Health (MxMH) survey.

The survey stores genre listening as ordered categories
(Never → Very frequently). Mental-health items are self-reported
0–10 scores. Cleaning keeps missing values as missing: filling them
with 0 would invent a real age, BPM, or "Never" answer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "mxmh_survey.csv"

FREQ_MAP = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Very frequently": 3,
}
MENTAL_HEALTH_COLS = ["Anxiety", "Depression", "Insomnia", "OCD"]
CATEGORICAL_COLS = [
    "Primary streaming service",
    "While working",
    "Instrumentalist",
    "Composer",
    "Fav genre",
    "Exploratory",
    "Foreign languages",
    "Music effects",
]
# Plausible recorded-music tempo. The raw file contains 999,999,999 BPM.
BPM_MIN = 20
BPM_MAX = 300


def frequency_columns(df: pd.DataFrame) -> list[str]:
    """Return the 16 `Frequency [Genre]` columns, in file order."""
    return [c for c in df.columns if c.startswith("Frequency [")]


def genre_label(column: str) -> str:
    """Strip the survey prefix: `Frequency [Rock]` → `Rock`."""
    return column.removeprefix("Frequency [").removesuffix("]")


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Read the CSV as published, with no transformations."""
    csv_path = Path(path) if path is not None else DEFAULT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Survey CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def encode_frequencies(df: pd.DataFrame) -> pd.DataFrame:
    """Map genre frequency labels to an ordinal 0–3 scale."""
    out = df.copy()
    cols = frequency_columns(out)
    out[cols] = out[cols].apply(lambda s: s.map(FREQ_MAP))
    return out


def clean_survey(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types, drop the consent column, and null out impossible BPM.

    Does not impute Age, streaming service, or Music effects. Callers
    decide whether to drop those rows for a given chart or model.
    """
    out = df.copy()
    if "Permissions" in out.columns:
        out = out.drop(columns=["Permissions"])
    if "Timestamp" in out.columns:
        out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")

    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["Hours per day"] = pd.to_numeric(out["Hours per day"], errors="coerce")
    for col in MENTAL_HEALTH_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "BPM" in out.columns:
        bpm = pd.to_numeric(out["BPM"], errors="coerce")
        valid = bpm.between(BPM_MIN, BPM_MAX)
        out["BPM_outlier"] = bpm.notna() & ~valid
        out["BPM"] = bpm.where(valid)

    for col in CATEGORICAL_COLS:
        if col in out.columns:
            out[col] = out[col].replace({"": np.nan, 0: np.nan})

    return out


def load_clean(path: str | Path | None = None) -> pd.DataFrame:
    """Load, clean, and encode frequencies — ready for EDA and models."""
    return encode_frequencies(clean_survey(load_raw(path)))
