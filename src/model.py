"""Interpretable model for the MxMH survey.

The survey's own music×mental-health question is `Music effects`
(Improve / No effect / Worsen). Worsen has only 17 rows, so a three-class
model would be theatre. The honest target is binary: Improve vs not.

Logistic regression (L2, scaled features) is the right tool here: n is
small (~700), features are correlated genre frequencies, and we need
signed coefficients a hiring manager can read. A random forest would
overfit the noise and hide the direction of each genre.
"""

from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import frequency_columns, genre_label

FEATURE_EXTRA = ["Hours per day", "Age"]


def _feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    work = df.dropna(subset=["Age", "Music effects"]).copy()
    work = work[work["Music effects"].isin(["Improve", "No effect", "Worsen"])]
    feats = frequency_columns(work) + FEATURE_EXTRA
    x = work[feats]
    y = work["Music effects"].eq("Improve").astype(int)
    return x, y


def fit_improve_model(df: pd.DataFrame, random_state: int = 42) -> dict:
    """Fit scaled logistic regression: Improve (1) vs not (0)."""
    x, y = _feature_frame(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=2000, random_state=random_state)),
        ]
    )
    pipe.fit(x_train, y_train)
    proba = pipe.predict_proba(x_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    dummy = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    dummy_pred = dummy.predict(x_test)
    dummy_proba = dummy.predict_proba(x_test)[:, 1]

    labels = [
        genre_label(c) if str(c).startswith("Frequency") else c for c in x.columns
    ]
    coefficients = (
        pd.Series(pipe.named_steps["logreg"].coef_[0], index=labels)
        .rename("log_odds")
        .rename_axis("feature")
        .sort_values()
        .reset_index()
    )
    fpr, tpr, _ = roc_curve(y_test, proba)
    return {
        "n": int(len(x)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "base_rate": float(y.mean()),
        "auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "dummy_accuracy": float(accuracy_score(y_test, dummy_pred)),
        "dummy_auc": float(roc_auc_score(y_test, dummy_proba)),
        "brier": float(brier_score_loss(y_test, proba)),
        "coefficients": coefficients,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "confusion": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
    }
