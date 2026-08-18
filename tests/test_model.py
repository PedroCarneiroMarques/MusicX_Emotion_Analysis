"""Tests for the Improve-vs-not logistic model."""

from src.data import load_clean
from src.model import fit_improve_model


def test_improve_model_is_not_leaked() -> None:
    result = fit_improve_model(load_clean())
    assert 0.4 <= result["auc"] <= 0.85
    assert result["n"] > 500
    assert len(result["coefficients"]) == 18
    assert result["dummy_accuracy"] > 0.7
    assert set(result["coefficients"].columns) >= {"feature", "log_odds"}
