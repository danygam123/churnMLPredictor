"""Tests for SHAP explanation consistency and prediction-engine integration."""

import numpy as np
import pandas as pd
import pytest

from ml.explain import ChurnExplainer
from ml.predict import ChurnPredictor
from ml.preprocessing import INPUT_FEATURES, CustomerDataValidator


def _sample(offset=0):
    values = {
        "Account length": 100 + offset,
        "International plan": "yes" if offset else "no",
        "Voice mail plan": "yes",
        "Number vmail messages": 20 + offset,
        "Total day minutes": 180.0 + offset * 3,
        "Total day calls": 100,
        "Total eve minutes": 160.0,
        "Total eve calls": 90,
        "Total night minutes": 200.0,
        "Total night calls": 100,
        "Total intl minutes": 8.0,
        "Total intl calls": 3,
        "Customer service calls": offset % 4,
    }
    return values


def test_local_shap_values_reconstruct_model_probability_and_waterfall(tmp_path):
    predictor = ChurnPredictor()
    background = pd.DataFrame([_sample(i) for i in range(1, 9)], columns=INPUT_FEATURES)
    explainer = ChurnExplainer(predictor.model, background, max_evals=27)

    result = explainer.explain(_sample(), max_evals=27)
    direct = predictor.model.predict_proba(pd.DataFrame([_sample()]))[0, 1]

    assert np.isclose(result["churn_probability"], direct)
    assert np.isclose(result["reconstructed_probability"], direct, atol=1e-6)
    assert len(result["all_contributions"]) == len(INPUT_FEATURES)
    assert set(result["increasing_risk_factors"]).isdisjoint(result["decreasing_risk_factors"])
    assert result["explanation_method"].endswith("output is churn probability")

    waterfall = ChurnExplainer.save_waterfall(result, tmp_path / "waterfall.png")
    assert waterfall.is_file() and waterfall.stat().st_size > 0


def test_global_importance_and_predictor_integration():
    predictor = ChurnPredictor()
    background = pd.DataFrame([_sample(i) for i in range(1, 9)], columns=INPUT_FEATURES)
    explanation = predictor.explain_single(_sample())
    assert 0.0 <= explanation["churn_probability"] <= 1.0
    assert explanation["prediction"] in ("Likely to churn", "Less likely to churn")

    global_values = predictor._explainer.explain_global(background.iloc[:2], max_evals=27)
    importance = predictor._explainer.global_importance(global_values)
    assert list(importance.columns) == [
        "feature", "mean_abs_contribution", "mean_contribution", "positive_fraction", "negative_fraction"
    ]
    assert len(importance) == len(INPUT_FEATURES)
    assert importance["mean_abs_contribution"].is_monotonic_decreasing


def test_missing_input_is_reported_as_estimated():
    predictor = ChurnPredictor()
    profile = _sample()
    del profile["Total day calls"]

    result = predictor.explain_single(profile)

    assert result["estimated_features"] == ["Total day calls"]
    assert result["genuinely_missing_features"] == ["Total day calls"]
    assert result["rejected_or_coerced_features"] == []
    item = next(x for x in result["all_contributions"] if x["feature"] == "Total day calls")
    assert item["input_status"] == "genuinely_missing_imputed"
    assert np.isclose(result["reconstructed_probability"], result["churn_probability"], atol=1e-6)


@pytest.mark.parametrize(
    ("feature", "invalid_value"),
    [
        ("Total day minutes", "nonsense"),
        ("International plan", "maybe"),
        ("Total day calls", 9999),
    ],
)
def test_rejected_values_are_distinguished_and_prediction_still_reconstructs(feature, invalid_value):
    predictor = ChurnPredictor()
    profile = _sample()
    profile[feature] = invalid_value

    result = predictor.explain_single(profile)
    item = next(x for x in result["all_contributions"] if x["feature"] == feature)
    validated = CustomerDataValidator().validate(pd.DataFrame([profile]), remove_duplicates=False).cleaned_df
    direct = predictor.model.predict_proba(validated[INPUT_FEATURES])[0, 1]

    assert result["estimated_features"] == [feature]
    assert result["genuinely_missing_features"] == []
    assert result["rejected_or_coerced_features"] == [feature]
    assert item["value"] == invalid_value
    assert item["input_status"] == "invalid_or_out_of_range_coerced_then_imputed"
    assert np.isclose(result["churn_probability"], direct)
    assert np.isclose(result["reconstructed_probability"], direct, atol=1e-6)


def test_serving_explainer_uses_one_fixed_reference_and_rejects_overrides():
    predictor = ChurnPredictor()
    result = predictor.explain_single(_sample())
    cached = predictor._explainer
    expected = pd.read_csv("models/explanation_reference.csv")

    assert expected.shape == (50, len(INPUT_FEATURES))
    assert list(expected.columns) == INPUT_FEATURES
    pd.testing.assert_frame_equal(cached.background_data, expected)
    assert np.isclose(result["reconstructed_probability"], result["churn_probability"], atol=1e-6)

    different_reference = expected.copy()
    different_reference["Customer service calls"] = 0
    with pytest.raises(ValueError, match="fixed reference population"):
        predictor.explain_single(_sample(2), different_reference)

    predictor.explain_single(_sample(3))
    assert predictor._explainer is cached
