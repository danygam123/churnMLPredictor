"""Safety and pipeline tests for independent customer what-if scenarios."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from ml.predict import ChurnPredictor
from ml.preprocessing import INPUT_FEATURES
from ml.simulation import simulate_scenario


@pytest.fixture(scope="module")
def predictor():
    return ChurnPredictor()


def _profile(offset=0):
    return {
        "Customer ID": f"SIM-{offset}",
        "Account length": 100 + offset,
        "International plan": "No",
        "Voice mail plan": "Yes",
        "Number vmail messages": 20 + offset,
        "Total day minutes": 180.0 + offset,
        "Total day calls": 100,
        "Total eve minutes": 160.0,
        "Total eve calls": 90,
        "Total night minutes": 200.0,
        "Total night calls": 100,
        "Total intl minutes": 8.0,
        "Total intl calls": 3,
        "Customer service calls": 2,
        "Churn": False,
    }


def test_no_change_returns_same_prediction_and_zero_percentage_points(predictor):
    profile = _profile()
    result = simulate_scenario(predictor, profile, {})
    direct = predictor.predict_single(profile, explain_sensitivities=False)

    assert result.original_probability == result.simulated_probability == direct["churn_probability"]
    assert result.difference == 0.0
    assert result.difference_percentage_points == 0.0
    assert result.original_prediction == result.simulated_prediction == direct["prediction"]
    assert result.original_risk_category == result.simulated_risk_category
    assert result.changed_features == {}
    assert set(result.scenario_inputs) == set(INPUT_FEATURES)


def test_valid_multiple_changes_match_the_existing_serving_pipeline(predictor):
    profile = _profile()
    changes = {"International plan": "Yes", "Customer service calls": 5, "Total day minutes": 210.5}
    result = simulate_scenario(predictor, profile, changes)
    candidate = {feature: profile[feature] for feature in INPUT_FEATURES}
    candidate.update(changes)
    direct = predictor.predict_single(candidate, explain_sensitivities=False)

    assert set(result.changed_features) == set(changes)
    assert result.changed_features["International plan"] == {"original_value": "No", "simulated_value": "Yes"}
    assert result.simulated_probability == direct["churn_probability"]
    assert np.isclose(result.difference, result.simulated_probability - result.original_probability)
    assert np.isclose(result.difference_percentage_points, result.difference * 100)
    assert result.simulated_prediction == direct["prediction"]
    assert result.decision_threshold == predictor.optimal_threshold


@pytest.mark.parametrize(
    ("feature", "value", "message"),
    [
        ("Total day minutes", 601, "between 0 and 600"),
        ("Customer service calls", 2.5, "whole number"),
        ("International plan", "Maybe", "Yes or No"),
        ("Voice mail plan", 0.5, "exactly 0/1"),
        ("Total intl calls", None, "non-missing"),
        ("Customer service calls", True, "not a binary value"),
        ("Customer ID", "OTHER", "Only the 13 canonical"),
        ("Churn", True, "Only the 13 canonical"),
        ("Total day charge", 35.0, "Only the 13 canonical"),
    ],
)
def test_invalid_immutable_or_unsupported_changes_are_rejected(predictor, feature, value, message):
    with pytest.raises(ValueError, match=message):
        simulate_scenario(predictor, _profile(), {feature: value})


def test_invalid_numeric_string_and_missing_baseline_are_disclosed(predictor):
    with pytest.raises(ValueError, match="must be numeric"):
        simulate_scenario(predictor, _profile(), {"Total eve minutes": "unknown"})

    profile = _profile()
    del profile["Total night calls"]
    result = simulate_scenario(predictor, profile, {})
    assert "Total night calls" in result.estimated_features
    assert result.original_probability == result.simulated_probability


def test_original_record_and_dataset_are_never_mutated_and_scenarios_are_independent(predictor):
    profile = _profile()
    dataset = pd.DataFrame([profile])
    original_profile = deepcopy(profile)
    original_dataset = dataset.copy(deep=True)

    scenario_a = simulate_scenario(predictor, profile, {"Customer service calls": 5})
    scenario_b = simulate_scenario(predictor, profile, {"Total intl minutes": 16.0})
    reset = simulate_scenario(predictor, profile, {})

    assert profile == original_profile
    pd.testing.assert_frame_equal(dataset, original_dataset)
    assert scenario_a.scenario_inputs["Total intl minutes"] == 8.0
    assert scenario_b.scenario_inputs["Customer service calls"] == 2
    assert reset.changed_features == {}
    assert reset.simulated_probability == reset.original_probability
    assert scenario_a.original_probability == scenario_b.original_probability == reset.original_probability


def test_observed_training_range_warning_is_labeled_marginal_not_ood(predictor):
    result = simulate_scenario(predictor, _profile(), {"Customer service calls": 10})
    assert "Customer service calls" in result.out_of_training_range_features
