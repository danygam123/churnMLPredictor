"""Phase 4 rule, explanation, prioritization, and portfolio tests."""

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing import INPUT_FEATURES
from ml.recommendations import recommend_customer, recommend_portfolio


def _profile(**overrides):
    result = {
        "Customer ID": "C-1",
        "Account length": 100,
        "International plan": "No",
        "Voice mail plan": "Yes",
        "Number vmail messages": 20,
        "Total day minutes": 180.0,
        "Total day calls": 100,
        "Total eve minutes": 160.0,
        "Total eve calls": 90,
        "Total night minutes": 200.0,
        "Total night calls": 100,
        "Total intl minutes": 12.0,
        "Total intl calls": 5,
        "Customer service calls": 4,
    }
    result.update(overrides)
    return result


def _explanation(status="provided", contribution=0.08):
    return {
        "genuinely_missing_features": [],
        "rejected_or_coerced_features": [],
        "all_contributions": [
            {"feature": "Customer service calls", "value": 4, "contribution": contribution, "direction": "increases", "input_status": status},
            {"feature": "International plan", "value": "No", "contribution": -0.02, "direction": "decreases", "input_status": "provided"},
            {"feature": "Total intl minutes", "value": 12.0, "contribution": 0.03, "direction": "increases", "input_status": "provided"},
            {"feature": "Total intl calls", "value": 5, "contribution": 0.01, "direction": "increases", "input_status": "provided"},
        ],
    }


def test_rules_trigger_and_shap_context_keeps_direction_and_human_review():
    profile = _profile()
    recs = recommend_customer(profile, 0.62, "High Risk", _explanation(), prediction_threshold=0.0773)

    by_id = {item["rule_id"]: item for item in recs}
    assert set(by_id) == {"service_contact_review", "international_usage_fit_review"}
    service = by_id["service_contact_review"]
    assert service["triggering_characteristics"]["Customer service calls"] == 4
    assert service["model_context"][0]["direction"] == "increases"
    assert service["human_approval_required"] is True
    assert service["causal_claim"] is False
    international = by_id["international_usage_fit_review"]
    context = {entry["feature"]: entry["direction"] for entry in international["model_context"]}
    assert context["International plan"] == "decreases"
    assert context["Total intl minutes"] == "increases"
    assert "does not reveal" in service["limitations"]
    assert recs == recommend_customer(profile, 0.62, "High Risk", _explanation(), prediction_threshold=0.0773)


@pytest.mark.parametrize("risk,probability", [("Low Risk", 0.4), ("Medium Risk", 0.05)])
def test_no_recommendations_below_risk_or_prediction_gate(risk, probability):
    assert recommend_customer(_profile(), probability, risk, _explanation(), prediction_threshold=0.0773) == []


def test_missing_or_imputed_attributes_do_not_trigger_rules_or_get_shap_context():
    profile = _profile(**{"Customer service calls": 99})
    explanation = _explanation(status="invalid_or_out_of_range_coerced_then_imputed")
    recs = recommend_customer(profile, 0.4, "Medium Risk", explanation, prediction_threshold=0.0773)
    assert [item["rule_id"] for item in recs] == ["international_usage_fit_review"]
    assert all(row["feature"] != "Customer service calls" for row in recs[0]["model_context"])

    profile["Customer service calls"] = np.nan
    profile["International plan"] = np.nan
    profile["Total intl minutes"] = np.nan
    assert recommend_customer(profile, 0.4, "Medium Risk", None, prediction_threshold=0.0773) == []


def test_individual_api_validates_rules_even_when_no_explanation_is_supplied():
    profile = _profile(**{"Customer service calls": 99, "Total intl minutes": 999, "Total intl calls": 0})
    recs = recommend_customer(profile, 0.4, "Medium Risk", None, prediction_threshold=0.0773)
    assert recs == []


def test_recommendation_priority_uses_risk_and_rule_strength_not_success_probability():
    profile = _profile(**{"Customer service calls": 3, "Total intl minutes": 0, "Total intl calls": 0})
    medium = recommend_customer(profile, 0.2, "Medium Risk", None, prediction_threshold=0.0773)[0]
    high = recommend_customer(profile, 0.8, "High Risk", None, prediction_threshold=0.0773)[0]
    assert high["priority_score"] > medium["priority_score"]
    assert "probability" not in medium["priority"].casefold()
    assert medium["model_context_status"].startswith("Open profile")


def test_portfolio_keeps_customer_mapping_and_suppresses_invalid_or_missing_values():
    valid = _profile()
    invalid = _profile(**{"Customer ID": "C-invalid", "Customer service calls": 99, "International plan": "Yes", "Total intl minutes": 0, "Total intl calls": 0})
    missing = _profile(**{"Customer ID": "C-missing", "Customer service calls": np.nan, "International plan": "Yes", "Total intl minutes": 0, "Total intl calls": 0})
    rows = []
    for customer, probability, risk in ((valid, 0.6, "High Risk"), (invalid, 0.6, "High Risk"), (missing, 0.6, "High Risk")):
        row = {name: customer[name] for name in INPUT_FEATURES}
        row.update({"Customer ID": customer["Customer ID"], "Churn_Probability": probability, "Risk_Category": risk})
        rows.append(row)
    frame = pd.DataFrame(rows)
    misleading_explanation = {customer["Customer ID"]: _explanation() for customer in (valid, invalid, missing)}
    output = recommend_portfolio(frame, prediction_threshold=0.0773, explanations=misleading_explanation)
    assert set(output["customer_id"]) == {"C-1"}
    assert output["customer_id"].nunique() == 1
    assert output["model_context_status"].str.startswith("Local SHAP context attached").all()

    with pytest.raises(ValueError, match="missing required columns"):
        recommend_portfolio(frame.drop(columns=["Total day minutes"]), prediction_threshold=0.0773)
