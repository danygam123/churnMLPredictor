"""Tests for Phase 3 portfolio scoring, risk bands, and summary calculations."""

import numpy as np
import pandas as pd
import pytest

from ml.predict import ChurnPredictor
from ml.preprocessing import INPUT_FEATURES
from ml.risk_intelligence import (
    RiskThresholds,
    analyze_portfolio,
    classify_risk,
    choose_dimension_column,
    churn_rate_by_group,
    derive_usage_charge_proxy,
    portfolio_summary,
)


def _customer(offset=0):
    return {
        "Account length": 100 + offset,
        "International plan": "yes" if offset % 2 else "no",
        "Voice mail plan": "yes",
        "Number vmail messages": 20 + offset,
        "Total day minutes": 180.0 + offset * 5,
        "Total day calls": 100,
        "Total eve minutes": 160.0,
        "Total eve calls": 90,
        "Total night minutes": 200.0,
        "Total night calls": 100,
        "Total intl minutes": 8.0,
        "Total intl calls": 3,
        "Customer service calls": offset % 4,
    }


def test_risk_thresholds_use_validation_anchors_and_separate_bands():
    thresholds = RiskThresholds(medium_threshold=0.08, high_threshold=0.50)
    metadata_thresholds = RiskThresholds.from_metadata({"optimal_threshold": 0.08, "default_threshold": 0.50})
    assert metadata_thresholds == thresholds
    assert classify_risk([0.079, 0.08, 0.4999, 0.50, 1.0], thresholds) == [
        "Low Risk", "Medium Risk", "Medium Risk", "High Risk", "High Risk"
    ]
    assert thresholds.medium_threshold < thresholds.high_threshold
    with pytest.raises(ValueError, match="0 <= medium < high <= 1"):
        RiskThresholds(medium_threshold=0.50, high_threshold=0.08)
    with pytest.raises(ValueError):
        RiskThresholds(medium_threshold=-0.1, high_threshold=0.5)

    predictor = ChurnPredictor()
    assert predictor.categorize_risk(0.07) == "Low Risk"
    assert predictor.categorize_risk(0.08) == "Medium Risk"
    assert predictor.categorize_risk(0.50) == "High Risk"
    assert predictor.optimal_threshold == pytest.approx(0.0772727273)


def test_portfolio_scores_labels_missing_and_invalid_inputs_consistently():
    predictor = ChurnPredictor()
    rows = []
    for i, label in enumerate((True, False, None, "yes")):
        row = _customer(i)
        row["Customer ID"] = "same" if i < 2 else None if i == 2 else "four"
        row["Churn"] = label
        rows.append(row)
    rows[1]["Total day minutes"] = "not-a-number"
    rows[0]["Total day minutes"] = None
    rows[2]["International plan"] = "maybe"
    rows[3]["Total night calls"] = 9999
    data = pd.DataFrame(rows)

    result = analyze_portfolio(data, predictor)
    scored = result.customers
    direct_cleaned = predictor.validator.validate(data, remove_duplicates=False).cleaned_df
    direct = predictor.model.predict_proba(direct_cleaned[INPUT_FEATURES])[:, 1]
    summary = portfolio_summary(scored)

    assert len(scored) == 4
    assert np.allclose(scored["Churn_Probability"], direct)
    assert scored["Customer ID"].tolist() == ["same", "same (row 2)", "Record 000003", "four"]
    assert scored["Rejected_Input_Count"].tolist() == [0, 1, 1, 1]
    assert scored["Genuinely_Missing_Input_Count"].tolist() == [1, 0, 0, 0]
    assert result.label_column_present
    assert summary["customers_analyzed"] == 4
    assert summary["predicted_churners"] == int((direct >= predictor.optimal_threshold).sum())
    assert summary["high_risk_customers"] == int((scored["Risk_Category"] == "High Risk").sum())
    assert np.isclose(summary["average_predicted_probability"], direct.mean())
    assert np.isclose(summary["observed_churn_rate"], 2 / 3)
    assert summary["observed_label_count"] == 3
    assert any("invalid or out-of-range" in warning for warning in result.warnings)
    assert scored["Risk_Category"].tolist() == [predictor.categorize_risk(p) for p in direct]
    profile = {feature: data.iloc[1][feature] for feature in INPUT_FEATURES}
    explanation = predictor.explain_single(profile)
    assert np.isclose(explanation["churn_probability"], scored.iloc[1]["Churn_Probability"])
    assert explanation["risk_category"] == scored.iloc[1]["Risk_Category"]


def test_upload_schema_and_observed_labels_have_actionable_errors():
    predictor = ChurnPredictor()
    with pytest.raises(ValueError, match="missing required model input columns"):
        analyze_portfolio(pd.DataFrame([{INPUT_FEATURES[0]: 100}]), predictor)
    with pytest.raises(ValueError, match="no data rows"):
        analyze_portfolio(pd.DataFrame(columns=INPUT_FEATURES), predictor)

    row = _customer()
    row["Churn"] = "possibly"
    with pytest.raises(ValueError, match="CSV data row 2.*Use true/false, yes/no, or 1/0"):
        analyze_portfolio(pd.DataFrame([row]), predictor)


def test_usage_charge_proxy_and_observed_group_rates():
    customers = pd.DataFrame([
        {**_customer(0), "Observed_Churn": True, "Churn_Probability": 0.2, "Plan": "A"},
        {**_customer(1), "Observed_Churn": False, "Churn_Probability": 0.1, "Plan": "A"},
        {**_customer(2), "Observed_Churn": True, "Churn_Probability": 0.3, "Plan": "B"},
    ])
    charges = derive_usage_charge_proxy(customers)
    expected = round(180 * 0.17 + 160 * 0.085 + 200 * 0.045 + 8 * 0.27, 2)
    assert charges.iloc[0] == expected
    grouped = churn_rate_by_group(customers, "Plan").set_index("Plan")
    assert grouped.loc["A", "rate"] == 0.5
    assert grouped.loc["A", "customers"] == 2
    assert choose_dimension_column(customers, ("plan",)) == "Plan"


def test_portfolio_without_observed_labels_does_not_invent_observed_rate():
    predictor = ChurnPredictor()
    result = analyze_portfolio(pd.DataFrame([_customer(0), _customer(1)]), predictor)
    summary = portfolio_summary(result.customers)
    assert not result.label_column_present
    assert summary["observed_churn_rate"] is None
    assert summary["observed_label_count"] == 0
    assert "Observed_Churn" not in result.customers
