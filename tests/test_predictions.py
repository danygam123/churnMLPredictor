"""Unit tests for C.R.A.M. V2 prediction and inference engine."""

import pandas as pd
import pytest
from ml.predict import ChurnPredictor


def test_single_customer_prediction():
    predictor = ChurnPredictor()
    sample_customer = {
        "Account length": 128,
        "International plan": "no",
        "Voice mail plan": "yes",
        "Number vmail messages": 25,
        "Total day minutes": 265.1,
        "Total day calls": 110,
        "Total eve minutes": 197.4,
        "Total eve calls": 99,
        "Total night minutes": 244.7,
        "Total night calls": 91,
        "Total intl minutes": 10.0,
        "Total intl calls": 3,
        "Customer service calls": 1,
    }
    result = predictor.predict_single(sample_customer)
    assert "churn_probability" in result
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_category"] in ["High Risk", "Medium Risk", "Low Risk"]
    assert result["prediction"] in ["Likely to churn", "Less likely to churn"]
    assert result["estimated_fields_count"] == 0


def test_batch_customer_prediction():
    predictor = ChurnPredictor()
    df = pd.DataFrame([
        {
            "Account length": 50,
            "International plan": "yes",
            "Voice mail plan": "no",
            "Number vmail messages": 0,
            "Total day minutes": 320.0,
            "Total day calls": 100,
            "Total eve minutes": 210.0,
            "Total eve calls": 100,
            "Total night minutes": 180.0,
            "Total night calls": 100,
            "Total intl minutes": 15.0,
            "Total intl calls": 5,
            "Customer service calls": 5,
        },
        {
            "Account length": 150,
            "International plan": "no",
            "Voice mail plan": "yes",
            "Number vmail messages": 30,
            "Total day minutes": 120.0,
            "Total day calls": 80,
            "Total eve minutes": 140.0,
            "Total eve calls": 80,
            "Total night minutes": 150.0,
            "Total night calls": 80,
            "Total intl minutes": 8.0,
            "Total intl calls": 2,
            "Customer service calls": 0,
        },
    ])
    results, warnings = predictor.predict_batch(df)
    assert len(results) == 2
    assert "Churn_Probability" in results.columns
    assert "Risk_Category" in results.columns
    assert "Prediction" in results.columns
    # Row 0 has high usage + 5 customer service calls + international plan, probability should be higher than row 1
    assert results["Churn_Probability"].iloc[0] > results["Churn_Probability"].iloc[1]
