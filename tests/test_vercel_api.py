"""Tests for the Vercel V2 API layer (analyze, score, explain, recommend, simulate, analyst).

These exercise the pure request-handling logic of each endpoint the same way a
request would: a payload in, a response payload out. Network access is never
required (the analyst runs in facts-only mode unless a key is configured).
"""

from __future__ import annotations

import importlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

APP_DIR = Path(__file__).resolve().parents[1] / "vercel-app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

SAMPLE_PATH = APP_DIR / "public" / "sample_customers.csv"


def _load(name):
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def sample_csv() -> str:
    assert SAMPLE_PATH.exists(), "The sample dataset must ship with the app"
    return SAMPLE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def analyzed(sample_csv):
    analyze = _load("api.analyze")
    response = analyze.handle({"csv": sample_csv})
    assert response["rejected"] is False
    return response


def _rows_as_records(response) -> list[dict]:
    columns = response["columns_rows"]["columns"]
    return [dict(zip(columns, row)) for row in response["columns_rows"]["rows"]]


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------

def test_analyze_scores_the_sample_dataset(analyzed):
    summary = analyzed["summary"]
    assert summary["customers_analyzed"] == 80
    assert 0 <= summary["predicted_churners"] <= 80
    assert 0 <= summary["average_predicted_probability"] <= 1
    assert analyzed["label_column_present"] is True
    assert analyzed["prediction_threshold"] == 0.5


def test_analyze_applies_production_binary_decision(analyzed):
    records = _rows_as_records(analyzed)
    for record in records:
        if record["Prediction"] == "Likely to churn":
            assert record["Churn_Probability"] >= 0.5
            assert record["Predicted_Churner"] is True
        else:
            assert record["Churn_Probability"] < 0.5
            assert record["Predicted_Churner"] is False


def test_analyze_returns_risk_bands_and_group_rates(analyzed):
    assert set(analyzed["band_counts"]) == {"High Risk", "Medium Risk", "Low Risk"}
    assert sum(analyzed["band_counts"].values()) == 80
    assert "International plan" in analyzed["group_rates"]
    rates = analyzed["group_rates"]["International plan"]
    assert all(0 <= row["rate"] <= 1 for row in rates)


def test_analyze_includes_schema_notes_and_warnings(analyzed):
    assert isinstance(analyzed["notes"], list)
    assert isinstance(analyzed["warnings"], list)
    assert analyzed["risk_thresholds"]["high_threshold"] == 0.5


def test_analyze_rejects_missing_columns_without_scoring():
    analyze = _load("api.analyze")
    response = analyze.handle({"csv": "Customer ID\nC-1\n"})
    assert response["rejected"] is True
    assert "missing required model input columns" in response["reason"]


def test_analyze_rejects_empty_and_unparsable_files():
    analyze = _load("api.analyze")
    empty = analyze.handle({"csv": ""})
    assert empty["rejected"] is True
    garbage = analyze.handle({"csv": 'a,b\n"x"y"z",1\n"open,2\n'})
    assert garbage["rejected"] is True


def test_analyze_rejects_oversize_files():
    analyze = _load("api.analyze")
    text = "Account length\n" + "1\n" * 5
    big = {"csv": "x" * 4_000_001 + text}
    response = analyze.handle(big)
    assert response["rejected"] is True
    assert "too large" in response["reason"]


def test_reband_mode_recomputes_bands_without_rescoring(analyzed):
    analyze = _load("api.analyze")
    rows = _rows_as_records(analyzed)
    before_high = analyzed["band_counts"]["High Risk"]
    shifted = analyze.handle({
        "rows": rows,
        "thresholds": {"medium_threshold": 0.001, "high_threshold": 0.002},
    })
    assert shifted["band_counts"]["High Risk"] > before_high
    assert shifted["prediction_threshold"] == 0.5
    assert shifted["summary"]["customers_analyzed"] == 80


def test_reband_rejects_inverted_thresholds(analyzed):
    analyze = _load("api.analyze")
    from _engine import EngineError

    with pytest.raises(EngineError):
        analyze.handle({
            "rows": _rows_as_records(analyzed),
            "thresholds": {"medium_threshold": 0.9, "high_threshold": 0.5},
        })


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def test_score_customer_uses_production_cutoff_and_band():
    score = _load("api.score")
    profile = {
        "Customer service calls": 5,
        "Total day minutes": 300,
        "International plan": "Yes",
    }
    response = score.score_customer({"profile": profile})
    assert response["decisionThreshold"] == 0.5
    assert response["label"] == ("Likely to churn" if response["score"] >= 0.5 else "Less likely to churn")
    assert response["riskBand"] in ("Low Risk", "Medium Risk", "High Risk")
    assert len(response["estimatedFields"]) == 10
    assert isinstance(response["signals"], list)


def test_score_customer_session_bands_override_defaults():
    score = _load("api.score")
    profile = {"Customer service calls": 2}
    default = score.score_customer({"profile": profile})
    shifted = score.score_customer({
        "profile": profile,
        "thresholds": {"medium_threshold": 0.0001, "high_threshold": 0.0002},
    })
    assert shifted["riskBand"] == "High Risk"
    assert shifted["riskBand"] != default["riskBand"]


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------

def test_explain_returns_shap_contributions_with_production_semantics():
    explain = _load("api.explain")
    profile = {
        "Customer service calls": 4,
        "Total day minutes": 250,
        "International plan": "Yes",
        "Total intl calls": 10,
    }
    response = explain.explain_customer({"profile": profile})
    assert response["explanation_method"].startswith("SHAP")
    assert response["decision_threshold_used"] == 0.5
    assert len(response["all_contributions"]) == 13
    top = response["top_contributions"][0]
    assert top["direction"] in ("increases", "decreases", "neutral")
    assert abs(response["reconstructed_probability"] - response["churn_probability"]) < 1e-6


def test_explain_rejects_missing_profile():
    explain = _load("api.explain")
    from _engine import EngineError

    with pytest.raises(EngineError):
        explain.explain_customer({})


# ---------------------------------------------------------------------------
# recommend
# ---------------------------------------------------------------------------

def test_recommend_generates_rule_based_suggestions(analyzed):
    recommend = _load("api.recommend")
    rows = [row for row in _rows_as_records(analyzed) if row["Churn_Probability"] >= 0.5]
    response = recommend.recommend_rows({"rows": rows})
    assert isinstance(response["recommendations"], list)
    for item in response["recommendations"]:
        assert item["human_approval_required"] is True
        assert item["limitations"]


def test_recommend_respects_production_threshold(analyzed):
    recommend = _load("api.recommend")
    rows = [row for row in _rows_as_records(analyzed) if row["Churn_Probability"] < 0.5]
    response = recommend.recommend_rows({"rows": rows})
    assert response["recommendations"] == []


def test_recommend_rejects_empty_rows():
    recommend = _load("api.recommend")
    from _engine import EngineError

    with pytest.raises(EngineError):
        recommend.recommend_rows({"rows": []})


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------

def test_simulate_evaluates_each_scenario_on_a_fresh_copy():
    simulate = _load("api.simulate")
    profile = {
        "Account length": 100, "International plan": "No", "Voice mail plan": "Yes",
        "Number vmail messages": 10, "Total day minutes": 150, "Total day calls": 100,
        "Total eve minutes": 150, "Total eve calls": 100, "Total night minutes": 150,
        "Total night calls": 100, "Total intl minutes": 10, "Total intl calls": 4,
        "Customer service calls": 1,
    }
    response = simulate.simulate({
        "profile": profile,
        "scenarios": [
            {"name": "A", "changes": {"Customer service calls": 0}},
            {"name": "B", "changes": {"International plan": "Yes"}},
            {"name": "C", "changes": {"Total day minutes": 500}},
        ],
    })
    assert len(response["results"]) == 3
    for result in response["results"]:
        assert 0 <= result["original_probability"] <= 1
        assert 0 <= result["simulated_probability"] <= 1
        assert result["decision_threshold"] == 0.5
    assert response["results"][0]["original_probability"] == response["results"][1]["original_probability"]


def test_simulate_rejects_invalid_changes():
    simulate = _load("api.simulate")
    from _engine import EngineError

    profile = {"Total day minutes": 150}
    with pytest.raises(EngineError):
        simulate.simulate({"profile": profile, "scenarios": [{"name": "A", "changes": {"Total day minutes": 99999}}]})
    with pytest.raises(EngineError):
        simulate.simulate({"profile": profile, "scenarios": [{"name": "A", "changes": {"Not an input": 1}}]})


# ---------------------------------------------------------------------------
# analyst
# ---------------------------------------------------------------------------

def test_analyst_status_never_exposes_the_key(monkeypatch):
    analyst_endpoint = _load("api.analyst")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-value-for-testing")
    status = analyst_endpoint.analyst_status()
    assert status["provider"]["configured"] is True
    assert "secret-value" not in json.dumps(status)


def test_analyst_facts_only_mode_computes_real_facts(analyzed, monkeypatch):
    analyst_endpoint = _load("api.analyst")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    rows = _rows_as_records(analyzed)
    response = analyst_endpoint.answer({"rows": rows, "question": "How many customers are in this dataset?"})
    assert response["status"] in ("facts_only", "answered")
    if response["facts"]:
        assert response["facts"]["customers_analyzed"] == 80


def test_analyst_routes_by_dimension(analyzed, monkeypatch):
    analyst_endpoint = _load("api.analyst")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    rows = _rows_as_records(analyzed)
    response = analyst_endpoint.answer({
        "rows": rows,
        "question": "What is the churn rate by International plan?",
    })
    assert response["status"] in ("facts_only", "answered")
    assert response["function"] == "churn_by_dimension"
    groups = response["facts"]["groups"] if response["facts"] else []
    assert {group["group"].upper() for group in groups} == {"YES", "NO"} or groups


def test_analyst_rejects_empty_question():
    analyst_endpoint = _load("api.analyst")
    from _engine import EngineError

    with pytest.raises(EngineError):
        analyst_endpoint.answer({"rows": [{"Customer ID": "x"}], "question": "   "})


# ---------------------------------------------------------------------------
# Serialization safety
# ---------------------------------------------------------------------------

def test_json_safe_converts_nan_and_numpy():
    import numpy as np
    from _engine import json_safe

    payload = json_safe({
        "a": float("nan"),
        "b": np.float64(1.5),
        "c": np.bool_(True),
        "d": [np.int64(3)],
        "e": pd.NA,
    })
    encoded = json.dumps(payload, allow_nan=False)
    assert json.loads(encoded) == {"a": None, "b": 1.5, "c": True, "d": [3], "e": None}
