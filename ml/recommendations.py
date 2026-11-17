"""Transparent, rule-based retention review suggestions for C.R.A.M. V2.

Rules use observed supported inputs only. SHAP is attached as model context, never
treated as evidence that an intervention will work. The portfolio path can rank
rule matches cheaply and attach local SHAP when a customer profile is opened.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from ml.preprocessing import INPUT_FEATURES, CustomerDataValidator


RISK_ORDER = {"High Risk": 2, "Medium Risk": 1, "Low Risk": 0}


def _usable(profile: Mapping[str, Any], explanation: Optional[Mapping[str, Any]], feature: str) -> bool:
    """Only use present, accepted values; exclude genuine missing and coerced values."""
    if feature not in profile or pd.isna(profile[feature]):
        return False
    if explanation is None:
        return True
    status = {
        item.get("feature"): item.get("input_status")
        for item in explanation.get("all_contributions", [])
    }
    return status.get(feature, "provided") == "provided"


def _shap_context(explanation: Optional[Mapping[str, Any]], features: Iterable[str]) -> list[dict[str, Any]]:
    if explanation is None or not explanation.get("_shap_available", True):
        return []
    wanted = set(features)
    rows = []
    for item in explanation.get("all_contributions", []):
        if item.get("feature") not in wanted or item.get("input_status", "provided") != "provided":
            continue
        contribution = float(item.get("contribution", 0.0))
        rows.append({
            "feature": item["feature"],
            "direction": "increases" if contribution > 0 else "decreases" if contribution < 0 else "neutral",
            "contribution": contribution,
            "value": item.get("value"),
        })
    return rows


def _rules(profile: Mapping[str, Any], explanation: Optional[Mapping[str, Any]]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    service = "Customer service calls"
    if _usable(profile, explanation, service):
        try:
            calls = float(profile[service])
        except (TypeError, ValueError):
            calls = 0.0
        if np.isfinite(calls) and calls >= 3:
            matches.append({
                "rule_id": "service_contact_review",
                "rule_name": "Repeated support contact review",
                "title": "Review recent support experience",
                "action": "Have a staff member review recent support contacts and check whether the customer's issue was resolved.",
                "why_triggered": f"The accepted record shows {calls:g} customer service calls, meeting the rule threshold of 3.",
                "triggering_characteristics": {service: calls},
                "related_shap_features": [service],
                "objective": "Identify and address possible unresolved support friction.",
                "rule_strength": 3,
                "limitations": "Call count does not reveal the reason, quality, or outcome of contacts. It does not establish dissatisfaction or service failure.",
            })

    plan, minutes, calls = "International plan", "Total intl minutes", "Total intl calls"
    if _usable(profile, explanation, plan) and str(profile[plan]).strip().casefold() in {"no", "0", "0.0", "false"}:
        evidence = {}
        related = [plan]
        if _usable(profile, explanation, minutes):
            try:
                value = float(profile[minutes])
                if np.isfinite(value) and value >= 10:
                    evidence[minutes] = value
                    related.append(minutes)
            except (TypeError, ValueError):
                pass
        if _usable(profile, explanation, calls):
            try:
                value = float(profile[calls])
                if np.isfinite(value) and value >= 4:
                    evidence[calls] = value
                    related.append(calls)
            except (TypeError, ValueError):
                pass
        if evidence:
            evidence[plan] = profile[plan]
            matches.append({
                "rule_id": "international_usage_fit_review",
                "rule_name": "International usage and plan fit review",
                "title": "Check international service fit",
                "action": "Ask a staff member to review whether the current international service options fit the customer's recorded usage.",
                "why_triggered": "The accepted record has no international plan and shows elevated international usage (at least 10 minutes or 4 calls).",
                "triggering_characteristics": evidence,
                "related_shap_features": related,
                "objective": "Review whether available service options match the customer's usage pattern.",
                "rule_strength": 2,
                "limitations": "Usage is not a bill or proof of plan mismatch; prices, customer preferences, and available plans are not in the model inputs.",
            })
    return matches


def recommend_customer(
    customer: Mapping[str, Any],
    churn_probability: float,
    risk_category: str,
    explanation: Optional[Mapping[str, Any]],
    *,
    prediction_threshold: float,
    _input_status: Optional[Mapping[str, str]] = None,
) -> list[dict[str, Any]]:
    """Return deterministic, traceable recommendations for one scored customer."""
    probability = float(churn_probability)
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("churn_probability must be finite and within [0, 1]")
    if risk_category not in RISK_ORDER:
        raise ValueError(f"Unsupported risk category: {risk_category!r}")
    if not 0.0 <= prediction_threshold <= 1.0:
        raise ValueError("prediction_threshold must be within [0, 1]")
    if probability < prediction_threshold or risk_category == "Low Risk":
        return []

    if _input_status is None:
        raw = pd.DataFrame([dict(customer)]).reindex(columns=INPUT_FEATURES)
        cleaned = CustomerDataValidator().validate(raw, remove_duplicates=False).cleaned_df
        input_status = {}
        for feature in INPUT_FEATURES:
            if pd.isna(raw.iloc[0][feature]):
                input_status[feature] = "genuinely_missing_imputed"
            elif pd.isna(cleaned.iloc[0][feature]):
                input_status[feature] = "invalid_or_out_of_range_coerced_then_imputed"
            else:
                input_status[feature] = "provided"
    else:
        input_status = dict(_input_status)

    local_explanation = dict(explanation or {})
    shap_by_feature = {
        item.get("feature"): item
        for item in (explanation or {}).get("all_contributions", [])
    }
    local_explanation["all_contributions"] = [
        {
            **shap_by_feature.get(feature, {}),
            "feature": feature,
            "input_status": input_status.get(feature, "genuinely_missing_imputed"),
        }
        for feature in INPUT_FEATURES
    ]
    local_explanation["_shap_available"] = bool(
        explanation is not None and explanation.get("_shap_available", True)
    )

    results = []
    for rule in _rules(customer, local_explanation):
        contexts = _shap_context(local_explanation, rule["related_shap_features"])
        positive_pp = sum(max(0.0, row["contribution"] * 100.0) for row in contexts)
        # Heuristic ordering only: rule strength + risk-band adjustment + a
        # capped positive SHAP alignment. This is not intervention efficacy.
        score = min(100, rule["rule_strength"] * 20 + RISK_ORDER[risk_category] * 10 + min(10, positive_pp))
        priority = "High" if score >= 85 else "Standard"
        results.append({
            **rule,
            "customer_id": str(customer.get("Customer ID", customer.get("Customer_ID", ""))),
            "churn_probability": probability,
            "risk_category": risk_category,
            "priority": priority,
            "priority_score": round(float(score), 2),
            "model_context": contexts,
            "model_context_status": "Local SHAP context attached" if contexts else "Open profile to compute local SHAP context" if not local_explanation["_shap_available"] else "No reliable local SHAP context for the rule features",
            "human_approval_required": True,
            "causal_claim": False,
        })
    results.sort(key=lambda item: (-item["priority_score"], item["rule_id"]))
    return results


def recommend_portfolio(
    customers: pd.DataFrame,
    *,
    prediction_threshold: float,
    explanations: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> pd.DataFrame:
    """Generate customer-linked portfolio suggestions; SHAP can be attached lazily."""
    if not isinstance(customers, pd.DataFrame):
        raise TypeError("customers must be a pandas DataFrame")
    required = {"Customer ID", "Churn_Probability", "Risk_Category", *INPUT_FEATURES}
    missing = sorted(required - set(customers.columns))
    if missing:
        raise ValueError("customers is missing required columns: " + ", ".join(missing))
    validated = CustomerDataValidator().validate(customers, remove_duplicates=False).cleaned_df
    rows = []
    records = customers.to_dict(orient="records")
    validated_records = validated.to_dict(orient="records")
    for customer, validated_row in zip(records, validated_records):
        customer_id = str(customer["Customer ID"])
        explanation = (explanations or {}).get(customer_id)
        statuses = []
        for feature in INPUT_FEATURES:
            original, cleaned = customer[feature], validated_row[feature]
            if pd.isna(original):
                status = "genuinely_missing_imputed"
            elif pd.isna(cleaned):
                status = "invalid_or_out_of_range_coerced_then_imputed"
            else:
                status = "provided"
            statuses.append({"feature": feature, "input_status": status})
        for recommendation in recommend_customer(
            customer, customer["Churn_Probability"], customer["Risk_Category"], explanation,
            prediction_threshold=prediction_threshold,
            _input_status={item["feature"]: item["input_status"] for item in statuses},
        ):
            rows.append(recommendation)
    columns = [
        "customer_id", "rule_id", "rule_name", "title", "priority", "priority_score",
        "churn_probability", "risk_category", "action", "why_triggered",
        "triggering_characteristics", "model_context", "model_context_status", "objective",
        "human_approval_required", "limitations",
    ]
    return pd.DataFrame(rows, columns=columns)
