"""Portfolio analysis endpoint.

Two modes:

1. POST {"csv": "<csv text>", "thresholds": {optional bands}} — validate the
   uploaded dataset with the Phase 7 precheck, score it with the V2 engine via
   ``analyze_portfolio``, and return the full scored table (compact columns +
   rows), summary, warnings, schema notes, group rates, and band counts.
2. POST {"rows": [scored rows], "thresholds": {bands}} — re-apply session risk
   bands to the client's current view and recompute summary/counts/rates
   without resending the table or re-scoring.

A rejected upload is never analyzed; the client keeps its previous dataset.

Production semantics: the binary customer decision is applied at the
established 0.5 cutoff (overriding the engine's stored operating point, which
the V2 metadata reserves for Model Performance); risk bands are the separate
session-adjustable classification.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict

import numpy as np

from _engine import (
    BINARY_DECISION_THRESHOLD,
    EngineError,
    MAX_UPLOAD_ROWS,
    get_predictor,
    group_rates,
    json_safe,
    frame_to_columns_rows,
    summarize_schema_issues,
    thresholds_from_payload,
    validate_upload_text,
)
from ml.risk_intelligence import (
    portfolio_summary,
    RiskThresholds,
    analyze_portfolio,
)


def _band_counts(frame, thresholds: RiskThresholds) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in ("High Risk", "Medium Risk", "Low Risk"):
        counts[label] = int((frame["Risk_Category"] == label).sum())
    return counts


def _production_decision(frame) -> None:
    """Apply the production 0.5 binary cutoff to a freshly scored portfolio."""
    probabilities = frame["Churn_Probability"]
    frame["Prediction"] = np.where(
        probabilities >= BINARY_DECISION_THRESHOLD, "Likely to churn", "Less likely to churn"
    )
    frame["Predicted_Churner"] = probabilities >= BINARY_DECISION_THRESHOLD


def analyze_csv(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = payload.get("csv")
    if not isinstance(text, str):
        raise EngineError("A customer CSV is required.")
    ok, message, frame = validate_upload_text(text)
    if not ok:
        return {"rejected": True, "reason": message}
    if len(frame) > MAX_UPLOAD_ROWS:
        raise EngineError(
            f"This application accepts up to {MAX_UPLOAD_ROWS:,} customers per dataset. "
            f"The file contains {len(frame):,} rows."
        )
    predictor = get_predictor()
    thresholds = thresholds_from_payload(payload, predictor)
    try:
        portfolio = analyze_portfolio(frame, predictor, thresholds)
    except (ValueError, TypeError) as error:
        raise EngineError(str(error)) from error
    _production_decision(portfolio.customers)
    portfolio.prediction_threshold = BINARY_DECISION_THRESHOLD
    summary = portfolio_summary(portfolio.customers)
    response = {
        "rejected": False,
        "columns_rows": frame_to_columns_rows(portfolio.customers),
        "summary": summary,
        "warnings": portfolio.warnings,
        "notes": summarize_schema_issues(frame),
        "label_column_present": portfolio.label_column_present,
        "id_source_column": portfolio.id_source_column,
        "prediction_threshold": BINARY_DECISION_THRESHOLD,
        "risk_thresholds": {
            "medium_threshold": thresholds.medium_threshold,
            "high_threshold": thresholds.high_threshold,
        },
        "band_counts": _band_counts(portfolio.customers, thresholds),
        "group_rates": group_rates(portfolio.customers),
    }
    if len(portfolio.customers) > MAX_UPLOAD_ROWS:
        raise EngineError("The analyzed dataset exceeded the response limit.")
    return response


def reband_rows(payload: Dict[str, Any]) -> Dict[str, Any]:
    from _engine import reconstruct_portfolio

    rows = payload.get("rows")
    predictor = get_predictor()
    thresholds = thresholds_from_payload(payload, predictor)
    portfolio = reconstruct_portfolio(rows, thresholds, BINARY_DECISION_THRESHOLD)
    customers = portfolio.customers
    if "Risk_Category" in customers.columns:
        from ml.risk_intelligence import classify_risk

        customers["Risk_Category"] = classify_risk(
            customers["Churn_Probability"].tolist(), thresholds
        )
    summary = portfolio_summary(customers)
    return {
        "summary": summary,
        "prediction_threshold": BINARY_DECISION_THRESHOLD,
        "risk_thresholds": {
            "medium_threshold": thresholds.medium_threshold,
            "high_threshold": thresholds.high_threshold,
        },
        "band_counts": _band_counts(customers, thresholds),
        "group_rates": group_rates(customers),
    }


def handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    if payload.get("rows") is not None:
        return reband_rows(payload)
    return analyze_csv(payload)


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(json_safe(payload), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send(200, {
            "status": "ready",
            "limits": {"max_file_mb": 4, "max_rows": MAX_UPLOAD_ROWS},
        })

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4_500_000:
                self._send(413, {"error": "The dataset could not be received. Requests must stay under 4.5 MB."})
                return
            payload = json.loads(self.rfile.read(length))
            self._send(200, handle(payload))
        except EngineError as error:
            self._send(400, {"error": error.message})
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            self._send(400, {"error": f"Could not analyze this dataset: {error}"})
