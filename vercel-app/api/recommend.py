"""Retention recommendation endpoint (rule-based, V2 engine).

Contract: POST {"rows": [scored rows of the current filtered view],
"thresholds": {optional bands}} → customer-linked review suggestions from
``ml.recommendations.recommend_portfolio``.

These are transparent, rule-based review suggestions — not experimentally
proven interventions and not guaranteed churn-reduction strategies.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict, List

from _engine import (
    BINARY_DECISION_THRESHOLD,
    EngineError,
    get_predictor,
    json_safe,
    reconstruct_portfolio,
    thresholds_from_payload,
)
from ml.recommendations import recommend_portfolio


def recommend_rows(payload: Dict[str, Any]) -> Dict[str, Any]:
    rows = payload.get("rows")
    predictor = get_predictor()
    thresholds = thresholds_from_payload(payload, predictor)
    portfolio = reconstruct_portfolio(rows, thresholds, BINARY_DECISION_THRESHOLD)
    try:
        recommendations = recommend_portfolio(
            portfolio.customers,
            prediction_threshold=BINARY_DECISION_THRESHOLD,
        )
    except (ValueError, TypeError) as error:
        raise EngineError(str(error)) from error
    records: List[Dict[str, Any]] = recommendations.to_dict(orient="records")
    return {
        "recommendations": json_safe(records),
        "customer_count": int(recommendations["customer_id"].nunique()) if len(recommendations) else 0,
        "prediction_threshold": BINARY_DECISION_THRESHOLD,
        "risk_thresholds": {
            "medium_threshold": thresholds.medium_threshold,
            "high_threshold": thresholds.high_threshold,
        },
    }


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
        self._send(200, {"status": "ready", "engine": "rule-based retention review suggestions"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4_500_000:
                self._send(413, {"error": "The dataset could not be received. Requests must stay under 4.5 MB."})
                return
            payload = json.loads(self.rfile.read(length))
            self._send(200, recommend_rows(payload))
        except EngineError as error:
            self._send(400, {"error": error.message})
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            self._send(400, {"error": f"Could not build recommendations: {error}"})
