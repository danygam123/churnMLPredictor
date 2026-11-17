"""Single-customer V2 scoring endpoint.

Contract: POST {"profile": {13 model inputs}, "thresholds": {optional session
risk bands}} → score, binary label at the production 0.5 cutoff, V2 risk band,
estimated fields, and one-at-a-time sensitivity signals against typical values.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict

from _engine import (
    BINARY_DECISION_THRESHOLD,
    EngineError,
    get_predictor,
    json_safe,
    thresholds_from_payload,
)
from ml.risk_intelligence import classify_risk


def score_customer(payload: Dict[str, Any]) -> Dict[str, Any]:
    profile = payload.get("profile")
    if not isinstance(profile, dict):
        raise EngineError("A customer profile is required.")
    predictor = get_predictor()
    thresholds = thresholds_from_payload(payload, predictor)
    result = predictor.predict_single(profile, threshold=BINARY_DECISION_THRESHOLD)
    probability = float(result["churn_probability"])
    return {
        "score": probability,
        "label": result["prediction"],
        "estimatedFields": result["estimated_fields"],
        "riskBand": classify_risk([probability], thresholds)[0],
        "riskThresholds": {
            "medium_threshold": thresholds.medium_threshold,
            "high_threshold": thresholds.high_threshold,
        },
        "decisionThreshold": BINARY_DECISION_THRESHOLD,
        "signals": [
            {"name": signal["feature"], "change": signal["score_delta_vs_typical"]}
            for signal in result.get("top_sensitivity_signals", [])
        ],
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
        self._send(200, {"status": "ready", "engine": "C.R.A.M. V2"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4_500_000:
                self._send(413, {"error": "Request must contain customer data and stay under 4.5 MB."})
                return
            payload = json.loads(self.rfile.read(length))
            self._send(200, score_customer(payload))
        except EngineError as error:
            self._send(400, {"error": error.message})
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            self._send(400, {"error": f"Could not score this request: {error}"})
