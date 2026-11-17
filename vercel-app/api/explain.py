"""Per-customer SHAP explanation endpoint (V2 engine).

Contract: POST {"profile": {13 model inputs}, "thresholds": {optional bands}}
→ signed SHAP contributions from ``ChurnPredictor.explain_single`` against the
fixed V2 serving reference population.

The production binary decision is applied at 0.5 after the explanation, and
the risk band is re-classified with the session bands, so the displayed
semantics match the rest of the application.
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


def explain_customer(payload: Dict[str, Any]) -> Dict[str, Any]:
    profile = payload.get("profile")
    if not isinstance(profile, dict):
        raise EngineError("A customer profile is required.")
    predictor = get_predictor()
    thresholds = thresholds_from_payload(payload, predictor)
    try:
        result = predictor.explain_single(profile)
    except FileNotFoundError as error:
        raise EngineError(
            "The SHAP reference population is missing from the deployment, so "
            "explanations are unavailable. Please contact the application administrator."
        ) from error
    probability = float(result["churn_probability"])
    result["prediction"] = (
        "Likely to churn" if probability >= BINARY_DECISION_THRESHOLD else "Less likely to churn"
    )
    result["risk_category"] = classify_risk([probability], thresholds)[0]
    result["decision_threshold_used"] = BINARY_DECISION_THRESHOLD
    return result


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
        self._send(200, {"status": "ready", "method": "SHAP PermutationExplainer"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4_500_000:
                self._send(413, {"error": "Request must contain customer data and stay under 4.5 MB."})
                return
            payload = json.loads(self.rfile.read(length))
            self._send(200, explain_customer(payload))
        except EngineError as error:
            self._send(400, {"error": error.message})
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            self._send(400, {"error": f"Could not explain this customer: {error}"})
