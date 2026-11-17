"""What-if scenario simulation endpoint (V2 engine).

Contract: POST {"profile": {baseline 13 inputs}, "scenarios": [{"name",
"changes": {input: value}}], "thresholds": {optional bands}} → one
``simulate_scenario`` evaluation per scenario, evaluated on a fresh copy of the
baseline (scenario state is never shared or mutated).

The production binary decision is applied at the 0.5 cutoff and risk bands are
the separate session classification.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict, List

from _engine import (
    BINARY_DECISION_THRESHOLD,
    EngineError,
    get_predictor,
    json_safe,
    thresholds_from_payload,
)
from ml.risk_intelligence import classify_risk
from ml.simulation import simulate_scenario


def simulate(payload: Dict[str, Any]) -> Dict[str, Any]:
    profile = payload.get("profile")
    scenarios = payload.get("scenarios")
    if not isinstance(profile, dict):
        raise EngineError("A baseline customer profile is required.")
    if not isinstance(scenarios, list) or not scenarios or len(scenarios) > 3:
        raise EngineError("Provide one to three scenarios to compare.")
    predictor = get_predictor()
    thresholds = thresholds_from_payload(payload, predictor)
    results: List[Dict[str, Any]] = []
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise EngineError("Each scenario must be an object with a name and its input changes.")
        name = str(scenario.get("name") or "Scenario")
        changes = scenario.get("changes")
        if not isinstance(changes, dict):
            raise EngineError(f"Scenario “{name}” needs a set of input changes.")
        try:
            result = simulate_scenario(
                predictor, profile, changes, threshold=BINARY_DECISION_THRESHOLD
            )
        except (ValueError, TypeError) as error:
            raise EngineError(f"Scenario “{name}” could not be simulated: {error}") from error
        record = asdict(result)
        probability = record["simulated_probability"]
        record["simulated_risk_category"] = classify_risk([probability], thresholds)[0]
        record["baseline_risk_category"] = classify_risk(
            [record["original_probability"]], thresholds
        )[0]
        results.append({"name": name, **record})
    return {
        "results": results,
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
        self._send(200, {"status": "ready", "engine": "what-if scenario simulation"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4_500_000:
                self._send(413, {"error": "Request must stay under 4.5 MB."})
                return
            payload = json.loads(self.rfile.read(length))
            self._send(200, simulate(payload))
        except EngineError as error:
            self._send(400, {"error": error.message})
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            self._send(400, {"error": f"Could not run the simulation: {error}"})
