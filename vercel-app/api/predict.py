from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd

from model_features import FEATURES, INPUT_FEATURES, build_model_matrix


ROOT = Path(__file__).resolve().parent.parent
with (ROOT / "model_metadata.json").open(encoding="utf-8") as file:
    METADATA = json.load(file)
RANGES = METADATA["input_ranges"]
MEDIANS = METADATA["served_model"]["imputation_values"]
MODEL = joblib.load(ROOT / "random_forest_model.joblib")
INTEGER_FIELDS = {
    "Account length", "Number vmail messages", "Total day calls", "Total eve calls",
    "Total night calls", "Total intl calls", "Customer service calls",
}
BINARY_FIELDS = {"International plan", "Voice mail plan"}


def clean_profile(source: dict) -> tuple[dict, list[str]]:
    profile = {}
    missing = []
    for feature in INPUT_FEATURES:
        raw = source.get(feature)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            profile[feature] = None
            missing.append(feature)
            continue
        if feature in BINARY_FIELDS and isinstance(raw, str):
            raw = {"yes": 1, "y": 1, "true": 1, "no": 0, "n": 0, "false": 0}.get(raw.strip().lower())
        try:
            value = float(raw)
        except (TypeError, ValueError):
            profile[feature] = None
            missing.append(feature)
            continue
        if not np.isfinite(value) or value < RANGES[feature]["min"] or value > RANGES[feature]["max"]:
            profile[feature] = None
            missing.append(feature)
            continue
        profile[feature] = float(round(value)) if feature in INTEGER_FIELDS else value
    return profile, missing


def score_profile(source: dict, explain: bool = False) -> dict:
    profile, missing = clean_profile(source)
    row = build_model_matrix(profile, MEDIANS)
    score = float(MODEL.predict_proba(row)[0, 1])
    result = {
        "score": score,
        "label": "Likely to churn" if score >= 0.5 else "Less likely to churn",
        "estimatedFields": missing,
    }
    if explain:
        signals = []
        for feature in INPUT_FEATURES:
            if profile[feature] is None:
                continue
            comparison = dict(profile)
            comparison[feature] = MEDIANS[feature]
            baseline = float(MODEL.predict_proba(build_model_matrix(comparison, MEDIANS))[0, 1])
            signals.append({"name": feature, "change": score - baseline})
        result["signals"] = sorted(signals, key=lambda item: abs(item["change"]), reverse=True)[:5]
    return result


class handler(BaseHTTPRequestHandler):
    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if parse_qs(urlparse(self.path).query).get("view") == ["stats"]:
            importance = {feature: float(MODEL.feature_importances_[FEATURES.index(feature)]) for feature in FEATURES}
            for charge, formula in METADATA["charge_formulas"].items():
                minutes = formula["minutes_feature"]
                importance[minutes] += importance.pop(charge)
            ranked = sorted(
                ({"name": name, "value": importance[name]} for name in INPUT_FEATURES),
                key=lambda item: item["value"], reverse=True,
            )
            self._send(200, {"status": "ready", "model": "Churn Compass", "importance": ranked})
            return
        self._send(200, {"status": "ready", "model": "Churn Compass"})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4_500_000:
                self._send(413, {"error": "Request must contain customer data and stay under 4.5 MB."})
                return
            payload = json.loads(self.rfile.read(length))
            if isinstance(payload.get("profiles"), list):
                profiles = payload["profiles"]
                if len(profiles) > 1000:
                    self._send(400, {"error": "Please send at most 1,000 customer profiles per request."})
                    return
                predictions = [score_profile(profile) for profile in profiles]
                self._send(200, {"predictions": predictions})
                return
            profile = payload.get("profile")
            if not isinstance(profile, dict):
                self._send(400, {"error": "A customer profile is required."})
                return
            self._send(200, score_profile(profile, explain=True))
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            self._send(400, {"error": f"Could not score this request: {error}"})

