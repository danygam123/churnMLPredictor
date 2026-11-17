"""Restricted AI Analyst endpoint (V2 engine, Phase 6 architecture).

Contract:
- GET → provider status only: whether a language-model key is configured.
  The credential itself is never returned, logged, or displayed.
- POST {"rows": [scored rows of the current filtered view],
  "thresholds": {bands}, "question": "<text>"} → plan (whitelisted function
  only), compute the facts in Python, and (when configured) narrate with
  grounding validation. Without a key the deterministic planner returns
  verified facts directly (status ``facts_only``).

The Gemini API key is read from the environment (Vercel environment variable
``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``) on the server side only.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler
from typing import Any, Dict

from _engine import (
    BINARY_DECISION_THRESHOLD,
    EngineError,
    get_predictor,
    json_safe,
    reconstruct_portfolio,
    thresholds_from_payload,
)
from ml.analyst import (
    AnalystError,
    MAX_QUESTION_CHARS,
    answer_question,
    get_api_key,
)


def analyst_status() -> Dict[str, Any]:
    predictor_status = "ready"
    try:
        get_predictor()
    except EngineError as error:
        predictor_status = error.message
    return {
        "status": "ready",
        "provider": {"configured": get_api_key() is not None},
        "model": predictor_status == "ready",
    }


def answer(payload: Dict[str, Any]) -> Dict[str, Any]:
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        raise EngineError("Please enter a question before asking.")
    if len(question) > MAX_QUESTION_CHARS:
        question = question[:MAX_QUESTION_CHARS]
    predictor = get_predictor()
    thresholds = thresholds_from_payload(payload, predictor)
    portfolio = reconstruct_portfolio(payload.get("rows"), thresholds, BINARY_DECISION_THRESHOLD)
    try:
        timeout = float(os.environ.get("CRAM_ANALYST_TIMEOUT_SECONDS", "12"))
    except ValueError:
        timeout = 12.0
    try:
        answer_result = answer_question(
            portfolio,
            question,
            champion_model_name=str(predictor.metadata.get("champion_model_name", "unknown")),
            timeout=timeout,
        )
    except AnalystError as error:
        return {"status": "error", "note": error.user_message, "question": question}
    return json_safe({
        "status": answer_result.status,
        "question": answer_result.question,
        "function": answer_result.function,
        "facts": answer_result.facts,
        "caveats": answer_result.caveats,
        "explanation": answer_result.explanation,
        "note": answer_result.note,
        "elapsed_seconds": answer_result.elapsed_seconds,
    })


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
        try:
            self._send(200, analyst_status())
        except Exception:
            self._send(200, {"status": "ready", "provider": {"configured": False}, "model": False})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4_500_000:
                self._send(413, {"error": "The dataset could not be received. Requests must stay under 4.5 MB."})
                return
            payload = json.loads(self.rfile.read(length))
            self._send(200, answer(payload))
        except EngineError as error:
            self._send(400, {"error": error.message})
        except (json.JSONDecodeError, TypeError, ValueError) as error:
            self._send(400, {"error": f"Could not complete the analysis: {error}"})
