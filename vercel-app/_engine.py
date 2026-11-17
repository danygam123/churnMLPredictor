"""Shared V2 engine wiring for the Churn Compass serverless endpoints.

Every V2 endpoint is deliberately stateless: a request carries either raw CSV
text or the already-scored rows of the client's current (optionally filtered)
view, and the engine recomputes what it needs. No uploaded data is written to
disk, logged, or shared between requests; each user's dataset lives entirely in
their browser session, which gives per-user isolation by construction.

Production limits (verified against Vercel serverless constraints):
- The platform accepts request bodies up to 4.5 MB, so this product accepts at
  most 4 MB of CSV text in one request. (The Phase 7 Streamlit reference used a
  10 MB precheck on a long-running process; serverless cannot.)
- Function responses are also bounded, so analysis caps the active dataset at
  10,000 customers.

Prediction semantics follow the production decision recorded for this
migration: the binary customer decision stays at the established 0.5 cutoff
("Likely / less likely to churn"), while V2 risk bands (Low / Medium / High)
are a separate, session-adjustable classification that defaults to the Phase 5
metadata thresholds. The Phase 1 cost-optimal threshold (~0.0773) is only ever
presented as an evaluated operating point on the Model Performance page.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.predict import ChurnPredictor
from ml.risk_intelligence import (
    INPUT_FEATURES,
    PortfolioResult,
    RiskThresholds,
    churn_rate_by_group,
    choose_dimension_column,
    classify_risk,
)

ROOT = Path(__file__).resolve().parent.parent

MAX_UPLOAD_CHARS = 4_000_000
MAX_UPLOAD_ROWS = 10_000
BINARY_DECISION_THRESHOLD = 0.5

OPTIONAL_COLUMNS = ("Customer ID", "Churn")
PLAN_CATEGORIES: Dict[str, set] = {
    "International plan": {"yes", "no", "y", "n", "true", "false", "1", "0", "1.0", "0.0"},
    "Voice mail plan": {"yes", "no", "y", "n", "true", "false", "1", "0", "1.0", "0.0"},
}

_PREDICTOR: Optional[ChurnPredictor] = None


class EngineError(Exception):
    """An engine failure that should be surfaced to the user as a message."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def get_predictor() -> ChurnPredictor:
    """Load the serving V2 champion model once per warm function instance."""
    global _PREDICTOR
    if _PREDICTOR is None:
        try:
            _PREDICTOR = ChurnPredictor()
        except Exception as error:  # corrupted or missing protected artifact
            raise EngineError(
                "The churn model could not be loaded, so analysis is unavailable. "
                "The model files may be missing or damaged; please contact the "
                "application administrator."
            ) from error
    return _PREDICTOR


def default_thresholds(predictor: ChurnPredictor) -> RiskThresholds:
    return RiskThresholds.from_metadata(predictor.metadata)


def thresholds_from_payload(
    payload: Dict[str, Any], predictor: ChurnPredictor
) -> RiskThresholds:
    """Session risk bands from the request, or the metadata defaults."""
    requested = payload.get("thresholds") or {}
    try:
        medium = float(requested["medium_threshold"])
        high = float(requested["high_threshold"])
    except (KeyError, TypeError, ValueError):
        return default_thresholds(predictor)
    try:
        return RiskThresholds(medium_threshold=medium, high_threshold=high)
    except ValueError as error:
        raise EngineError(
            "Risk bands must satisfy 0 <= medium < high <= 1. Showing the default bands."
        ) from error


# ---------------------------------------------------------------------------
# Upload precheck — ported from the Phase 7 reference implementation
# ---------------------------------------------------------------------------

def parse_csv_text(text: str) -> pd.DataFrame:
    try:
        return pd.read_csv(io.StringIO(text))
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        raise EngineError(
            "The file could not be read as a CSV. Please export it again as plain "
            "CSV (comma-separated) with one header row."
        ) from error


def validate_upload_text(text: str) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    """Precheck size, parseability, emptiness, and the required schema.

    A rejected upload never reaches the model, and the client keeps its
    previous working dataset. This gate does not replace the Phase 3
    ``CustomerDataValidator``, which cleans rows and reports per-value problems
    after the precheck passes.
    """
    if not text or not text.strip():
        return False, "The selected file is empty. Please choose a non-empty customer CSV.", None
    if len(text) > MAX_UPLOAD_CHARS:
        return False, (
            f"The file is too large ({len(text) / 1_000_000:.1f} MB). "
            "This application accepts customer files up to 4 MB."
        ), None
    try:
        frame = parse_csv_text(text)
    except EngineError as error:
        return False, error.message, None
    if frame.empty:
        return False, (
            "The file contains a header but no customer rows. Add at least one customer row."
        ), None
    missing = [column for column in INPUT_FEATURES if column not in frame.columns]
    if missing:
        return False, (
            "This file is missing required model input columns, so it cannot be analyzed: "
            + ", ".join(missing)
            + ". The supported schema is described on the Project info page under “Supported data schema”."
        ), None
    return True, "", frame


def summarize_schema_issues(frame: pd.DataFrame) -> List[str]:
    """Human-readable notes about values the validator will impute or reject."""
    notes: List[str] = []
    for column in INPUT_FEATURES:
        if column not in frame.columns:
            continue
        values = frame[column]
        blank = int(values.isna().sum())
        if column in PLAN_CATEGORIES:
            text = values.dropna().astype(str).str.strip().str.casefold()
            unexpected = int((~text.isin(PLAN_CATEGORIES[column])).sum())
            if unexpected:
                notes.append(
                    f"{column}: {unexpected} value(s) are not Yes/No and will be treated as missing or rejected during validation."
                )
            if blank:
                notes.append(f"{column}: {blank} blank value(s) will be imputed and disclosed per customer.")
        else:
            numeric = pd.to_numeric(values, errors="coerce")
            invalid = int((values.notna() & numeric.isna()).sum())
            if invalid:
                notes.append(
                    f"{column}: {invalid} value(s) are not usable numbers and will be rejected and imputed during validation."
                )
            if blank:
                notes.append(f"{column}: {blank} blank value(s) will be imputed and disclosed per customer.")
    for column in OPTIONAL_COLUMNS:
        if column in frame.columns and frame[column].isna().any():
            label = (
                "row identifiers will be generated"
                if column == "Customer ID"
                else "those rows are excluded from observed-rate statistics"
            )
            notes.append(f"{column}: some values are blank; {label}.")
    return notes


# ---------------------------------------------------------------------------
# Portfolio reconstruction from scored rows (the client's filtered view)
# ---------------------------------------------------------------------------

def rebuild_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Rebuild a typed DataFrame from the scored rows the client sent back."""
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise EngineError("The scored customer rows were not received correctly. Please re-analyze your dataset.")
    frame = pd.DataFrame(rows)
    if "Churn_Probability" not in frame.columns or "Risk_Category" not in frame.columns:
        raise EngineError("The scored customer rows are incomplete. Please re-analyze your dataset.")
    if "Predicted_Churner" in frame.columns:
        frame["Predicted_Churner"] = frame["Predicted_Churner"].astype(bool)
    if "Observed_Churn" in frame.columns:
        values = [pd.NA if value is None else bool(value) for value in frame["Observed_Churn"].tolist()]
        frame["Observed_Churn"] = pd.array(values, dtype="boolean")
    return frame


def reconstruct_portfolio(
    rows: List[Dict[str, Any]],
    thresholds: RiskThresholds,
    prediction_threshold: float,
) -> PortfolioResult:
    """Rebuild a PortfolioResult from the client's current scored view.

    The analysis (validation, scoring, band classification with the session
    thresholds) already happened in ``analyze_portfolio`` when the dataset was
    uploaded; this only restores the structure the analytics modules expect,
    with the session's bands applied.
    """
    customers = rebuild_frame(rows)
    if "Observed_Churn" in customers.columns:
        customers["Observed_Churn"] = pd.array(
            [pd.NA if value is None else bool(value) for value in customers["Observed_Churn"].tolist()],
            dtype="boolean",
        )
        labeled = customers["Observed_Churn"].notna()
    else:
        labeled = pd.Series(False, index=customers.index)
    return PortfolioResult(
        customers=customers,
        warnings=[],
        label_column_present=bool(labeled.any()),
        id_source_column="Customer ID" if "Customer ID" in customers.columns else None,
        prediction_threshold=float(prediction_threshold),
        risk_thresholds=thresholds,
    )


def reclassify(
    rows: List[Dict[str, Any]],
    thresholds: RiskThresholds,
) -> List[str]:
    frame = rebuild_frame(rows)
    return classify_risk(frame["Churn_Probability"].tolist(), thresholds)


# ---------------------------------------------------------------------------
# Group-rate context for the Portfolio page
# ---------------------------------------------------------------------------

GROUP_CANDIDATES = (
    ("Contract Type", "Contract", "Contract type", "ContractType"),
    ("International plan",),
    ("Voice mail plan",),
    ("Payment Method", "PaymentMethod", "Payment method"),
    ("Customer service calls",),
)


def group_rates(frame: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
    """Churn rate (or mean predicted probability) per available dimension."""
    result: Dict[str, List[Dict[str, Any]]] = {}
    for candidates in GROUP_CANDIDATES:
        column = choose_dimension_column(frame, candidates)
        if not column:
            continue
        try:
            grouped = churn_rate_by_group(frame, column)
        except ValueError:
            continue
        if grouped.empty:
            continue
        grouped = grouped.sort_values("rate", ascending=False).head(25)
        result[column] = [
            {
                "group": _scalar(row[column]),
                "rate": round(float(row["rate"]), 4),
                "customers": int(row["customers"]),
            }
            for _, row in grouped.iterrows()
        ]
    return result


# ---------------------------------------------------------------------------
# JSON safety and compact serialization
# ---------------------------------------------------------------------------

def _scalar(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.generic):
        converted = value.item()
        return _scalar(converted)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def json_safe(value: Any) -> Any:
    """Recursively convert a result so ``json.dumps(..., allow_nan=False)`` succeeds."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    return _scalar(value)


def frame_to_columns_rows(frame: pd.DataFrame) -> Dict[str, Any]:
    """Compact column-oriented serialization that keeps responses small."""
    columns = [str(column) for column in frame.columns]
    rows = [
        [_scalar(cell) for cell in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    return {"columns": columns, "rows": rows}
