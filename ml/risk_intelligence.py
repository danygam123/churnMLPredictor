"""Portfolio-level customer risk intelligence built on the C.R.A.M. V2 predictor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ml.preprocessing import CHARGE_RATES, INPUT_FEATURES, CustomerDataValidator


@dataclass(frozen=True)
class RiskThresholds:
    """Transparent portfolio segments; separate from the binary prediction cutoff."""

    medium_threshold: float
    high_threshold: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.medium_threshold < self.high_threshold <= 1.0):
            raise ValueError("Risk thresholds must satisfy 0 <= medium < high <= 1")

    @classmethod
    def from_metadata(cls, metadata: Dict[str, Any]) -> "RiskThresholds":
        # The medium floor uses Phase 1's cost-optimized validation threshold;
        # high risk uses its default 0.50 classification cutoff.
        return cls(
            medium_threshold=float(metadata["optimal_threshold"]),
            high_threshold=float(metadata["default_threshold"]),
        )


RISK_LABELS = ("Low Risk", "Medium Risk", "High Risk")


def classify_risk(probabilities: Iterable[float], thresholds: RiskThresholds) -> List[str]:
    labels: List[str] = []
    for value in probabilities:
        probability = float(value)
        if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"Churn probability must be finite and within [0, 1], got {value!r}")
        if probability >= thresholds.high_threshold:
            labels.append("High Risk")
        elif probability >= thresholds.medium_threshold:
            labels.append("Medium Risk")
        else:
            labels.append("Low Risk")
    return labels


def _parse_observed_label(value: Any, row_number: int) -> Any:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"yes", "y", "true", "1", "1.0", "churn"}:
            return True
        if normalized in {"no", "n", "false", "0", "0.0", "not churn", "no churn"}:
            return False
    raise ValueError(
        f"Invalid Churn label at CSV data row {row_number}: {value!r}. "
        "Use true/false, yes/no, or 1/0; leave unknown labels blank."
    )


def _customer_ids(data: pd.DataFrame) -> tuple[List[str], Optional[str]]:
    candidates = ("Customer ID", "Customer_ID", "customer_id", "CustomerID")
    source = next((name for name in candidates if name in data.columns), None)
    if source is None:
        return [f"Record {i + 1:06d}" for i in range(len(data))], None

    identifiers: List[str] = []
    seen: set[str] = set()
    for i, value in enumerate(data[source].tolist()):
        identifier = "" if pd.isna(value) else str(value).strip()
        if not identifier:
            identifier = f"Record {i + 1:06d}"
        if identifier in seen:
            identifier = f"{identifier} (row {i + 1})"
        seen.add(identifier)
        identifiers.append(identifier)
    return identifiers, source


@dataclass
class PortfolioResult:
    customers: pd.DataFrame
    warnings: List[str]
    label_column_present: bool
    id_source_column: Optional[str]
    prediction_threshold: float
    risk_thresholds: RiskThresholds


def analyze_portfolio(
    data: pd.DataFrame,
    predictor: Any,
    thresholds: Optional[RiskThresholds] = None,
) -> PortfolioResult:
    """Validate, score, and segment an uploaded portfolio without dropping rows."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Customer data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("The uploaded customer file has no data rows")
    if len(data) > 100_000:
        raise ValueError("This research dashboard supports at most 100,000 rows per upload")

    missing_columns = [feature for feature in INPUT_FEATURES if feature not in data.columns]
    if missing_columns:
        raise ValueError(
            "The file is missing required model input columns: " + ", ".join(missing_columns)
        )

    thresholds = thresholds or RiskThresholds.from_metadata(predictor.metadata)
    raw_features = data.loc[:, INPUT_FEATURES].copy()
    report = CustomerDataValidator().validate(data, remove_duplicates=False)
    cleaned = report.cleaned_df[INPUT_FEATURES]

    raw_missing = raw_features.isna()
    rejected = raw_features.notna() & cleaned.isna()
    probabilities = np.asarray(predictor.model.predict_proba(cleaned)[:, 1], dtype=float)
    result = data.copy()
    result["Customer ID"], id_source = _customer_ids(data)
    result["Churn_Probability"] = probabilities
    result["Prediction"] = np.where(
        probabilities >= predictor.optimal_threshold,
        "Likely to churn",
        "Less likely to churn",
    )
    result["Predicted_Churner"] = probabilities >= predictor.optimal_threshold
    result["Risk_Category"] = classify_risk(probabilities, thresholds)
    result["Derived Usage Charge Total"] = derive_usage_charge_proxy(raw_features)
    result["Genuinely_Missing_Input_Count"] = raw_missing.sum(axis=1).to_numpy()
    result["Rejected_Input_Count"] = rejected.sum(axis=1).to_numpy()
    result["Estimated_Input_Count"] = (
        result["Genuinely_Missing_Input_Count"] + result["Rejected_Input_Count"]
    )

    has_labels = "Churn" in data.columns
    if has_labels:
        result["Observed_Churn"] = pd.array(
            [_parse_observed_label(value, i + 2) for i, value in enumerate(data["Churn"].tolist())],
            dtype="boolean",
        )

    warnings = list(report.warnings)
    invalid_cells = int(rejected.sum().sum())
    missing_cells = int(raw_missing.sum().sum())
    if missing_cells:
        warnings.append(f"{missing_cells} input cell(s) were genuinely blank and imputed.")
    if invalid_cells:
        warnings.append(f"{invalid_cells} invalid or out-of-range input cell(s) were coerced and imputed.")
    if (result["Estimated_Input_Count"] > 3).any():
        rows = int((result["Estimated_Input_Count"] > 3).sum())
        warnings.append(
            f"{rows} customer row(s) have more than 3 estimated inputs; their scores may be unreliable."
        )
    return PortfolioResult(
        customers=result,
        warnings=warnings,
        label_column_present=has_labels,
        id_source_column=id_source,
        prediction_threshold=float(predictor.optimal_threshold),
        risk_thresholds=thresholds,
    )


def portfolio_summary(customers: pd.DataFrame) -> Dict[str, Any]:
    """Calculate headline metrics directly from scored portfolio rows."""
    if customers.empty:
        raise ValueError("Cannot summarize an empty customer portfolio")
    summary: Dict[str, Any] = {
        "customers_analyzed": int(len(customers)),
        "predicted_churners": int(customers["Predicted_Churner"].sum()),
        "average_predicted_probability": float(customers["Churn_Probability"].mean()),
        "high_risk_customers": int((customers["Risk_Category"] == "High Risk").sum()),
    }
    if "Observed_Churn" in customers.columns:
        observed = customers["Observed_Churn"].dropna()
        summary["observed_churn_rate"] = float(observed.mean()) if len(observed) else None
        summary["observed_label_count"] = int(len(observed))
    else:
        summary["observed_churn_rate"] = None
        summary["observed_label_count"] = 0
    return summary


def derive_usage_charge_proxy(customers: pd.DataFrame) -> pd.Series:
    """Derive the model's usage-charge total; this is not an account monthly bill."""
    components = []
    for _charge_name, (minutes_name, rate) in CHARGE_RATES.items():
        if minutes_name in customers:
            components.append(pd.to_numeric(customers[minutes_name], errors="coerce") * rate)
    if not components:
        return pd.Series(np.nan, index=customers.index, dtype=float)
    charges = pd.concat(components, axis=1)
    return charges.sum(axis=1, min_count=len(components)).round(2)


def churn_rate_by_group(customers: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """Return observed churn rate by group, or mean predicted probability without labels."""
    if group_column not in customers.columns:
        raise ValueError(f"Group column {group_column!r} is not present")
    frame = customers.loc[customers[group_column].notna()]
    if frame.empty:
        return pd.DataFrame(columns=[group_column, "rate", "customers"])
    if "Observed_Churn" in frame.columns and frame["Observed_Churn"].notna().any():
        labeled = frame.loc[frame["Observed_Churn"].notna()]
        grouped = labeled.groupby(group_column, observed=False)["Observed_Churn"].agg(["mean", "count"])
    else:
        grouped = frame.groupby(group_column, observed=False)["Churn_Probability"].agg(["mean", "count"])
    return grouped.rename(columns={"mean": "rate", "count": "customers"}).reset_index()


def choose_dimension_column(data: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    """Find an optional business dimension case-insensitively without inventing one."""
    normalized = {str(column).strip().casefold(): column for column in data.columns}
    for name in names:
        match = normalized.get(name.strip().casefold())
        if match is not None:
            return str(match)
    return None
