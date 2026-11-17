"""Independent, non-causal what-if simulations through the serving predictor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from ml.preprocessing import (
    CATEGORICAL_INPUTS,
    INPUT_FEATURES,
    INTEGER_FIELDS,
    VALID_RANGES,
    CustomerDataValidator,
)


@dataclass(frozen=True)
class SimulationResult:
    """Baseline-versus-scenario predictions and an auditable change record."""

    original_probability: float
    simulated_probability: float
    difference: float
    difference_percentage_points: float
    original_prediction: str
    simulated_prediction: str
    original_risk_category: str
    simulated_risk_category: str
    decision_threshold: float
    changed_features: Dict[str, Dict[str, Any]]
    scenario_inputs: Dict[str, Any]
    estimated_features: tuple[str, ...]
    out_of_training_range_features: tuple[str, ...]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set, np.ndarray, pd.Series)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _native(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _binary(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"yes", "y", "true", "1", "1.0"}:
            return 1.0
        if normalized in {"no", "n", "false", "0", "0.0"}:
            return 0.0
        raise ValueError("Use Yes or No for binary model inputs.")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        raise ValueError("Use Yes or No for binary model inputs.") from None
    if numeric not in (0.0, 1.0):
        raise ValueError("Binary model inputs must be exactly 0/1 or Yes/No.")
    return numeric


def _input_frame(profile: Mapping[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{feature: profile.get(feature, np.nan) for feature in INPUT_FEATURES}])


def _validated_inputs(profile: Mapping[str, Any]) -> pd.DataFrame:
    return CustomerDataValidator().validate(_input_frame(profile), remove_duplicates=False).cleaned_df[INPUT_FEATURES]


def _display_value(feature: str, value: Any) -> Any:
    if _is_missing(value):
        return None
    if feature in CATEGORICAL_INPUTS:
        return "Yes" if float(value) == 1.0 else "No"
    return _native(value)


def simulate_scenario(
    predictor: Any,
    original_record: Mapping[str, Any],
    changes: Mapping[str, Any],
    *,
    threshold: float | None = None,
) -> SimulationResult:
    """Apply validated hypothetical input changes on a fresh customer copy.

    Only the canonical 13 model inputs can be changed. Identifiers, observed
    labels, derived charge fields, and arbitrary uploaded metadata are not
    model inputs and are therefore rejected as scenario edits.
    """
    if not isinstance(original_record, Mapping):
        raise TypeError("original_record must be a mapping of customer inputs")
    if not isinstance(changes, Mapping):
        raise TypeError("changes must map supported model input names to new values")

    unsupported = sorted(set(changes) - set(INPUT_FEATURES))
    if unsupported:
        raise ValueError(
            "Only the 13 canonical model inputs are editable; unsupported or immutable fields: "
            + ", ".join(unsupported)
        )

    chosen_threshold = float(predictor.optimal_threshold if threshold is None else threshold)
    if not np.isfinite(chosen_threshold) or not 0.0 <= chosen_threshold <= 1.0:
        raise ValueError("threshold must be finite and within [0, 1]")

    # Copy first and preserve the source mapping. This candidate is private to
    # this call; another scenario always starts from original_record again.
    baseline_raw = {feature: original_record.get(feature, np.nan) for feature in INPUT_FEATURES}
    candidate_raw = dict(baseline_raw)
    candidate_raw.update(dict(changes))

    for feature, value in changes.items():
        if _is_missing(value) or isinstance(value, (list, tuple, dict, set, np.ndarray)):
            raise ValueError(f"{feature} must be a single valid, non-missing model input value")
        if feature in CATEGORICAL_INPUTS:
            _binary(value)
        else:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{feature} must be numeric, not a binary value")
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{feature} must be numeric") from None
            if not np.isfinite(numeric):
                raise ValueError(f"{feature} must be finite")
            low, high = VALID_RANGES[feature]
            if not low <= numeric <= high:
                raise ValueError(f"{feature} must be between {low:g} and {high:g}")
            if feature in INTEGER_FIELDS and not numeric.is_integer():
                raise ValueError(f"{feature} must be a whole number")

    baseline_clean = _validated_inputs(baseline_raw)
    scenario_clean = _validated_inputs(candidate_raw)

    # The range validator also protects binary categories, including numeric
    # category inputs. Reject any changed cell that became missing after clean.
    for feature in changes:
        if _is_missing(scenario_clean.iloc[0][feature]):
            raise ValueError(f"{feature} was rejected by input validation; enter a value in its valid domain")
        if feature in CATEGORICAL_INPUTS and float(scenario_clean.iloc[0][feature]) not in (0.0, 1.0):
            raise ValueError(f"{feature} must be exactly 0/1 or Yes/No")

    changed_features: Dict[str, Dict[str, Any]] = {}
    for feature in changes:
        before = baseline_clean.iloc[0][feature]
        after = scenario_clean.iloc[0][feature]
        same = not _is_missing(before) and not _is_missing(after) and float(before) == float(after)
        if not same:
            changed_features[feature] = {
                "original_value": _native(baseline_raw[feature]),
                "simulated_value": _display_value(feature, after),
            }

    # Both evaluations use the same fitted predictor and preprocessing pipeline
    # used by normal serving. Sensitivity/SHAP work is deliberately disabled.
    original_output = predictor.predict_single(
        baseline_raw,
        threshold=chosen_threshold,
        explain_sensitivities=False,
    )
    simulated_output = predictor.predict_single(
        scenario_clean.iloc[0].to_dict(),
        threshold=chosen_threshold,
        explain_sensitivities=False,
    )
    original_probability = float(original_output["churn_probability"])
    simulated_probability = float(simulated_output["churn_probability"])
    difference = simulated_probability - original_probability

    training_ranges = getattr(predictor, "metadata", {}).get("input_ranges", {})
    out_of_range = []
    for feature in INPUT_FEATURES:
        bounds = training_ranges.get(feature, {})
        value = scenario_clean.iloc[0][feature]
        if pd.notna(value) and "min" in bounds and "max" in bounds:
            if float(value) < float(bounds["min"]) or float(value) > float(bounds["max"]):
                out_of_range.append(feature)

    estimated = tuple(simulated_output.get("estimated_fields", []))
    return SimulationResult(
        original_probability=original_probability,
        simulated_probability=simulated_probability,
        difference=difference,
        difference_percentage_points=difference * 100.0,
        original_prediction=str(original_output["prediction"]),
        simulated_prediction=str(simulated_output["prediction"]),
        original_risk_category=str(original_output["risk_category"]),
        simulated_risk_category=str(simulated_output["risk_category"]),
        decision_threshold=chosen_threshold,
        changed_features=changed_features,
        scenario_inputs={feature: _native(scenario_clean.iloc[0][feature]) for feature in INPUT_FEATURES},
        estimated_features=estimated,
        out_of_training_range_features=tuple(out_of_range),
    )
