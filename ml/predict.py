"""Inference and scoring engine for C.R.A.M. V2.

Decouples ML prediction logic from the Streamlit user interface, supporting:
- Single-customer scoring
- Batch CSV dataset scoring
- Configurable risk threshold segmentation (High, Medium, Low)
- Transparent reporting of estimated / imputed fields
- Feature sensitivity signals for initial decision support
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from ml.preprocessing import (
    ALL_FEATURES,
    CHARGE_RATES,
    INPUT_FEATURES,
    CustomerDataValidator,
)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "champion_churn_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
EXPLANATION_REFERENCE_PATH = MODELS_DIR / "explanation_reference.csv"

# Fallback to root paths if not yet in models/
if not MODEL_PATH.exists() and (ROOT / "random_forest_model.joblib").exists():
    MODEL_PATH = ROOT / "random_forest_model.joblib"
if not METADATA_PATH.exists() and (ROOT / "model_metadata.json").exists():
    METADATA_PATH = ROOT / "model_metadata.json"


class ChurnPredictor:
    """Production predictor for Customer Retention & Analytics Model (C.R.A.M. V2)."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
    ):
        self.model_path = Path(model_path or MODEL_PATH)
        self.metadata_path = Path(metadata_path or METADATA_PATH)

        self.model = joblib.load(self.model_path)
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.input_features = self.metadata.get("input_features", INPUT_FEATURES)
        self.ranges = self.metadata.get("input_ranges", {})
        self.optimal_threshold = float(self.metadata.get("optimal_threshold", 0.50))
        self.default_threshold = float(self.metadata.get("default_threshold", 0.50))
        self.medians = {
            col: info["median"] for col, info in self.ranges.items()
        } if self.ranges else {}
        self.validator = CustomerDataValidator()
        self._explainer = None

    def categorize_risk(
        self,
        probability: float,
        high_threshold: Optional[float] = None,
        low_threshold: Optional[float] = None,
    ) -> str:
        """Assigns risk bands separately from the binary churn decision.

        Defaults anchor High Risk at the Phase 1 default 0.50 cutoff and
        Medium Risk at the Phase 1 cost-optimized threshold. ``low_threshold``
        remains the legacy argument name for the Medium Risk lower bound.
        """
        high = high_threshold if high_threshold is not None else self.default_threshold
        medium = low_threshold if low_threshold is not None else self.optimal_threshold
        if not (0.0 <= medium < high <= 1.0):
            raise ValueError("Risk thresholds must satisfy 0 <= medium < high <= 1")
        if probability >= high:
            return "High Risk"
        elif probability >= medium:
            return "Medium Risk"
        else:
            return "Low Risk"

    def predict_single(
        self,
        customer_profile: Dict[str, Any],
        threshold: Optional[float] = None,
        explain_sensitivities: bool = True,
    ) -> Dict[str, Any]:
        """Scores an individual customer record."""
        thresh = threshold if threshold is not None else self.optimal_threshold
        df_raw = pd.DataFrame([customer_profile])

        # Track missing / estimated inputs
        missing_fields = [
            f for f in self.input_features
            if f not in customer_profile or pd.isna(customer_profile[f]) or customer_profile[f] is None
        ]

        # In case charges aren't passed, validator and pipeline derive them
        report = self.validator.validate(df_raw, remove_duplicates=False)
        cleaned_df = report.cleaned_df

        # Predict probability
        probs = self.model.predict_proba(cleaned_df)[0]
        churn_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
        prediction_label = "Likely to churn" if churn_prob >= thresh else "Less likely to churn"
        risk_category = self.categorize_risk(churn_prob)

        result: Dict[str, Any] = {
            "churn_probability": churn_prob,
            "prediction": prediction_label,
            "risk_category": risk_category,
            "decision_threshold_used": thresh,
            "estimated_fields": missing_fields,
            "estimated_fields_count": len(missing_fields),
        }

        # Calculate directional sensitivities vs typical customer profile
        if explain_sensitivities and self.medians:
            signals = []
            for feat in self.input_features:
                if feat in missing_fields:
                    continue
                val = customer_profile.get(feat)
                if val is None or pd.isna(val):
                    continue
                # Perturb single feature to median
                perturbed = dict(customer_profile)
                perturbed[feat] = self.medians.get(feat, 0.0)
                p_df = self.validator.validate(pd.DataFrame([perturbed]), remove_duplicates=False).cleaned_df
                perturbed_prob = float(self.model.predict_proba(p_df)[0][1])
                diff = churn_prob - perturbed_prob
                signals.append({
                    "feature": feat,
                    "value": val,
                    "score_delta_vs_typical": float(diff),
                })
            signals.sort(key=lambda s: abs(s["score_delta_vs_typical"]), reverse=True)
            result["top_sensitivity_signals"] = signals[:5]

        return result

    def explain_single(self, customer_profile: Dict[str, Any], background_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Return a SHAP explanation against the fixed V2 serving reference population.

        Custom background data is rejected so a cached explainer cannot silently use
        a different population from the one a caller just supplied. For offline
        analyses with custom populations, use ``ChurnExplainer`` directly.
        """
        if background_data is not None:
            raise ValueError(
                "Serving explanations use the fixed reference population at "
                f"{EXPLANATION_REFERENCE_PATH}. Custom backgrounds are not accepted; "
                "use ChurnExplainer directly for offline analyses."
            )
        if self._explainer is None:
            if not EXPLANATION_REFERENCE_PATH.exists():
                raise FileNotFoundError(
                    "The fixed V2 SHAP reference population is missing: "
                    f"{EXPLANATION_REFERENCE_PATH}"
                )
            from ml.explain import ChurnExplainer

            serving_reference = pd.read_csv(EXPLANATION_REFERENCE_PATH)
            self._explainer = ChurnExplainer(self.model, serving_reference)
        result = self._explainer.explain(customer_profile)
        result["risk_category"] = self.categorize_risk(result["churn_probability"])
        result["prediction"] = (
            "Likely to churn"
            if result["churn_probability"] >= self.optimal_threshold
            else "Less likely to churn"
        )
        result["decision_threshold_used"] = self.optimal_threshold
        return result

    def predict_batch(
        self,
        data: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Scores a batch customer dataset (e.g. uploaded CSV)."""
        thresh = threshold if threshold is not None else self.optimal_threshold
        report = self.validator.validate(data, remove_duplicates=False)
        cleaned = report.cleaned_df

        # Count missing inputs per row
        missing_counts = cleaned[self.input_features].isna().sum(axis=1)

        probs = self.model.predict_proba(cleaned)[:, 1]
        results = data.copy()
        results["Churn_Probability"] = np.round(probs, 4)
        results["Risk_Category"] = [
            self.categorize_risk(p) for p in probs
        ]
        results["Prediction"] = np.where(probs >= thresh, "Likely to churn", "Less likely to churn")
        results["Estimated_Input_Count"] = missing_counts

        return results, report.warnings
