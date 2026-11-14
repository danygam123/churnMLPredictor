"""SHAP explanations for the C.R.A.M. V2 raw customer input schema.

Explanations are computed against the complete fitted prediction pipeline so they
refer to the 13 fields accepted by the application, not transformed columns or
derived charge fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import shap

from ml.preprocessing import INPUT_FEATURES, CustomerDataValidator


class ChurnExplainer:
    """Model-agnostic SHAP explanations for a fitted churn prediction pipeline."""

    def __init__(
        self,
        model: Any,
        background_data: pd.DataFrame,
        *,
        max_evals: Optional[int] = None,
        random_state: int = 42,
        negligible_contribution: float = 0.01,
    ) -> None:
        if not isinstance(background_data, pd.DataFrame) or background_data.empty:
            raise ValueError("background_data must be a non-empty pandas DataFrame")
        missing = sorted(set(INPUT_FEATURES) - set(background_data.columns))
        if missing:
            raise ValueError(f"background_data is missing required inputs: {missing}")
        if negligible_contribution < 0:
            raise ValueError("negligible_contribution must be non-negative")

        self.model = model
        self.validator = CustomerDataValidator()
        self.background_data = self._validated(background_data[INPUT_FEATURES])
        self.max_evals = max_evals or (10 * (2 * len(INPUT_FEATURES) + 1))
        self.negligible_contribution = float(negligible_contribution)
        self.explainer = shap.PermutationExplainer(
            self._predict_churn_probability,
            shap.maskers.Independent(self.background_data, max_samples=len(self.background_data)),
            feature_names=INPUT_FEATURES,
            seed=random_state,
        )

    def _validated(self, data: pd.DataFrame) -> pd.DataFrame:
        report = self.validator.validate(data, remove_duplicates=False)
        return report.cleaned_df[INPUT_FEATURES]

    def _predict_churn_probability(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
        else:
            frame = pd.DataFrame(data, columns=INPUT_FEATURES)
        cleaned = self._validated(frame)
        return np.asarray(self.model.predict_proba(cleaned)[:, 1], dtype=float)

    def explain(self, customer_profile: Dict[str, Any], *, max_evals: Optional[int] = None) -> Dict[str, Any]:
        """Return signed SHAP contributions, grouped by direction, in probability units."""
        raw = pd.DataFrame([customer_profile]).reindex(columns=INPUT_FEATURES)
        genuinely_missing = [name for name in INPUT_FEATURES if pd.isna(raw.iloc[0][name])]
        cleaned = self._validated(raw)
        rejected_or_coerced = [
            name for name in INPUT_FEATURES
            if name not in genuinely_missing and pd.isna(cleaned.iloc[0][name])
        ]
        estimated_features = [
            name for name in INPUT_FEATURES
            if name in genuinely_missing or name in rejected_or_coerced
        ]
        shap_result = self.explainer(
            cleaned,
            max_evals=max_evals or self.max_evals,
            silent=True,
        )
        values = np.asarray(shap_result.values, dtype=float).reshape(-1)
        base_value = float(np.asarray(shap_result.base_values).reshape(-1)[0])
        probability = float(self.model.predict_proba(cleaned)[0, 1])

        contributions = []
        for name, value in zip(INPUT_FEATURES, values):
            number = float(value)
            original_value = raw.iloc[0][name]
            validated_value = cleaned.iloc[0][name]
            if name in genuinely_missing:
                input_status = "genuinely_missing_imputed"
                display_value: Any = None
            elif name in rejected_or_coerced:
                input_status = "invalid_or_out_of_range_coerced_then_imputed"
                display_value = original_value
            else:
                input_status = "provided"
                display_value = validated_value
            if isinstance(display_value, np.generic):
                display_value = display_value.item()
            if (
                input_status == "provided"
                and name in ("International plan", "Voice mail plan")
                and display_value is not None
            ):
                display_value = "Yes" if float(display_value) >= 0.5 else "No"
            contributions.append({
                "feature": name,
                "value": display_value,
                "input_status": input_status,
                "contribution": number,
                "direction": "increases" if number > 0 else "decreases" if number < 0 else "neutral",
                "magnitude": abs(number),
            })
        contributions.sort(key=lambda item: item["magnitude"], reverse=True)
        influential = [x for x in contributions if x["magnitude"] >= self.negligible_contribution]
        return {
            "churn_probability": probability,
            "base_value": base_value,
            "contributions_sum": float(values.sum()),
            "reconstructed_probability": base_value + float(values.sum()),
            "max_evals": max_evals or self.max_evals,
            "explanation_method": "SHAP PermutationExplainer; output is churn probability",
            "estimated_features": estimated_features,
            "genuinely_missing_features": genuinely_missing,
            "rejected_or_coerced_features": rejected_or_coerced,
            "top_contributions": contributions[:5],
            "increasing_risk_factors": [x for x in influential if x["contribution"] > 0],
            "decreasing_risk_factors": [x for x in influential if x["contribution"] < 0],
            "limited_influence_factors": [x for x in contributions if x["magnitude"] < self.negligible_contribution],
            "all_contributions": contributions,
            "summary": self._summary(probability, influential),
        }

    @staticmethod
    def _summary(probability: float, influential: list[dict[str, Any]]) -> str:
        if not influential:
            return f"The model estimates churn probability at {probability:.1%}; no input had a contribution of at least one percentage point."
        up = [item["feature"] for item in influential if item["contribution"] > 0][:3]
        down = [item["feature"] for item in influential if item["contribution"] < 0][:3]
        parts = [f"The model estimates churn probability at {probability:.1%}."]
        if up:
            parts.append("The largest factors pushing the estimate higher are " + ", ".join(up) + ".")
        if down:
            parts.append("Factors pushing it lower include " + ", ".join(down) + ".")
        return " ".join(parts) + " These are model associations, not causal effects."

    def explain_global(self, customers: pd.DataFrame, *, max_evals: Optional[int] = None) -> shap.Explanation:
        """Compute SHAP values for a dataset; callers should pass a bounded sample."""
        if not isinstance(customers, pd.DataFrame) or customers.empty:
            raise ValueError("customers must be a non-empty pandas DataFrame")
        missing = sorted(set(INPUT_FEATURES) - set(customers.columns))
        if missing:
            raise ValueError(f"customers is missing required inputs: {missing}")
        cleaned = self._validated(customers[INPUT_FEATURES])
        return self.explainer(cleaned, max_evals=max_evals or self.max_evals, silent=True)

    @staticmethod
    def global_importance(explanation: shap.Explanation) -> pd.DataFrame:
        """Rank global features by mean absolute contribution in probability units."""
        values = np.asarray(explanation.values, dtype=float)
        result = pd.DataFrame({
            "feature": list(explanation.feature_names),
            "mean_abs_contribution": np.mean(np.abs(values), axis=0),
            "mean_contribution": np.mean(values, axis=0),
            "positive_fraction": np.mean(values > 0, axis=0),
            "negative_fraction": np.mean(values < 0, axis=0),
        })
        return result.sort_values("mean_abs_contribution", ascending=False).reset_index(drop=True)

    @staticmethod
    def save_beeswarm(explanation: shap.Explanation, output_path: Union[str, Path]) -> Path:
        """Render and save a SHAP beeswarm plot; matplotlib is required."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("Beeswarm plotting requires matplotlib") from exc
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shap.plots.beeswarm(explanation, show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        return path

    @staticmethod
    def save_waterfall(local_explanation: Dict[str, Any], output_path: Union[str, Path]) -> Path:
        """Render and save a local SHAP waterfall plot from explain() output."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("Waterfall plotting requires matplotlib") from exc
        explanation = shap.Explanation(
            values=np.array([x["contribution"] for x in local_explanation["all_contributions"]]),
            base_values=local_explanation["base_value"],
            data=np.array([x["value"] for x in local_explanation["all_contributions"]], dtype=object),
            feature_names=[x["feature"] for x in local_explanation["all_contributions"]],
        )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        return path


def generate_training_global_report(
    *,
    output_csv: Union[str, Path],
    output_plot: Union[str, Path],
    n_background: int = 50,
    n_explain: int = 100,
    max_evals: Optional[int] = None,
) -> pd.DataFrame:
    """Generate reproducible global summaries from train split features only."""
    from sklearn.model_selection import train_test_split

    from ml.train import RANDOM_STATE, TEST_SIZE, load_raw_dataset

    root = Path(__file__).resolve().parent.parent
    X, y, _ = load_raw_dataset()
    X_train, _, _, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    model = __import__("joblib").load(root / "models" / "champion_model_evaluation.joblib")
    background = X_train.sample(min(n_background, len(X_train)), random_state=RANDOM_STATE)
    sample = X_train.sample(min(n_explain, len(X_train)), random_state=RANDOM_STATE + 1)
    explainer = ChurnExplainer(model, background, max_evals=max_evals)
    explanation = explainer.explain_global(sample)
    importance = explainer.global_importance(explanation)
    importance.to_csv(output_csv, index=False)
    explainer.save_beeswarm(explanation, output_plot)
    return importance


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    table = generate_training_global_report(
        output_csv=root / "docs" / "phase_2_global_importance.csv",
        output_plot=root / "docs" / "images" / "phase_2_shap_beeswarm.png",
    )
    print(table.to_string(index=False))
