"""Probability calibration module for C.R.A.M. V2.

Evaluates calibration reliability, computes Brier scores, and provides Platt scaling
(sigmoid) and Isotonic calibration options via CalibratedClassifierCV.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from ml.evaluate import compute_calibration_assessment

logger = logging.getLogger(__name__)


def evaluate_calibration_methods(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Compares uncalibrated model against Platt scaling (sigmoid) and Isotonic calibration.

    Fits candidates strictly on cross-validation folds to avoid data leakage.
    Returns comparison metrics and recommendation.
    """
    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)

    # 1. Uncalibrated baseline
    pipeline.fit(X_train, y_train)
    raw_probs = pipeline.predict_proba(X_test)[:, 1]
    raw_assessment = compute_calibration_assessment(y_test, raw_probs)

    # 2. Platt Scaling (Sigmoid) with 5-fold cross-validation
    sigmoid_calibrator = CalibratedClassifierCV(estimator=pipeline, method="sigmoid", cv=5)
    sigmoid_calibrator.fit(X_train, y_train)
    sigmoid_probs = sigmoid_calibrator.predict_proba(X_test)[:, 1]
    sigmoid_assessment = compute_calibration_assessment(y_test, sigmoid_probs)

    # 3. Isotonic Regression with 5-fold cross-validation
    isotonic_calibrator = CalibratedClassifierCV(estimator=pipeline, method="isotonic", cv=5)
    isotonic_calibrator.fit(X_train, y_train)
    isotonic_probs = isotonic_calibrator.predict_proba(X_test)[:, 1]
    isotonic_assessment = compute_calibration_assessment(y_test, isotonic_probs)

    results = {
        "uncalibrated": raw_assessment,
        "sigmoid_platt": sigmoid_assessment,
        "isotonic": isotonic_assessment,
    }

    # Determine champion based on lowest Brier Score
    brier_scores = {
        "uncalibrated": raw_assessment["brier_score"],
        "sigmoid_platt": sigmoid_assessment["brier_score"],
        "isotonic": isotonic_assessment["brier_score"],
    }
    best_method = min(brier_scores, key=brier_scores.get)
    improvement = raw_assessment["brier_score"] - brier_scores[best_method]

    results["recommendation"] = {
        "best_method": best_method,
        "raw_brier": raw_assessment["brier_score"],
        "best_brier": brier_scores[best_method],
        "brier_improvement": float(improvement),
        "should_calibrate": bool(improvement > 0.002),  # Meaningful margin
    }

    return results


def get_calibrated_pipeline(
    pipeline: Any,
    method: str = "sigmoid",
    cv: int = 5,
) -> CalibratedClassifierCV:
    """Wraps an entire pipeline into a cross-validated CalibratedClassifierCV."""
    return CalibratedClassifierCV(estimator=pipeline, method=method, cv=cv)
