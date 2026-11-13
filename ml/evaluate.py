"""Evaluation and business cost optimization metrics for C.R.A.M. V2.

Computes classification metrics, PR-AUC, ROC-AUC, Brier score, confusion matrix,
cost-sensitive business metrics, and threshold tuning curves.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Explicit Business Cost Framework (documented for telecom retention)
# Cost of False Negative (Customer churns undetected: loss of Annual Customer Lifetime Value)
DEFAULT_COST_FN: float = 500.0
# Cost of False Positive (Unnecessary retention discount / high-touch outreach intervention)
DEFAULT_COST_FP: float = 50.0


def compute_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Computes all primary evaluation metrics at a specific probability threshold."""
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, predictions))
    prec = float(precision_score(y_true, predictions, zero_division=0))
    rec = float(recall_score(y_true, predictions, zero_division=0))
    f1 = float(f1_score(y_true, predictions, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, probabilities))
    pr_auc = float(average_precision_score(y_true, probabilities))
    brier = float(brier_score_loss(y_true, probabilities))

    # Business cost calculation
    total_cost = float(fn * DEFAULT_COST_FN + fp * DEFAULT_COST_FP)
    # Baseline cost if no model existed (all churners missed: FN = total churners)
    baseline_cost = float(np.sum(y_true == 1) * DEFAULT_COST_FN)
    cost_savings = float(baseline_cost - total_cost)

    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "business_cost": total_cost,
        "baseline_cost": baseline_cost,
        "cost_savings": cost_savings,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    criterion: str = "cost",
    n_thresholds: int = 100,
) -> Dict[str, Any]:
    """Finds the optimal decision threshold on validation data.

    Parameters
    ----------
    criterion : str, default='cost'
        Optimization criterion: 'cost' (minimizes business cost) or 'f1' (maximizes F1).
    """
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)

    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    best_thresh = 0.5
    best_score = float("inf") if criterion == "cost" else -1.0
    records = []

    for t in thresholds:
        preds = (probabilities >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        cost = fn * cost_fn + fp * cost_fp
        f1 = f1_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        prec = precision_score(y_true, preds, zero_division=0)

        records.append({
            "threshold": float(t),
            "cost": float(cost),
            "f1": float(f1),
            "recall": float(rec),
            "precision": float(prec),
        })

        if criterion == "cost" and cost < best_score:
            best_score = cost
            best_thresh = t
        elif criterion == "f1" and f1 > best_score:
            best_score = f1
            best_thresh = t

    return {
        "criterion": criterion,
        "optimal_threshold": float(best_thresh),
        "optimal_score": float(best_score),
        "threshold_curve": records,
    }


def compute_calibration_assessment(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Calculates calibration curves and Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)

    prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=n_bins, strategy="uniform")
    brier = float(brier_score_loss(y_true, probabilities))

    # Calculate Expected Calibration Error (ECE)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probabilities, bins) - 1
    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        if np.any(mask):
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(probabilities[mask])
            ece += np.sum(mask) / len(y_true) * np.abs(bin_acc - bin_conf)

    return {
        "brier_score": brier,
        "expected_calibration_error": float(ece),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }
