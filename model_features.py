"""Shared input schema and deterministic feature preparation for the churn model."""

from __future__ import annotations

import pandas as pd


INPUT_FEATURES = [
    "Account length",
    "International plan",
    "Voice mail plan",
    "Number vmail messages",
    "Total day minutes",
    "Total day calls",
    "Total eve minutes",
    "Total eve calls",
    "Total night minutes",
    "Total night calls",
    "Total intl minutes",
    "Total intl calls",
    "Customer service calls",
]

CHARGE_RATES = {
    "Total day charge": ("Total day minutes", 0.17),
    "Total eve charge": ("Total eve minutes", 0.085),
    "Total night charge": ("Total night minutes", 0.045),
    "Total intl charge": ("Total intl minutes", 0.27),
}

FEATURES = [
    "Account length",
    "International plan",
    "Voice mail plan",
    "Number vmail messages",
    "Total day minutes",
    "Total day calls",
    "Total day charge",
    "Total eve minutes",
    "Total eve calls",
    "Total eve charge",
    "Total night minutes",
    "Total night calls",
    "Total night charge",
    "Total intl minutes",
    "Total intl calls",
    "Total intl charge",
    "Customer service calls",
]


def build_model_matrix(inputs, medians) -> pd.DataFrame:
    """Fill unknown inputs and derive usage charges in the trained feature order."""
    if isinstance(inputs, dict):
        frame = pd.DataFrame([inputs])
    else:
        frame = pd.DataFrame(inputs).copy()
    for column in INPUT_FEATURES:
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].fillna(medians[column])
    for charge, (minutes, rate) in CHARGE_RATES.items():
        frame[charge] = (frame[minutes] * rate).round(2)
    return frame[FEATURES]
