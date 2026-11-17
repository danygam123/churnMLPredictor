"""Reusable, leak-free preprocessing pipeline for C.R.A.M. V2.

Handles missing values, duplicate records, categorical variables, numerical variables,
feature scaling, unexpected types, and validation of customer records using scikit-learn
ColumnTransformer and Pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Canonical input schema (13 features collected from user or base records)
INPUT_FEATURES: List[str] = [
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

# Charge rates documented from research dataset
CHARGE_RATES: Dict[str, Tuple[str, float]] = {
    "Total day charge": ("Total day minutes", 0.17),
    "Total eve charge": ("Total eve minutes", 0.085),
    "Total night charge": ("Total night minutes", 0.045),
    "Total intl charge": ("Total intl minutes", 0.27),
}

# Complete model feature schema (17 features)
ALL_FEATURES: List[str] = [
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

NUMERICAL_INPUTS: List[str] = [
    "Account length",
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

CATEGORICAL_INPUTS: List[str] = [
    "International plan",
    "Voice mail plan",
]

INTEGER_FIELDS: set[str] = {
    "Account length",
    "Number vmail messages",
    "Total day calls",
    "Total eve calls",
    "Total night calls",
    "Total intl calls",
    "Customer service calls",
}

VALID_RANGES: Dict[str, Tuple[float, float]] = {
    "Account length": (0.0, 500.0),
    "International plan": (0.0, 1.0),
    "Voice mail plan": (0.0, 1.0),
    "Number vmail messages": (0.0, 150.0),
    "Total day minutes": (0.0, 600.0),
    "Total day calls": (0.0, 300.0),
    "Total eve minutes": (0.0, 600.0),
    "Total eve calls": (0.0, 300.0),
    "Total night minutes": (0.0, 600.0),
    "Total night calls": (0.0, 300.0),
    "Total intl minutes": (0.0, 100.0),
    "Total intl calls": (0.0, 100.0),
    "Customer service calls": (0.0, 50.0),
}


class ChargeFeatureDeriver(BaseEstimator, TransformerMixin):
    """Derives telecommunication charge features deterministically from usage minutes.

    Ensures that whether data arrives with 13 input features or 17 features,
    the output maintains consistent calculation and ordering.
    """

    def __init__(self, charge_rates: Optional[Dict[str, Tuple[str, float]]] = None):
        self.charge_rates = charge_rates or CHARGE_RATES

    def fit(self, X: pd.DataFrame, y: Any = None) -> "ChargeFeatureDeriver":
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=INPUT_FEATURES)
        else:
            df = X.copy()

        for charge, (minutes_col, rate) in self.charge_rates.items():
            if minutes_col in df.columns:
                df[charge] = (pd.to_numeric(df[minutes_col], errors="coerce") * rate).round(2)
            else:
                df[charge] = 0.0

        for col in ALL_FEATURES:
            if col not in df.columns:
                df[col] = np.nan

        return df[ALL_FEATURES]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.array(ALL_FEATURES, dtype=object)


class RobustBinaryCleaner(BaseEstimator, TransformerMixin):
    """Cleans and standardizes binary text/categorical inputs (Yes/No, true/false, 1/0)."""

    def fit(self, X: Any, y: Any = None) -> "RobustBinaryCleaner":
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X, columns=CATEGORICAL_INPUTS[:X.shape[1]])
        else:
            df = X.copy()
        mapping = {
            "yes": 1.0, "y": 1.0, "true": 1.0, "1": 1.0, "1.0": 1.0,
            "no": 0.0, "n": 0.0, "false": 0.0, "0": 0.0, "0.0": 0.0,
        }
        for col in df.columns:
            s = df[col]
            str_vals = s.astype(str).str.strip().str.lower()
            mapped = str_vals.map(mapping)
            numeric = pd.to_numeric(s, errors="coerce")
            df[col] = mapped.where(mapped.notna(), numeric)
        return df

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is not None:
            return np.array(input_features, dtype=object)
        return np.array(CATEGORICAL_INPUTS, dtype=object)


@dataclass
class ValidationReport:
    """Detailed summary of dataset hygiene and validation."""
    total_records: int = 0
    duplicate_records: int = 0
    missing_value_cells: int = 0
    missing_by_column: Dict[str, int] = field(default_factory=dict)
    invalid_range_cells: int = 0
    unexpected_type_cells: int = 0
    warnings: List[str] = field(default_factory=list)
    cleaned_df: Optional[pd.DataFrame] = None


class CustomerDataValidator:
    """Validates raw customer records before ingestion into model pipelines."""

    def __init__(self, allowed_ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        self.allowed_ranges = allowed_ranges or VALID_RANGES

    def validate(self, data: pd.DataFrame, remove_duplicates: bool = True) -> ValidationReport:
        report = ValidationReport(total_records=len(data))
        df = data.copy()

        # 1. Duplicate check
        dup_mask = df.duplicated()
        report.duplicate_records = int(dup_mask.sum())
        if report.duplicate_records > 0:
            report.warnings.append(f"Found {report.duplicate_records} exact duplicate row(s).")
            if remove_duplicates:
                df = df.drop_duplicates().reset_index(drop=True)
                report.warnings.append("Duplicate rows removed from active pipeline.")

        # 2. Check for missing required columns
        for col in INPUT_FEATURES:
            if col not in df.columns:
                report.warnings.append(f"Missing column '{col}' will be initialized with NaN.")
                df[col] = np.nan

        # 3. Clean binary / categorical variables
        binary_cols = [c for c in CATEGORICAL_INPUTS if c in df.columns]
        bin_cleaner = RobustBinaryCleaner()
        df[binary_cols] = bin_cleaner.transform(df[binary_cols])

        # 4. Check numerical inputs & types
        for col in NUMERICAL_INPUTS:
            if col in df.columns:
                orig_series = df[col]
                numeric_series = pd.to_numeric(orig_series, errors="coerce")
                unexpected = orig_series.notna() & numeric_series.isna()
                if unexpected.any():
                    count = int(unexpected.sum())
                    report.unexpected_type_cells += count
                    report.warnings.append(f"Column '{col}' had {count} non-numeric values coerced to NaN.")
                df[col] = numeric_series

        # 5. Missing value accounting
        col_missing = df[INPUT_FEATURES].isna().sum()
        report.missing_by_column = col_missing[col_missing > 0].to_dict()
        report.missing_value_cells = int(col_missing.sum())

        # 6. Check range bounds
        for col, (low, high) in self.allowed_ranges.items():
            if col in df.columns:
                out_of_bounds = df[col].notna() & ((df[col] < low) | (df[col] > high))
                if out_of_bounds.any():
                    count = int(out_of_bounds.sum())
                    report.invalid_range_cells += count
                    report.warnings.append(f"Column '{col}' had {count} value(s) outside [{low}, {high}]. Coerced to NaN.")
                    df.loc[out_of_bounds, col] = np.nan

        # 7. Integer rounding where appropriate
        for col in INTEGER_FIELDS:
            if col in df.columns:
                df[col] = df[col].apply(lambda v: round(v) if pd.notna(v) else np.nan)

        report.cleaned_df = df
        return report


def build_preprocessing_pipeline(
    scale_numeric: bool = False,
    include_charges: bool = True,
) -> Pipeline:
    """Builds a scikit-learn Pipeline with ColumnTransformer for C.R.A.M. V2.

    Parameters
    ----------
    scale_numeric : bool, default=False
        Whether to apply StandardScaler to numerical features. Recommended True
        for linear models (LogisticRegression), False for tree ensembles (RandomForest, XGBoost).
    include_charges : bool, default=True
        Whether the pipeline should derive the 4 charge rates from minutes.

    Returns
    -------
    Pipeline
        Fit-ready scikit-learn Pipeline.
    """
    # Numerical transformer: median imputation + optional standard scaling
    num_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numerical_transformer = Pipeline(steps=num_steps)

    # Categorical transformer: most-frequent imputation + pass through 0/1 numeric
    cat_steps: List[Tuple[str, Any]] = [
        ("cleaner", RobustBinaryCleaner()),
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ]
    categorical_transformer = Pipeline(steps=cat_steps)

    features_to_transform = ALL_FEATURES if include_charges else INPUT_FEATURES
    all_num = [f for f in features_to_transform if f in NUMERICAL_INPUTS or f in CHARGE_RATES]
    all_cat = [f for f in features_to_transform if f in CATEGORICAL_INPUTS]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, all_num),
            ("cat", categorical_transformer, all_cat),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    steps: List[Tuple[str, Any]] = []
    if include_charges:
        steps.append(("derive_charges", ChargeFeatureDeriver()))
    steps.append(("col_transform", preprocessor))

    return Pipeline(steps=steps)


def get_feature_names_after_preprocessing(pipeline: Pipeline) -> List[str]:
    """Extracts output feature names from the fitted pipeline."""
    col_transform = pipeline.named_steps.get("col_transform")
    if col_transform and hasattr(col_transform, "get_feature_names_out"):
        return list(col_transform.get_feature_names_out())
    return ALL_FEATURES
