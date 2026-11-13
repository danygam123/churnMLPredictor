"""Unit tests for C.R.A.M. V2 preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest
from ml.preprocessing import (
    ALL_FEATURES,
    CHARGE_RATES,
    INPUT_FEATURES,
    CustomerDataValidator,
    build_preprocessing_pipeline,
    get_feature_names_after_preprocessing,
)


def test_customer_data_validator_clean():
    validator = CustomerDataValidator()
    raw = pd.DataFrame([{
        "Account length": 100,
        "International plan": "yes",
        "Voice mail plan": "no",
        "Number vmail messages": 0,
        "Total day minutes": 120.0,
        "Total day calls": 80,
        "Total eve minutes": 100.0,
        "Total eve calls": 70,
        "Total night minutes": 90.0,
        "Total night calls": 60,
        "Total intl minutes": 10.0,
        "Total intl calls": 4,
        "Customer service calls": 2,
    }])
    report = validator.validate(raw)
    assert report.total_records == 1
    assert report.duplicate_records == 0
    assert report.cleaned_df["International plan"].iloc[0] == 1.0
    assert report.cleaned_df["Voice mail plan"].iloc[0] == 0.0


def test_customer_data_validator_handles_invalid_types_and_ranges():
    validator = CustomerDataValidator()
    raw = pd.DataFrame([{
        "Account length": "invalid_number",
        "International plan": "not_sure",
        "Voice mail plan": "True",
        "Number vmail messages": 99999,  # Out of range
        "Total day minutes": -50.0,  # Negative out of range
        "Total day calls": 80,
        "Total eve minutes": 100.0,
        "Total eve calls": 70,
        "Total night minutes": 90.0,
        "Total night calls": 60,
        "Total intl minutes": 10.0,
        "Total intl calls": 4,
        "Customer service calls": 2,
    }])
    report = validator.validate(raw)
    assert pd.isna(report.cleaned_df["Account length"].iloc[0])
    assert pd.isna(report.cleaned_df["Number vmail messages"].iloc[0])
    assert pd.isna(report.cleaned_df["Total day minutes"].iloc[0])
    assert report.cleaned_df["Voice mail plan"].iloc[0] == 1.0


def test_pipeline_transformation_shape_and_leak_free():
    pipe = build_preprocessing_pipeline(scale_numeric=True)
    df_train = pd.DataFrame([{feat: 10.0 for feat in INPUT_FEATURES}])
    df_train["International plan"] = "No"
    df_train["Voice mail plan"] = "Yes"

    # Fit on train
    pipe.fit(df_train)
    out_train = pipe.transform(df_train)
    assert out_train.shape == (1, 17)

    # Transform new test row without refitting
    df_test = pd.DataFrame([{feat: 20.0 for feat in INPUT_FEATURES}])
    df_test["International plan"] = "Yes"
    df_test["Voice mail plan"] = "No"
    out_test = pipe.transform(df_test)
    assert out_test.shape == (1, 17)
    assert not np.isnan(out_test).any()
