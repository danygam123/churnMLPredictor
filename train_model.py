"""Rebuild and evaluate the telecom churn model used by the Streamlit app.

The optional --download flag retrieves the two CSV files referenced by the
original Colab notebook. CSV files and generated artifacts stay local to the
project; raw data is ignored by git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from model_features import CHARGE_RATES, FEATURES, INPUT_FEATURES, build_model_matrix


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "random_forest_model.joblib"
METADATA_PATH = ROOT / "model_metadata.json"
SOURCES = {
    "churn-20.csv": "1rNbCnwc4RfNxcZZfMuWjJPK8QSivVumx",
    "churn-80.csv": "1wsx-kXJVImWHaTpRjXsFPU5UU1lWAvKZ",
}
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 100,
    "class_weight": "balanced",
    "random_state": 0,
}
TEST_SIZE = 0.2
SPLIT_SEED = 42


def download_sources() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, file_id in SOURCES.items():
        destination = RAW_DIR / filename
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        with urlopen(url, timeout=60) as response:
            destination.write_bytes(response.read())
        print(f"Downloaded {destination.relative_to(ROOT)}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data() -> tuple[pd.DataFrame, pd.Series, dict[str, str]]:
    paths = [RAW_DIR / name for name in SOURCES]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(missing)}. Run: python train_model.py --download"
        )

    frames = [pd.read_csv(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    required = set(FEATURES) | {"Churn", "State", "Area code"}
    absent = sorted(required - set(data.columns))
    if absent:
        raise ValueError(f"Source data is missing columns: {absent}")
    if data[list(required)].isna().any().any():
        raise ValueError("Source data contains missing required values.")

    data["International plan"] = data["International plan"].map(
        {"No": 0, "Yes": 1}
    )
    data["Voice mail plan"] = data["Voice mail plan"].map({"No": 0, "Yes": 1})
    if data[["International plan", "Voice mail plan"]].isna().any().any():
        raise ValueError("Plan columns contain values other than 'No' or 'Yes'.")
    y = data["Churn"].astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Churn target must contain only False/True or 0/1 values.")
    X = data[INPUT_FEATURES].astype(float)
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Feature data contains non-finite values.")
    derived = build_model_matrix(X, X.median())
    for charge in CHARGE_RATES:
        if not np.allclose(derived[charge], data[charge], atol=0.011):
            raise ValueError(f"The source data does not match the expected {charge} formula.")
    hashes = {path.name: sha256(path) for path in paths}
    return data[FEATURES].astype(float), y, hashes


def evaluate(y_true: pd.Series, predictions, probabilities) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "churn_precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "churn_recall": float(recall_score(y_true, predictions, zero_division=0)),
        "churn_f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc_average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["no_churn", "churn"],
            output_dict=True,
            zero_division=0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the two source CSVs from the Google Drive IDs in the notebook.",
    )
    args = parser.parse_args()
    if args.download:
        download_sources()

    X, y, hashes = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SPLIT_SEED,
        stratify=y,
    )
    input_train = X_train[INPUT_FEATURES]
    input_test = X_test[INPUT_FEATURES]
    train_medians = input_train.median()
    model_train = build_model_matrix(input_train, train_medians)
    model_test = build_model_matrix(input_test, train_medians)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(model_train, y_train)
    baseline_predictions = baseline.predict(model_test)

    evaluation_model = RandomForestClassifier(**MODEL_PARAMS)
    evaluation_model.fit(model_train, y_train)
    test_predictions = evaluation_model.predict(model_test)
    test_probabilities = evaluation_model.predict_proba(model_test)[:, 1]

    # Check how median-filled inputs behave when a customer does not know
    # some fields. Each scenario is evaluated with five different masks.
    missing_data_metrics = {}
    for missing_count in (1, 3, 5):
        repeats = []
        for repeat in range(5):
            rng = np.random.default_rng(1000 + missing_count * 10 + repeat)
            masked = input_test.copy()
            mask = np.zeros(masked.shape, dtype=bool)
            for row_index in range(len(masked)):
                columns = rng.choice(len(INPUT_FEATURES), size=missing_count, replace=False)
                mask[row_index, columns] = True
            masked = masked.mask(mask)
            filled_test = build_model_matrix(masked, train_medians)
            probabilities = evaluation_model.predict_proba(filled_test)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            repeats.append(evaluate(y_test, predictions, probabilities))
        summary_fields = (
            "accuracy",
            "churn_precision",
            "churn_recall",
            "churn_f1",
            "roc_auc",
            "pr_auc_average_precision",
        )
        missing_data_metrics[str(missing_count)] = {
            field: {
                "mean": float(np.mean([result[field] for result in repeats])),
                "std": float(np.std([result[field] for result in repeats])),
            }
            for field in summary_fields
        }

    # The one-time held-out test evaluation is complete. Refit the selected
    # configuration on all rows for the model that the app will serve.
    final_model = RandomForestClassifier(**MODEL_PARAMS)
    full_medians = X[INPUT_FEATURES].median()
    final_training_matrix = build_model_matrix(X[INPUT_FEATURES], full_medians)
    final_model.fit(final_training_matrix, y)
    joblib.dump(final_model, MODEL_PATH, compress=3)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "source": "Two CSV files referenced by the attached Colab notebook",
            "files": hashes,
            "rows": int(len(X)),
            "feature_count": int(X.shape[1]),
            "user_input_feature_count": len(INPUT_FEATURES),
            "churn_rows": int(y.sum()),
            "churn_rate": float(y.mean()),
            "duplicate_rows": int(pd.concat([X, y.rename("Churn")], axis=1).duplicated().sum()),
        },
        "target": {"name": "Churn", "positive_class": 1},
        "features": FEATURES,
        "input_features": INPUT_FEATURES,
        "input_ranges": {
            column: {
                "min": float(X[column].min()),
                "max": float(X[column].max()),
                "median": float(X[column].median()),
            }
            for column in INPUT_FEATURES
        },
        "charge_formulas": {
            charge: {"minutes_feature": minutes, "rate": rate, "rounding": 2}
            for charge, (minutes, rate) in CHARGE_RATES.items()
        },
        "split": {
            "strategy": "single stratified holdout",
            "test_size": TEST_SIZE,
            "random_state": SPLIT_SEED,
            "training_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "baseline": {
            "strategy": "most_frequent",
            "test_accuracy": float(accuracy_score(y_test, baseline_predictions)),
            "test_churn_recall": float(
                recall_score(y_test, baseline_predictions, zero_division=0)
            ),
        },
        "evaluation_model": {
            "estimator": "RandomForestClassifier",
            "parameters": MODEL_PARAMS,
            "test_metrics": evaluate(y_test, test_predictions, test_probabilities),
            "median_imputation_sensitivity": missing_data_metrics,
        },
        "served_model": {
            "path": MODEL_PATH.name,
            "fit_rows": int(len(X)),
            "imputation_values": {
                column: float(value) for column, value in full_medians.items()
            },
            "note": "Refit on all rows after the held-out evaluation; use held-out metrics above as the performance estimate.",
        },
        "versions": {
            "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}.{__import__('sys').version_info.micro}",
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {MODEL_PATH.name} and {METADATA_PATH.name}")
    print(json.dumps(metadata["evaluation_model"]["test_metrics"], indent=2))
    print("Baseline:", json.dumps(metadata["baseline"]))


if __name__ == "__main__":
    main()
