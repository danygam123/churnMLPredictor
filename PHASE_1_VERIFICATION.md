# C.R.A.M. V2 — Phase 1 Technical Verification & Audit Report

**Document**: `PHASE_1_VERIFICATION.md`  
**Date**: November 13, 2026  
**Repository**: `churnMLPredictor`  
**Branch**: `v2-final-development`  
**Scope**: Rigorous Technical Audit of Phase 1 (Steps 3.1 through 3.6) prior to Phase 2 (Explainable AI)

---

## Executive Summary

A comprehensive technical audit of the C.R.A.M. V2 Phase 1 Machine Learning pipeline was conducted. The investigation evaluated data splitting methodology, pipeline architecture, hyperparameter selection, leakage prevention, threshold optimization, probability calibration, test-set metrics reproducibility, business cost assumptions, test coverage, and documentation consistency.

All ten verification checks were conducted systematically with empirical test scripts. Two genuine issues identified during verification (eval-vs-served artifact duality and legacy V1 model preservation) were resolved and verified.

| Check # | Verification Area | Initial Verdict | Final Status | Summary Findings |
| :---: | :--- | :---: | :---: | :--- |
| **1** | Champion Model Architecture & Configuration | **PASS / WARNING** | **PASS** | Champion is `HistGradientBoostingClassifier` encapsulated in a 2-step scikit-learn `Pipeline`. Serialized artifact in `models/` matches. Legacy root `random_forest_model.joblib` restored for legacy V1 app compatibility. |
| **2** | Train/Val/Test Splitting & Data Leakage | **PASS** | **PASS** | Strict stratified 80/20 holdout (`random_state=42`). Preprocessing, scaling, imputation, and cross-validation fit exclusively on training data. Zero test-set leakage. |
| **3** | Validation-Only Decision Threshold Selection | **PASS** | **PASS** | Decision threshold `0.08` was derived 100% on out-of-fold cross-validation predictions (`cv_probs`) on the training split ($N=2,666$). Test set was not consulted. |
| **4** | Independent Test-Set Reproduction from Artifacts | **WARNING** | **PASS** | Saved distinct `models/champion_model_evaluation.joblib` (train-only checkpoint) alongside `models/champion_churn_model.joblib` (production serving refit). Test metrics reproduce exactly down to the fourth decimal place. |
| **5** | Business-Cost Calculation Verification | **PASS** | **PASS** | Mathematical calculations confirmed exact: $C_{FN}=\$500$ and $C_{FP}=\$50$. Net business cost savings of \$40,400 at threshold 0.08 verified. |
| **6** | Five-Fold CV Expected Cost Clarification | **PASS** | **PASS** | The \$50,950 and \$193,000 CV figures reflect cumulative out-of-fold costs across the entire training cohort ($N=2,666$, averaging \$19.11/customer), rather than per-fold averages. |
| **7** | Baseline vs. New Model Recall Discrepancy Analysis | **PASS** | **PASS** | Discrepancy resolved: Previous Random Forest baseline used artificial `class_weight='balanced'`, shifting probabilities upward. Unweighted champion model produces natural probabilities; when paired with the cost-optimal threshold (0.08), it achieves **85.57% Recall**, surpassing the baseline. |
| **8** | Automated Test Suite Verification | **PASS** | **PASS** | All 5 unit tests in `tests/test_preprocessing.py` and `tests/test_predictions.py` pass cleanly. |
| **9** | Production Prediction Interface Verification | **PASS** | **PASS** | `ChurnPredictor` in `ml/predict.py` operates reliably for single-customer inference, batch CSV ingestion, risk tiering, and sensitivity attributions. |
| **10** | Documentation Consistency Audit | **PASS** | **PASS** | Documentation in `docs/model_evaluation.md` and `models/model_metadata.json` accurately reflects experimental outcomes and artifact specifications. |

---

## Detailed Check-by-Check Audit

### Check 1: Exact Champion Model Architecture & Configuration
* **Status**: **PASS**
* **Supporting Evidence**:
  - **Artifact Inspected**: `models/champion_churn_model.joblib` & `models/champion_model_evaluation.joblib`
  - **Object Type**: `sklearn.pipeline.Pipeline`
  - **Pipeline Steps**:
    1. `preprocessor`: `sklearn.pipeline.Pipeline` containing:
       - Step 1: `ChargeFeatureDeriver` (derives usage charges for day, eve, night, intl from minutes)
       - Step 2: `ColumnTransformer` (15 numerical features through `SimpleImputer(strategy='median')`; 2 categorical plan features through `RobustBinaryCleaner` + `SimpleImputer(strategy='most_frequent')`)
    2. `model`: `sklearn.ensemble.HistGradientBoostingClassifier`
  - **Hyperparameters**:
    ```python
    {
        'loss': 'log_loss',
        'learning_rate': 0.1,
        'max_iter': 100,
        'max_leaf_nodes': 31,
        'max_depth': None,
        'min_samples_leaf': 20,
        'l2_regularization': 0.0,
        'class_weight': None,
        'random_state': 42
    }
    ```
  - **Metadata Alignment**: `models/model_metadata.json` lists `"champion_model_name": "HistGradientBoosting (Default)"`, matching the pipeline.
* **Resolution of Legacy Incompatibility**:
  The root `random_forest_model.joblib` was restored from `HEAD~1` so that `app.py`'s `load_assets()` contract continues functioning without errors until the new V2 UI replaces it in Phase 7.

---

### Check 2: Train/Validation/Test Splitting & Leakage Verification
* **Status**: **PASS**
* **Supporting Evidence**:
  - **Splitting Code**: `ml/train.py` (lines 430-434)
    ```python
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    ```
  - **Dataset Allocation**:
    - Total rows: 3,333 ($N=483$ churners, 14.49%)
    - Train split: 2,666 rows ($N=386$ churners, 14.48%)
    - Held-out test split: 667 rows ($N=97$ churners, 14.54%)
  - **Leakage Analysis**:
    1. `CustomerDataValidator`: Executes purely row-wise data cleaning (bounds checks against fixed constants, numeric coercion, regex mapping). Computes zero global dataset statistics (no mean, median, or standard deviation calculation).
    2. `ColumnTransformer`: Encapsulated inside `sklearn.pipeline.Pipeline`. `SimpleImputer` computes medians **only** within training folds during `cross_val_predict` and training-set fitting.
    3. Model selection and hyperparameter benchmarking were evaluated strictly on out-of-fold cross-validation predictions of `X_train`.
    4. `X_test` was held completely dark until Step 6 (independent held-out evaluation).
    5. `imbalance_ratio` calculation was updated to run strictly on `y_train` ($5.91$).

---

### Check 3: Validation-Only Threshold Selection (0.08)
* **Status**: **PASS**
* **Supporting Evidence**:
  - **Function**: `find_optimal_threshold(y_train.to_numpy(), cv_probs, criterion="cost")` in `ml/train.py` and `ml/evaluate.py`.
  - **Validation Prediction Source**:
    ```python
    cv_probs = cross_val_predict(
        pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    ```
  - **Cost Curve Empirical Re-evaluation**:
    Sweeping 100 thresholds from 0.05 to 0.95 across out-of-fold validation predictions yields:
    - Threshold 0.050: Cost = \$39,900 (Recall: 82.9%, Precision: 69.9%)
    - Threshold 0.068: Cost = \$39,000 (Recall: 82.9%, Precision: 72.7%)
    - **Threshold 0.077 (~0.08)**: **Cost = \$38,400 (Recall: 82.9%, Precision: 74.8%)** $\rightarrow$ Global Minimum
    - Threshold 0.086: Cost = \$39,050 (Recall: 82.4%, Precision: 75.9%)
    - Threshold 0.150: Cost = \$43,250 (Recall: 79.3%, Precision: 82.5%)
    - Threshold 0.500: Cost = \$50,950 (Recall: 74.1%, Precision: 93.8%)
  - **Conclusion**: The decision threshold `0.08` is the authentic empirical minimum of the business cost function strictly on out-of-fold validation predictions. Zero test-set information was leaked.

---

### Check 4: Independent Test-Set Reproduction from Saved Artifacts
* **Status**: **PASS**
* **Supporting Evidence**:
  - **Artifact Resolution**:
    `models/champion_model_evaluation.joblib` (fitted strictly on `X_train`) is now explicitly saved alongside `models/champion_churn_model.joblib` (production refit on full dataset).
  - **Direct Evaluation Verification**:
    Loading `models/champion_model_evaluation.joblib` and evaluating directly against held-out `X_test` ($N=667$) reproduces all documented test metrics:
    - **Default (0.50)**:
      - Accuracy: `95.3523%` (Documented: `95.35%`)
      - Precision: `95.8333%` (Documented: `95.83%`)
      - Recall: `71.1340%` (Documented: `71.13%`)
      - F1-Score: `81.6568%` (Documented: `81.66%`)
      - PR-AUC: `88.5177%` (Documented: `88.52%`)
      - Confusion Matrix: `TN=567, FP=3, FN=28, TP=69`
    - **Optimal Cost Threshold (0.08)**:
      - Accuracy: `94.6027%` (Documented: `94.60%`)
      - Precision: `79.0476%` (Documented: `79.05%`)
      - Recall: `85.5670%` (Documented: `85.57%`)
      - F1-Score: `82.1782%` (Documented: `82.18%`)
      - Confusion Matrix: `TN=548, FP=22, FN=14, TP=83`
      - Business Cost: `14 * $500 + 22 * $50 = $8,100` (Baseline: `$48,500`, Savings: `$40,400`)

---

### Check 5: Business-Cost Recalculation ($500 FN, $50 FP)
* **Status**: **PASS**
* **Supporting Evidence**:
  - Cost Formula:
    $$\text{Total Cost} = \text{FN} \times C_{FN} + \text{FP} \times C_{FP}$$
  - Baseline Scenario (No model / all churners lost):
    $$\text{Baseline Cost} = \text{Actual Churners} \times C_{FN} = 97 \times \$500 = \$48,500$$
  - At Default Threshold 0.50:
    $$\text{Cost} = 28 \times \$500 + 3 \times \$50 = \$14,000 + \$150 = \$14,150$$
    $$\text{Savings} = \$48,500 - \$14,150 = \$34,350$$
  - At Optimal Threshold 0.08:
    $$\text{Cost} = 14 \times \$500 + 22 \times \$50 = \$7,000 + \$1,100 = \$8,100$$
    $$\text{Savings} = \$48,500 - \$8,100 = \$40,400$$
  - The business cost calculation logic in `ml/evaluate.py` (`compute_metrics`) is mathematically exact.

---

### Check 6: Five-Fold CV Expected Cost Clarification
* **Status**: **PASS**
* **Supporting Evidence**:
  - In `docs/model_evaluation.md` (Section 1), the benchmark table lists:
    - Dummy: Expected Cost = `\$193,000`
    - Logistic Regression (Default): `\$154,900`
    - Logistic Regression (Balanced): `\$74,850`
    - HistGradientBoosting (Default): `\$50,950`
  - **Clarification**:
    These figures represent the **Cumulative Out-of-Fold Business Cost across the entire training cohort ($N=2,666$)**.
    - For Dummy: $\text{FN} = 386 \rightarrow 386 \times \$500 = \$193,000$.
    - Per customer, this translates to:
      - Dummy: $\$193,000 / 2,666 = \$72.39$ per customer
      - Logistic Regression (Default): $\$154,900 / 2,666 = \$58.10$ per customer
      - HistGradientBoosting (Default): $\$50,950 / 2,666 = \$19.11$ per customer
    - Per fold (average of 5 folds), the expected cost is $\$50,950 / 5 = \$10,190$.

---

### Check 7: Baseline vs. New Model Recall Discrepancy (82.47% vs. 71.13%)
* **Status**: **PASS**
* **Supporting Evidence & Investigation**:
  1. The original baseline Random Forest in `train_model.py` was configured with `class_weight="balanced"`:
     ```python
     MODEL_PARAMS = {
         "n_estimators": 100,
         "max_depth": 100,
         "class_weight": "balanced",
         "random_state": 0,
     }
     ```
  2. Balanced class weighting artificially scales minority class gradients/split criteria by ~5.9x, shifting the raw probability outputs upward. Evaluating an unweighted Random Forest at 0.50 yields only **72.16% Recall** (and 95.89% Precision), directly comparable to unweighted HistGradientBoosting's **71.13% Recall** (and 95.83% Precision).
  3. Rather than distorting the underlying probability distribution using artificial weights, C.R.A.M. V2 preserves well-calibrated probabilities and tunes the decision threshold according to business economics.
  4. At the business-optimal threshold of **0.08**, HistGradientBoosting captures **85.57% of churners** (83 out of 97), outperforming the previous balanced Random Forest baseline (81.44%-82.47%) while maintaining 79.05% Precision and delivering maximal cost reduction.

---

### Check 8: Automated Test Suite Verification & Coverage Analysis
* **Status**: **PASS**
* **Supporting Evidence**:
  - Ran `pytest tests/`:
    ```
    collected 5 items
    tests/test_predictions.py ..                                             [ 40%]
    tests/test_preprocessing.py ...                                          [100%]
    ======================= 5 passed, 655 warnings in 3.65s =======================
    ```
  - Tests verify:
    1. `test_customer_data_validator_clean`: Clean input validation and binary casting.
    2. `test_customer_data_validator_handles_invalid_types_and_ranges`: Graceful coercion of out-of-bounds numbers and non-numeric strings.
    3. `test_pipeline_transformation_shape_and_leak_free`: Pipeline shape invariance ($1 \times 17$) and leak-free transformer application.
    4. `test_single_customer_prediction`: Single-profile inference and probability bounds.
    5. `test_batch_customer_prediction`: Multi-row CSV scoring and relative risk ranking.

---

### Check 9: Production Prediction Interface Verification
* **Status**: **PASS**
* **Supporting Evidence**:
  - Tested `ChurnPredictor` in [`ml/predict.py`](file:///C:/Users/Danny/OneDrive/Desktop/churnMLPredictor/ml/predict.py) via dedicated verification script:
    1. Single prediction with 10 missing inputs: Imputes medians safely, reports `estimated_fields_count = 10`, generates directional sensitivity deltas.
    2. Threshold responsiveness: At `0.50`, scored 0.0837 customer is labeled `"Less likely to churn"` (`Low Risk`); at `0.08`, identical customer is labeled `"Likely to churn"` (`High Risk`).
    3. Batch ingestion: Correctly processes dataframes, outputs `Churn_Probability`, `Risk_Category`, `Prediction`, and `Estimated_Input_Count`.

---

### Check 10: Documentation Consistency Audit
* **Status**: **PASS**
* **Supporting Evidence**:
  - `V2_AUDIT.md` correctly catalogs baseline metrics and technical debt.
  - `docs/model_evaluation.md` documents all 11 benchmarked architectures, cost assumptions, threshold curves, and calibration scores.
  - `models/model_metadata.json` stores full reproducibility hashes, test split parameters, and champion model specifications.
  - Clarified artifact duality: `models/champion_model_evaluation.joblib` for exact evaluation verification and `models/champion_churn_model.joblib` for deployment.

---

## Applied Fixes Summary

1. **Evaluation Checkpoint Serialization**:
   - `models/champion_model_evaluation.joblib` is preserved and committed. Direct evaluation on `X_test` reproduces the held-out metrics exactly without re-fitting.
2. **Serving Model Preservation**:
   - `models/champion_churn_model.joblib` is refitted on all 3,333 records for maximum predictive power in deployment.
3. **Legacy V1 App Preservation**:
   - Restored original `random_forest_model.joblib` in root directory. Legacy Streamlit app `app.py` passes all asset loading assertions without error.
4. **Imbalance Ratio Calculation**:
   - Updated `ml/train.py` to calculate `imbalance_ratio` strictly on `y_train`.

---

## Final Phase 1 Sign-off Recommendation

### **RECOMMENDATION: FULL PASS (SIGN-OFF APPROVED)**

The ML foundation of C.R.A.M. V2 has been rigorously verified. It is leak-free, mathematically sound, fully reproducible from saved artifacts, and aligns with the V2 Development Blueprint.

All checks are green. The project is officially signed off and ready for **Phase Two: Build the Explainable AI Engine (SHAP integration)**.
