# C.R.A.M. V2 — System Audit & Baseline Report (Phase Zero)

**Document**: `V2_AUDIT.md`  
**Date**: November 2026  
**Repository**: `churnMLPredictor`  
**Branch**: `v2-final-development`  
**Target**: C.R.A.M. V2 (Customer Retention & Analytics Model)

---

## 1. Executive Summary

This audit assesses the pre-V2 architecture of C.R.A.M. across both its Streamlit application and Vercel Next.js/Python serverless prototype. The audit establishes the baseline model performance, evaluates technical debt, documents reproducibility risks, and categorizes every core capability into standard status flags: **Working**, **Partially working**, **Broken**, **Missing**, or **Needs improvement**.

---

## 2. Component-by-Component Audit

| Component / Subsystem | Current Implementation | Status | Notes & Action Required |
| :--- | :--- | :--- | :--- |
| **Streamlit Interface & Pages** | Monolithic single-file app (`app.py`) with tab/guided chat views | **Partially working** | Needs modular multi-page hierarchy (`app/pages/` or standard navigation: Overview, Customers, Customer Insights, Retention Strategies, Simulation Lab, Model Center). Currently UI is tightly coupled with inference logic. |
| **Prediction Functionality** | Single profile scoring & batch scoring functions in `app.py` & Vercel API | **Working** | Functions produce churn probability and threshold flag. Needs decoupling into reusable `ml/predict.py`. |
| **Dataset Preprocessing** | Hardcoded pandas transformations in `model_features.py` using manual median imputation | **Needs improvement** | Lacks standard scikit-learn `Pipeline` and `ColumnTransformer`. Leakage potential if applied across full dataset prior to splitting. Must be encapsulated into reproducible training/serving pipeline. |
| **Model Training Scripts** | `train_model.py` runs random forest training on concatenated 80/20 CSVs | **Partially working** | Does not benchmark competing architectures (Logistic Regression, XGBoost, LightGBM, HistGradientBoosting). Uses fixed hyperparameters with no systematic threshold optimization. |
| **Model Serialization** | `joblib.dump(..., compress=3)` saving `random_forest_model.joblib` | **Working** | Model and metadata are saved cleanly; needs migration into dedicated `models/` directory with version tracking. |
| **Charts and Metrics** | Basic Streamlit metrics & Altair/matplotlib bar charts | **Needs improvement** | Needs interactive Plotly visualizations (risk distributions, ROC/PR curves, confusion matrices, SHAP breakdown) with dark enterprise aesthetic. |
| **Input Validation** | Manual regex checks and range filters in `app.py` | **Partially working** | Fragile for production datasets; requires robust schema validation (e.g., Pydantic or Pandera) for customer CSV uploads. |
| **Dependencies & Environment** | Frozen `requirements.txt` with Python 3.14+ dependencies | **Working** | Upgraded with `xgboost`, `lightgbm`, `shap` in local virtual environment. |
| **Deployment Configuration** | Streamlit Cloud configuration + Vercel Next.js hybrid config | **Working** | Streamlit community deployment active; Next.js edition present in `vercel-app/`. |
| **Explainable AI (SHAP)** | Crude 1-feature perturbation sensitivity in `app.py` | **Needs improvement** | Full TreeSHAP / LinearSHAP explainability engine missing; required for global and local customer waterfall attribution. |
| **Retention Recommendations** | Ad-hoc text suggestions | **Missing** | Rule-based decision-support engine linking feature importance to actionable retention interventions is missing. |
| **What-If Simulation Lab** | Static manual form fields | **Missing** | Interactive simulation lab comparing baseline vs counterfactual profiles side-by-side is missing. |
| **Class Imbalance Handling** | Basic `class_weight='balanced'` on Random Forest | **Partially working** | No formal cost-sensitive evaluation or validation-based decision threshold selection. |
| **Probability Calibration** | Raw tree output probabilities | **Missing** | No Brier score tracking, Platt scaling, or Isotonic calibration assessment. |

---

## 3. Baseline Performance Record (Step 2.3)

Prior research and baseline benchmarking on the Telecom Customer Churn dataset established the following reference points:

### 3.1 Historical Logistic Regression Baseline
* **Accuracy**: 85.46%
* **Churn Recall**: 18.00%
* *Critical Finding*: While nominal accuracy appears acceptable (85.46%), the model misses 82% of actual churners. This demonstrates severe vulnerability to class imbalance (~14.5% churn prevalence), rendering the model inadequate for retention intervention without recalibration and threshold tuning.

### 3.2 Majority Class Dummy Classifier
* **Strategy**: Most frequent class (`No Churn`)
* **Test Accuracy**: 85.46%
* **Churn Recall**: 0.00%
* **Churn Precision**: 0.00%
* **Churn F1**: 0.00%

### 3.3 Previous Uncalibrated Random Forest
* **Test Accuracy**: 95.50%
* **Churn Precision**: 86.02%
* **Churn Recall**: 82.47%
* **Churn F1**: 84.21%
* **ROC-AUC**: 92.18%
* **PR-AUC**: 86.54%

---

## 4. Phase One Rebuild Objectives

To transition C.R.A.M. to V2, the following technical gates must be executed:
1. **Scikit-Learn Preprocessing Pipeline (`ml/preprocessing.py`)**:
   - `ColumnTransformer` with `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` / binary encoders.
   - Strict fitting on training splits only to eliminate data leakage.
2. **Multi-Model Benchmarking (`ml/train.py` & `ml/evaluate.py`)**:
   - Compare Logistic Regression (L2 regularized, scaled), Random Forest, XGBoost, LightGBM, and HistGradientBoosting.
3. **Class Imbalance & Cost Optimization**:
   - Class weight balancing, cost-sensitive threshold analysis ($C_{FN} = \$500$, $C_{FP} = \$50$).
4. **Probability Calibration (`ml/calibrate.py`)**:
   - Brier score calculation, reliability diagrams, CalibratedClassifierCV comparison.
5. **Final Evaluation Report (`docs/model_evaluation.md`)**:
   - Multi-metric benchmark table and reproducible pipeline artifacts.
