# C.R.A.M. V2 — Phase 2 Completion Report

**Status:** Implemented and locally verified  
**Branch:** `v2-final-development`  
**Scope:** Reusable SHAP explainability for the Phase 1 HistGradientBoosting pipeline  
**Production status:** No push, merge, deployment, or production change was made.

## What was delivered

- `ml/explain.py` adds `ChurnExplainer`, backed by SHAP `PermutationExplainer`.
- Explanations run through the complete fitted pipeline, including the existing input validation, derived charge features, imputers, and classifier.
- Local explanation results include the model's churn probability, base value, signed contribution for each of the 13 customer inputs, factors that raise or lower the estimate, factors below a configurable one percentage point magnitude, and a plain-language summary.
- Contributions are in **probability units**. Their sum with the SHAP base value reconstructs the model's churn probability, subject to floating-point precision.
- Global analysis returns a ranked table with mean absolute and mean signed contributions and positive/negative contribution rates. Beeswarm and individual waterfall image export are supported.
- `ChurnPredictor.explain_single()` exposes individual explanations through the existing prediction engine and caches the explainer after initialization.
- Serving explanations now use the fixed, version-controlled 50-customer feature-only reference sample in `models/explanation_reference.csv`. It is sampled deterministically (`random_state=42`) from the Phase 1 training split. Serving calls reject custom reference DataFrames; offline comparisons can still use `ChurnExplainer` directly.
- Explanation responses distinguish `genuinely_missing_features` from `rejected_or_coerced_features`. `estimated_features` contains both categories, and each contribution includes an `input_status`; rejected raw values remain visible in the response as the submitted value.
- `requirements.txt` now includes the pinned SHAP and Matplotlib dependencies required for computation and plot export.
- `tests/test_explanations.py` checks local probability reconstruction, output shape, global ranking, and integration with `ChurnPredictor`.

## Method and statistical safeguards

The champion is a scikit-learn pipeline around `HistGradientBoostingClassifier`. The chosen permutation explainer calls the **whole pipeline** on the original 13 inputs. This avoids presenting internal transformed columns or derived charge fields as separate customer factors. The explainer output is the positive-class churn probability, so the base value and SHAP contributions are on the probability scale rather than log-odds.

The global analysis command uses the train-only evaluation artifact (`models/champion_model_evaluation.joblib`) and recreates the Phase 1 stratified training split. It uses no held-out test labels. With default settings it draws 50 background rows and explains 100 training rows, using 270 model evaluations per explained row (10 permutation cycles for 13 features). The result is a bounded, approximate global summary; SHAP permutation values are estimates and can vary with the sample, background, and evaluation budget.

The explanations describe how this fitted model responds to inputs. They do not establish that changing a factor will cause a customer's churn probability or real-world behavior to change. They are not validated intervention effects.

## Global result from this run

The leading features by mean absolute SHAP contribution were:

| Rank | Input feature | Mean absolute contribution |
| ---: | --- | ---: |
| 1 | International plan | 0.0658 |
| 2 | Total day minutes | 0.0620 |
| 3 | Customer service calls | 0.0485 |
| 4 | Total intl calls | 0.0363 |
| 5 | Total eve minutes | 0.0299 |

These are average magnitudes of probability contributions on the sampled training rows, not churn rates or causal effects. The tail rows are especially approximate due to the limited sample and permutation budget. See the complete 13-feature table and beeswarm image linked below.

## Artifacts

- Global importance table: [`phase_2_global_importance.csv`](phase_2_global_importance.csv)
- Global beeswarm: [`phase_2_shap_beeswarm.png`](images/phase_2_shap_beeswarm.png)
- Implementation: [`ml/explain.py`](../ml/explain.py)
- Predictor integration: [`ml/predict.py`](../ml/predict.py)
- Explainability checks: [`tests/test_explanations.py`](../tests/test_explanations.py)

## How to use

Serving explanations use the same fixed reference population across predictor instances. Its 50 rows are feature values from the Phase 1 stratified training split, sampled with seed 42; the churn labels are not stored. This reference defines what the local SHAP baseline means. Do not replace it silently or compare explanations created against a different reference population.

```python
from ml.predict import ChurnPredictor

predictor = ChurnPredictor()
result = predictor.explain_single(customer_profile)
```

Custom reference populations remain available for offline methodological comparisons by constructing `ChurnExplainer` directly. `ChurnPredictor.explain_single()` rejects a supplied custom DataFrame with a clear error instead of silently keeping its cached reference.

To regenerate the train-only global table and plot from repository data:

```powershell
venv\Scripts\python.exe -m ml.explain
```

## Verification

Ran the project suite with the repository virtual environment:

```text
12 passed, 2945 warnings in 8.18s
```

Warnings are emitted by dependency compatibility paths (including NumPy/joblib serialization and the Windows core-count probe); no test failed. The local global report command completed and wrote the CSV and PNG. The generated global table contains all 13 original inputs. Tests verify probability reconstruction within `1e-6`, distinguish missing from rejected/coerced inputs, and ensure serving calls cannot silently override the fixed reference population.

## Limitations and next work

- The global explanation is based on a fixed deterministic sample of 100 rows and an empirical background sample of 50 rows. It is not an external validation of model quality or a population-level causal study.
- SHAP calculations cost more than ordinary predictions. A reference set and explanation evaluation budget should remain bounded in an interactive application.
- The fixed reference is a small sample of the Phase 1 public research training split; a different real-world serving population may warrant a separately governed reference version. The serving API deliberately does not accept per-call reference changes.
- There is not yet a Streamlit customer profile interface; UI integration belongs to later product work.
- No full retraining, model selection, threshold tuning, or calibration change was made.
- Phase 3 (customer risk intelligence) remains pending.
