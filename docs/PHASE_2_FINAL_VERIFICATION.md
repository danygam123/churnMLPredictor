# C.R.A.M. V2 — Phase 2 Final Verification

**Original recommendation: CONDITIONAL PASS (superseded by the follow-up PASS below)**  
**Branch reviewed:** `v2-final-development`  
**Scope:** Independent verification of the existing SHAP work only  
**Model:** Existing train-only `champion_model_evaluation.joblib` for controlled verification runs  
**Changes made for this review:** This report only. No feature, model, architecture, or Phase 3 changes; no push.

> Follow-up fixes were completed after this independent review. The invalid-input disclosure issue is resolved, and serving explanations now use a fixed reference sample that rejects per-call overrides. See [PHASE_2_FINAL_SIGNOFF.md](../PHASE_2_FINAL_SIGNOFF.md) for the final post-fix recommendation and verification evidence.

## Executive finding

The SHAP implementation is mathematically faithful to the existing pipeline for the profiles tested. Its output is on the churn-probability scale, its global rankings are reasonably stable across the deterministic samples tested, and its reference population has a large and expected effect on the baseline and local attribution values. Cached individual explanations took about 0.33 seconds in this environment. The 100-row global analysis took about 34.5 seconds and should be run offline or cached before a future interface needs global results.

One input-reporting defect prevents an unconditional pass: numeric strings that cannot be parsed, invalid binary values, and out-of-range numeric values are coerced to missing by the validator and imputed by the model, but `estimated_features` only checks whether the raw input was initially null. Those coerced values are therefore reported with `value: null` in the attribution while absent from `estimated_features`. The model explanation remains faithful to the imputed prediction, but the response does not reliably disclose that the submitted value was replaced. Before user-facing sign-off, the serving response should accurately identify values that become missing during validation, or reject those records clearly.

## 1. Explanation fidelity — PASS for tested valid profiles

**Procedure:** Loaded the train-only evaluation artifact. Selected six distinct customer feature rows from the reproducible training split. Computed SHAP explanations with two independently sampled 50-row reference sets (`random_state=42` and `99`), using `max_evals=270`. Compared `base_value + sum(all 13 contributions)` against a direct positive-class pipeline probability for each profile.

| Measure | Result |
| --- | ---: |
| Profile/reference comparisons | 12 |
| Maximum absolute reconstruction error | `3.82e-17` |
| Mean absolute reconstruction error | `1.48e-17` |
| Required numerical tolerance | `1e-6` |

All tested values reconstruct well within tolerance. This confirms the current SHAP values telescope to the direct probability for these cases; it does not establish exact Shapley values, because `PermutationExplainer` estimates them with a finite evaluation budget.

## 2. Feature attribution stability — PASS with documented variation

**Procedure:** Compared three deterministic global analyses. Each used 50 background rows, 100 explained rows, and `max_evals=270` per row. The samples differed across runs; the first two used independent random training samples, while the third used a separate background and explained sample. No churn labels were passed to the explainer or importance calculation.

| Run | Top five by mean absolute SHAP contribution |
| --- | --- |
| A | International plan; Total day minutes; Customer service calls; Total intl calls; Total eve minutes |
| B | International plan; Total day minutes; Customer service calls; Number vmail messages; Total eve minutes |
| C | Total day minutes; Customer service calls; International plan; Total eve minutes; Total intl minutes |

Pairwise Spearman correlations of all 13 feature ranks were `0.934` (A–B), `0.945` (A–C), and `0.918` (B–C). Each pair shared four of its top five features. International plan, Total day minutes, and Customer service calls remained in the top three across all runs, although their ordering changed. Lower-ranked features moved more. The ranking is reasonably stable at the top under these samples, not identical or guaranteed to generalize beyond this dataset.

## 3. Reference population — material, explicit modeling assumption

The same customer profile was explained against four deterministic 50-row references: two random samples, a low model-score training population, and a high model-score training population. The direct pipeline prediction stayed at `0.001521` for that profile, while the SHAP base value and attribution allocation changed:

| Reference population | SHAP base value | Direct profile probability |
| --- | ---: | ---: |
| Random sample, seed 42 | 0.095939 | 0.001521 |
| Random sample, seed 99 | 0.179699 | 0.001521 |
| Low model-score population | 0.000547 | 0.001521 |
| High model-score population | 0.994428 | 0.001521 |

This is expected: the base value describes the model output for the chosen reference distribution, and contributions explain the difference from that base to the customer’s prediction. Changing the population can change both the base and which inputs receive credit, while leaving the customer’s direct model probability unchanged. A UI must identify or standardize the reference population and keep it consistent when comparing customers. The predictor caches the explainer after the first reference DataFrame; subsequent calls on that predictor silently continue using that first reference even if another DataFrame is supplied.

## 4. Performance and caching

Measured on this workstation with the current virtual environment and evaluation artifact:

| Measurement | Result |
| --- | ---: |
| First `ChurnPredictor.explain_single()` call, including construction | 0.346 s |
| Five warm calls, median | 0.332 s |
| Warm call range | 0.302–0.342 s |
| Explainer object identity after warm calls | unchanged (cached) |
| Full default global command (50 background, 100 explained, 270 evaluations, CSV + beeswarm) | 34.509 s |
| Independent global sample runs | 33.406–35.256 s each |

One individual explanation appears usable for a future synchronous single-customer view under this measured local environment. Performance can differ with hardware, concurrent users, and background size. Recreating `ChurnPredictor` on each Streamlit rerun would also lose its per-instance explainer cache. The global analysis should not execute on an ordinary page render; precompute or cache its result. No Streamlit integration was implemented or performance-tested here.

## 5. Tests, warnings, and edge behavior

**Complete test suite:** `venv\Scripts\python.exe -m pytest -q` — **8 passed, 1,637 warnings in 8.90 seconds**.

Warning categories from the pytest summary:

| Count | Category and source | Assessment |
| ---: | --- | --- |
| 1,635 | `DeprecationWarning` from `joblib/numpy_pickle.py:207`, assigning an array shape while loading serialized artifacts under NumPy 2.5.0 (981 attributed to the explanation tests; 654 to prediction tests). | Current artifact loading succeeds, but this is a real forward-compatibility concern in the joblib/NumPy path. Recheck or upgrade compatible dependencies before a future NumPy release removes the deprecated behavior. |
| 1 | `PytestRemovedIn10Warning` from the installed `anyio` pytest plugin using the renamed `CallSpec2` alias. | Tests pass now; plugin compatibility should be checked before moving to pytest 10. |
| 1 | `UserWarning` from joblib/loky failing to query physical cores because `wmic` is unavailable on this Windows environment; loky falls back to logical cores. | Environment discovery warning, not an explanation failure. |

Edge checks performed:

- A genuinely null input is listed in `estimated_features`; the prediction still reconstructs. All 13 null inputs are accepted and imputed, with all 13 listed as estimated. Extensive imputation should not be treated as reliable customer evidence.
- An invalid numeric string (`"nonsense"`) and an invalid binary value (`"maybe"`) are coerced to missing. In both cases `estimated_features` is empty even though the feature display value becomes null. This is the conditional-pass defect described above.
- Out-of-range numeric values are also coerced to missing by the validator; the same estimated-field reporting gap follows from the code path.
- Unexpected keys in an individual customer dictionary are ignored by reindexing to the known 13 inputs; the verified profile probability was unchanged. Extra columns in a background DataFrame are likewise sliced away. There is no warning to the caller about ignored keys.
- `explain_global()` rejects an empty frame or one missing required feature columns, but accepts and discards extra columns.

## 6. Documentation and leakage review

- **13 input features:** The SHAP module takes names from `INPUT_FEATURES`, iterates contributions over that schema, and the produced importance CSV contains all 13. The inputs are Account length; International plan; Voice mail plan; Number vmail messages; Total day minutes; Total day calls; Total eve minutes; Total eve calls; Total night minutes; Total night calls; Total intl minutes; Total intl calls; Customer service calls.
- **Probability scale:** The explainer callback selects `predict_proba()[:, 1]`. Direct comparisons confirmed the reconstruction is the churn probability, not log-odds.
- **Causality:** The Phase 2 completion documentation states that explanations describe model associations and do not establish causal effects. The generated local summary also includes a causal limitation.
- **Held-out test labels:** The global report recreates the stratified train/test split, uses only `X_train` for background and explained rows, and loads the train-only evaluation checkpoint. The split function receives labels to reproduce stratification, but `y_test` is discarded and never enters SHAP, ranking, or report metrics. No held-out test labels are used to create the global explanation.
- **Derived fields:** The complete fitted pipeline is called from the 13 raw inputs. The four deterministic charge fields are not exposed as separate attributions, which avoids duplicating credit between a usage input and its derived charge.

## Recommendation and limitations

**Historical recommendation: CONDITIONAL PASS.** The specific issue was resolved in the follow-up. See the final sign-off report for current test results and the PASS recommendation. The original benchmarks cover one local machine, the project’s public dataset, a finite set of profiles and references, and the existing finite SHAP budget; they do not establish production latency or cross-population explanation stability.

## Follow-up verification after the conditional-sign-off fixes

- `ChurnExplainer.explain()` now reports `genuinely_missing_features` separately from `rejected_or_coerced_features`. `estimated_features` includes both. Per-feature records retain the submitted invalid value and identify it as `invalid_or_out_of_range_coerced_then_imputed`; genuinely empty values use `genuinely_missing_imputed`.
- `ChurnPredictor.explain_single()` now uses `models/explanation_reference.csv` (50 feature-only rows sampled with seed 42 from the Phase 1 stratified training split). Supplying a different per-call DataFrame raises a clear `ValueError`; custom populations remain available for direct offline use of `ChurnExplainer`.
- Regression tests cover an invalid numeric string, invalid binary input, out-of-range value, genuinely missing value, prediction/SHAP reconstruction, and rejection of changed serving references.
- Post-fix complete suite: **12 passed, 2,945 warnings in 8.18 seconds**. Warning categories remain joblib/NumPy array-shape deprecations, the anyio pytest alias deprecation, and loky’s Windows physical-core probe fallback.
- The final post-fix recommendation is **PASS**, subject to the limitations recorded in `PHASE_2_FINAL_SIGNOFF.md`.
