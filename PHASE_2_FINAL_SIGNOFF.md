# C.R.A.M. V2 — Phase 2 Final Sign-Off

**Final recommendation: PASS**  
**Branch:** `v2-final-development`  
**Scope:** Two narrow correctness fixes from the Phase 2 conditional review  
**Production:** No production, V1 app, Git remote, or deployment changes were made.

## Sign-off summary

Both conditional findings are resolved. The explanation response now distinguishes genuinely missing inputs from non-missing inputs rejected or coerced by validation, and it keeps the original invalid value visible. Serving explanations now use a single version-controlled reference dataset; a caller-supplied alternate reference is rejected instead of silently ignored. Regression tests verify these behaviors and verify that predictions and SHAP reconstruction agree for valid, missing, and coerced inputs.

The SHAP algorithm, feature attribution methodology, model architecture, champion artifacts, model-selection process, calibration, and decision threshold were not changed. Phase 3 was not started.

## 1. Invalid-input disclosure fix

`ChurnExplainer.explain()` now checks each original profile value against the validated row before model imputation:

- `genuinely_missing_features` lists fields that arrived empty.
- `rejected_or_coerced_features` lists non-empty fields that validation changed to missing, including unparseable numeric strings, invalid binary values, and out-of-range numeric values.
- `estimated_features` is the union of those lists because both groups are ultimately imputed by the model pipeline.
- Each item in `all_contributions` includes `input_status`. Invalid submitted values remain displayed as submitted and are marked `invalid_or_out_of_range_coerced_then_imputed`; genuinely missing values remain `null` and are marked `genuinely_missing_imputed`.

The regression suite exercises all three invalid cases and a genuine missing value. For each case it checks the disclosure category and confirms that the SHAP reconstruction equals the direct prediction through the validated model pipeline within `1e-6`. Existing valid-input prediction behavior and probability reconstruction remain covered.

## 2. Fixed serving reference population

Serving explanations use [`models/explanation_reference.csv`](models/explanation_reference.csv): 50 feature-only rows selected deterministically with `random_state=42` from the same stratified training partition used in Phase 1. It contains the 13 original customer inputs and no churn-label column. The sampling population and selection rule are documented in [`PHASE_2_COMPLETION.md`](docs/PHASE_2_COMPLETION.md).

`ChurnPredictor.explain_single()` loads and caches this fixed reference. If a caller supplies `background_data`, the method raises a clear `ValueError` directing offline custom-reference analyses to `ChurnExplainer`. It cannot silently accept a new reference while reusing the old cached one. A regression test verifies the 50-by-13 schema, exact loaded reference, cache reuse, and rejection of a different supplied reference.

This makes the SHAP baseline consistent within the V2 serving API. The baseline still represents this public research training sample; it is a documented reference assumption, not a claim that it represents every future business’s customer population.

## 3. Verification evidence

Ran the full repository test suite after the fixes:

```text
venv\Scripts\python.exe -m pytest -q
12 passed, 2945 warnings in 8.18s
```

The 2,945 warnings are repeated artifact-load `DeprecationWarning`s from joblib under NumPy 2.5.0 (2,943 total across test modules), one `PytestRemovedIn10Warning` from the installed anyio plugin, and one loky warning because this Windows environment has no `wmic` physical-core probe; loky falls back to logical cores. No test failed. The joblib/NumPy deprecation is a forward-compatibility item to revisit when dependencies are next updated; it did not prevent the saved artifacts from loading in this run.

Earlier independent checks remain applicable: 12 profile/reference comparisons reconstructed direct probabilities with maximum absolute error `3.82e-17`; three sampled global runs had pairwise rank correlations `0.918`–`0.945` and four-of-five top-feature overlap; measured default global analysis took about 34.5 seconds. The fixed serving population addresses reference consistency. Global analyses should remain precomputed or cached rather than run on ordinary page renders.

## 4. Files changed and purpose

| File | Purpose |
| --- | --- |
| `ml/explain.py` | Distinguishes null inputs from values rejected by validation, reports status and preserves submitted invalid values in local explanation output. |
| `ml/predict.py` | Loads one canonical serving reference and rejects caller-supplied alternatives to prevent stale-reference assumptions. |
| `models/explanation_reference.csv` | Fixed 50-row, 13-feature serving reference sample, selected reproducibly from the Phase 1 training split. |
| `requirements.txt` | Adds pinned Matplotlib for SHAP plots; the pinned SHAP dependency is also listed here. |
| `tests/test_explanations.py` | Adds regression coverage for invalid numeric strings, invalid binary values, out-of-range values, genuinely missing inputs, output fidelity, canonical reference loading, caching, and override rejection. |
| `docs/PHASE_2_COMPLETION.md` | Updates serving usage, reference-population documentation, invalid-input response semantics, and test results. |
| `docs/PHASE_2_FINAL_VERIFICATION.md` | Preserves the independent conditional review and adds the post-fix verification outcome so the historical finding is not mistaken for current status. |
| `docs/phase_2_global_importance.csv` | Saved deterministic 13-feature global SHAP ranking from the Phase 1 training-only evaluation run. |
| `docs/images/phase_2_shap_beeswarm.png` | Saved global SHAP beeswarm visualization from the same training-only run. |
| `PHASE_2_FINAL_SIGNOFF.md` | Records this final recommendation, evidence, limitations, and file-by-file change summary. |

## Final recommendation

**PASS.** The two conditional-sign-off issues are resolved and covered by regression tests. No Phase 3 work remains in this authorized scope. The project is ready for user review of Phase 2; Phase 3 remains pending and requires separate approval.
