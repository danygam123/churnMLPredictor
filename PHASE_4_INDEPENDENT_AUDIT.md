# C.R.A.M. V2 — Phase 4 Independent Audit

**Readiness gate: PASS after one isolated correctness fix**  
**Branch:** `v2-final-development`  
**Audit scope:** Phase 4 recommendation behavior, tests, model compatibility, and readiness to build Phase 5

## Repository state and inspection

- Repository root confirmed: `C:\Users\Danny\OneDrive\Desktop\churnMLPredictor`.
- Active branch confirmed: `v2-final-development`; HEAD at audit start was `4cdd6a8` (`Document Phase 2 preservation checkpoint`).
- The worktree already contained uncommitted Phase 3 work and Phase 4 work. It was not cleaned or reset. The pre-existing Phase 3 changes include `README.md`, `ml/explain.py`, `ml/predict.py`, `ml/risk_intelligence.py`, `v2_app.py`, Phase 3 docs, and associated tests. Phase 4 added its recommendation module, tests, docs, and README section.
- Inspected Phase 4 rule code, portfolio and profile integration, recommendation and app tests, sign-off/verification/handoff docs, plus Phase 1 prediction/preprocessing, model metadata, Phase 2 explanation response fields, and Phase 3 portfolio ID handling.

## Findings

### PASS — Rule logic and decision gates

The current code implements the documented rule conditions:

1. `service_contact_review`: accepted `Customer service calls >= 3`; wording asks staff to investigate contacts and possible resolution. It does not state that poor service or failure occurred.
2. `international_usage_fit_review`: accepted `International plan = No` and either accepted international minutes `>= 10` or calls `>= 4`; wording asks staff to review plan fit. It does not assert overcharging or a proven mismatch.

Both rules are traceable to IDs, names, actual trigger values, objectives, SHAP-linked input features, and limitations. The supported data fields are existing model inputs; no contract, payment method, true tenure, monthly invoice, satisfaction, or service resolution is invented.

Recommendations are gated to `Medium Risk` or `High Risk` and probability at or above the passed `prediction_threshold`. In the V2 integration, that value is `PortfolioResult.prediction_threshold`, taken from `predictor.optimal_threshold` and Phase 1 model metadata (`0.07727272727272727`). The value is not redefined inside the rules. Risk band and binary prediction cutoff remain distinct as intended.

Phase 3 generates unique customer IDs when uploaded IDs are blank or repeated; the profile selector filters on that unique ID. Portfolio recommendations carry the resulting `Customer ID`. The tests verify portfolio customer-to-recommendation mapping. Profile explanations are calculated from the selected profile itself.

### PASS after fix — Input validation consistency

The first independent code review found an API-level gap: portfolio calls validated raw input status, and the V2 profile supplied Phase 2 statuses, but a direct `recommend_customer()` call with no explanation could consider raw, invalid values. That could allow an invalid service-call count or invalid usage value to trigger a rule if a caller bypassed the current UI path.

This was an isolated Phase 4 correctness issue. `recommend_customer()` now independently applies the existing `CustomerDataValidator` to the canonical 13 inputs, marks missing and rejected/coerced features, and uses those statuses to suppress rule triggers and SHAP context. Portfolio calls reuse statuses already computed for each row to avoid repeat validation. A regression test covers direct calls without an explanation. No model, feature schema, recommendation conditions, or Phase 1–3 decision methodology changed.

After the fix, focused recommendation plus V2 app tests passed: **8 passed, 329 warnings**. The complete suite also passed, as recorded below.

### PASS — SHAP mapping and score semantics

Recommendation context selects SHAP contributions only for the rule's related canonical input names, with `input_status == "provided"`. Positive contributions are described as increasing the model's predicted churn probability; negative contributions as decreasing it. A portfolio row without a local explanation explicitly says to open the profile for SHAP context; the portfolio path does not compute SHAP for every customer. An explanation with no accepted reliable related feature is reported as having no reliable local context. SHAP is described as model behavior, not causal evidence.

The priority formula in code and handoff matches:

```text
min(100,
    20 * rule_strength
    + 10 * risk_band_rank
    + min(10, sum(max(0, related_SHAP_contribution * 100)))
)
```

Rule strengths are 3 for support-contact review and 2 for international fit. Risk ranks are High=2, Medium=1, Low=0 (Low Risk is gated out). Only positive SHAP values for related features add to the score, capped at 10 points; the total score is capped at 100. Scores `>= 85` are `High`; the remainder are `Standard`. The formula is identified as a ranking heuristic, never as an intervention success probability.

## Tests, warnings, model and timing reproduced

The existing recommendation tests exercise both rules, gates, input status, SHAP directions, deterministic output, priority, and customer mapping. The Streamlit `AppTest` uploads a CSV, loads the actual serving predictor/model, renders the V2 dashboard and profile, and completes without app exceptions. These relevant tests are not skipped; the model-loading path is not mocked.

```text
Focused recommendation + V2 AppTest: 8 passed, 329 warnings in 5.80s
Complete suite: 25 passed, 4,580 warnings in 9.73s
```

The original Phase 4 final report recorded 24 passing tests. The audit added one safety regression test for direct invalid-input handling, which accounts for the current total of 25; the earlier result was accurate for its earlier test inventory.

Warnings reproduced in the complete run:

- **4,578 NumPy 2.5/joblib `DeprecationWarning`s:** occur while restoring saved arrays during the existing model artifact loading paths. They do not currently prevent loading or prediction, but are a forward compatibility concern for future NumPy/joblib versions. Keep model artifacts and dependency compatibility under review.
- **1 anyio/pytest `PytestRemovedIn10Warning`:** plugin imports pytest's renamed `CallSpec2`; non-blocking under the installed pytest 9.2.0, but should be addressed before pytest 10 compatibility is required.
- **1 loky physical-core `UserWarning`:** the Windows environment has no `wmic` probe; loky falls back to logical core count. Prediction, explanation, and tests complete successfully.

The fitted model is loadable and a real row scored through `ChurnPredictor.predict_single()` with the configured Phase 1 threshold. On the available 2,666-row `churn-80.csv` sample, this audit measured portfolio scoring at **0.485 seconds** and recommendation generation at **0.112 seconds**, producing 394 recommendations for 290 customers. These timings are local and workload-specific, not a scale guarantee.

## Preservation evidence

`git diff --exit-code -- app.py models vercel-app` completed with no changes. Current SHA-256 digests were recorded for `app.py`, the serving and evaluation models, model metadata, and canonical explanation reference in the audit evidence stream. No model retraining or artifact replacement occurred. No push, merge, deployment, or remote branch operation occurred.

## Limitations of the audit

This audit verifies local source behavior, the repository's checked-in artifacts, automated tests, and the available 2,666-row sample. It does not validate recommendations against a production customer population or intervention experiment. The machine-specific timing does not establish production capacity. The data still lacks verified contract, payment, true tenure, invoice, satisfaction, and issue-resolution fields.

## Phase 4 readiness decision

**Gate: PASS.** The identified API-level invalid-input gap was narrow, fixed, and covered by a regression test. The full test suite passes afterward; rule conditions, thresholds, feature mappings, customer associations, priority formula, and fitted model compatibility are verified. The warnings are non-blocking and their future-compatibility implications are documented. Phase 4 is safe as the foundation for the authorized Phase 5 simulation work.
