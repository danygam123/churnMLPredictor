# C.R.A.M. V2 — Phase 4 Final Verification

**Recommendation: PASS**  
**Branch:** `v2-final-development`  
**Baseline before Phase 4:** 18 passed, 4,580 warnings (22.01 s)  
**Production, push, merge, deployment:** None

## Scope and preparation

The working directory was confirmed as `C:\Users\Danny\OneDrive\Desktop\churnMLPredictor` and the active branch as `v2-final-development`. Before Phase 4, the worktree contained the expected, uncommitted Phase 3 files: modified `README.md`, `ml/explain.py`, and `ml/predict.py`; new `v2_app.py`, `ml/risk_intelligence.py`, `tests/test_risk_intelligence.py`, `tests/test_v2_app.py`, and the Phase 3 verification and handoff reports. Those changes were retained. The Phase 2 preservation report identifies checkpoint `06f0fecc907a5da72d332c4d1761ae836fafa8c2`.

The requested master roadmap PDF was accessible at `C:\Users\Danny\Downloads\CRAM project next steps.pdf`. Its Phase Four pages specify a structured rule engine, connection to customer attributes and SHAP, traceable recommendation reasons, objectives, human approval, and explicit non-causal language. The Phase 4 implementation follows those requirements within the verified 13-feature schema. `PHASE_2_FINAL_VERIFICATION.md` is under `docs/`, not the repository root; no root copy exists. The other named project reports were present.

## Evidence and test results

The complete test command was run before changes to record the baseline and after implementation to check regressions:

```text
Before Phase 4: 18 passed, 4580 warnings in 22.01s
After Phase 4: 24 passed, 4580 warnings in 10.64s
```

Phase 4 added six rule and portfolio tests. These verify both rules, their non-trigger cases, risk and threshold gates, deterministic ordering, accurate positive and negative SHAP interpretation, invalid or missing value suppression, priority ordering, customer ID mapping, and input schema validation. The existing V2 Streamlit `AppTest` uploaded a sample portfolio and rendered without exceptions. It also exercises the recommendation section and preserves the existing customer-profile SHAP workflow.

The pre-existing 4,580 warnings are unchanged from baseline: 4,578 repeated NumPy 2.5/joblib saved-array shape deprecation warnings, one anyio/pytest `CallSpec2` deprecation warning, and one Windows loky warning where physical core detection falls back to logical cores. No Phase 4-specific warning class or compatibility failure was observed.

## Functional verification

- The portfolio workflow produces customer-linked recommendations, risk and rule-category counts, and downloadable CSV data.
- The profile workflow reuses the profile's local Phase 2 SHAP explanation. It displays matching feature direction in probability units where the corresponding input was accepted.
- Missing or validation-rejected rule features do not trigger recommendations. Portfolio evaluation checks raw inputs against the validator's cleaned values; profile evaluation uses Phase 2's `input_status` semantics.
- Recommendation output states its trigger, characteristics, model context availability, objective, human approval requirement, and limitations.
- Low-risk customers and records below the Phase 1 prediction threshold receive no recommendation.
- Portfolio rendering does not compute SHAP for every record. Local SHAP context is calculated only for the selected profile, keeping the large-upload path bounded.
- On the repository's 2,666-row `churn-80.csv` sample, local portfolio scoring took 0.488 s and rule recommendation generation took 0.117 s after removing the repeated per-row pandas iteration overhead. It produced 394 recommendation rows for 290 customers. These are local timings for this machine and dataset, not production capacity guarantees.

## Preservation checks

No Phase 1 model, training/evaluation artifacts, calibration, prediction threshold, or metadata was modified. Phase 2 SHAP method, fixed reference population, contribution semantics, and invalid/missing disclosures were not modified. Phase 3 risk-intelligence logic and the separate V2 entry point were retained. `app.py` and the `vercel-app` production V1 files were not changed. No model was retrained. No Git remote, deployment, push, or merge operation occurred.

## Limitations and recommendation

The source data supports only two defensible initial rules: repeated customer service contacts and international-plan/usage mismatch review. It does not verify satisfaction, contact resolution, service reliability, contract type, payment method, true tenure, monthly billing, or intervention outcomes. Suggestions are hypotheses and cannot be described as effective churn-prevention actions. The prioritization score is an ordering heuristic, not an estimated success probability. Portfolio-wide rules are visible without per-customer SHAP; the detailed profile provides local SHAP context.

**Final recommendation: PASS.** Implemented scope is complete, the whole suite passes, and regression coverage protects Phase 1–3 behavior and the non-causal recommendation boundary. Phase 5 is the next roadmap phase and was not started.
