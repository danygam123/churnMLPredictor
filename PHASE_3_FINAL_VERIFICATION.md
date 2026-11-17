# C.R.A.M. V2 — Phase 3 Final Verification

**Recommendation: PASS**  
**Phase:** Customer Risk Intelligence  
**Branch:** `v2-final-development`  
**Phase 2 preservation checkpoint:** `06f0fecc907a5da72d332c4d1761ae836fafa8c2` (`codex/phase2-signed-off-backup`)  
**Production, push, merge to main:** None

## Scope and roadmap alignment

The Phase 3 roadmap calls for configurable High/Medium/Low risk segmentation; customer-base KPIs and risk distribution; churn-pattern charts; filters; a searchable, sortable customer list; selected customer profiles; and CSV export. The implementation delivers those capabilities in a separate V2 Streamlit entry point, leaving the V1 `app.py` untouched.

The source telecom schema has 13 model inputs plus an optional observed `Churn` label. It does not include contract type, payment method, or a monthly bill. The dashboard supports those dimensions when an uploaded file supplies them, reports them unavailable when absent, and uses explicitly named account-length and derived usage-charge proxies for the current schema. No missing business dimensions are fabricated.

## Verification results

### Business logic and statistics

- Risk-band validation and boundary tests pass. Defaults use Phase 1's validation-based cost threshold (`0.0772727`) as the Medium Risk floor and its default `0.50` cutoff as the High Risk floor. Low is below the Medium floor; Medium is at or above it and below High; High is at or above the High floor. UI adjustments change risk bands only.
- The binary predicted-churn decision continues to use the Phase 1 cost threshold. It is distinct from risk labels and the high-risk count.
- Portfolio statistics were checked against independent calculations from the scored probability array. The integration test verifies counts, average probability, observed churn rate, and labeled-customer denominator.
- Observed churn labels accept booleans, yes/no, and 1/0 values. Missing labels remain unknown and are excluded from observed-rate denominators; invalid labels produce a row-specific actionable error.
- Missing and rejected inputs are counted separately. The app reports imputation warnings, including rows with more than three estimated inputs.
- Duplicate customer IDs are made unique for profile selection. Missing IDs receive temporary row IDs. Exact duplicate input records are not silently removed.
- Usage-charge proxy calculation uses the existing fixed minute rates and is explicitly not presented as a monthly invoice.

### Phase 1 and Phase 2 regression protection

- The complete suite includes the original preprocessing/prediction tests and Phase 2 explanation tests. It confirms the champion can still score records, the canonical reference remains usable, and SHAP contributions still reconstruct the directly scored probability.
- Phase 3 risk categories are returned separately from the existing binary decision. `ChurnPredictor.categorize_risk()` now uses the same validation-anchored Medium and High defaults as the portfolio module; it does not change `optimal_threshold` or model probabilities.
- The SHAP plotting helpers select Matplotlib's non-interactive Agg backend for the Streamlit/server workflow. The tested app completed without Tk cleanup warnings.
- No training script was run. Model artifact files, calibration results, and probability threshold metadata were not modified.

### Application smoke test and performance

Streamlit `AppTest` uploaded and rendered the actual 2,666-row `churn-80.csv` file, including risk KPIs, charts, the customer table, a selected profile, and its Phase 2 SHAP explanation. It reported no application exceptions.

| Measurement | Result |
| --- | ---: |
| Streamlit app setup / model load | 2.80 s |
| 2,666-row upload, dashboard, and profile render | 2.18 s |
| Uploaded row count | 2,666 |
| Predicted churners at 0.0772727 | 416 |
| Average predicted probability | 14.5% |
| High-risk customers at 0.50 | 388 |
| Observed churn rate in labeled sample | 14.6% |

These are local AppTest timings and descriptive results for this dataset and environment, not production load benchmarks or generalization claims.

### Complete test suite

Command: `venv\Scripts\python.exe -m pytest -q`

```text
18 passed, 4580 warnings in 9.72s
```

Warnings are 4,578 repeated `DeprecationWarning`s from joblib's NumPy 2.5 array-shape restoration while loading the saved artifacts; one `PytestRemovedIn10Warning` from anyio's plugin using the renamed `CallSpec2`; and one loky warning because this Windows environment has no `wmic` physical-core probe (loky falls back to logical cores). The model artifacts load and all tests pass. The joblib/NumPy warning is a forward-compatibility issue to revisit with a compatible dependency update; no dependency update was needed for Phase 3.

## Findings and limitations

- Risk-band thresholds are transparent and configurable, but remain business settings anchored to this model's validation thresholds; they are not universal risk definitions.
- Contract/payment/monthly-bill charts and filters require those fields in the uploaded file. The model itself still consumes only the fixed 13-input schema.
- Account length and derived usage charges are proxies when true tenure and monthly billing fields are absent.
- Observed rates are descriptive associations in uploaded historical labels. Neither model scores nor SHAP explanations establish causal effects or intervention outcomes.
- The model remains trained on one public telecom dataset. It is not validated for other industries or customer populations.
- The dashboard is session-based. It does not persist uploaded customer records, provide user authentication, or change the V1 app/deployment.
- Portfolio uploads are limited to 100,000 rows by the analysis module. Larger files require a future, separately scoped performance design.

## Final assessment

**PASS.** Phase 3's risk intelligence logic and user workflow are implemented, tested, documented, and integrated with Phases 1 and 2. Current-schema field gaps are disclosed with appropriate proxies. Known dependency warnings and product limitations are recorded above and in the final handoff.
