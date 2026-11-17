# C.R.A.M. V2 — Phase 3 Final Handoff

**Status:** Phase 3 implementation complete  
**Recommendation:** PASS  
**Active branch:** `v2-final-development`  
**Phase 2 checkpoint:** `codex/phase2-signed-off-backup` at `06f0fecc907a5da72d332c4d1761ae836fafa8c2`  
**Remote/deployment:** No push, merge to `main`, or deployment occurred

## A. Executive summary

Phase 3 turns customer-level churn probabilities into a portfolio risk workspace. It adds explicit High/Medium/Low bands, portfolio KPIs, risk and churn-pattern charts, filters, search and sort, customer profiles with existing Phase 2 SHAP explanations, and filtered CSV export. It runs as a new `v2_app.py` entry point so the existing V1 app stays untouched.

The implementation was exercised on the actual 2,666-row telecom training CSV and passed the full 18-test suite. Field gaps in the source dataset are disclosed rather than filled with invented contract, payment, tenure, or monthly-bill data.

## B. Phase 2 preservation

Before Phase 3 work, the complete Phase 2 signed-off state was preserved:

- Backup branch: `codex/phase2-signed-off-backup`
- Preserved Phase 2 commit: `06f0fecc907a5da72d332c4d1761ae836fafa8c2`
- Phase 2 sign-off: `PHASE_2_FINAL_SIGNOFF.md`, recommendation PASS
- Separate pre-checkpoint uncommitted snapshot: `stash@{0}`, object `df4114b3f20ea83f41c15894ac1a8b0cdb3b2b04`
- Preservation procedure and contents: [`PHASE_2_PRESERVATION_REPORT.md`](PHASE_2_PRESERVATION_REPORT.md)

The local development branch was fast-forwarded to the same Phase 2 checkpoint and then received a separate preservation-report commit (`4cdd6a8ee1cfdbbe337dd4c998401e44cd76b546`). The backup branch was not moved. Phase 3 edits are in the development working tree; the exact signed-off Phase 2 commit and stash remain recoverable.

## C. Implementation details

### Risk bands and prediction decisions

- `RiskThresholds` validates the constraint `0 <= medium < high <= 1` and classifies probabilities into Low, Medium, and High Risk.
- Default Medium floor: Phase 1 validation cost threshold `0.077272727...`.
- Default High floor: Phase 1 default classification cutoff `0.50`.
- Binary `Prediction` remains `Likely to churn` when probability is at or above the Phase 1 cost threshold. This is deliberately separate from risk band: the medium band begins at the binary threshold; the high band begins at 0.50.
- Users can adjust the risk-band thresholds in the dashboard sidebar. The selected thresholds are shown and do not alter predictions or the model.
- `ChurnPredictor.categorize_risk()` was aligned to the same defaults. Its legacy `low_threshold` argument now represents the Medium Risk lower boundary.

### Portfolio analysis

`ml/risk_intelligence.py` provides the pure portfolio functions:

- Requires all 13 expected model input columns; keeps every record and does not silently deduplicate.
- Applies the existing `CustomerDataValidator` and fitted champion pipeline.
- Preserves uploaded fields for analysis and export while adding probability, binary decision, predicted-churn flag, risk category, ID, and missing/rejected input counts.
- Reports exact observed churn rates only when a valid `Churn` column exists. Unknown labels are excluded from denominators; unsupported labels produce a row-specific actionable error.
- Uses uploaded customer IDs where available; otherwise generates temporary row IDs. Duplicate IDs are made unique for selection.
- Warns about blank and invalid/rejected inputs separately and flags rows with more than three estimated values.
- Calculates the current model's derived usage-charge total from its existing fixed rates. It is presented as a charge proxy, not an account's monthly invoice.
- Computes headline totals and grouped observed rates or, when labels are absent, grouped mean predicted probabilities.

### V2 Streamlit workspace

`v2_app.py` provides:

- CSV upload with actionable schema and label validation.
- Portfolio KPIs: customers analyzed, predicted churners, average probability, high-risk count, and observed churn rate when labeled.
- Risk distribution chart and grouped customer-pattern charts.
- Configurable risk thresholds and filters for risk band, search text, service plans, optional contract type, tenure/account-length, and monthly charges/usage-charge proxy.
- Charts by contract type, payment method, tenure, and monthly charges when supplied; current-schema proxies and unavailable dimensions are clearly identified.
- Phase 2 model-wide SHAP importance from the training-sample report, explicitly labeled as model-wide rather than upload-specific.
- Searchable customer table, probability sort in either direction, selected profile with probability/binary decision/risk/estimated fields, top local SHAP factors, and filtered CSV download.
- Session-based uploaded data handling; the dashboard does not persist uploaded records.

The upload limits analysis to 100,000 rows. Actual contract, payment, monthly-billing, or tenure dimensions are used if present as extra CSV columns; they are not model features unless part of the existing 13-input schema.

## D. Repository changes

### Created

| File | Purpose |
| --- | --- |
| `ml/risk_intelligence.py` | Risk bands, validated portfolio scoring, labels, IDs, summary statistics, grouping, and usage-charge proxy. |
| `v2_app.py` | Separate Phase 3 Streamlit dashboard; preserves V1 `app.py`. |
| `tests/test_risk_intelligence.py` | Risk threshold, statistics, labels, invalid/missing inputs, IDs, and Phase 1/2 integration checks. |
| `tests/test_v2_app.py` | Streamlit upload, KPI, dashboard, and selected-profile smoke test. |
| `PHASE_2_PRESERVATION_REPORT.md` | Records the mandatory pre-Phase 3 checkpoint and recovery references. |
| `PHASE_3_FINAL_VERIFICATION.md` | Technical review and PASS recommendation. |
| `PHASE_3_FINAL_HANDOFF.md` | This end-to-end engineering record. |

### Modified

| File | Purpose |
| --- | --- |
| `ml/predict.py` | Aligns predictor risk categories with the independent Phase 3 bands while preserving the binary threshold. |
| `ml/explain.py` | Forces Matplotlib's non-interactive Agg backend for server-side SHAP plot generation; the SHAP method and values are unchanged. |
| `README.md` | Documents how to run the separate V2 dashboard and its input/threshold boundaries. |

### Deliberately untouched

- `app.py`, root V1 model, Vercel application, and production deployment.
- Champion model artifacts, training scripts, calibration, decision threshold, and Phase 2 canonical explanation reference data.
- GitHub remote, `main`, and deployment configuration.
- Phase 4 recommendations, Phase 5 simulation, and Phase 6 AI analyst.

## E. Verification evidence

### Automated tests

Command: `venv\Scripts\python.exe -m pytest -q`

```text
18 passed, 4580 warnings in 9.72s
```

The suite covers existing preprocessing and predictions, Phase 2 local/global SHAP and reconstruction, invalid versus missing explanation inputs, fixed-reference enforcement, portfolio thresholds and metric calculations, invalid labels/schema, IDs, charge proxies, and the Streamlit upload/profile route.

Warnings:

- 4,578 repeated `DeprecationWarning`s from joblib's NumPy 2.5 array-shape restoration while loading model artifacts. Artifacts load successfully; revisit dependency compatibility before a future NumPy upgrade.
- One `PytestRemovedIn10Warning` from the installed anyio pytest plugin.
- One loky warning because Windows `wmic` is unavailable for physical-core discovery; loky uses logical cores.
- No test failures. The earlier Tkinter cleanup warnings in UI smoke tests were eliminated by using Matplotlib's Agg backend for saved SHAP plots.

### User-facing smoke test and measured data

Streamlit `AppTest` successfully uploaded the actual local `data/raw/churn-80.csv` sample with 2,666 customers. It generated no application exception. The measured local run took 2.80 seconds for app/model setup and 2.18 seconds for upload, portfolio rendering, customer selection, and profile SHAP explanation.

Observed dashboard values on that sample:

- 2,666 customers analyzed.
- 416 predicted churners at the 0.0772727 binary threshold.
- 14.5% average predicted probability.
- 388 High Risk customers at the 0.50 band threshold.
- 14.6% observed churn rate among supplied labels.

These are sample-specific descriptive outputs and local AppTest timings, not externally validated performance or realized business impact.

### Repository and model integrity

- Phase 2 backup branch and stash remain available at the references listed above.
- Development branch is `v2-final-development`; Phase 3 edits and reports are local/uncommitted.
- `app.py` is unchanged; no V1 or live deployment changes occurred.
- No retraining command was run. Champion joblib files, model metadata thresholds, calibration values, and Phase 2 reference CSV are unchanged.
- Phase 2 SHAP reconstruction tests remain green; Phase 2 signed-off files are recoverable from the backup commit and current branch base.

## F. Compatibility summary

| Area | Result |
| --- | --- |
| Phase 1 scoring/preprocessing | PASS — existing tests and full portfolio model scoring pass. |
| Phase 2 local SHAP | PASS — profile explanation renders in V2 app; all original explanation tests pass. |
| SHAP probability reconstruction | PASS — existing tolerance checks remain green. |
| Canonical reference population | PASS — fixed 50-row reference is unchanged and still used. |
| Model artifacts, calibration, probability threshold | PASS — no changes or retraining. |
| V1 `app.py` and deployment | PASS — untouched. |
| V2 portfolio statistics/UI flow | PASS — unit and Streamlit upload smoke tests pass. |

## G. Limitations and outstanding work

- Risk labels are configurable business bands grounded in Phase 1's two operating thresholds, not universal standards.
- This telecom dataset lacks contract type, payment method, true tenure, and monthly bills. The app reports absent dimensions and labels proxies; a business dataset must provide those fields for corresponding charts/filters.
- Grouped observed churn is descriptive. It does not prove why customers churn or that a retention action would change outcomes.
- The model was trained on one public dataset and should not be assumed to generalize outside its source population.
- The workspace does not add persistence, account management, or production upload controls; these are outside Phase 3 scope.
- The joblib/NumPy warning indicates future compatibility work is prudent but did not block this release.

## H. Next steps

Phase 3 is complete and ready for review. The roadmap's next phase is **Phase 4: Retention Recommendation Engine**. It should use model explanations and explicit rules, explain why each candidate recommendation appears, and label interventions as hypotheses rather than proven churn-prevention effects. Do not begin that phase until separately authorized.
