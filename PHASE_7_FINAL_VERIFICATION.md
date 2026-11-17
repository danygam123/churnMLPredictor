# Phase 7 Final Verification — C.R.A.M. V2 "Build the Final SaaS Experience"

*Prepared 2026-11-16 on branch `v2-final-development` (work tree, not yet committed). Roadmap
authority: `CRAM project next steps.pdf`, Phase Seven, Steps 9.1–9.6, pages 17–20.*

This document records what was executed and observed. It distinguishes
**automated verification** (tests), **manual verification** (live browser), and
**outstanding** (not performed or not possible in this environment).

---

## 1. Test-suite verification (automated)

| Suite | Result |
|---|---|
| Full suite (`venv\Scripts\python.exe -m pytest -q`) | **115 passed, 0 failed, 0 skipped** |
| Phase 6 baseline | 85 passing — all preserved |
| New Phase 7 AppTests (`tests/test_v2_app.py`) | **35 passed** (replaces the 5 pre-existing single-page AppTests; 80 Phase 2–6 tests unchanged) |
| Warnings | 5,233 — identical pre-existing joblib/NumPy-2.5 deprecation categories; no new warning source |

Test coverage per acceptance area:

**Navigation** — `test_all_eight_pages_render_with_sample_data` (all 8 pages render by
`url_path` with sample data); `test_navigation_works_without_a_dataset`; `test_data_pages_prompt_
for_a_dataset_when_none_is_loaded`; `test_sidebar_uploader_visible_across_pages`.

**Onboarding / sample data** — `test_onboarding_explains_workflow_and_schema`;
`test_sample_dataset_loads_from_onboarding_and_marks_itself_as_sample`;
`test_tracked_sample_dataset_exists_and_passes_validation` (80 rows, 24 churners, 13 inputs,
synthetic IDs, passes the same validator as uploads);
`test_clear_session_data_returns_to_onboarding_despite_warm_caches`.

**Uploads/validation** — valid CSV activates and renders (`test_valid_csv_upload…`);
missing columns rejected with named columns and dataset **not** replaced
(`test_missing_columns…`, `test_failed_upload_does_not_replace_a_working_dataset`); invalid plan
values accepted but disclosed on Settings (`test_invalid_plan_values…`); malformed (ragged) CSV
rejected safely (`test_malformed_csv_is_rejected_safely`); header-only file rejected
(`test_empty_dataset_is_rejected`); ~12 MB file rejected by size check
(`test_oversized_file_is_rejected_by_size_check`); uploads never touch disk
(`test_uploaded_data_is_session_only_and_never_written_to_disk`).

**Existing functionality preserved** — simulation regression (full A/B/C independent
create/edit/reset/remove flow, ported from the Phase 5 test: `test_simulation_lab_regression…`);
SHAP profile explanations (`test_customer_insights_render_profile_shap_and_export_wiring`);
portfolio + individual recommendations (`test_retention_strategies_render…`); search filtering
runtime-verified (`test_customer_search_filters_the_table_and_export_matches_filters`); risk-band
reclassification propagates across pages (`test_risk_band_reclassification…`); AI Analyst
facts-only and mocked-LLM flows ported from Phase 6.

**Exports** — analytics-summary content (`test_overview_analytics_summary_export_content`);
recommendation export serialization (`test_retention_recommendation_export_serializes…`, using
the sample portfolio which triggers 26 suggestions); per-customer SHAP export wiring;
global-importance export (`test_model_center_exports_global_importance`); a wiring matrix
asserts every required export exists on its page (`test_all_required_export_keys_are_wired…`).

**Reliability/security** — model-load failure handled with a friendly message and no unhandled
exception (`test_model_load_failure_is_handled_gracefully`, simulating artifact corruption);
clear-session control (`test_settings_clear_session_control`); bounded analyst chat + clear
action (`test_analyst_chat_history_is_bounded_and_clearable`); unexpected analyst failure
becomes a session answer, not a crash (`test_analyst_unexpected_failure…`); privacy
disclosures on AI Analyst and Settings (`test_analyst_privacy_disclosures…`); credentials never
echoed; schema documentation (`test_settings_documents_supported_schema`); Model Center works
without a dataset.

**Test-tooling note.** This Streamlit version's AppTest cannot inspect `st.download_button`
payloads, so export *content* is verified at the data level with the same shared functions the
pages call, plus source-level wiring assertions. AppTest also crashes when a run follows an
in-run widget-state cleanup (seen after the Settings clear control); that flow is verified via
session-state assertions plus a fresh-session check instead. These are harness limits, not app
defects — both behaviors were additionally verified in the live browser (§2).

## 2. Live manual verification (browser)

The app was run with `venv\Scripts\python.exe -m streamlit run v2_app.py --server.port 8599`
and driven in a real Chromium browser:

- First run shows the onboarding workflow on Overview (Welcome → Upload → Validate → Run
  Analysis → Explore) with the "Load the built-in sample dataset" action.
- One click on the sample button produced "Analysis ready — Sample data … (80 rows)", the
  sidebar portfolio filters, and the Overview metrics: **80 customers analyzed, 24 predicted
  churners, 28.3% average predicted probability, 24 high-risk customers, 30.0% observed churn
  rate** — matching the tracked sample (24 churners / 30%).
- Sidebar navigation (no reload) verified every page: Customers (table + export),
  Customer Insights (profile C-0002: 95.4% probability, "Likely to churn", High Risk, SHAP
  summary, export), Retention Strategies (portfolio + individual sections), Simulation Lab
  (baseline metric), AI Analyst (question input, suggested questions), Model Center (champion
  model, held-out metrics), Settings (thresholds, schema, privacy).
- Settings page loads the sample independently (`test_settings_clear_session_control` +
  browser check "Active dataset:"), confirming dataset management lives in Settings as approved.
- No unhandled Streamlit error screens appeared; console contained no application errors (the
  only 404s were `_stcore/health` under sub-paths caused by hard full-page reloads during
  testing, not by in-app navigation).
- Screenshots were not capturable in this environment (no visible desktop window); the
  accessibility-tree walkthrough above is the manual-verification record.

## 3. Protected artifact verification

- `git diff 06f0fecc907a5da72d332c4d1761ae836fafa8c2 HEAD -- app.py train_model.py
  model_features.py random_forest_model.joblib model_metadata.json models vercel-app` →
  **empty output** (all 28 protected V1 artifacts byte-identical). Exit code 0.
- `ml/analyst.py` **not modified** (git status confirms).
- `requirements.txt` not modified — **zero new dependencies**.
- `git tag cram-v2-phase5-complete`, `stash@{0}`
  ("C.R.A.M. Phase 2 signed-off snapshot before Phase 3 (2026-11-13)"), and branch
  `codex/phase2-signed-off-backup` all intact.

## 4. Work-tree status (not yet committed)

```
 M tests/test_v2_app.py   (Phase 7 suite, 35 tests)
 M v2_app.py              (thin entry: 30 lines; was 752)
?? .streamlit/config.toml (enterprise theme)
?? assets/sample_customers.csv + assets/README.md (tracked de-identified sample)
?? v2_pages/ (8 page modules + __init__.py)
?? v2_shared.py           (shared state/caching/validation/filters/exports)
```

No push, merge, deployment, or Phase 8 work occurred. **The implementation awaits final
review and commit authorization.**
