# Phase 6 — AI Business Analyst: Final Verification

**Date:** 2026-11-16
**Branch:** `v2-final-development` (baseline tag `cram-v2-phase5-complete` at `3d9c795`)
**Scope:** restricted natural-language analytics assistant added to C.R.A.M. V2.
**Verdict: PASS** — all dedicated tests, AppTests, and the full regression suite pass; protected artifacts unchanged.

---

## 1. What was verified

| Requirement | Status | Evidence |
| --- | --- | --- |
| Restricted analytics-function architecture (LLM never computes statistics) | PASS | `ml/analyst.py` — 8 whitelisted functions; planner may only select one of them or emit `cannot_answer`; arguments validated against a whitelist and column existence check |
| Actual dataset calculations on the active upload | PASS | All functions operate on `PortfolioResult.customers` from `analyze_portfolio`; facts are computed in-process, never by the model |
| Structured results separated from interpretation | PASS | `AnalystAnswer` dataclass: `facts`, `caveats`, `explanation`, `note`, `status` are distinct fields; UI renders them separately |
| Dataset-aware context | PASS | `build_context` reports rows, present/missing model inputs, extra columns, groupable columns, label availability and count, thresholds |
| Observed vs. predicted distinction | PASS | `observed_vs_predicted` function; `rate_is` labels whether a group rate is observed (labeled rows) or mean predicted probability; UI captions distinguish model estimates from observed data |
| Honest "cannot answer" behavior | PASS | LLM planner can emit `cannot_answer`; keyword planner refuses unroutable questions; functions raise `ValueError` for missing columns/labels which the UI surfaces as warnings. Tests: `test_planner_cannot_answer_is_honored`, `test_keyword_planner_admits_unroutable_questions`, `test_observed_vs_predicted_requires_labels`, `test_churn_by_dimension_rejects_unknown_and_ungroupable_columns` |
| No arbitrary code/SQL execution | PASS | No `eval`/`exec`/SQL anywhere in the analyst path; the only computation entry point is `_execute`, which dispatches to a hardcoded function registry with validated arguments. Test: `test_injected_planner_function_is_rejected` (`run_sql`/`DROP TABLE` rejected) |
| Prompt-injection resistance | PASS | Planner prompt marks the question as untrusted data; only whitelisted structured output is accepted; thresholds/data/artifacts unchanged during injection attempts. Test: `test_prompt_injection_cannot_change_state_or_execute` |
| Grounding of generated numbers | PASS | `validate_grounding` extracts every numeral from the LLM explanation and accepts it only if it derives from computed facts (incl. percent variants for shares in [0,1]); ungrounded explanations are withheld and raw facts shown. Tests: `test_grounding_validator_rejects_unsupported_numbers`, `test_explanation_with_invented_number_is_withheld`, `test_explanation_with_only_fact_numbers_is_accepted` |
| Malformed LLM responses handled | PASS | Missing fields, empty text, unknown functions, invalid JSON (with one corrective retry), over-length output. Tests: `test_llm_response_missing_fields_is_reported`, `test_unknown_function_from_llm_falls_back_to_keyword_planner`, `test_malformed_planner_json_falls_back`, `test_explanation_over_length_is_withheld` |
| API failures / timeouts / rate limits | PASS | Injectable transport; `requests.Timeout`, connection errors, 429, 401/403, 4xx/5xx and non-JSON bodies map to friendly fallback messages; planner failure falls back to the deterministic planner; explainer failure shows computed facts. Tests: `test_default_transport_maps_timeout_and_rate_limit`, `test_api_failure_on_unroutable_question_reports_error`, `test_rate_limit_message_is_friendly`, `test_explainer_failure_shows_facts` |
| Secure credential handling | PASS | Key read only from `GEMINI_API_KEY`/`GOOGLE_API_KEY` env vars (app layer also checks `st.secrets`); sent only in the `x-goog-api-key` header to the configured endpoint; never logged, echoed, or included in prompts. Tests: `test_get_api_key_reads_environment_only`; injection test asserts the key never appears in output |
| Mocked LLM in tests (no network, no paid API) | PASS | All 36 `tests/test_analyst.py` tests use an injectable transport or monkeypatched `_call_llm`; one test actively fails if any network call is attempted without a key |
| Streamlit AppTest coverage | PASS | 2 new AppTests: facts-only fallback (caption, facts table, disabled Ask on empty question) and mocked-LLM answered path (planner + explainer calls mocked, grounded explanation rendered, verification caption) |
| Phase 2–5 regression | PASS | Full suite: **79 passed** (41 pre-existing + 36 analyst + 2 new AppTests), 0 failures |

## 2. Test runs

| Run | Command | Result |
| --- | --- | --- |
| Analyst unit suite | `venv\Scripts\python.exe -m pytest tests\test_analyst.py -q` | **36 passed** |
| V2 AppTests | `venv\Scripts\python.exe -m pytest tests\test_v2_app.py -q` | **5 passed** (3 pre-existing + 2 new) |
| Full suite | `venv\Scripts\python.exe -m pytest -q` | **79 passed, 5,233 warnings in 17.95s** |

Warning total rose from ~4,906 to 5,233 because six test modules now load the model artifacts; the categories are the same pre-existing joblib/NumPy-2.5 deprecations plus the known anyio/loky probes. No new warning category was introduced by Phase 6. Debugging history: the first analyst run was 31/36; all five failures (empty-case facts schema, keyword routing for "churn rate by \<column\>", percent-variant grounding, transport fake signature, empty-response message wording) were fixed and re-verified.

## 3. Performance observations

- Facts-only answer end-to-end (plan → compute → package): **~0.006 s** on an 8-row portfolio.
- Dataset context build: **~0.006 s**.
- The LLM path adds two HTTP round-trips (planner + explainer), bounded by a 30 s timeout each; on transport failure the UI degrades to computed facts within the same request.

## 4. Protected-file preservation

After implementation, `git diff 06f0fecc907a5da72d332c4d1761ae836fafa8c2 -- app.py train_model.py model_features.py random_forest_model.joblib model_metadata.json models vercel-app` remains **empty**: all 28 protected files still match the preserved Phase 2 checkpoint. No model was retrained or replaced. `stash@{0}` and `codex/phase2-signed-off-backup` remain untouched.

## 5. Known limitations (honest boundaries)

1. The assistant answers questions about the **currently uploaded, filtered view**; it has no memory across uploads or sessions beyond the rendered chat list in session state.
2. Group comparisons are descriptive associations in the upload; they never establish causation, intervention effects, or business savings.
3. The keyword planner (no-API-key mode) routes simple phrasings only; unmatched questions get an honest "cannot route" note listing supported phrasings.
4. Grounding validation checks numerals, not wording: an explanation could still describe a fact loosely. Numbers themselves cannot be invented or altered.
5. The Gemini planner/explainer quality depends on the provider; every structured output is validated and every explanation is grounding-checked regardless.
6. Sub-500-character questions are truncated rather than rejected; extremely long questions lose their tail.
7. `Observed_Churn` grouping was deliberately excluded from groupable dimensions (grouping by the outcome label is not a segment comparison).
