# C.R.A.M. V2 — Phase 6 Final Review & Phase 7 Readiness Audit

**Date:** 2026-11-16
**Auditor:** independent engineering audit session (read-only; no code, tests, or documentation modified)
**Branch audited:** `v2-final-development` @ `535fb20` (Phase 6 commit)
**Method:** repository inspection, full-suite test rerun, source review of `ml/analyst.py` / `v2_app.py` / both analyst test files, empirical probes of the planner, grounding validator, and transport, protected-artifact Git-blob comparison, and extraction of the original Phase 7 specification from the master roadmap PDF.

---

## 1. Executive summary

Phase 6 (AI Business Analyst) is genuinely implemented, tested, and committed. The restricted
analytics-function architecture is real, not cosmetic: the LLM has no computation path, all facts
come from approved functions over the scored portfolio, planner output is strictly validated, and
explanations are numeral-grounded before display. The full suite independently reruns green at
**79 passed / 0 failed**, and all 28 protected V1 artifacts remain byte-identical to checkpoint
`06f0fecc…` by Git blob comparison.

The audit found **one confirmed defect** (an unhandled `TypeError` on a narrow edge path that
violates the "handles errors without crashing" completion criterion), **one grounding-validator
weakness** (spurious percent forms for values > 1 pass validation), and several minor/informational
items. None undermines the core architecture; all have small, contained fixes.

**Overall Phase 6 verdict: PASS WITH FINDINGS.**
**Phase 7 readiness verdict: READY AFTER FIXES** (two small fixes recommended before starting; see §8–§10).

## 2. Repository state (verified, not assumed)

| Item | Verified state |
| --- | --- |
| Branch / HEAD | `v2-final-development` @ `535fb20` ("Phase 6: AI Business Analyst - restricted analytics layer, grounded LLM explanations, 38 new tests") |
| Working tree | Clean — no modified, staged, or untracked files |
| Phase 5 baseline commit | `3d9c795` present in history; origin `v2-final-development` still points at `3d9c795` |
| Branch vs. origin | **Ahead 1** (`git rev-list --left-right --count` → `0 1`); Phase 6 is committed locally, **not pushed** |
| Phase 5 tag | `cram-v2-phase5-complete` (annotated, at `3d9c795`, local only) — intact |
| Backup branch | `codex/phase2-signed-off-backup` @ `06f0fec` — intact |
| Stash | `stash@{0}` ("C.R.A.M. Phase 2 signed-off snapshot before Phase 3 (2026-11-13)") — intact |
| `main` / `origin/main` | `9f5edb6` — untouched |
| Phase 6 commit contents | Exactly 10 files, +2,169/−0: `ml/analyst.py`, `tests/test_analyst.py`, `tests/test_v2_app.py`, `v2_app.py`, `README.md`, `CRAM_MASTER_ENGINEERING_HANDOFF.md`, `CRAM_FINAL_INTEGRATION_READINESS.md`, three `PHASE_6_FINAL_*.md` reports. No `requirements.txt` change (zero new dependencies confirmed) |

## 3. Architecture and implementation assessment

**Whitelist registry — correct.** `FUNCTIONS` contains exactly the 8 approved functions
(`portfolio_overview`, `risk_band_breakdown`, `churn_by_dimension`, `high_risk_profile`,
`feature_distribution`, `probability_distribution`, `observed_vs_predicted`, `dataset_summary`).
The only computation entry point is `_execute` (line 785), which dispatches to the registry with
arguments that survived `_validate_plan` (unknown args, missing args, non-string/empty values, and
non-existent columns all rejected; `churn_by_dimension` additionally re-validates groupability
inside the function). No `eval`, `exec`, SQL, shell, filesystem, or raw-data path exists anywhere
in the module.

**Computation/interpretation separation — maintained.** Facts are computed from
`PortfolioResult.customers` (Phase 3's scored output) in-process. The explainer receives only a
JSON blob of computed facts, caveats, and dataset metadata — never the raw frame — and its output
is a display string, never merged back into results. `AnalystAnswer` keeps `facts`, `caveats`,
`explanation`, and `note` in separate fields, and the UI renders them distinctly with a
verification caption.

**Function design — appropriate and reusable.** The functions are pure over `(customers, context)`,
return `{"facts": …, "caveats": …}`, raise `ValueError` with actionable messages when data cannot
support them, and reuse existing Phase 3 services (`portfolio_summary`, `churn_rate_by_group`)
rather than duplicating analytics. `high_risk_profile`'s standardized-difference comparison and
plan-share extraction are dataset-descriptive only, with explicit non-causal caveats. Notably,
`_render_ai_analyst(portfolio, predictor)` is already a self-contained function — this will make
the Phase 7 "AI Analyst" page extraction straightforward.

**Dataset-aware context — correct.** `build_context` correctly reports rows, present/missing model
inputs, extra columns, groupable columns (1 < unique ≤ 25, `Customer ID` and the raw `Churn` label
excluded), label availability/count, and all three thresholds. `suggested_questions` adapts to
labels and available dimensions.

**No-API-key fallback — reliable.** The deterministic keyword planner routes simple phrasings;
unroutable questions get an honest "cannot route" note listing supported phrasings. The facts-only
AppTest confirms the UI works end-to-end with no key and no network.

## 4. Correctness and security findings

### F-1 (Medium) — Unhandled `TypeError` in the keyword-planner fallback

- **Where:** `ml/analyst.py` — `_keyword_plan` (lines 530–531) interacting with `answer_question`'s computation catch (line 869–872, which catches only `ValueError`).
- **Evidence (reproduced during this audit):** a portfolio with valid observed labels but **zero groupable columns** (all features constant or > 25 distinct; labels all one value) plus a question containing "churn rate" returns `("churn_by_dimension", {}, None)`; `_execute` then calls `churn_by_dimension(customers, context)` without `column`, raising `TypeError: churn_by_dimension() missing 1 required positional argument: 'column'`, which escapes `answer_question` uncaught. In `v2_app.py` the call (line 264) has no try/except, so this surfaces as a Streamlit error screen for that run.
- **Why it matters:** violates the Phase 6 completion criterion "the assistant handles errors without crashing the application" on a reachable (if uncommon) dataset shape, and shows the error-handling net catches `ValueError` only.
- **Recommended resolution:** in `_keyword_plan`, return the honest cannot-answer note when the label-based fallback has no dimension candidates (e.g. treat empty `dimension_candidates` as unroutable); optionally broaden the computation catch to `Exception` as defense-in-depth; add a regression test for "labeled but zero groupable columns".

### F-2 (Low–Medium) — Grounding validator admits spurious percent forms for values > 1

- **Where:** `ml/analyst.py` — `_add_number_variants` (lines 659–661): `f"{value * 100:.1f}"`, `f"{value * 100:.2f}"`, and `f"{value * 100:,.1f}"` are added unconditionally, while the whole-percent form (line 664–666) is correctly gated to `0 < value <= 1`.
- **Evidence (probe):** with facts `{"customers_analyzed": 8}`, `"800"` is rejected but **"800.0" passes** (`validate_grounding` returns `None`).
- **Why it matters:** the explainer could misstate a count as a percentage (e.g. "churn could reach 800.0") and the explanation would display with its verification caption — a misleading claim that passes the very control meant to prevent it.
- **Recommended resolution:** gate all `value * 100` variants behind `0 < value <= 1`, matching the whole-percent gate; add a test asserting counts cannot be quoted as percentages.

### F-3 (Low) — Sign-flipped numerals pass grounding

- **Where:** `ml/analyst.py` — `validate_grounding` line 680 (`normalized.lstrip("-") in allowed`).
- **Evidence (probe):** with `{"customers_analyzed": 8}`, "About **-8** customers will churn." passes.
- **Why it matters:** a flipped-sign claim is factually wrong yet displays under the verification caption.
- **Recommended resolution:** require the signed token to match an allowed signed variant instead of stripping the sign.

### F-4 (Low, by design but should be disclosed in Phase 7) — Uploaded column names leave the machine

- **Where:** `ml/analyst.py` — `_build_planner_prompt` (lines 598–609) and the explainer payload (lines 880–899).
- **Evidence:** the planner prompt sends the uploaded dataset's column names, thresholds, and label availability to the Gemini endpoint; the explainer sends aggregate facts only. No raw customer rows or identifiers are transmitted.
- **Why it matters:** with a live key, metadata about a private upload leaves the deployment. This is inherent to the LLM narration design and consistent with the roadmap's later security step (9.6), but it should be stated in Settings/help text when Phase 7 adds deployment.
- **Recommended resolution:** document it; Phase 7's Settings page is the natural place to surface "analyst requests send column names and aggregate statistics to the configured provider."

### F-5 (Informational) — Misleading wording passes grounding

- **Where:** `validate_grounding` validates numerals only. Probe: "All 8 customers will definitely churn." passes with facts of 8 customers / 2 churners.
- **Why it matters:** tone/quantifier misdirection is not caught. This is a known, documented limitation (verification §5.4); the risk is bounded because numbers themselves cannot be invented and caveats are always displayed.
- **Recommended resolution:** accept as documented limitation; optionally add an explainer-prompt instruction review in a future iteration.

### F-6 (Informational) — Rounding drift is permitted by design

`_add_number_variants` allows 1/2/4-dp roundings of fact values (probe: "0.25" accepted for fact 0.2537). Reasonable for readability; documented; no change required.

### F-7 (Informational) — App-level call is not defensively wrapped

`v2_app.py` line 262–269 calls `answer_question` without try/except. The module is designed to return statuses rather than raise, but F-1 demonstrates the net has holes. A `try/except Exception` producing an `st.error` entry would be cheap defense-in-depth; the root fix remains F-1.

### F-8 (Informational) — Session chat history grows unbounded

`cram_analyst_chat` accumulates every Q&A (including full fact dicts) for the session. Trivial memory cost for realistic use; cap or "clear chat" control is a nice-to-have for Phase 7's UX pass.

### F-9 (Informational) — Worst-case latency of two sequential LLM calls

Planner + explainer each carry a 30 s timeout → worst case ~60 s behind the spinner. Failures degrade to facts within the same request. Acceptable; consider a shorter planner timeout in Phase 7 polish.

**Security summary:** prompt-injection resistance is real and tested — planner output must be strict whitelisted JSON (with one corrective retry), `run_sql`/invented functions are rejected, thresholds/data/artifacts are never writable by the analyst path, and the injected-key question cannot leak the API key (asserted in tests). Credentials are env-only (`GEMINI_API_KEY`/`GOOGLE_API_KEY`, app layer also checks `st.secrets`), transmitted only via the `x-goog-api-key` header, never logged or echoed. The code-fence extractor (`_strip_code_fences`) correctly recovers fenced JSON (probed) and its greedy two-object failure mode safely falls into the retry path.

## 5. Testing results

| Run | Result |
| --- | --- |
| Full suite (`pytest -q`, run fresh during this audit) | **79 passed, 5,233 warnings, 0 failures** in 15.7 s |
| Phase 2–5 regression (41 pre-existing tests) | All green |
| Analyst unit tests (36) | All green |
| AppTests (5, incl. 2 new) | All green |

**Tests are not passing vacuously.** The analytics-correctness tests exercise real computation
against real model artifacts (no mocking); the facts-only AppTest runs the real planner and real
functions end-to-end in the Streamlit harness; only the two LLM calls are mocked — which is the
correct, mandated design (no network/paid-API dependency). The facts-only transport test actively
fails if any network call is attempted without a key.

**Missing coverage (gaps, ranked):**

1. No test for the F-1 path (labeled dataset with zero groupable columns) — the one gap that would have caught the confirmed defect.
2. `_default_transport` branches for 401/403, generic 4xx/5xx, and non-JSON bodies are untested (only timeout + 429; other statuses are exercised only by injecting `AnalystError` at the answer level).
3. No test for `_strip_code_fences` with realistic fenced planner output.
4. No AppTest that a suggestion button populates the question box.
5. No test for `MAX_QUESTION_CHARS` truncation of over-long input.
6. No test for `feature_distribution` on a non-model-input column or `churn_by_dimension` grouped by `Observed_Churn` (both currently behave safely — verified by code read — but unverified by tests).
7. No test for `probability_distribution` bucket labels feeding the grounding key-collection path.

## 6. Repository integrity verification

- **Protected artifacts:** `git diff 06f0fecc907a5da72d332c4d1761ae836fafa8c2 -- app.py train_model.py model_features.py random_forest_model.joblib model_metadata.json models vercel-app` → **empty diff (exit 0)**: all 28 protected files remain byte-identical to the preserved Phase 2 checkpoint. No model artifact was modified or retrained.
- **Dependencies:** Phase 6 commit contains no `requirements.txt` change; the only import added is the already-pinned `requests==2.34.2` (used by `ml/analyst.py`). Zero new dependencies confirmed.
- **Phase 5 historical references:** tag `cram-v2-phase5-complete`, stash `stash@{0}`, backup branch `codex/phase2-signed-off-backup`, and commit `3d9c795` all intact and unmodified.
- **Documentation accuracy:** the Phase 6 reports' factual claims (8-function whitelist, 36 dedicated tests, 79 total, no new dependencies, ~0.006 s facts-only latency, protected-diff empty, "not exercised against the live API") were each checked against the repository and all hold. The master handoff's Phase 6 addendum correctly describes the Git state (ahead 1, tag local-only). One documentation nuance: the verification/signoff documents describe error handling as fully graceful; F-1 shows one reachable edge path escapes that claim — the documents should gain a one-line errata note when F-1 is fixed.

## 7. Outstanding bugs and weaknesses (ranked)

| # | Severity | Item | Location |
| --- | --- | --- | --- |
| F-1 | **Medium** | Unhandled `TypeError` (labeled + zero groupable columns → planner fallback → crash of the run) | `ml/analyst.py` `_keyword_plan` L530–531, `answer_question` L869–872; surfaces via `v2_app.py` L264 |
| F-2 | Low–Medium | Spurious percent forms for values > 1 pass grounding ("800.0" from count 8) | `ml/analyst.py` `_add_number_variants` L659–661 |
| F-3 | Low | Sign-flipped numerals pass grounding ("-8" for 8) | `ml/analyst.py` `validate_grounding` L680 |
| F-4 | Low (disclosure) | Uploaded column names/thresholds sent to provider when key configured | `ml/analyst.py` `_build_planner_prompt`, explainer payload |
| F-5 | Informational | Misleading wording passes numeral-only grounding (documented limitation) | `validate_grounding` |
| F-6 | Informational | Rounding drift permitted by design | `_add_number_variants` |
| F-7 | Informational | App call lacks defensive try/except | `v2_app.py` L262–269 |
| F-8 | Informational | Unbounded chat history in session state | `v2_app.py` `cram_analyst_chat` |
| F-9 | Informational | Worst-case ~60 s wait (two 30 s-timeout LLM calls) | `_call_llm` timeout usage |

## 8. Recommended fixes (all small and contained)

1. **Fix F-1 (do before Phase 7):** make the label-based keyword fallback return the honest cannot-answer note when `dimension_candidates` is empty; optionally catch `Exception` (not just `ValueError`) around `_execute` as defense-in-depth; add the missing regression test. ~10 lines + 1–2 tests.
2. **Fix F-2 (do before Phase 7):** gate the three `value * 100` decimal variants behind `0 < value <= 1`; add a test. ~3 lines + 1 test.
3. **Fix F-3 (cheap, do with F-2):** match signed tokens exactly; add a test.
4. **F-4:** no code change now; require a disclosure line in Phase 7's Settings/privacy text.
5. **F-7/F-8/F-9:** fold into Phase 7's UX/reliability pass (roadmap step 9.6), not blocking.
6. **Add the missing transport-branch and code-fence tests** (§5 gaps 2–3) when F-1/F-2 are fixed — same test file, no behavior change.

Nothing found requires re-architecture, re-testing of Phases 2–5, or any change to protected artifacts.

## 9. Original Phase 7 objectives and dependencies (verified from the master roadmap)

Source: `CRAM project next steps.pdf`, **"Phase Seven — Build the Final SaaS Experience" (Days 26–28, Steps 9.1–9.6, pages 17–20)** — quoted-equivalent, not reinterpreted:

- **Objective:** "Make C.R.A.M. look and feel like a real product. This is where we consolidate everything."
- **9.1 Redesign:** modern enterprise analytics design — deep navy/charcoal background, blue primary accent, green for positive indicators, red for elevated risk, amber for warnings, clean typography, consistent spacing, minimal clutter, interactive charts, subtle transitions; avoid excessive gradients/animations and decorative AI graphics.
- **9.2 Navigation:** pages — Overview (business-wide churn insights), Customers (searchable database), Customer Insights (individual explanations), Retention Strategies (potential actions), Simulation Lab (what-if), **AI Analyst (natural-language analytics)**, Model Center (model performance/technical details), Settings (dataset and application configuration).
- **9.3 Onboarding:** first-run workflow (Welcome → Upload → Validate → Run analysis → Explore) plus a built-in sample dataset so reviewers can explore without their own data.
- **9.4 Proper CSV upload:** accept valid CSVs, check required columns, identify missing fields, detect invalid values, explain validation errors, handle unexpected categories, prevent incompatible datasets from reaching the model; support one clearly documented telecom schema for the initial release.
- **9.5 Exports and reports:** export predictions, risk classifications, SHAP explanations, retention recommendations, and overall analytics summaries; a polished PDF report as an optional finishing touch.
- **9.6 Reliability and security before launch:** do not expose API keys; store secrets in deployment secrets; avoid logging sensitive customer information; avoid retaining uploaded datasets unnecessarily; validate file size and content; handle model-loading failures; handle empty datasets; useful error messages; uploaded data must not be accessible by another user; use de-identified/synthetic data for demos.

**Dependencies on existing components:** 9.2 consolidates every existing phase surface (dashboard → Overview, profile/SHAP → Customer Insights, recommendations → Retention Strategies, simulation → Simulation Lab, Phase 6 analyst → AI Analyst, Phase 1 metadata → Model Center); 9.4 builds directly on the Phase 3 `CustomerDataValidator`; 9.5 exports derive from Phase 2/3/4 outputs; 9.6 subsumes Phase 6's key/secrets handling and error behavior.

## 10. Phase 7 readiness verdict: READY AFTER FIXES

**Phase 6 introduces no architectural obstacle to Phase 7** — the analyst is already a standalone, page-extractable component, its state keys are namespaced, and it degrades gracefully without a key (which Phase 7's Settings page will manage). The foundation (clean tree, committed local history, green 79-test suite, intact protected artifacts) is suitable.

**Before starting Phase 7, fix F-1 and F-2 (and cheaply F-3):** F-1 is a genuine violation of a Phase 6 completion criterion and directly conflicts with Phase 7 step 9.6's "useful error messages / handle empty and unusual datasets" bar; F-2 weakens a headline Phase 6 security control that Phase 7 will advertise. Both are small, isolated changes in `ml/analyst.py` plus a handful of tests — no risk to Phases 2–5.

**Not ready "as-is" only because of F-1;** with those fixes (estimated well under an hour of work) the project is READY.

---

*This audit modified nothing except creating this report. No push, merge, deploy, retrain, or deletion of any ref was performed. Awaiting authorization for any next action.*
