# Phase 6 — AI Business Analyst: Engineering Handoff

**Date:** 2026-11-16
**Branch:** `v2-final-development`
**Author:** current engineering session (Phase 6 assignee)
**Status:** Implementation complete; 79 tests pass; pending user-authorized commit.

---

## 1. Objective

Add a natural-language analytics assistant to C.R.A.M. V2 so a business user can ask questions
about the active customer dataset and receive answers grounded in **actual computed analytics**.
The LLM interprets verified results; it never invents statistics, executes code or SQL, touches
the filesystem, or modifies data, thresholds, or artifacts.

## 2. Architecture

```text
User question (max 500 chars)
  ↓
Planner ─────────────────────────────────────────────┐
  • With API key: Gemini structured call            │ no API key → deterministic
    (pick ONE whitelisted function + validated      │ keyword planner (facts-only
    args, or cannot_answer; one corrective retry)   │ mode, zero network)
  • Without key: keyword planner                    │
  ↓                                                 │
Approved analytics functions (hardcoded registry)   │
  ↓                                                 │
Actual dataset calculations (scored portfolio)      │
  ↓                                                 │
Structured facts + caveats                          │
  ↓                                                 │
Explainer (LLM narrates facts; facts-only on failure/withhold)
  ↓
Grounding validator (every numeral must derive from computed facts)
  ↓
UI: explanation + facts table + caveats, or facts-only fallback
```

Two LLM calls maximum per question. The explainer is skipped (facts shown directly) whenever it
errors, returns over-length output, or produces an ungrounded number.

## 3. Files created and modified

| File | Change |
| --- | --- |
| `ml/analyst.py` | **New.** Full Phase 6 module: context builder, 8 approved analytics functions, keyword planner, Gemini planner/explainer, grounding validator, injectable HTTP transport, `answer_question` public API |
| `v2_app.py` | **Modified.** Imports `from ml import analyst`; adds `_flatten_facts`, `_render_analyst_answer`, `_set_analyst_question`, `_render_ai_analyst`; renders the "AI Business Analyst" section between the customer-table CSV download and "Potential retention actions for review" |
| `tests/test_analyst.py` | **New.** 36 dedicated tests (context, analytics correctness, planning, mocked-LLM behavior, grounding, security/injection, transport error mapping) |
| `tests/test_v2_app.py` | **Modified.** 2 new AppTests (facts-only fallback; mocked-LLM answered path) |
| `README.md`, `CRAM_MASTER_ENGINEERING_HANDOFF.md` | **Modified.** Phase 6 sections appended (append-only; existing text untouched) |
| `CRAM_FINAL_INTEGRATION_READINESS.md` | Pre-Phase-6 preservation audit (created earlier in this session; included in the commit) |

No protected file (`app.py`, root V1 files, `models/`, `vercel-app/`) was modified: all 28 still
match checkpoint `06f0fecc…` by Git blob comparison.

## 4. Approved analytics functions

| Function | Arguments | Returns (facts) |
| --- | --- | --- |
| `portfolio_overview` | — | customers analyzed, predicted churners + share, average probability, high-risk count, threshold, observed churn rate when labels exist |
| `risk_band_breakdown` | — | per-band counts, shares, average probability; observed rates over labeled rows only; band thresholds |
| `churn_by_dimension` | `column` | churn rate (observed if labeled, else mean predicted probability) and count per group, sorted descending; rejects unknown or ungroupable columns |
| `high_risk_profile` | — | High-Risk vs. other-customer feature means with standardized differences, plan Yes-shares, threshold; consistent schema even when the band is empty |
| `feature_distribution` | `feature` | count, missing/rejected count, mean, std, quartiles, min/max for one model input |
| `probability_distribution` | — | min/mean/max, quantiles, probability buckets, customers at/above the decision threshold |
| `observed_vs_predicted` | — | observed churn rate vs. predicted rate on labeled rows, TP/FP/FN/TN agreement counts; hard-requires valid labels |
| `dataset_summary` | — | rows, present/missing model inputs, extra columns, groupable columns, label availability, missing/rejected cell counts, thresholds, model name |

Existing Phase 3 services (`portfolio_summary`, `churn_rate_by_group`) are reused; no analytics are duplicated.

## 5. LLM provider and integration decisions

- **Provider:** Gemini REST API (`generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`).
- **Model:** `gemini-2.5-flash` default; override with `CRAM_ANALYST_MODEL`.
- **Credentials:** environment variables `GEMINI_API_KEY` or `GOOGLE_API_KEY` (the app layer additionally checks `st.secrets`). The key is passed only via the `x-goog-api-key` header, never logged, displayed, or embedded in prompts.
- **Dependencies:** zero new packages — uses the already-pinned `requests==2.34.2`.
- **Degradation:** no API key ⇒ deterministic keyword planner + computed facts (`facts_only`), so the feature works fully without any paid service. Planner failure ⇒ keyword fallback; explainer failure ⇒ facts only.

## 6. Security controls

1. The planner may select only registry function names; argument names and values are validated (unknown args, missing args, non-existent columns, empty strings all rejected).
2. No `eval`, `exec`, SQL, shell, filesystem, or network access exists in the computation path; the LLM response is pure data.
3. User questions are length-capped and embedded as untrusted data; planner output must be strict JSON matching the whitelist (with one corrective retry, then graceful fallback).
4. The grounding validator rejects any explanation numeral not derivable from computed facts (including percent variants of shares in [0,1]); withheld explanations are replaced by the raw facts with a visible note.
5. Explanations are capped at 2,000 characters.
6. Model artifacts, thresholds, uploaded data, and session state are never writable by the analyst path.

## 7. Test results

- `tests/test_analyst.py`: **36 passed** (all LLM traffic mocked; no network, no paid API).
- `tests/test_v2_app.py`: **5 passed** (3 pre-existing + 2 new Phase 6 AppTests).
- Full suite: **79 passed, 0 failed** in 17.95 s (5,233 warnings — all pre-existing joblib/NumPy deprecation categories; count grew only because more tests load model artifacts).

## 8. Performance observations

- Facts-only answer (plan → compute → package): ~0.006 s on an 8-row portfolio.
- Context build: ~0.006 s.
- LLM mode adds two bounded HTTP calls (30 s timeout each); all failures degrade to computed facts.

## 9. Known limitations

See `PHASE_6_FINAL_VERIFICATION.md` §5: upload-scoped answers, descriptive (non-causal) associations, keyword-planner phrasing limits, numeral-only grounding, provider dependence, question truncation at 500 chars, and exclusion of the outcome label from groupable dimensions.

## 10. Git state

- Branch `v2-final-development`; baseline tag `cram-v2-phase5-complete` at `3d9c795` (local only).
- Modified: `v2_app.py`, `tests/test_v2_app.py`, plus appended `README.md` / `CRAM_MASTER_ENGINEERING_HANDOFF.md`.
- New: `ml/analyst.py`, `tests/test_analyst.py`, `PHASE_6_FINAL_VERIFICATION.md`, `PHASE_6_FINAL_HANDOFF.md`, `PHASE_6_FINAL_SIGNOFF.md`, `CRAM_FINAL_INTEGRATION_READINESS.md`.
- Protected artifacts unchanged; stash `stash@{0}` and backup branch intact. No push, merge, deploy, reset, or retrain was performed.

## 11. Next Codex — Phase 6 orientation

Read `ml/analyst.py` top-to-bottom before changing it: the invariants are the whitelist registry,
argument validation, the two-call LLM flow, and the grounding validator. When adding an analytics
function: register it in `FUNCTIONS`, keep it pure over `customers`/`context`, return
`{"facts": ..., "caveats": ...}`, raise `ValueError` with a user-actionable message when the data
cannot support it, and add both a correctness test and a routing/grounding test. Never let a code
path send computed facts to the LLM for computation, and never display an unvalidated explanation.

---

## 12. Remediation addendum (2026-11-16)

Findings F-1/F-2/F-3 from `PHASE_6_FINAL_REVIEW_AND_PHASE_7_READINESS.md` were fixed under user
authorization (see `PHASE_6_REMEDIATION_REPORT.md` for root causes and evidence):

- `_keyword_plan` can no longer emit `churn_by_dimension` without a column; a labeled dataset with
  zero groupable columns now yields an honest cannot-route note, and `answer_question` converts
  unexpected computation exceptions into a graceful `error` status.
- `_add_number_variants` generates percent forms only for values in `(0, 1]`.
- `validate_grounding` requires exact signed matches (a positive fact cannot authorize a negative
  claim, and vice versa).

Test totals after remediation: **42 analyst tests, 85 total, 0 failures**. The §7 guidance above is
unchanged and now includes these three regressions as examples of the required paired tests.
