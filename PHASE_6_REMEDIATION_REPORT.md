# Phase 6 Remediation Report

**Date:** 2026-11-16
**Authorized scope:** findings F-1, F-2, F-3 from `PHASE_6_FINAL_REVIEW_AND_PHASE_7_READINESS.md`
**Branch:** `v2-final-development` (uncommitted work on top of `535fb20`; **not committed or pushed** — awaiting authorization)
**Result:** all three findings fixed, probes and full suite green; architecture and preservation unchanged.

---

## 1. F-1 — Keyword planner TypeError (Medium)

- **Original finding:** a labeled dataset with zero groupable columns plus a question containing "churn rate"/"highest churn" made `_keyword_plan` return `("churn_by_dimension", {}, None)`; `_execute` then called `churn_by_dimension(customers, context)` without the required `column` argument, raising `TypeError`, which escaped `answer_question` (only `ValueError` was caught) and would surface as a Streamlit error screen. Reproduced empirically during the audit.
- **Root cause:** the label-based fallback branch built its argument dict with an inline conditional that produced an **empty argument dict** instead of declining to route; combined with the narrow `except ValueError` in the computation step, the malformed plan became an unhandled exception.
- **Exact fix (two layers):**
  1. `ml/analyst.py` `_keyword_plan` — when the label-based fallback has no `dimension_candidates`, it now returns `(None, {}, <honest note>)` explaining that no groupable column exists and listing alternative supported topics. The planner can no longer generate invalid arguments.
  2. `ml/analyst.py` `answer_question` — defense-in-depth: the computation step now also catches unexpected `Exception` and converts it into a graceful `error` status with a user-friendly note ("The dashboard could not complete this analysis safely. Please try a different question."), so no computation failure can crash a Streamlit run.
- **Regression tests added (`tests/test_analyst.py`):**
  - `test_keyword_churn_rate_without_groupable_columns_is_honest` — reproduces the exact failure scenario (labeled portfolio, zero groupable columns) at the planner level, plus a positive control that a valid "churn rate" question with groupable columns still routes to `churn_by_dimension` with `dimension_candidates[0]`.
  - `test_churn_rate_question_without_groupable_columns_answers_gracefully` — end-to-end `answer_question`: status `facts_only`, no facts, honest note, no exception.
  - `test_unexpected_computation_error_is_reported_not_raised` — a function that raises `RuntimeError` yields `status="error"` with the friendly note instead of propagating.

## 2. F-2 — Spurious percent variants for values > 1 (Low–Medium)

- **Original finding:** `_add_number_variants` added `value * 100` decimal forms unconditionally, so a fact count of 8 authorized "800.0" in an explanation ("800" was already rejected, but "800.0" passed).
- **Root cause:** the three percent-decimal variants lived in the unconditional variant tuple while only the whole-percent form was gated to `(0, 1]`.
- **Exact fix:** removed the three unconditional variants; all four `value * 100` forms (`.0f`, `.1f`, `.2f`, comma `.1f`) are now generated only when `0 < value <= 1`. Share/rate facts in [0, 1] keep all legitimate percentage representations; counts and values above 1 authorize none.
- **Regression tests added:**
  - `test_grounding_rejects_percent_variants_for_values_above_one` — "800", "800.0", "800,000", "200", "20.0" all rejected against facts `{8, 2}` (the confirmed 800.0 case); boundary: fact `1.0` still authorizes "100%" and "100.0%".
  - `test_grounding_allows_legitimate_percent_and_decimal_forms` — whole-percent ("25%" / "25.0%" for fact 0.25), decimal-percent ("7.7%" / "7.73%" for fact 0.0773), and an unrelated percentage ("41.3%") still rejected.

## 3. F-3 — Sign-flipped numerals (Low)

- **Original finding:** `validate_grounding` accepted a token whose sign was stripped (`normalized.lstrip("-") in allowed`), so a positive fact of 8 authorized "-8" and (vice versa) a negative fact of -0.5 authorized "0.5".
- **Root cause:** the lenient sign-insensitive fallback line.
- **Exact fix:** removed the sign-stripping acceptance; tokens must exactly match an allowed variant. Facts that are themselves negative get pre-generated signed variants (`str(round(-0.5, 1))` → "-0.5"), so legitimate negative claims still validate.
- **Regression test added:** `test_grounding_rejects_sign_flipped_claims` — "-8" rejected against fact 8; "-0.5" accepted against fact -0.5; "0.5" rejected against fact -0.5 (positive and negative directions both covered).

## 4. Verification results

| Check | Result |
| --- | --- |
| Targeted regression tests (`pytest tests/test_analyst.py`) | **42 passed** (36 prior + 6 new), 0 failed |
| Complete suite (`pytest -q`) | **85 passed, 5,233 warnings, 0 failed** in 16.9 s |
| Phase 2–6 regression | All prior tests green; no behavior change outside the three fixes |
| F-1 reproduction probe (exact audit scenario) | `status=facts_only`, `facts=None`, honest note — **no exception** |
| F-2 reproduction probe | "800" → rejected; "800.0" → rejected; legitimate "25%" of 0.25 → accepted |
| F-3 reproduction probe | "-8" → rejected; legitimate "-0.5" → accepted |
| Protected artifacts | `git diff 06f0fec -- app.py train_model.py model_features.py random_forest_model.joblib model_metadata.json models vercel-app` → **empty (exit 0)**; all 28 files unchanged |
| Final diff inspection | `ml/analyst.py` +28/−7 (three fixes + defense-in-depth only), `tests/test_analyst.py` +91 (six tests only); no unrelated changes |

Warning total (5,233) is unchanged from the pre-remediation run — no new warning sources.

## 5. Preservation confirmation

- Eight-function whitelist, two-call LLM flow, credential handling, and all Phase 6 architecture: **unchanged**.
- Phase 2–5 behavior: **unchanged** (only `ml/analyst.py` and `tests/test_analyst.py` modified; both are Phase 6 assets created in this development line).
- Protected V1 artifacts: **verified unchanged** (28/28 match checkpoint `06f0fecc…`).
- No dependencies added; no model retraining; no refactors beyond the authorized fixes.
- `stash@{0}`, `codex/phase2-signed-off-backup`, and tag `cram-v2-phase5-complete`: **intact, untouched**.
- No commit, push, merge, or deploy performed.

## 6. Remaining known limitations (unchanged, by design)

- Wording/quantifier misdirection passes numeral-only grounding (F-5, documented limitation).
- Rounding drift within 1/2/4 decimal places remains permitted (F-6, by design).
- App-level call still lacks its own try/except (F-7) — mitigated by the new in-module `except Exception`; recommended for Phase 7's UX pass.
- Chat history grows unbounded per session (F-8) and worst-case LLM latency is ~60 s (F-9) — both deferred to Phase 7 polish per the review.
- Data-privacy disclosure for analyst requests (F-4) belongs in Phase 7's Settings page.
- The Gemini integration remains unexercised against the live API (tests mock all traffic by design).

## 7. Files modified

| File | Change |
| --- | --- |
| `ml/analyst.py` | F-1 planner fix + computation defense-in-depth; F-2 percent-variant gating; F-3 exact signed match |
| `tests/test_analyst.py` | 6 regression tests (F-1 ×3, F-2 ×2, F-3 ×1) |
| `PHASE_6_FINAL_VERIFICATION.md`, `PHASE_6_FINAL_HANDOFF.md`, `PHASE_6_FINAL_SIGNOFF.md` | Remediation addenda appended (existing text untouched) |
| `PHASE_6_REMEDIATION_REPORT.md` | This report |

**Status:** awaiting user authorization to commit (and separately, to push or begin Phase 7).
