# Phase 7 Final Handoff — C.R.A.M. V2 "Build the Final SaaS Experience"

*2026-11-16, branch `v2-final-development`, implementation complete and fully tested; nothing
committed, pushed, merged, or deployed — awaiting review and commit authorization.*

---

## 1. Executive summary

Phase 7 consolidated the Phase 2–6 technical capabilities into a cohesive eight-page,
enterprise-styled SaaS experience. The 752-line single-page dashboard became a thin entry point
plus eight page modules and one shared-foundation module. Every existing feature was migrated
to its roadmap-designated destination, the first-run onboarding workflow and a tracked
de-identified sample dataset were added, CSV upload gained a friendly validation gate in front
of the unchanged Phase 3 validator, five required export families were implemented, and the
Step 9.6 reliability/security items plus the four Phase 6 carry-forward items (F-4, F-7, F-8,
F-9) were completed. The full test suite stands at **115 passed** (85 Phase 6 baseline
preserved, 5 old single-page AppTests replaced by 35 new eight-page AppTests).

## 2. Roadmap step coverage (Steps 9.1–9.6)

| Step | Requirement | Status |
|---|---|---|
| 9.1 | Enterprise redesign | ✅ `.streamlit/config.toml` (deep-navy surfaces #0b1220/#111a2e, blue accent #3b82f6, light text), consistent headers/captions, altair risk-band colors (red #ef4444 / amber #f59e0b / green #22c55e), blue group charts, `width="stretch"` charts, no decorative AI graphics, no extra animation libraries |
| 9.2 | Eight pages, thin entry, feature migration | ✅ see §3 |
| 9.3 | Onboarding + sample dataset | ✅ see §5 |
| 9.4 | CSV upload + validation | ✅ see §6 |
| 9.5 | Exports and reports | ✅ five required exports (PDF deferred — see §8) |
| 9.6 | Reliability and security | ✅ see §7 |

## 3. Final eight-page feature map

| Page (url_path) | Migrated features (all preserved) |
|---|---|
| **Overview** (`overview`) | Portfolio metrics (5), risk-distribution chart, customer-pattern group charts (contract/tenure/payment/charge with proxy disclosures), portfolio warnings, analytics-summary export; hosts the first-run onboarding when no dataset is loaded |
| **Customers** (`customers`) | Searchable scored table (sort by probability, ID/probability/decision/band/estimated-inputs + optional columns), filtered CSV export |
| **Customer Insights** (`customer_insights`) | Customer profile metrics, observed-label disclosure, local SHAP explanation table + summary + imputation warnings, per-customer SHAP CSV export |
| **Retention Strategies** (`retention_strategies`) | Portfolio rule-based suggestions (chart, table, CSV export with serialized rule context) **and** per-customer recommendation details with its own customer selector (preserved from the old profile section) |
| **Simulation Lab** (`simulation_lab`) | A/B/C What-If scenarios lifted verbatim (independent widget state, reset, remove, out-of-range and imputation disclosures, comparison table), own customer selector |
| **AI Analyst** (`ai_analyst`) | Restricted Phase 6 analyst unchanged; adds F-4/F-7/F-8/F-9 (§9) |
| **Model Center** (`model_center`) | Champion model, binary threshold rationale, held-out metrics table (default + optimal operating points with costs), calibration, training-dataset and version metadata, input-range table, global SHAP importance + CSV export, risk-band defaults; works without a dataset |
| **Settings** (`settings`) | Dataset management (active-dataset status, validation notes, load sample, clear session), risk-band threshold sliders (identical bounds/steps to the old sidebar) + restore defaults, supported-schema table + validation explanation, F-4 privacy disclosure + safe credential-status indicator, session-isolation notes |

Nothing was silently removed. The one deliberate structural change: the old app rendered the
profile/SHAP section above the simulation lab on one page; those are now Customer Insights and
Simulation Lab pages, each with its own selector, matching the roadmap's migration table.

## 4. Architecture and file-structure changes

```
v2_app.py            30 lines  — page config, sidebar, st.navigation (8 st.Page entries)
v2_shared.py        419 lines  — session contract, model-load failure handling, upload
                                  precheck, cached parsing/analysis, dataset activation and
                                  clearing, sidebar (uploader/filters/thresholds caption),
                                  filtered view, explanation cache, exports
v2_pages/*.py                  — 8 page modules, each exposing render()
.streamlit/config.toml          — enterprise theme (Streamlit-native only)
assets/sample_customers.csv     — tracked de-identified sample (80 rows, 24 churners)
assets/README.md                — sample provenance and de-identification record
tests/test_v2_app.py 559 lines  — 35 AppTests
```

**Session contract** (Step 9.6 session isolation): the active dataset lives only in
`st.session_state["cram_upload_bytes"]`, activated **only** after `validate_upload_bytes`
accepts it — a rejected file never replaces a working dataset and never reaches the model.
Parsing is cached on the bytes; portfolio analysis is cached on the parsed frame; risk
classification is applied per render from session thresholds (never cached). Uploaded bytes are
processed in memory, never written to disk, never logged. `clear_session_data()` removes every
user-specific key (dataset, name, sample flag, validation notes, chat, simulation state, cached
explanation, thresholds). Shared model-resource caches contain no user data.

**Navigation testability.** AppTest cannot switch `st.navigation` pages via query params; the
suite uses `app._page_hash = calc_hash(url_path)` (`streamlit.navigation.page.calc_hash`,
deterministic BLAKE2b) wrapped in a `_goto(app, name)` helper. Verified empirically before
committing to native `st.navigation`.

## 5. Onboarding and sample dataset (Step 9.3)

- **Workflow** — Overview hosts the first-run experience: Welcome → Upload → Validate → Run
  Analysis → Explore, with a plain-language explanation of what C.R.A.M. does, the expected
  schema, and pointers to all pages. After activation, a one-time "Analysis ready …" banner
  names every page. Data pages show `no_dataset_prompt` when nothing is loaded; Model Center
  and Settings are fully functional without a dataset.
- **Sample dataset** — `assets/sample_customers.csv`: 80 customers (56 non-churn + 24 churn,
  30% churn rate) deterministically sampled (`random_state=42`) from the public 20% benchmark
  split `data/raw/churn-20.csv` (667 rows). De-identification: geographic quasi-identifiers
  (`State`, `Area code`) dropped (not model inputs); synthetic sequential `Customer ID`s
  (`C-0001`…); no names or phone numbers exist in the source. `assets/README.md` documents
  provenance. The sample passes the **same** `validate_upload_bytes` gate as uploads and is
  clearly labeled "Sample data" in the sidebar and in exports. One-click load is available on
  Overview (onboarding) and Settings. Training/production model data was not touched.

## 6. CSV upload and validation (Step 9.4)

Two layers, with the Phase 3 `CustomerDataValidator` unchanged as the authority:

1. **Friendly precheck (`validate_upload_bytes`)** — empty file, size > 10 MB, unparseable
   CSV (ragged rows / undecodable bytes), header-only table, and missing required columns
   (named explicitly) are each rejected with a specific, actionable message. A rejected file
   never activates; the previous dataset remains live.
2. **Post-activation disclosure (`summarize_schema_issues`)** — non-Yes/No plan values,
   non-numeric values, blanks, and blank optional columns are summarized on Settings as
   validation notes before analysis; per-customer handling remains the validator's job and is
   disclosed on the customer pages (rejected/coerced vs. genuinely-missing imputation).

The supported schema (13 required inputs + optional `Customer ID`/`Churn`) is documented on
Settings (table) and Model Center (inputs, ranges, and schema summary). The sidebar uploader is
compact and persistent across pages (approved placement); full management lives in Settings.

## 7. Exports (Step 9.5)

| Export | Page | File name | Content |
|---|---|---|---|
| Customer predictions + risk classifications | Customers | `cram_customer_risk_results.csv` | Full scored rows for the filtered view (ID, inputs, probability, decision, band, explanation inputs) |
| SHAP explanations | Customer Insights | `cram_shap_explanation_<ID>.csv` | Per-input submitted value, validation status, direction, contribution (probability points) + model summary |
| Retention recommendations | Retention Strategies | `cram_customer_recommendations.csv` | Per-suggestion rows with serialized `triggering_characteristics` and `model_context` |
| Analytics summary | Overview | `cram_analytics_summary.csv` | Portfolio metrics, risk-band counts, group churn rates (sections/metrics/values) |
| Global feature importance | Model Center | `cram_global_feature_importance.csv` | Full Phase 2 global SHAP table |

All exports respect the active dataset and current sidebar filters, name the dataset in
captions, include customer identifiers where appropriate, and contain no sensitive fields (the
schema has none; asserted by test). Every pre-existing export was preserved.

## 8. PDF report — deferred (documented decision)

A polished PDF would require a new rendering dependency (reportlab/weasyprint/pdfkit), which
Step 9.5 explicitly permits only if justified. The five required exports are complete, tested,
and dependency-free; the analytics-summary CSV carries the same content a PDF report would
present. **PDF is documented as a deferred optional enhancement**, per the authorization's
fallback clause. No dependency was added.

## 9. Reliability and security (Step 9.6) + Phase 6 carry-forward

- **Model-load failure**: `load_predictor_safe()` shows a friendly explanation plus a
  technical-detail caption (exception class only — no traceback) and stops gracefully; every
  model-consuming path routes through it, including the sidebar. Test-simulated artifact
  corruption produces no unhandled error screen.
- **Secrets**: keys are read from `st.secrets`/environment, used only by the analyst, never
  displayed or logged. Settings and AI Analyst show only a configured/not-configured indicator.
  Disclosure (F-4) states accurately that requests leave the application and contain column
  names, metadata/thresholds, and aggregate facts — never raw rows or identifiers.
- **Analyst robustness**: F-7 wraps the analyst call so unexpected failures become a session
  answer (friendly note + exception-class caveat), never an unhandled screen. F-8 bounds chat
  history to 10 entries with a visible "Clear chat" action. F-9 adds staged spinner feedback,
  disables Ask on empty input, and labels fallback behavior ("facts shown directly…").
  `ml/analyst.py` was **not modified**; the whitelist/planner/grounding architecture is intact.
- **Data privacy**: uploads stay in session memory (verified: no files written); no sensitive
  logging (nothing logs uploads or identifiers); clear-session control on the sidebar and
  Settings restores the onboarding state; content-keyed caches hold no identifiers, so cleared
  sessions do not resurrect data from warm caches (tested explicitly).
- **Session isolation**: all user-specific state lives in `st.session_state`; a fresh browser
  session starts at onboarding. Shared caches are model resources and content-keyed only.
- **Empty data**: empty uploads are rejected pre-activation; filter combinations that match no
  customers produce explicit warnings on every data page.

## 10. Full test results

- **115 passed, 0 failed, 0 skipped** (80 Phase 2–6 tests + 35 Phase 7 AppTests).
- Baseline regression: all Phase 2–6 behavioral guarantees maintained; the simulation
  regression test was ported verbatim in intent (independence, reset, remove, customer switch,
  explanation-cache stability).
- All LLM traffic mocked; **live Gemini API verification remains outstanding** (unchanged from
  Phase 6 policy — never exercised against the paid API by design).
- Live manual browser review of all eight pages: performed (see
  `PHASE_7_FINAL_VERIFICATION.md` §2).

## 11. Phase 2–6 behavior confirmation

No ML, analytics, preprocessing, recommendation, or simulation logic changed. `ml/analyst.py`
untouched. Protected V1 artifacts byte-identical to checkpoint `06f0fecc…` (verified). Model
artifacts untouched; no retraining. Tag `cram-v2-phase5-complete`, `stash@{0}`, and
`codex/phase2-signed-off-backup` intact.

## 12. New dependencies

**None.** Theme and charts use Streamlit + altair (already present). PDF deferred (§8).

## 13. Outstanding limitations and deferred work

- Live Gemini verification outstanding (mocked-only, by policy).
- PDF report deferred as an optional enhancement (§8).
- AppTest harness limits (download-button payloads; post-clear widget-state collection) —
  worked around in tests and covered by live-browser verification; documented in
  `PHASE_7_FINAL_VERIFICATION.md`.
- Pre-existing joblib/NumPy-2.5 deprecation warnings (5,233) unchanged — same backlog as
  A§8 in the Phase 6 handoff.
- Visual polish beyond the Streamlit-native theme (custom CSS, transitions) intentionally
  minimal per the "no unnecessary frontend" constraint; further polish is discretionary.

## 14. Git status and next move

Work tree contains the Phase 7 changes only (2 modified, 4 untracked paths — see
`PHASE_7_FINAL_VERIFICATION.md` §4). **Nothing committed.** No push, merge, deployment, or
Phase 8 work occurred. Next step: user review, then commit authorization.
