# C.R.A.M. Master Engineering Handoff

**Prepared:** 2026-11-14  
**Repository:** `C:\Users\Danny\OneDrive\Desktop\churnMLPredictor`  
**Branch:** `v2-final-development`  
**Purpose:** self-contained orientation for a fresh Codex session; distinguish repository evidence, earlier phase reports, and attribution evidence.

## 1. Project overview

C.R.A.M. means **Customer Retention & Analytics Model**. It began as a telecom customer churn prediction project: use historical customer records to estimate churn probability and help analysts prioritize review. The original V1 is a Streamlit application and separate Vercel application. V2 preserves that baseline and develops an auditable ML system around the 13 source customer inputs, with portfolio analysis, model explanations, cautious recommendations, and hypothetical scenario comparisons.

The intended users are analysts and customer-retention staff reviewing a customer or uploaded telecom portfolio. Predictions are estimates from a single public telecom dataset; rules are review heuristics; SHAP describes model behavior; scenario changes are hypothetical. None establishes causality or proves a business action retains a customer.

Engineering aims visible in the repository include preserving the original product, separating UI and ML logic, building repeatable preprocessing and evaluation, handling invalid/missing values explicitly, testing end-to-end Streamlit behavior, and documenting statistical and business-claim limits.

## 2. Ownership and development history

Attribution below follows the user's explicit provenance statement for Phases 0–1 and the later phase reports, code, and development-session context for Phases 2–5. Current uncommitted files do not have commit author metadata, so Git alone cannot independently prove their author. Do not infer that a Codex session built the whole project from scratch.

| Work | Original contributor | Current status |
| --- | --- | --- |
| Phase 0 — Preservation and initial audit | Antigravity | Inherited foundation |
| Phase 1 — ML Engine | Antigravity | Inherited foundation |
| Phase 2 — Explainable AI | Codex | Implemented; see Phase 2 records and source |
| Phase 3 — Customer Risk Intelligence | Codex | Implemented; see Phase 3 records and source |
| Phase 4 — Retention Recommendation Engine | Codex | Implemented and independently audited, subject to final review |
| Phase 5 — What-If Simulation Lab | Codex | Implemented; final independent review completed |

| Phase | Objective and implementation | Evidence, tests, limitations, status |
| --- | --- | --- |
| **0 — Preservation and audit** | Preserve original V1 and assess repository/model; establish V2 foundation. Original contributor: Antigravity. | `V2_AUDIT.md` and repository history document the original audit. The current V1 files match preserved checkpoint objects as described in Section 7. Historical Phase 0 authorship comes from the user's provenance statement; Git commit author fields do not identify Antigravity. Status: inherited foundation. |
| **1 — ML Engine** | Original Antigravity work: preprocessing and training, 11-model comparison, cost-threshold analysis, evaluation checkpoint, full-data serving model, inference API and metadata. Champion: HistGradientBoosting default configuration. | `PHASE_1_FINAL_SIGNOFF.md`, `PHASE_1_VERIFICATION.md`, `docs/model_evaluation.md`, `ml/train.py`, `ml/preprocessing.py`, `ml/predict.py`, `models/`. Original Phase 1 test totals are reported in those records, not rerun as a Phase 1-only suite here. One notable methodological boundary: evaluation artifact was fitted on 2,666 training records; serving artifact was refit on all 3,333. The serving artifact itself has no independent held-out evaluation. Status: inherited foundation. |
| **2 — Explainable AI** | Codex work: SHAP local/global explanation layer using a probability-output permutation explainer; fixed serving reference population; local reconstruction and stability/performance verification; input status for provided, missing, or rejected/coerced inputs. `ChurnPredictor.explain_single` uses the fixed `models/explanation_reference.csv` and rejects caller-supplied backgrounds to avoid stale-cache ambiguity. | `ml/explain.py`, `ml/predict.py`, `models/explanation_reference.csv`, `tests/test_explanations.py`, `docs/PHASE_2_FINAL_VERIFICATION.md`, `PHASE_2_FINAL_SIGNOFF.md`, `PHASE_2_PRESERVATION_REPORT.md`. Phase 2 records include reconstruction within numerical tolerance, reference-population tests, stability and runtime evidence. Current full suite rerun covers present tests but is not a rerun of every historical benchmark. SHAP is attribution, not causality. Status: signed off in historical records. |
| **3 — Customer Risk Intelligence** | Codex work: V2 portfolio dashboard; CSV validation, customer identity handling, risk filters/bands, summaries/charts/exports, profile selection and lazy local explanations. `ml/risk_intelligence.py` separates Low/Medium/High bands from binary classification. Duplicate/missing IDs are made unique; optional observed labels are parsed separately from predictors. | `v2_app.py`, `ml/risk_intelligence.py`, Phase 3 reports and `tests/test_risk_intelligence.py`, `tests/test_v2_app.py`. Prior Phase 3 report records AppTest and full-suite outcomes; current rerun is 41 passed. Some current Phase 3 tracked edits remain uncommitted. Status: implementation present; report evidence should be checked against the current worktree by the next integration review. |
| **4 — Retention Recommendation Engine** | Codex work: `ml/recommendations.py`; two transparent review rules, individual/portfolio APIs, SHAP context only, explicit human approval, deterministic heuristic priority. An independent audit found direct API validation gap; source was corrected to validate in `recommend_customer`, with a regression test. | `PHASE_4_FINAL_HANDOFF.md`, `PHASE_4_FINAL_VERIFICATION.md`, `PHASE_4_INDEPENDENT_AUDIT.md`, `PHASE_4_FINAL_SIGNOFF.md`, `tests/test_recommendations.py`. Earlier documented totals differ by stage (24 before audit regression; 25 after the fix at the independent audit). Current full suite: 41 passed. Reported 2,666-row timing: about 0.49s portfolio scoring and 0.12s rule generation; machine-specific. No validation against real intervention results. Status: implementation and independent audit PASS; consolidated final readiness is recommended before further phases. |
| **5 — What-If Simulation Lab** | Codex work: `ml/simulation.py` plus V2 UI integration. Up to three independent A/B/C scenarios. Each changes only canonical model inputs, validates values, reuses the serving predictor, computes signed probability deltas and classifications, supports reset/remove/customer switch, and warns on per-feature range exceedances. | `PHASE_5_FINAL_HANDOFF.md`, `PHASE_5_FINAL_VERIFICATION.md`, `PHASE_5_FINAL_SIGNOFF.md`, new `PHASE_5_FINAL_INDEPENDENT_REVIEW.md`, `tests/test_simulation.py`, `tests/test_v2_app.py`. Current complete suite 41 passed. Earlier benchmark on 150 calls: 11.3032s total; 75.355ms per scenario; 226.064ms for three scenarios, including validation and two prediction calls. The new review found no code defect. Important doc discrepancy: range metadata derives from all 3,333 feature rows, not only the 80% training partition. No causal interpretation. Status: PASS with this documentation precision issue and dependency warnings recorded. |

### 2.1 Earlier work on inherited Phase 0/1 foundation

Codex later audited, integrated with, and documented the inherited predictor and model behavior, including Phase 2 prediction/explanation boundary work and subsequent product additions. These later changes do not make Codex the original author of Phase 0 or Phase 1. Exact authorship of current uncommitted changes cannot be established from `git log`; use the explicit provenance above and describe uncertain attribution as uncertain.

## 3. Current architecture and data flow

### Entry points

- `app.py`: original V1 Streamlit application. Protected; unchanged versus preserved checkpoint.
- `v2_app.py`: V2 Streamlit portfolio/dashboard/profile/recommendation/simulation interface.
- `train_model.py`: legacy V1 training script; do not confuse with V2's `ml/train.py`.
- `vercel-app/`: existing V1 web deployment source and its model artifacts; protected and unchanged against preserved checkpoint.

### ML and product layers

- `ml/preprocessing.py`: canonical input schema, feature definitions, range and integer constraints, robust categorical cleaning, deterministic derived usage charges, validation, and scikit-learn transforms.
- `ml/train.py`: split/CV/model comparison, threshold selection, evaluation artifact, metadata, and full-data serving refit. Running it would retrain and overwrite artifacts; do not run without explicit authorization.
- `ml/predict.py`: loads `models/champion_churn_model.joblib` and metadata, validates records, predicts through the fitted pipeline, classifies at a requested decision threshold, and lazily caches the SHAP explainer against the fixed reference.
- `ml/explain.py`: local/global SHAP explanation logic and attribution response formatting.
- `ml/risk_intelligence.py`: portfolio validation, scoring, risk bands, label parsing, IDs, and results.
- `ml/recommendations.py`: two rule-based suggestions and heuristic review ordering; no outreach or automatic action.
- `ml/simulation.py`: validates hypothetical changes and compares baseline/scenario through the existing serving predictor.
- `v2_app.py`: Streamlit UI, upload/profile state, cached portfolio/profile results, controls and visual outputs.

### Main flow

CSV upload → `CustomerDataValidator` cleans and preserves rows/columns → `analyze_portfolio` scores with the already-fitted predictor → risk bands are assigned separately from binary classification → selected profile calls local SHAP and rule recommendation logic → simulation copies the selected 13-feature profile and scores baseline and each candidate through the same serving predictor. Scenario computations skip SHAP; profile explanation is cached by canonical profile. Portfolio results are cached by upload data. No scenario persistence or user database is present.

### Relevant tree

```text
app.py                           # protected original V1
v2_app.py                        # V2 Streamlit interface
ml/
  preprocessing.py               # schema, validation, fitted preprocessing definitions
  train.py                       # training/evaluation; do not run casually
  predict.py                     # serving predictor and explanation cache
  explain.py                     # SHAP local/global layer
  risk_intelligence.py           # portfolio scoring/segmentation
  recommendations.py             # Phase 4 suggestion rules
  simulation.py                  # Phase 5 what-if engine
models/
  champion_churn_model.joblib    # protected full-data serving model
  champion_model_evaluation.joblib # protected train-only evaluation pipeline
  model_metadata.json            # serving metadata/thresholds/ranges
  explanation_reference.csv     # fixed SHAP reference population
tests/                            # unit tests and Streamlit AppTests
docs/                             # model evaluation and Phase 2 verification
vercel-app/                       # protected V1 web app/model deployment source
PHASE_*                           # phase sign-offs, verifications and handoffs
```

### Validation and errors

`CustomerDataValidator` fills missing model inputs, normalizes binary values, derives charges and enforces configured ranges while recording warnings. In prediction, `estimated_fields` identifies originally missing values; rejected/coerced values are separately tracked in explanation and recommendation paths. Scenario edits use stricter per-edit checks and reject missing or invalid edits before model calls. Portfolio CSV ingestion can retain incomplete rows with imputation, so users must read disclosure fields and validation warnings.

## 4. Consolidated model and methodology reference

### Dataset and 13 inputs

The model metadata and Phase 1 reports describe 3,333 telecom records with 483 churn records (14.49%, approximately 5.9:1 non-churn to churn). Phase 1 split used 80% train (2,666 rows) and 20% held-out test (667), with stratification and random state 42. The five-fold CV comparison was run on the training split.

Canonical inputs (all are used by the serving pipeline): `Account length`; `International plan`; `Voice mail plan`; `Number vmail messages`; `Total day minutes`; `Total day calls`; `Total eve minutes`; `Total eve calls`; `Total night minutes`; `Total night calls`; `Total intl minutes`; `Total intl calls`; `Customer service calls`.

`ml/preprocessing.py` has explicit broad valid bounds: account length 0–500; binary plan inputs 0–1; voicemail messages 0–150; day/evening/night minutes 0–600; day/evening/night calls 0–300; international minutes/calls 0–100; customer-service calls 0–50. Account length, message count, all call counts, and service calls require whole numbers. The UI encodes binary inputs as Yes/No. These are validation bounds, not claims about realistic customer combinations.

### Model artifacts and evaluation

- Champion: default `HistGradientBoostingClassifier` pipeline.
- `models/champion_model_evaluation.joblib`: train-only (2,666 row) checkpoint used to reproduce held-out test metrics.
- `models/champion_churn_model.joblib`: serving pipeline refit on all 3,333 rows; its operational performance is inferred from the evaluation artifact and has not itself received an independent holdout evaluation.
- Threshold: optimal cost threshold stored as approximately `0.0772727`; default threshold `0.50`. It was selected from cross-validated training predictions on a 100-point grid, under illustrative FN=$500 and FP=$50 costs. Those are assumptions, not validated real costs.
- Phase 1 held-out test at 0.50: accuracy 95.35%, precision 95.83%, recall 71.13%, F1 81.66%; confusion matrix TN567/FP3/FN28/TP69; modeled cost $14,150.
- At the selected 0.0773 threshold: accuracy 94.60%, precision 79.05%, recall 85.57%, F1 82.18%; TN548/FP22/FN14/TP83; modeled cost $8,100 versus $48,500 all-negative baseline. The $40,400 modeled difference is not demonstrated business savings.
- Benchmark report's CV champion row: precision 93.77%, recall 74.09%, F1 82.78%, PR-AUC 85.81%, ROC-AUC 90.91%, Brier 0.0396, expected illustrative cost $50,950.

### Calibration

Phase 1 report values: uncalibrated Brier 0.0397/ECE 0.0301; Platt Brier 0.0406/ECE 0.0313; isotonic Brier 0.0396/ECE 0.0266. Although isotonic is numerically best by a very small amount, the sign-off says the improvement was insufficient for applying calibration; the deployed serving champion remains uncalibrated. Preserve this distinction.

### Explanations and risk bands

Phase 2 uses SHAP `PermutationExplainer` to explain model churn probability against a fixed canonical reference population (`models/explanation_reference.csv`). Contributions reconstruct the direct probability within the tested tolerance; local/global computation and stability evidence is in `docs/PHASE_2_FINAL_VERIFICATION.md` and Phase 2 sign-off. Serving rejects alternate backgrounds; custom analyses can use `ChurnExplainer` directly. Explanations are model attribution, not causal effects.

Risk bands are separate from binary model classification. Defaults place Low below the approximately 0.0773 medium floor, Medium from there up to but excluding 0.50, and High at or above 0.50. Binary customer classification uses the configured decision threshold (normally the optimal value), not the risk-band label.

### Recommendation rules and priority

`ml/recommendations.py` gates suggestions to probability at or above the supplied prediction threshold and non-Low risk. The rules are:

1. **Repeated support contact review:** `Customer service calls >= 3`; suggestion is a human review of contacts/resolution, not a finding of dissatisfaction or failure.
2. **International usage and plan fit review:** International plan is No and either international minutes `>= 10` or calls `>= 4`; suggestion is to review fit, not evidence a mismatch or savings.

Priority score is `min(100, 20 * rule_strength + 10 * risk_band_rank + min(10, sum(max(0, related_SHAP_contribution * 100))))`, where rule strengths are 3 and 2 and risk ranks are High=2, Medium=1, Low=0. High priority starts at score 85. It is a heuristic ordering, not efficacy, financial impact, or probability of success. Recommendations require human approval.

### Simulation

Each scenario edits supported input values on a separate copy. Both predictions use the same fitted serving pipeline; signed difference is simulated minus baseline probability and percentage-point difference is `100 * difference`. Classifications use the same threshold. Each UI customer can have up to three scenarios; they reset/remove independently and are not persisted. OOR warning checks each scenario feature against metadata min/max. **Source correction:** metadata ranges are computed from all 3,333 rows in `X`, not just training rows; the check is marginal and is not a multivariate novelty detector. Usage attributes and account length may not be directly controllable interventions. The output estimates only how the model responds to hypothetical inputs.

### What is observed, estimated, heuristic, hypothetical

- **Observed:** values and labels supplied in uploaded records; descriptive portfolio statistics.
- **Model estimate:** predicted churn probability/classification from the fitted model; subject to dataset and serving limits.
- **Heuristic:** recommendation triggers and priority score.
- **Hypothetical:** scenario probability difference after changing model inputs.
- **Not validated:** effectiveness of offers, service actions, customer response, actual prevented churn, uplift, or financial savings.

## 5. File and artifact inventory

Phase attribution in this inventory is historical attribution from user-provided provenance and phase documentation; exact Git authorship of current uncommitted files is uncertain.

| Path | Purpose / phase | Essential? / type | Preservation notes |
| --- | --- | --- | --- |
| `app.py` | V1 Streamlit app; inherited Phase 0/1 | V1 application | Protected; verified against checkpoint. |
| `random_forest_model.joblib`, `model_metadata.json`, `model_features.py`, `train_model.py` | V1 root model/schema/training files | V1 model/code | Treat as protected unless a separate authorized V1 task is opened. |
| `vercel-app/**` | V1 web app/API/model/static assets | V1 app + model | All 19 tracked files compared to checkpoint; unchanged. |
| `ml/preprocessing.py`, `ml/train.py`, `ml/predict.py` | Phase 1 engine; later Codex integration/audit edits in predictor | V2 essential Python | Model/prediction foundation. Do not retrain casually. `ml/predict.py` has current unstaged edits. |
| `models/champion_churn_model.joblib` | Full-data serving pipeline | Essential model binary | Protected; matches checkpoint. |
| `models/champion_model_evaluation.joblib` | Train-only evaluation artifact | Evaluation model binary | Protected; matches checkpoint. |
| `models/model_metadata.json` | Thresholds, schema, evaluation/range metadata | Essential data/config | Model input ranges currently reflect all data rows. |
| `models/explanation_reference.csv` | Fixed SHAP serving reference | Explanation data | Must stay aligned with Phase 2 docs and predictor policy. |
| `ml/explain.py` | Phase 2 SHAP engine; Codex | Essential V2 Python | Current unstaged matplotlib backend edit. |
| `ml/risk_intelligence.py` | Phase 3 portfolio services; Codex | Essential V2 Python | Untracked. |
| `ml/recommendations.py` | Phase 4 rules/API; Codex | Essential V2 Python | Untracked; independently reviewed. |
| `ml/simulation.py` | Phase 5 what-if engine; Codex | Essential V2 Python | Untracked; independently reviewed. |
| `v2_app.py` | V2 Streamlit application | Essential V2 UI | Untracked; AppTest-covered. |
| `tests/test_*.py` | Regression and AppTests | Verification | Current test tree includes 41 passing tests. |
| `PHASE_1_*`, `V2_AUDIT.md`, `docs/model_evaluation.md` | Inherited history and model record | Documentation | Keep historical values and caveats intact. |
| `PHASE_2_*`, `docs/PHASE_2_FINAL_VERIFICATION.md` | Phase 2 review/signoff | Documentation | Authority for explanation specifics. |
| `PHASE_3_*`, `PHASE_4_*`, `PHASE_5_*` | Phase-specific verification/handoff/signoff | Documentation | Compare to current source; reports can reflect earlier test counts. |
| `CRAM_MASTER_ENGINEERING_HANDOFF.md`, `CRAM_NEXT_CODEX_QUICKSTART.md`, `PHASE_5_FINAL_INDEPENDENT_REVIEW.md` | Current task handoff/review | Documentation | Newly prepared in this review. |

## 6. Test and verification record

### Independently rerun for this review

Command: `.\venv\Scripts\python.exe -m pytest -q`  
Result: **41 passed, 4,907 warnings in 14.49 seconds**. No failures.

Warning breakdown: 4,905 repeated `joblib`/NumPy 2.5 array-shape-restoration deprecation warnings; one anyio plugin warning that pytest's `CallSpec2` alias will be removed in pytest 10; one Windows loky physical-core detection warning and fallback to logical cores. Current suite loads and uses saved models successfully. Address NumPy/joblib compatibility before dependency upgrades; revisit pytest plugin before pytest 10. Core probe warning is environment-specific.

The complete run includes present Phase 2 explanation tests, Phase 3 risk tests, Phase 4 recommendation tests, Phase 5 simulation tests, and Streamlit AppTests. Earlier Phase 5 report describes AppTest interactions including no upload, profile selection, disclaimer, create/rename/edit, independent scenarios, three-scenario cap, reset/remove, compare, and customer switching. Current aggregate suite was rerun; historical focused run times below are not fresh benchmarks.

### Earlier phase-reported results (not combined into one run)

- Phase 3 reports cite a then-current 18-test full suite and AppTest dashboard verification.
- Phase 4 handoff cites 24 tests before the independent audit regression; Phase 4 independent audit reports 25 tests after direct API validation fix.
- Phase 5 verification cites 17 focused simulation/AppTest passes and 41 full-suite passes. The present run independently repeats the 41 full-suite result.
- Phase 4's 2,666-row portfolio benchmark and Phase 5's 150-call scenario benchmark are machine-specific earlier reports, not production throughput claims.

The independent review did not repeat all Phase 2 attribution/stability sweeps or historic performance measurements; their numerical claims remain historical report evidence, while current automated tests reran.

## 7. Git status and preservation

At this review:

- Branch: `v2-final-development`
- HEAD: `4cdd6a8ee1cfdbbe337dd4c998401e44cd76b546`, `Document Phase 2 preservation checkpoint`
- Backup branch: `codex/phase2-signed-off-backup` at `06f0fecc907a5da72d332c4d1761ae836fafa8c2`, `Preserve Phase 2 signed-off state`
- `main` and `origin/main`: `9f5edb6`, `Include Vercel chat parser in deployment`
- Stash: `stash@{0}` — “C.R.A.M. Phase 2 signed-off snapshot before Phase 3 (2026-11-13)”
- Worktree is dirty; no staged changes.

`git status --short --branch` at inspection:

```text
## v2-final-development
 M README.md
 M ml/explain.py
 M ml/predict.py
?? PHASE_3_FINAL_HANDOFF.md
?? PHASE_3_FINAL_VERIFICATION.md
?? PHASE_4_FINAL_HANDOFF.md
?? PHASE_4_FINAL_SIGNOFF.md
?? PHASE_4_FINAL_VERIFICATION.md
?? PHASE_4_INDEPENDENT_AUDIT.md
?? PHASE_5_FINAL_HANDOFF.md
?? PHASE_5_FINAL_SIGNOFF.md
?? PHASE_5_FINAL_VERIFICATION.md
?? ml/recommendations.py
?? ml/risk_intelligence.py
?? ml/simulation.py
?? tests/test_recommendations.py
?? tests/test_risk_intelligence.py
?? tests/test_simulation.py
?? tests/test_v2_app.py
?? v2_app.py
```

There are no staged changes. `git diff --cached` is empty. The tracked unstaged diff comprises README documentation and Phase 3 edits in `ml/explain.py` and `ml/predict.py`. Protected paths `app.py`, `models/`, and `vercel-app/` are absent from worktree status and have empty Git diffs.

### Historical artifact comparison

Compared all **24 tracked files** in `app.py`, `models/`, and `vercel-app/` against preserved checkpoint `06f0fecc907a5da72d332c4d1761ae836fafa8c2` using Git-filtered object hashes. All 24 matched checkpoint blobs. This includes the original V1 app, both V2 model binaries, and all V1 deployment app/model files. Raw file SHA-256 can differ where Windows line ending conversion is involved; the Git-filtered comparison accounts for repository filters. This is valid preserved-object evidence, unlike relying on a current hash alone.

No push, merge, deploy, reset, clean, retraining, or remote Git modification was performed for this review. Current Phase 3–5 work is uncommitted and must not be discarded.

## 8. Limitations, issues, blockers

### Known product/model limitations

- One public telecom dataset; external population, time, and industry generalization is unproven.
- No live intervention experiment or causal design. No actual retention uplift, saved customers, or real financial savings have been demonstrated.
- Inputs omit contract type, payment method, actual bill/invoice, customer satisfaction, service-resolution outcome, and other business context.
- Phase 4 has two narrow rules; eligible high-risk customers may receive no suggestion. Priority is a review-order heuristic.
- SHAP reflects the fitted model and chosen reference population, not causal impact or a prescription.
- Simulation may alter customer attributes that a business cannot directly change. Allowed input ranges do not make all combinations realistic. Per-feature bounds are not multivariate OOD detection.
- Model metadata range min/max is computed from all 3,333 records, not train-only; current Phase 5 language should say observed dataset range or full-dataset range.
- 4,905 joblib/NumPy deprecation warnings create future compatibility risk; current tested environment passes.
- Phase 3–5 files remain uncommitted. This is the primary delivery/continuity concern, not a model correctness failure.

### Blockers vs. future work

No blocking correctness issue was found for Phase 4/5; review recommendation is PASS. Before a release or external demonstration, perform a user-directed final integration/readiness review and decide how to preserve the uncommitted work. Future improvement priority is dependency compatibility, clearer data provenance/validation reporting, usability review, and external validation—not automatically a Phase 6 feature.

## 9. Recommended next steps

1. Inspect current `git status`, full diff, and all untracked Phase 3–5 files before any edit. Preserve the existing work.
2. Read `PHASE_5_FINAL_INDEPENDENT_REVIEW.md` and the Phase 3–5 verification, signoff, and handoff reports; reconcile any report wording against actual source, especially range provenance.
3. Decide with the user whether to commit/archive the completed uncommitted implementation. No remote actions are authorized by this handoff.
4. Conduct final integrated local usability/readiness review if requested, including a clean-environment run and realistic demo workflow; keep it separate from historical test claims.
5. Address NumPy/joblib and pytest plugin compatibility in a deliberate maintenance task before dependency upgrades.
6. Only after the user chooses a direction, consider user experience, reliability, demonstration readiness, or a future optional Phase 6. Do not assume new features are the highest-value next step.

## 10. NEXT CODEX — START HERE

You are continuing **C.R.A.M. V2**, a telecom churn-risk analysis application. Antigravity originally completed Phase 0 preservation/audit and Phase 1 ML engine. Codex later completed Phases 2–5 to the extent supported by repository/session evidence. Do not claim Codex built the project from scratch.

Phases 2–5 add SHAP explanations, portfolio risk intelligence, review-only recommendation rules, and a hypothetical scenario lab. The Phase 4/5 independent review recommends PASS. The present full suite is 41 passed with 4,907 warnings.

Protected original files/artifacts: `app.py`, root V1 model files, `models/champion_churn_model.joblib`, `models/champion_model_evaluation.joblib`, and `vercel-app/**`. All tracked protected paths match preserved Git checkpoint `06f0fecc907a5da72d332c4d1761ae836fafa8c2` by Git-filtered object comparison. Do not modify or retrain them without explicit authorization.

The current branch is `v2-final-development` at `4cdd6a8`; Phases 3–5 changes are uncommitted and unstaged, and a Phase 2 backup branch and stash exist. First inspect `git status --short`, `git diff`, `git diff --cached`, untracked files, then read `PHASE_5_FINAL_INDEPENDENT_REVIEW.md` and the latest Phase 3/4/5 reports. Do not reset, clean, push, merge, deploy, or start Phase 6 without explicit authorization.

Recommended immediate task: perform user-approved final integration/readiness review and resolve delivery strategy for the existing uncommitted work. Keep model probabilities, recommendations, and what-if changes within their documented evidence boundaries. Treat `PHASE_1_FINAL_SIGNOFF.md` as the inherited ML methodology record, `docs/PHASE_2_FINAL_VERIFICATION.md` as historical explanation evidence, the latest phase verification/signoff reports as phase records, and this handoff plus the independent review as current cross-phase orientation.

---

## 11. Phase 6 addendum — AI Business Analyst (2026-11-16, current session)

*Appended 2026-11-16 by the Phase 6 engineering session. Sections 1–10 above describe the state
before Phase 6 and were written earlier; the stale uncommitted-work statements in them were
resolved by checkpoint commit `3d9c795` (see `CRAM_FINAL_INTEGRATION_READINESS.md`). This
addendum records what changed in Phase 6 and does not modify the historical text.*

### 11.1 What was built

The user authorized **Option A + B** preservation (accept commit `3d9c795` on
`origin/v2-final-development`; local annotated tag `cram-v2-phase5-complete` at `3d9c795`, not
pushed) and Phase 6 was implemented in this session:

- `ml/analyst.py` — restricted analytics layer: dataset-aware context builder; 8 approved
  analytics functions (`portfolio_overview`, `risk_band_breakdown`, `churn_by_dimension`,
  `high_risk_profile`, `feature_distribution`, `probability_distribution`, `observed_vs_predicted`,
  `dataset_summary`) that reuse Phase 3 services; deterministic keyword planner (facts-only mode
  with no API key); Gemini planner/explainer (two LLM calls maximum, model `gemini-2.5-flash`,
  override via `CRAM_ANALYST_MODEL`); grounding validator that rejects any explanation numeral not
  derivable from computed facts; injectable HTTP transport with timeout/429/401-403/4xx error
  mapping; `answer_question` public API returning a structured `AnalystAnswer`.
- `v2_app.py` — "AI Business Analyst" section between the customer-table CSV download and the
  retention-actions section: suggested questions, chat history in session state, facts table,
  caveats, status-specific messaging, disabled Ask on empty input, key from `st.secrets` or env.
- `tests/test_analyst.py` — 36 tests (analytics correctness, context capabilities, keyword and
  mocked-LLM planning, cannot-answer behavior, grounding/injection resistance, transport error
  mapping, credential handling). All LLM traffic mocked; no network or paid API in tests.
- `tests/test_v2_app.py` — 2 new AppTests (facts-only fallback UI; mocked-LLM answered path).
- `CRAM_FINAL_INTEGRATION_READINESS.md`, `PHASE_6_FINAL_VERIFICATION.md`,
  `PHASE_6_FINAL_HANDOFF.md`, `PHASE_6_FINAL_SIGNOFF.md` — audit and phase records.
- `README.md` — Phase 6 section appended.

### 11.2 Verification record

- Full suite: **79 passed, 0 failed** (41 pre-existing Phase 2–5 tests all green; 36 analyst
  tests; 2 new AppTests). Warning count 5,233 — same pre-existing joblib/NumPy-2.5 deprecation
  categories; the increase over 4,906 reflects more model-loading tests, not a new source.
- Facts-only answer latency ~0.006 s on an 8-row portfolio; context build ~0.006 s.
- Protected preservation re-verified after implementation: `git diff 06f0fec -- app.py
  train_model.py model_features.py random_forest_model.joblib model_metadata.json models
  vercel-app` is **empty** (all 28 protected files match checkpoint
  `06f0fecc907a5da72d332c4d1761ae836fafa8c2`). No retraining occurred.

### 11.3 Security invariants established in Phase 6

The LLM may only (a) select one whitelisted function with validated arguments or emit
`cannot_answer`, and (b) narrate computed facts. It cannot execute code or SQL, access the
filesystem, modify data/thresholds/artifacts, or see credentials. API keys come from
`GEMINI_API_KEY`/`GOOGLE_API_KEY` (env; app also checks `st.secrets`), travel only in the
`x-goog-api-key` header, and are never logged or echoed. Zero new dependencies
(`requests==2.34.2` was already pinned).

### 11.4 Git state after Phase 6 (this session)

Branch `v2-final-development` at `3d9c795` plus uncommitted Phase 6 files listed in 11.1; local
tag `cram-v2-phase5-complete` at `3d9c795`; `stash@{0}` and `codex/phase2-signed-off-backup`
intact. The session is authorized to commit locally but **not** to push, merge, or deploy; the
Gemini integration was not exercised against the live API (tests mock all traffic by design).

### 11.5 Next session orientation

Start from `PHASE_6_FINAL_HANDOFF.md` §11 ("Next Codex — Phase 6 orientation") before editing
`ml/analyst.py`. Preserve the whitelist registry, argument validation, two-call LLM flow, and
grounding validator as invariants. Maintenance backlog from §8 (artifact gap, threshold grid
floor, dependency warnings) still applies and remains unfinished.

---

## 12. Phase 7 addendum — Final SaaS experience (2026-11-16, current session)

*Appended 2026-11-16 by the Phase 7 engineering session. Sections 1–11 above are historical
records of earlier phases and were not modified. Phase 7 work is in the working tree and is
NOT committed; the user must authorize the commit after review.*

### 12.1 What was built

The complete Phase 7 (roadmap Steps 9.1–9.6, pages 17–20) was implemented in one continuous
assignment on top of `cb86df6`:

- **Architecture:** `v2_app.py` rewritten as a 30-line thin entry (page config, sidebar,
  native `st.navigation` over exactly eight `st.Page` entries with stable url_paths). New
  `v2_shared.py` (419 lines: session contract, `load_predictor_safe`, upload precheck
  `validate_upload_bytes`, cached parsing/analysis, dataset activation/clearing, sidebar with
  compact uploader + portfolio filters, filtered view, explanation cache, export helpers) and
  `v2_pages/` with eight `render()` modules (overview, customers, customer_insights,
  retention_strategies, simulation_lab, ai_analyst, model_center, settings).
- **Design (9.1):** `.streamlit/config.toml` — deep-navy surfaces (#0b1220 / #111a2e), blue
  accent (#3b82f6), light text; risk-band colors red/amber/green via altair scales.
- **Onboarding + sample (9.3):** first-run workflow on Overview; tracked de-identified
  `assets/sample_customers.csv` (80 rows, 24 churners, synthetic IDs, geography dropped,
  `random_state=42` from the public benchmark split) + `assets/README.md` provenance;
  one-click load on Overview and Settings; sample passes the same validation as uploads.
- **Upload validation (9.4):** friendly precheck (10 MB limit, readability, non-empty, all 13
  required columns named on failure) gates activation so incompatible files never replace a
  working dataset or reach the model; schema quirks disclosed on Settings; Phase 3
  `CustomerDataValidator` untouched as the per-value authority; schema documented on Settings
  and Model Center.
- **Exports (9.5):** customer predictions/risk classifications (`cram_customer_risk_results.csv`),
  per-customer SHAP explanations (`cram_shap_explanation_<ID>.csv`), retention recommendations
  (`cram_customer_recommendations.csv`), analytics summary (`cram_analytics_summary.csv`),
  global feature importance (`cram_global_feature_importance.csv`) — all filter- and
  dataset-aware, no new dependencies. PDF report deferred as an optional enhancement
  (dependency unjustified; documented).
- **Reliability/security (9.6):** `load_predictor_safe` friendly model-failure handling on
  every consuming path; uploads session-only (never on disk, never logged); clear-session on
  sidebar + Settings; session isolation by design (user state in `st.session_state` only;
  shared caches content-keyed model resources); secrets never displayed/logged with a safe
  configured-indicator; schema documentation.
- **Phase 6 carry-forward:** F-4 provider-privacy disclosure (AI Analyst + Settings, accurate
  wording), F-7 defensive analyst error handling (unexpected failures become session answers),
  F-8 bounded chat history (10 entries) + clear action, F-9 latency UX (staged spinner, no
  empty submissions, labeled fallbacks). `ml/analyst.py` NOT modified; whitelist, planner
  validation, grounding, and credential handling intact.
- **Tests:** `tests/test_v2_app.py` rewritten — 35 AppTests covering navigation (8 pages),
  onboarding/sample, upload success/failure modes (missing columns, invalid values, malformed,
  empty, oversized, no-replacement), filters/search/exports, the ported simulation regression,
  analyst facts-only + mocked-LLM + bounded-chat + defensive-error flows, privacy
  disclosures, model-load failure, threshold reclassification, clear-session, and schema docs.
  AppTest page switching uses `app._page_hash = calc_hash(url_path)` (verified empirically;
  query-param switching does not work).

### 12.2 Verification record

- Full suite: **115 passed, 0 failed, 0 skipped** (80 Phase 2–6 tests unchanged; 35 Phase 7
  AppTests replace the 5 single-page ones; 85-test Phase 6 baseline preserved). Warnings
  5,233 — same pre-existing joblib/NumPy-2.5 categories.
- Live manual review: all eight pages driven in Chromium against a real `streamlit run`
  server with the sample dataset — metrics matched the sample (80/24/28.3%/24/30.0%);
  Customer Insights SHAP verified live; no unhandled error screens. Screenshots unavailable
  (no visible desktop window); walkthrough recorded in `PHASE_7_FINAL_VERIFICATION.md` §2.
- Protected preservation re-verified: `git diff 06f0fecc907a5da72d332c4d1761ae836fafa8c2 HEAD
  -- app.py train_model.py model_features.py random_forest_model.joblib model_metadata.json
  models vercel-app` is **empty** (all 28 artifacts byte-identical). No retraining.
- Zero new dependencies (`requirements.txt` unchanged).
- Live Gemini API verification remains outstanding (mocked-only, per standing policy).

### 12.3 Work-tree state after Phase 7

Modified: `tests/test_v2_app.py`, `v2_app.py` (752 → 30 lines). New: `.streamlit/config.toml`,
`assets/sample_customers.csv`, `assets/README.md`, `v2_shared.py`, `v2_pages/` (9 files), plus
the three `PHASE_7_FINAL_*.md` documents and this addendum. `cram-v2-phase5-complete`,
`stash@{0}`, and `codex/phase2-signed-off-backup` intact. **Nothing committed, pushed, merged,
or deployed; Phase 8 not started.** The commit awaits user authorization.


---

## 13. Migration addendum — Vercel production release (2026-11-16, current session)

*Appended 2026-11-16 by the migration session, per the "FINAL VERCEL MIGRATION, PRODUCT
INTEGRATION & PRODUCTION RELEASE" authorization. Section 12 above records the Phase 7
Streamlit state; this section records its retirement into the deployed Vercel application.*

### 13.1 What was built

The complete C.R.A.M. V2 functionality (Phases 2–7) now runs in the existing production
Vercel project ("C.R.A.M.", Root Directory `vercel-app`), preserving its identity
(teal accent, deep navy surfaces, Manrope/DM Sans, sidebar + breadcrumb patterns, guided
interview, customer snapshot, research-preview positioning). New semantic green/amber/red
tokens were added only for the risk-band system.

- **Engine relocation:** `ml/` moved via `git mv` to `vercel-app/ml/`; the four V2 artifacts
  (`champion_churn_model.joblib`, `champion_model_evaluation.joblib`,
  `explanation_reference.csv`, `model_metadata.json`) moved to `vercel-app/models/`.
  `ml/train.py` searches upward for the repo root so offline training still works, and a
  repo-root `conftest.py` puts `vercel-app` on `sys.path` for the test suite. Root V1
  artifacts (`random_forest_model.joblib`, `model_metadata.json`, `model_features.py`) are
  byte-identical (`git ls-tree` blobs re-verified during this session).
- **Serverless endpoints:** shared `_engine.py` at the `vercel-app` task root (the proven
  `model_features.py` bundling pattern) plus six endpoints: `api/score.py`, `api/explain.py`,
  `api/analyze.py` (full-CSV mode + rows re-band mode), `api/recommend.py`, `api/simulate.py`,
  `api/analyst.py` (GET status exposes only `provider.configured`; 12 s per LLM call,
  `maxDuration` 30). No retraining; the V2 champion "HistGradientBoosting (Default)" serves
  from `vercel-app/models/model_metadata.json` with its held-out operating points,
  calibration, cost framework (FN $500 / FP $50), and CV benchmark record — Model
  Performance computes no new numbers.
- **Production semantics:** the binary decision stays at `BINARY_DECISION_THRESHOLD = 0.5`
  ("Likely / less likely"), applied uniformly in score/explain/analyze/simulate
  (`_production_decision` overwrites the engine's stored operating point in `analyze`).
  V2 risk bands (Low/Medium/High, `RiskThresholds`, metadata defaults 0.07727/0.5) are
  separate and session-adjustable. The 0.0773 threshold appears only as an evaluated
  operating point on Model Performance.
- **Uploads:** the Phase 7 precheck is ported with documented production limits — 4 MB CSV
  text and 10,000 rows, chosen against Vercel's 4.5 MB request/response limits. A rejected
  upload never replaces the working dataset; the sample dataset loads in one click from
  `public/sample_customers.csv`; the five export families are all present.
- **Client:** single client shell extended to the eight-entry navigation (Predict, Portfolio,
  Customers, Retention Strategies, Simulation Lab, AI Analyst, Model Performance, Project
  Info/Settings). Small deterministic engine semantics (risk-band classification, portfolio
  summary arithmetic, Phase 7 filters) are mirrored in `lib/portfolio.ts` for instant
  display; the server remains authoritative for all model work and exports.
- **SHAP in serverless (empirically resolved):** shap 0.54.1 has no numba/llvmlite hard
  dependency (~4 MB total); the champion pipeline is 146 KB, so explanations run in-function.

### 13.2 Verification record (Gates A–B)

- **Python suite: 103 passed, 0 failed** — the 23-test `tests/test_vercel_api.py` layer
  covers analyze (full + re-band), score, explain, recommend, simulate, analyst
  (facts-only + status privacy), and JSON-safety; the 35 Streamlit AppTests were retired
  with the Streamlit UI. Pre-existing joblib/NumPy-2.5 warnings unchanged.
- **Frontend: 24 Vitest unit tests** pin the mirrored semantics (classify boundaries at
  `>=`, summary arithmetic, group rates, filters, CSV parsing/quoting, export shapes).
- **E2E: 13 Playwright journeys pass** (chat interview, unknown-detail disclosure, quick
  form, SHAP on demand, sample load, rejected-upload preservation, search/filter/export,
  customer detail + Simulation Lab hand-off, retention groups, scenario comparison,
  facts-only analyst, V2 operating-point record, settings + clear-session). The harness
  serves the real Python handlers behind a local proxy, so the journeys exercise the
  production code paths. All six endpoints were additionally exercised over real HTTP
  (status 200, correct semantics) before the suite was written.
- `npm run typecheck` clean; `npm run build` green.
- **Next.js security upgrade:** 16.4.3 was flagged by the framework for GHSA-mfr8-8qww-v76h
  (moderate; information disclosure across root params with `use cache`/`unstable_cache`).
  Upgraded to 16.4.6 in a background worktree, verified there (pre-retirement suite +
  build green), merged, and re-verified in place; `npm audit` reports 0 vulnerabilities.
  The app does not use the affected APIs.
- **Credential safety:** no key material in the client bundle; `GEMINI_API_KEY`/
  `GOOGLE_API_KEY` are read server-side from the environment only, and no `.env`/secret
  files are tracked.

### 13.3 Streamlit retirement (Gate C)

After parity was proven, `app.py`, `v2_app.py`, `v2_shared.py`, `v2_pages/`, `.streamlit/`,
and `tests/test_v2_app.py` were removed (user-confirmed full retirement; the V1 app
included, recoverable in git history). `requirements.txt` drops Streamlit and its
exclusive dependencies (altair, pydeck, pyarrow, blinker, watchdog); training-era
dependencies (lightgbm, xgboost, matplotlib) remain. The engine, the sample dataset
(`assets/` + `public/sample_customers.csv`), all artifacts, and the Phase 1–7 documents
are kept.

### 13.4 Deployment notes (Gate D)

- Deployment flows through the existing GitHub-linked Vercel project with Root Directory
  `vercel-app`; production branch `main` (previously at `9f5edb6`). No second Vercel
  project, no force-push. Branch flow: work on `v2-final-development`, then merge to
  `main`.
- `vercel-app/vercel.json` bundles the engine into every Python function
  (`_engine.py`, `ml/*.py`, `models/*`, plus the protected V1 files) and sets
  `api/analyst.py` `maxDuration` 30.
- `vercel-app/requirements.txt` pins the deployment stack (adds shap 0.54.1, requests,
  scipy, cloudpickle, slicer, tqdm, packaging).
- `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is the only environment variable the AI Analyst
  needs; without it the analyst runs in facts-only mode. No credentials are invented here.
- Post-deploy verification: check `/` renders, `/api/score` GET returns ready,
  `/api/analyze` scores the sample, and the analyst status endpoint reports the
  configured/unconfigured state accurately.
