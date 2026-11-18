# C.R.A.M. — Customer Retention and Analytics Model

Telecom customer churn research application. The production experience is
**C.R.A.M. (Customer Retention and Analytics Model)**, a Next.js interface served by the
C.R.A.M. V2 Python engine
as Vercel serverless functions (`https://churn-ml-predictor.vercel.app/`). The
Streamlit interfaces (V1 `app.py` and the Phase 3–7 V2 dashboard) were retired
after the Vercel application reached full feature parity; they remain
recoverable in git history.

## Application features

- **Predict** — guided chat interview or quick form for a single customer,
  with an estimated churn score, a "Likely / less likely" label at the
  production 50% cutoff, a separate V2 risk band, one-at-a-time sensitivity
  signals, and on-demand SHAP explanation from the fixed V2 reference
  population.
- **Portfolio** — CSV upload (Phase 3 validator + Phase 7 precheck: 4 MB /
  10,000 rows on Vercel; a rejected file never replaces the working dataset),
  headline metrics, risk-band distribution, group patterns, and an analytics
  summary export. A de-identified 80-customer sample dataset loads in one
  click.
- **Customers** — searchable, filterable scored table with per-customer detail
  (SHAP contributions, rule-based review suggestions), row hand-off to the
  Simulation Lab, and the customer predictions export.
- **Retention Strategies** — transparent, rule-based review suggestions with
  priority, trigger explanations, and the retention recommendations export.
  These are review prompts for human approval, not proven interventions.
- **Simulation Lab** — up to three what-if scenarios per baseline, each
  evaluated on a fresh copy through the serving predictor (predictive, not
  causal).
- **AI Analyst** — restricted natural-language analytics (Phase 6): eight
  approved functions compute every number in Python; an optional server-side
  Gemini key enables grounded narration that is validated against the computed
  facts (facts-only mode works with no key). Chat history is bounded.
- **Model Performance** — held-out metrics, missing-input sensitivity,
  confusion matrix, the V2 champion's operating-point record (the Phase 1
  cost-optimal threshold ≈ 0.0773 appears here only as an *evaluated operating
  point*; the production binary decision stays at 0.5), calibration, global
  Phase 2 SHAP importance, and the original Random Forest importance.
- **Project Info / Settings** — research scope, supported schema, dataset
  management, session risk-band adjustment, AI provider status, and privacy
  disclosures.

Five CSV export families are supported: customer predictions, per-customer
SHAP explanations, retention recommendations, the portfolio analytics summary,
and global feature importance.

## Repository layout

```
vercel-app/            production application (Vercel Root Directory)
  app/                 Next.js client shell + views
  api/                 Python serverless endpoints (score, explain, analyze,
                       recommend, simulate, analyst)
  _engine.py           shared engine wiring (limits, thresholds, serialization)
  ml/                  the C.R.A.M. V2 engine (preprocessing, predict, explain,
                       risk_intelligence, recommendations, simulation, analyst,
                       evaluate, calibrate, train)
  models/              V2 champion artifacts (HistGradientBoosting pipeline,
                       evaluation record, SHAP reference population, metadata)
  model_metadata.json  V1 Random Forest metadata (protected, byte-identical)
  random_forest_model.joblib / model_features.py   protected V1 artifacts
  public/              sample dataset + global feature importance CSV
assets/                de-identified sample dataset + provenance
data/raw/              raw training data (not tracked)
tests/                 Python test suite (engine + API layer)
docs/                  training output reports
```

## Run the production stack locally

```powershell
cd vercel-app
npm install
npm run build
npm run start          # Next.js on :3000
# the Python endpoints run as Vercel functions in deployment; the full local
# parity harness (Next + engine + proxy) is exercised by the Playwright suite:
npm run test:e2e
```

## Tests

```powershell
# Python engine + API layer (from the repo root)
venv\Scripts\python.exe -m pytest -q

# Frontend unit tests (Vitest)
cd vercel-app && npm run test

# End-to-end journeys (Playwright; builds and orchestrates the full stack)
cd vercel-app && npm run test:e2e
```

## AI Analyst configuration

The analyst runs in facts-only mode with no configuration. To enable
language-model narration, set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) as a
Vercel environment variable. The key is read server-side only, is never
returned to the browser, and requests send only column names, dataset
metadata, and aggregate computed facts — never raw rows or identifiers.

## Rebuild and evaluate the model

The training script uses the two CSV files referenced by the Colab notebook.
From the project root, run:

```powershell
python train_model.py --download
```

The CSV files are kept under `data/raw/` and ignored by git. The script
reserves one 20% stratified test set with seed 42, compares the selected
Random Forest against a majority-class baseline, reports churn
precision/recall/F1 plus ROC-AUC and PR-AUC, then refits the model on all rows
for the app. It writes `random_forest_model.joblib` and `model_metadata.json`.
The metadata records the source-file checksums, split, metrics, model
parameters, and package versions.

The held-out metrics estimate performance on this telecom dataset. They do not
show that the model generalizes to other industries or customer populations.

## Prediction semantics

- The binary customer decision stays at the established **0.5** cutoff
  ("Likely / less likely to churn") everywhere in the product.
- V2 **risk bands** (Low / Medium / High) are a separate, session-adjustable
  classification defaulting to the Phase 5 metadata thresholds.
- Unknown numeric inputs are filled using feature medians saved from the
  training data and are disclosed per customer; predictions with several
  unknown details are flagged as less dependable.
- The interface uses 13 customer facts; usage-charge fields are derived from
  minutes using the dataset's per-minute rates.

## Phase history

Phase documents are preserved as the engineering record of the Streamlit-era
work (the engine and its guarantees carry over unchanged into the Vercel
application):

- Phase 1: baseline + cost framework (`PHASE_1_FINAL_SIGNOFF.md`)
- Phase 2: SHAP explanation layer (`PHASE_2_FINAL_SIGNOFF.md`,
  `PHASE_2_PRESERVATION_REPORT.md`)
- Phase 3: risk-intelligence dashboard (`PHASE_3_FINAL_HANDOFF.md`,
  `PHASE_3_FINAL_VERIFICATION.md`)
- Phase 4: retention recommendations (`PHASE_4_FINAL_HANDOFF.md`,
  `PHASE_4_FINAL_SIGNOFF.md`, `PHASE_4_FINAL_VERIFICATION.md`,
  `PHASE_4_INDEPENDENT_AUDIT.md`)
- Phase 5: simulation lab (`PHASE_5_FINAL_HANDOFF.md`,
  `PHASE_5_FINAL_SIGNOFF.md`, `PHASE_5_FINAL_VERIFICATION.md`,
  `PHASE_5_FINAL_INDEPENDENT_REVIEW.md`)
- Phase 6: restricted AI analyst (`PHASE_6_FINAL_HANDOFF.md`,
  `PHASE_6_FINAL_SIGNOFF.md`, `PHASE_6_FINAL_VERIFICATION.md`,
  `PHASE_6_REMEDIATION_REPORT.md`)
- Phase 7: final SaaS experience + migration
  (`PHASE_7_FINAL_HANDOFF.md`, `PHASE_7_FINAL_SIGNOFF.md`,
  `PHASE_7_FINAL_VERIFICATION.md`,
  `CRAM_MASTER_ENGINEERING_HANDOFF.md` §13)
