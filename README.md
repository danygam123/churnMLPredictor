# Churn ML Predictor

Streamlit research app for exploring telecom customer churn.

## App features

- Guided chat accepts a natural-language customer description, extracts
  recognizable details, and follows up on fields it could not identify. A help
  popover explains the next field and can fill it with a disclosed typical
  value.
- Quick form supports direct entry and clearly marks median-filled fields.
- Batch CSV scoring includes a downloadable template and prediction file.
- Model evaluation shows held-out metrics, missing-input sensitivity, a
  confusion matrix, and global feature importance.
- Project overview includes the research paper and scope notes.

Unknown numeric inputs are filled using feature medians saved from the final
training data. A prediction with several unknown details may be unreliable; the
evaluation tab shows a separate simulated missing-data check. The app reports
the number of estimated fields and warns when several details are missing.

The interface uses 13 customer facts. Four usage-charge fields are derived from
minutes using the per-minute rates in this telecom dataset. The CSV template
lists the required input columns.

The chat uses local phrase and number matching for common descriptions (for
example, “120 daytime minutes” and “three calls to support”). It does not call a
hosted language model. Details it cannot confidently identify are left for a
follow-up instead of being guessed.

## Rebuild and evaluate the model

The training script uses the two CSV files referenced by the Colab notebook. From
the project root, run:

```powershell
python train_model.py --download
```

The CSV files are kept under `data/raw/` and ignored by git. The script reserves
one 20% stratified test set with seed 42, compares the selected Random Forest
against a majority-class baseline, reports churn precision/recall/F1 plus ROC-AUC
and PR-AUC, then refits the model on all rows for the app. It writes
`random_forest_model.joblib` and `model_metadata.json`. The metadata records the
source-file checksums, split, metrics, model parameters, and package versions.

The held-out metrics estimate performance on this telecom dataset. They do not
show that the model generalizes to other industries or customer populations.

## Run the app

```powershell
streamlit run app.py
```

## Run the local C.R.A.M. V2 risk dashboard

The Phase 3 customer-risk workspace is a separate Streamlit entry point so the
existing V1 interface remains available unchanged:

```powershell
venv\Scripts\python.exe -m streamlit run v2_app.py
```

Upload a CSV containing the 13 model inputs. A `Customer ID` column is optional
(temporary row identifiers are used when absent), and an optional `Churn`
column enables observed-rate summaries. Risk bands use the Phase 1 validation
thresholds by default and can be adjusted in the sidebar. These bands are
separate from the binary churn prediction threshold. The dashboard labels
missing dataset fields rather than inventing contract, payment-method, tenure,
or monthly-bill values; for the current telecom schema it uses account length
and the model's derived usage-charge total as explicitly named proxies.

Phase 4 adds transparent, rule-based retention review suggestions to the V2
portfolio and customer profile. The current evidence-based rules flag repeated
support contacts for a human review of the support experience and elevated
international usage when no international plan is recorded for a human plan-fit
review. Recommendations are available only to Medium or High Risk customers
whose probability meets the Phase 1 prediction threshold, and only when an
accepted observed characteristic matches a rule. Portfolio matches are
generated cheaply; local SHAP context is attached when a customer profile is
opened, so the app does not calculate SHAP for every uploaded customer during
ordinary portfolio rendering.

These suggestions are hypotheses, require human approval, and are not
automatically sent to customers or applied to accounts. SHAP values describe
how features influenced this model's probability relative to its fixed Phase 2
reference; they do not show that an action causes churn reduction. The source
model has no verified contract type, customer tenure, payment method, monthly
invoice, satisfaction, or issue-resolution fields. See
[`PHASE_4_FINAL_HANDOFF.md`](PHASE_4_FINAL_HANDOFF.md) for rule thresholds,
priority logic, caveats, and verification evidence.

Phase 5 adds an interactive What-If Simulation Lab to V2. From an uploaded
customer profile, create up to three independent scenarios, vary supported
model inputs, reset or remove scenarios, and compare model probabilities with
the original baseline. Only the existing 13 validated inputs are editable;
customer IDs, observed churn labels, and derived charge fields are not. The
simulation uses the current serving model and pipeline and is predictive, not
causal. A different model probability does not prove that a real business
action would produce that result. See
[`PHASE_5_FINAL_HANDOFF.md`](PHASE_5_FINAL_HANDOFF.md) for the editable inputs,
validation domains, scenario behavior, and limitations.

## Phase 6 — AI Business Analyst

Phase 6 adds a natural-language analytics assistant to the V2 dashboard. Ask
questions about the currently uploaded, filtered dataset — for example "How
many customers are predicted to churn?", "What are the most common
characteristics among high-risk customers?", or "Churn rate by Contract Type" —
and the assistant answers from actual computed analytics.

The architecture is deliberately restricted: a planner (the Gemini API when a
key is configured, or a deterministic keyword planner otherwise) selects one of
eight approved analytics functions with validated arguments; the application
computes the facts from the scored portfolio; and the language model only
narrates those verified results. Every numeral in a generated explanation is
checked against the computed facts before display, and an explanation that
cites an unsupported number is withheld in favor of the raw results. When
questions cannot be answered from the available data, the assistant says so
instead of guessing. Questions unsupported by the dataset — no observed labels,
a missing column, or a phrasing the planner cannot route — receive honest
explanations rather than invented statistics.

No API key is required: without `GEMINI_API_KEY`/`GOOGLE_API_KEY` the section
works in facts-only mode and shows computed results directly. The key is read
from the environment (or Streamlit secrets), sent only in the request header,
and never logged or displayed. The assistant cannot execute code or SQL, modify
uploaded data, or change thresholds and model artifacts. Group comparisons and
high-risk profiles are descriptive associations in the upload; they do not
establish causation or intervention effects. See
[`PHASE_6_FINAL_HANDOFF.md`](PHASE_6_FINAL_HANDOFF.md),
[`PHASE_6_FINAL_VERIFICATION.md`](PHASE_6_FINAL_VERIFICATION.md), and
[`PHASE_6_FINAL_SIGNOFF.md`](PHASE_6_FINAL_SIGNOFF.md) for architecture,
analytics functions, security controls, and the 79-test verification record.
