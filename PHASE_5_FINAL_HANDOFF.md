# C.R.A.M. V2 — Phase 5 Final Handoff

**Phase:** What-If Simulation Lab  
**Status:** PASS  
**Entry point:** `v2_app.py`  
**Engine:** `ml/simulation.py`

## Open the Simulation Lab

From the project folder run:

```powershell
venv\Scripts\python.exe -m streamlit run v2_app.py
```

Upload a CSV with the existing 13 required model input columns. Use the dashboard's **Open a customer profile** selector. The Simulation Lab appears below that profile. Without a customer upload, the existing upload prompt is shown and no simulation controls are rendered.

## Work with scenarios

1. Review the selected customer's original 13 model inputs in **View original model inputs** and note the baseline predicted churn probability.
2. Choose **Create scenario** to add Scenario A. Add B and C the same way; three scenarios is the limit.
3. Rename a scenario if useful. Choose one or more supported inputs under **Attributes to vary**. The controls use model-specific binary choices and numeric/integer ranges.
4. Values recalculate automatically as you edit. Every scenario starts from the same selected customer input profile; changing one scenario does not carry over to another.
5. Compare the baseline and scenarios in the results table. It shows probability, probability-point difference, model classification, risk category, and changed inputs.
6. **Reset scenario** removes its edits and restores the baseline. **Remove scenario** deletes only that scenario. Selecting another customer resets the in-session scenarios for the prior customer.

Scenarios exist only in the current Streamlit session. The application does not save them to disk or a database.

## Editable fields and valid domains

Only features in the model's existing canonical input schema can be varied. Numeric bounds below come from the current `CustomerDataValidator`; integer features must also be whole numbers.

| Input | Scenario editor |
| --- | --- |
| Account length | Integer from 0 to 500; explicitly not verified tenure |
| International plan | Yes or No |
| Voice mail plan | Yes or No |
| Number vmail messages | Integer from 0 to 150 |
| Total day minutes | 0 to 600 |
| Total day calls | Integer from 0 to 300 |
| Total eve minutes | 0 to 600 |
| Total eve calls | Integer from 0 to 300 |
| Total night minutes | 0 to 600 |
| Total night calls | Integer from 0 to 300 |
| Total intl minutes | 0 to 100 |
| Total intl calls | Integer from 0 to 100 |
| Customer service calls | Integer from 0 to 50 |

The normal preprocessing pipeline converts Yes/No binary fields to 1/0 and derives the four charge fields from usage minutes. IDs, `Churn` labels, uploaded metadata, and derived charge columns cannot be edited. The application does not create a contract field, payment method, true tenure, monthly invoice, customer response, or business action variable.

The interface may allow an input within the validator bounds that is outside the model metadata's observed minimum/maximum. It flags such a value using a one-feature range comparison. This is not a joint-distribution or out-of-distribution detector; combinations within individual bounds may still be unfamiliar to the model.

## How the engine compares results

`simulate_scenario(predictor, original_record, changes)` validates each requested edit, copies the original canonical inputs, and sends both baseline and scenario through the existing `ChurnPredictor.predict_single()` path, including the fitted preprocessing pipeline. It disables prediction sensitivities because they are not needed for this comparison.

The reported difference is:

```text
simulated_probability - original_probability
```

The UI displays the result in percentage points. Negative means a lower model-assigned probability for the hypothetical profile; positive means higher; zero means unchanged. Model classification uses the unchanged Phase 1 threshold. The score is not an estimate of actual churn reduction, retention, revenue, or intervention success.

Each run builds its own copied profile. No caller dictionary, uploaded DataFrame, original prediction, fitted model, shared preprocessing pipeline, or other scenario is modified. The engine rejects unsupported fields, missing scenario edits, invalid numeric/binary values, fractional integer inputs, and values outside the validator bounds. If the original customer record contains missing or rejected fields, those remain subject to the model pipeline's existing imputation and are disclosed in the result/UI.

## Limitations shown to users

The visible notice says:

> These simulations show how the existing machine-learning model responds to hypothetical input changes. They are not causal estimates and do not prove that making these changes would produce the displayed outcome in real customers.

Predictions depend on the existing model and its training population. A hypothetical feature combination may not resemble an actual customer, and a probability change does not establish business effectiveness. The comparison does not model an intervention, its cost, customer response, or a causal outcome. Account length is not represented as verified tenure; a lower predicted probability is not described as churn being reduced.

## Verification

See [`PHASE_5_FINAL_VERIFICATION.md`](PHASE_5_FINAL_VERIFICATION.md) for the Phase 4 readiness gate, full test output, warning categories, simulation latency, model-preservation digests, and final recommendation. Phase 6 (optional AI Business Analyst) has not been started.
