# C.R.A.M. V2 — Phase 4 Final Handoff

**Status: PASS**  
**Branch: `v2-final-development`**  
**Scope: Rule-based retention recommendation engine**

## A. Executive summary

Phase 4 turns model risk and customer characteristics into a small set of transparent, reviewable suggestions. A business user can see rule categories across the current portfolio, download customer-level suggestions, and inspect the selected customer's local model factors alongside the suggestions. The work advances C.R.A.M. from risk reporting to decision support while keeping recommendations deterministic and human controlled.

The engine does not use an LLM, contact customers, modify accounts, estimate intervention effectiveness, or claim that an action prevents churn.

## B. Architecture and changed files

| File | Purpose |
| --- | --- |
| `ml/recommendations.py` | Defines the rule set and functions for individual and portfolio recommendation output; links accepted feature values to local SHAP context and priority. |
| `v2_app.py` | Adds filtered portfolio recommendation counts/table/export and detailed suggestions to the existing selected-customer profile. Keeps the established V2 app flow. |
| `tests/test_recommendations.py` | Covers the rule triggers, gates, SHAP mapping, missing/imputed inputs, priority and customer associations. |
| `tests/test_v2_app.py` | Existing V2 upload and profile integration smoke test remains in place and passes with the recommendation section. |
| `README.md` | Documents how to access Phase 4 and the recommendations' scope and limits. |
| `PHASE_4_FINAL_VERIFICATION.md` | Records preparation, test evidence, preservation checks and final verification. |
| `PHASE_4_FINAL_HANDOFF.md` | This implementation reference and operator handoff. |
| `PHASE_4_FINAL_SIGNOFF.md` | Records the final sign-off recommendation and boundaries. |

The rules are data logic in `ml/recommendations.py`, separate from Streamlit. `recommend_customer()` is the profile API. `recommend_portfolio()` returns normalized customer-linked records for counts and export. It accepts optional local explanations keyed by customer ID. The dashboard computes SHAP only when a profile is opened and then supplies the explanation to the individual API; it does not run expensive local explanations for every portfolio row.

## C. Implemented recommendation rules

Only the 13 canonical model inputs are treated as supported. A rule uses a field only when it is present and accepted by validation. Profile explanations supply per-feature `input_status`; portfolio evaluation compares each original input with the Phase 3 validator result to distinguish provided values from missing or rejected/coerced values.

| Rule ID | Trigger | Potential action | Objective | Key limitations |
| --- | --- | --- | --- | --- |
| `service_contact_review` | `Customer service calls >= 3` | Ask a staff member to review recent support contacts and determine whether the customer's issue was resolved. | Identify possible unresolved support friction. | Contact count does not establish the reason, quality, outcome, satisfaction, or service failure. |
| `international_usage_fit_review` | International plan is `No` and either `Total intl minutes >= 10` or `Total intl calls >= 4` | Have staff review whether available international service options fit recorded usage. | Review suitability of service options against usage. | Usage is not a bill or proof of a mismatch. Customer preferences, plan details, and price are unavailable. |

Recommendations are gated to customers whose risk is Medium or High and whose churn probability meets the existing Phase 1 prediction threshold. If no supported characteristic matches a rule, the profile shows a no-recommendation state. No unsupported contract, payment, tenure, monthly-invoice, engagement, satisfaction, or service-reliability field is inferred.

## D. SHAP context and prioritization

The rule trigger and SHAP result are separate parts of each recommendation. The observed feature value is the business-rule evidence; the feature's signed SHAP contribution is model context. The profile shows whether that contribution increased or decreased predicted churn probability and retains the Phase 2 fixed-reference semantics. Missing or coerced features cannot contribute as observed rule evidence or as reliable supporting SHAP facts.

Portfolio suggestions are created from rule characteristics and risk output; their `model_context_status` tells the user to open the customer profile for local SHAP. This avoids repeated SHAP computation across up to 100,000 uploaded rows. Portfolio export includes customer ID, rule ID, priority, risk/probability context, trigger, objective, limitations, human-review requirement, and SHAP-context status. Profile display adds available local SHAP feature directions.

Priority score is a transparent sorting heuristic:

```text
score = min(100,
            20 * rule_strength
            + 10 * risk_band_rank
            + min(10, sum(max(0, related_SHAP_contribution * 100)))
           )
```

Here `rule_strength` is 3 for repeated support contact and 2 for usage/plan-fit review; `risk_band_rank` is 2 for High Risk and 1 for Medium Risk. Only positive (risk-increasing) related SHAP contributions add points, capped at 10. Scores of 85 or more are labeled `High`; other matches are `Standard`. Risk category remains a separate field. The formula has not been validated against action outcomes and is not a probability of success or treatment-effect estimate.

## E. User workflow

1. Run the separate V2 dashboard: `venv\Scripts\python.exe -m streamlit run v2_app.py`.
2. Upload a CSV containing all 13 required model inputs. Customer ID is optional; a Churn column is optional and does not drive recommendation rules.
3. Use the existing risk filters and portfolio table. The recommendation section shows rule categories for the filtered rows and allows customer-level CSV export.
4. Open a customer profile to see the rule reason, observed triggering features, priority, local SHAP context where available, objective, human approval requirement and limitations.
5. Treat every suggestion as a review hypothesis. A person must approve any customer contact or account action.

## F. Tests and verification

Before Phase 4, the full suite was **18 passed, 4,580 warnings**. After Phase 4, the full suite was **24 passed, 4,580 warnings in 10.64 seconds**. The added tests cover each rule and its thresholds; below-risk cases; missing and invalid values; SHAP direction and field mapping; customer-to-recommendation associations; deterministic output; and required schema errors. The V2 Streamlit AppTest uploads a sample file, renders the dashboard and selected-customer profile without app exceptions. On the 2,666-row local `churn-80.csv` sample, scoring took 0.488 seconds and rule generation took 0.117 seconds, producing 394 suggestions for 290 customers. These machine-specific figures are not production capacity guarantees.

The warning categories are unchanged from baseline: 4,578 joblib warnings associated with NumPy 2.5's deprecated array shape restoration; one anyio plugin warning about pytest's renamed `CallSpec2`; one Windows loky warning because physical core detection is unavailable and logical cores are used instead. The current artifacts load and the tests pass. See the verification report for preservation checks and detail.

## G. Limitations and claims

- Rules represent supported associations or usage mismatches, not causal explanations.
- SHAP explains the model's probability output relative to its reference population; it does not establish business action effectiveness.
- No intervention trial was performed. Do not claim a suggestion will reduce churn, preserve revenue, or improve retention.
- Account length is not verified tenure. Derived usage-charge total is not a monthly bill. Neither is used as a Phase 4 rule.
- Two initial rules are intentionally narrow. Many customers, including some high-risk customers, will have no applicable recommendation when available data does not support one.
- Priority is only a review-order heuristic, not an effectiveness or financial value estimate.
- Suggestions require human approval. There is no automated outreach or account change.
- The source is a single public telecom dataset; transfer to other populations requires validation.

## H. Preservation and next phase

Phase 1 champion artifacts, model pipeline, calibration, threshold, and metadata were not modified. Phase 2 SHAP approach and fixed reference were not modified. Phase 3 risk segmentation and app features were retained. V1 `app.py` and production assets were untouched. No retraining, deploy, push, merge, or remote change occurred.

**Phase 4 completion: PASS. Phase 5 — What-If Simulation Lab — is the next roadmap phase. It has not been started.**
