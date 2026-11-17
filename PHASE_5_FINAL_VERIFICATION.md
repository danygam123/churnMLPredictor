# C.R.A.M. V2 — Phase 5 Final Verification

**Recommendation: PASS**  
**Branch:** `v2-final-development`  
**Phase 4 prerequisite gate:** PASS after isolated validation fix  
**Deployment, push, merge, remote changes:** None

## Implemented functionality

`ml/simulation.py` applies hypothetical changes to a fresh copy of one customer's canonical feature inputs, validates the edited values, and evaluates the original and scenario through the existing `ChurnPredictor.predict_single()` serving path. Results include both probabilities, difference in probability points, classifications, risk categories, decision threshold, changed values, imputed fields, and input features outside their observed single-feature training ranges.

The V2 interface selects the customer using the existing customer profile control. Users can add up to three scenarios (A/B/C), rename each, select inputs to vary, change values with constrained widgets, compare all scenarios with the unchanged baseline, reset a scenario, remove it, and switch customers. Scenario widgets are independently keyed and each simulation starts from the selected customer's original profile. There is no scenario database or persistence.

The app caches portfolio scoring by uploaded data, then reassigns selected risk bands from the cached probabilities without rescoring when the band sliders move. It caches only the currently selected customer's SHAP explanation, keyed by the canonical profile content. Scenario edits therefore reuse portfolio and explanation results while each hypothetical probability is recalculated through the serving predictor. SHAP is not recalculated for each slider change.

## Supported and editable inputs

The champion model and metadata declare the following 13 model inputs. The editor exposes each because it is an original serving feature, with the same allowed domains as `CustomerDataValidator` (`VALID_RANGES`) and the binary cleaner. Integer inputs require whole numbers.

| Editable input | Allowed scenario domain |
| --- | --- |
| Account length | Integer, 0–500; dataset field, not verified tenure |
| International plan | Yes / No (model values 1 / 0) |
| Voice mail plan | Yes / No (model values 1 / 0) |
| Number vmail messages | Integer, 0–150 |
| Total day minutes | 0–600 |
| Total day calls | Integer, 0–300 |
| Total eve minutes | 0–600 |
| Total eve calls | Integer, 0–300 |
| Total night minutes | 0–600 |
| Total night calls | Integer, 0–300 |
| Total intl minutes | 0–100 |
| Total intl calls | Integer, 0–100 |
| Customer service calls | Integer, 0–50 |

The validator also accepts binary forms `yes/no`, `y/n`, `true/false`, and `1/0` through `RobustBinaryCleaner`; the UI presents Yes/No. Hypothetical missing values, invalid categories, nonnumeric values, nonfinite values, fractional integer fields, and out-of-validator-range edits are rejected with field-specific messages.

Customer identifiers, observed `Churn` labels, uploaded metadata, and derived charge columns are not editable. The derived day/evening/night/international charges are recalculated inside the original preprocessing pipeline from usage inputs. Account length is labeled as not verified tenure. No monthly bill, contract type, payment method, or intervention field is introduced.

## Interpretation and validation limits

The baseline and simulated output use the existing Phase 1 optimal threshold for classification. The signed difference is `simulated_probability - original_probability`; the displayed point difference is that value multiplied by 100. Negative means the model assigns a lower probability to the hypothetical inputs, positive means higher, and zero means unchanged. This is a model-output comparison only.

The code compares scenario values with the training `input_ranges` min/max stored in model metadata and warns on individual features outside those marginal ranges. This is not a validated multivariate out-of-distribution detector; it cannot identify unusual combinations inside each feature's individual range. Predictions depend on one telecom model and its training population. No cost, customer response, actual intervention, or intervention effect is modeled.

The interface displays the required disclaimer:

> These simulations show how the existing machine-learning model responds to hypothetical input changes. They are not causal estimates and do not prove that making these changes would produce the displayed outcome in real customers.

It also explains training-data dependence, potentially unseen combinations, and omitted costs/responses. Baseline and scenario imputed features are disclosed. Original uploaded rows are only read; the simulation module copies the 13 input values before each scenario and does not mutate the source record or another scenario.

## Tests and performance

Focused simulation engine suite:

```text
14 passed (included in the combined focused run below)
```

V2 Streamlit AppTests:

```text
3 passed (included in the combined focused run below)
```

Combined simulation and V2 AppTest run: **17 passed, 656 warnings in 10.89s**.

The AppTests cover no-upload handling, upload/profile compatibility, disclaimer visibility, creation and renaming, two-way scenario independence, input editing, creation limit of three, reset, removal, side-by-side result rendering, and changing the selected customer. The simulation unit tests use the real saved serving predictor and cover empty changes, multiple changes, probability-point arithmetic, classification, invalid/out-of-domain inputs, unsupported/immutable fields, missing original values, non-mutation, independent repeated simulations, and observed training-range notices.

Full regression suite:

```text
41 passed, 4,907 warnings in 14.47s
```

On 150 independent simulations of one row from the available 2,666-row `churn-80.csv` file, total engine runtime was **11.3032 seconds**, averaging **75.355 ms per scenario** or **226.064 ms for three scenarios**, including two predictor evaluations and validation per scenario. This measures local engine calls, not browser rendering or production throughput.

## Warning review

The suite's 4,907 warnings consist of 4,905 repeated joblib array-restoration deprecation warnings under NumPy 2.5, one anyio plugin warning using pytest's renamed `CallSpec2`, and one Windows loky physical-core probe warning. Saved models load, scoring succeeds, and no simulation warning causes test failure. The joblib/NumPy warning is a forward-compatibility concern for future versions; pytest 10 plugin compatibility should be revisited; loky falls back to logical cores in this environment. None blocks this phase.

## Preservation verification

- Phase 4 independent audit and gate report: [`PHASE_4_INDEPENDENT_AUDIT.md`](PHASE_4_INDEPENDENT_AUDIT.md), PASS.
- No Phase 1 model training, selection, calibration, or threshold changes.
- No Phase 2 explainer, attribution method, or reference-population changes.
- No Phase 3 risk segmentation changes.
- Phase 4 recommendation rule logic is unchanged by Phase 5.
- `git diff --exit-code -- app.py models vercel-app` returned no changes. V1 and model artifacts were unchanged.
- Current SHA-256 values: `app.py` `86DDBEB5B714BC8D2A2BCB5BB9AF9DA2FB4EA85B35624567857597A2A20BE884`; serving model `CEE7B8CA54ED7074626E47F49B39DD56C993BDF88B3A447B6BBE4D3850E14C2A`; evaluation model `EF7CE6F797EC32E7FE49A824FE9F213ABC9CAC7BD20F616A9B7965DD9530E3CE`; model metadata `4CAD0B60E89F086825D0519737910BA5814E5DD7ACBDC90727FA00E174F1BCBC`; Phase 2 reference `5A8431194939F98BC1C914FBFF618827D2802FCBA93E13D85D25AF8EC70C72F8`.
- No model retraining, deployment, push, merge, or remote branch operation occurred. Phase 6 was not started.

## Final recommendation

**PASS.** Phase 4's gate passed after its isolated validation defect was corrected and covered. Phase 5 uses only supported fields, preserves baseline and scenario independence, uses the actual serving prediction path, exposes the causal boundary clearly, and passes the full regression suite.
