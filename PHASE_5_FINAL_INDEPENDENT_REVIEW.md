# C.R.A.M. V2 Phase 4–5 Final Independent Review

**Review date:** 2026-11-14  
**Branch / reviewed worktree:** `v2-final-development`  
**Recommendation: PASS, with documentation and environment warnings recorded below**

## Scope and method

Reviewed `ml/simulation.py`, `ml/preprocessing.py`, `ml/predict.py`, `ml/recommendations.py`, `ml/risk_intelligence.py`, `v2_app.py`, Phase 4/5 reports, tests, Git index/worktree state, and protected artifacts. Reran the complete Python test suite. No implementation files were modified during this review.

## Findings

### 1. Simulation inputs, validation, and isolation — PASS

The editor exposes exactly the 13 canonical model inputs in `INPUT_FEATURES`. Identifiers, observed `Churn`, derived charge fields, and arbitrary upload columns are rejected as changes. Binary plan values accept the validator's Yes/No forms; numeric values must parse, be finite, lie within the feature's declared `VALID_RANGES`, and be whole numbers for `INTEGER_FIELDS`. Missing and nonscalar edits are rejected. The UI widgets use the same ranges and integer-vs-continuous distinction.

`simulate_scenario` builds a new canonical baseline mapping and a separate candidate mapping for each call; it does not mutate the source mapping. Streamlit stores each scenario under its own slot and widgets are keyed by customer and scenario. Unit/AppTests cover two-way scenario independence, baseline preservation, reset/remove, customer switching, invalid values, unsupported fields, and scenario limits. All pass in the full suite.

### 2. Prediction path and arithmetic — PASS

Both baseline and scenario call the same `predictor.predict_single` instance, with `explain_sensitivities=False`; that method validates inputs and calls the same loaded fitted serving pipeline. The result is `scenario_probability - baseline_probability`, and percentage points are that difference multiplied by 100. Both classifications use the same explicitly selected threshold (defaulting to metadata `optimal_threshold`). The threshold comparison in `predict_single` is `probability >= threshold`. Risk category is a separate banding system; it is not substituted for the classification threshold.

The baseline may contain originally missing/invalid source fields and is scored using normal serving validation/imputation. Changed scenario fields cannot be missing or invalid. This preserves serving behavior and keeps accepted scenario predictions and comparison explanations on the same model.

### 3. UI language and causal boundary — PASS

The simulation UI says the result is a model output for hypothetical inputs, expressly says it is not causal and does not prove real-customer outcomes, and notes omitted intervention costs and customer responses. The per-input help describes hypothetical model inputs and says changing them does not imply a real-world action can change them. The summary reports a probability-point difference, never a proven retention effect or actual churn reduction. Account length is explicitly not verified tenure.

Phase 4 recommendations likewise require human approval and mark `causal_claim=False`; the rule limitations say observed usage does not prove service failure, dissatisfaction, plan mismatch, or intervention efficacy.

### 4. Range warning semantics — PASS with report discrepancy

The interface checks each input independently against metadata min/max and labels these as “observed single-feature training ranges.” The warning is a marginal range check; it is not a multivariate out-of-distribution detector, and combinations inside every individual range may still be unfamiliar.

**Documentation discrepancy:** in `ml/train.py`, `input_ranges` are calculated from `X` after the train/test split setup and before the full-data serving model fit; `X` contains the full 3,333-row dataset, including the held-out test partition. Therefore “training ranges” is not literally accurate. No test labels or held-out outcomes enter this calculation, but the min/max bounds are full-dataset feature ranges. Existing Phase 5 documents repeat the training-only wording. This is a documentation precision issue, not evidence that the simulation uses held-out labels. The new handoff records the accurate source. No code or prior sign-off document was changed.

### 5. Git state — PASS for preservation; work remains uncommitted

At review, branch is `v2-final-development`, HEAD `4cdd6a8ee1cfdbbe337dd4c998401e44cd76b546` (`Document Phase 2 preservation checkpoint`). No staged changes were present. The worktree had three modified tracked files (`README.md`, `ml/explain.py`, `ml/predict.py`) and untracked Phase 3–5 reports/modules/tests plus `v2_app.py`; see the master handoff for the exact inventory. `git diff -- app.py models vercel-app` and the staged equivalent were empty. No protected paths appeared in status.

### 6. V1 and model preservation — PASS

Compared all 24 tracked paths under `app.py`, `models/`, and `vercel-app/` to the preserved Phase 2 checkpoint `06f0fecc907a5da72d332c4d1761ae836fafa8c2` using Git-filtered object hashes (so Windows line-ending conversion does not create false differences). All 24 matched their checkpoint blobs. In particular, `app.py`, `models/champion_churn_model.joblib`, `models/champion_model_evaluation.joblib`, and all V1 model/app files in `vercel-app/` match. This is historical-object evidence, not merely a current-hash snapshot. The application model was not retrained.

### 7. Complete test suite — PASS

Reran from the repository virtual environment:

```text
.\venv\Scripts\python.exe -m pytest -q
41 passed, 4907 warnings in 14.49s
```

Warnings: 4,905 NumPy 2.5/joblib array shape-restoration deprecations (repeated across model loads); one anyio warning for pytest's renamed `CallSpec2`; one Windows loky warning because physical-core detection was unavailable and it fell back to logical cores. No test failed. The joblib warning is a forward compatibility concern for a future NumPy/joblib combination; it does not prevent current artifact loading or inference. The pytest plugin warning merits dependency maintenance before a future pytest 10 upgrade. The loky warning only describes this machine's core-count probe.

### 8. Discrepancies and limitations

- Earlier Phase 5 reports describe metadata min/max as training-only; implementation derives them from all 3,333 feature rows. See finding 4.
- Earlier Phase 5 verification says 4,907 warnings; the independent rerun confirms the count and updates elapsed time to 14.49s (the prior report recorded 14.47s).
- Phase 4's earlier handoff says 24 tests; the Phase 4 independent audit added a direct-API validation regression, and the current aggregate suite is 41 tests. These are different run stages, not contradictory runs.
- Scenario-domain bounds are broad validation bounds, not proof that every allowed combination is realistic. Marginal range checks cannot detect multivariate novelty.
- The underlying model uses one public telecom dataset and has no real intervention validation. All hypothetical comparisons describe model responses, not customer outcomes.
- Current Phase 3–5 implementation and reports are uncommitted in the worktree. Preserve this work and inspect it before any commit or later task.

## Final recommendation

**PASS.** The Phase 4/5 implementation has no correctness defect found in this review. Input editing is constrained to model features; scenarios are isolated; scoring and arithmetic use the existing serving pipeline and threshold; UI copy maintains the noncausal boundary; the complete suite passes; and protected artifacts match the preserved checkpoint. The full-dataset range provenance discrepancy and dependency warnings should remain visible in documentation and future maintenance.
