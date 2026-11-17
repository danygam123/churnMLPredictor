# Sample dataset provenance

`sample_customers.csv` is the built-in demonstration dataset for C.R.A.M. V2 (Phase 7, roadmap
Step 9.3). It is derived from the public "Telecom Customer Churn" benchmark dataset (the same
public research dataset used to train the champion model; see `models/model_metadata.json`
`dataset` section), specifically the published 20% evaluation sample `data/raw/churn-20.csv`.

De-identification applied when creating this file:

- Geographic quasi-identifiers (`State`, `Area code`) were dropped — they are not model inputs.
- No names, phone numbers, or account numbers exist in the source file; sequential synthetic
  identifiers (`C-0001` …) were added for the `Customer ID` field.
- The original class labels (`Churn`) were retained so observed-rate features can be demonstrated.

Contents: 80 customers (24 churners, 30% churn rate) selected deterministically
(`pandas.sample(random_state=42)`) from 667 public benchmark rows, containing exactly the
documented schema: 13 required model inputs plus optional `Customer ID` and `Churn`.

The file passes the same upload validation as user-supplied datasets. It is loaded read-only into
the session in memory; it is never written elsewhere and never mixed with the protected training
artifacts under `models/` or `data/raw/`.
