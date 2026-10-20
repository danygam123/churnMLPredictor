# Churn ML Predictor

Streamlit research app for exploring telecom customer churn.

## App features

- Guided chat asks one plain-language question at a time and accepts a rough
  number, yes/no, or “not sure.” It is a scripted intake flow; the prediction
  comes from the trained Random Forest, and no external language-model API is
  connected.
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
