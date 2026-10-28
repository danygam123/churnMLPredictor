# Churn Compass for Vercel

This is the Vercel version of Churn Compass. It is a separate Next.js web app with a Python scoring function. The existing Streamlit app remains in the parent project.

## What is included

- Guided, plain-language customer chat with follow-up questions and a live profile snapshot.
- Direct customer form with clearly marked estimated fields.
- CSV batch scoring and downloadable results.
- Held-out evaluation, missing-data sensitivity, and model-signal views.
- The already-trained Random Forest and its feature metadata; no model retraining is needed.

The chat uses a deterministic local parser for familiar phrases and numbers. It is not connected to a hosted language model.

## Run locally

Install Node.js 22+, Python 3.14, and the dependencies, then run the site and Python API in separate terminals:

```powershell
npm install
python -m pip install -r requirements.txt
npm run dev
vercel dev
```

`vercel dev` runs the Python function at `/api/predict` alongside the Next.js app. For a frontend-only preview, `npm run dev` serves the interface, but scoring requires the Python function.

## Create a separate Vercel project

Import the repository that contains this folder into Vercel and set the **Root Directory** to `vercel-app`. Vercel should detect Next.js; the Python runtime builds `api/predict.py`. The root-level `vercel.json` bundles the trained model and schema with that function. The Streamlit app stays at the repository root and can continue running on Streamlit Community Cloud.

The Vercel Python runtime must match the model’s Python version (`.python-version`) and install `requirements.txt`. The Python function returns one profile score, batch scores, and model feature importance.

## Research limits

This educational prototype was evaluated on a public telecom dataset. Scores estimate model output for those data and should not be the sole basis for real customer decisions.
