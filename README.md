# IPO Success Analyzer

IPO Success Analyzer is a Streamlit app that predicts whether an IPO is likely to have a positive listing outcome, using a tuned Random Forest pipeline trained on `_ipo_success_predictor.csv`.

## Features

- Single IPO prediction with confidence and probability breakdown.
- Batch CSV prediction with downloadable results.
- Automatic feature engineering for subscription demand, ratios, seasonality, and proceeds transformations.
- Model metadata display in-app (accuracy, CV performance, training date, and feature count).

## Project Structure

```text
IPO-Success-Analyzer/
|- app.py
|- ml_pipeline.py
|- retrain_model.py
|- style.css
|- _ipo_success_predictor.csv
|- ipo_success_model.pkl
|- feature_columns.pkl
|- model_metadata.pkl
|- scaler.pkl
|- imputer.pkl
|- README.md
```

## Tech Stack

- Python
- Streamlit
- pandas, numpy
- scikit-learn
- scipy
- plotly
- joblib

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install streamlit pandas numpy scikit-learn scipy joblib plotly
```

## Train / Retrain Model

Run training whenever dataset content or feature logic changes:

```bash
python retrain_model.py
```

Artifacts generated:

- `ipo_success_model.pkl` (full fitted pipeline)
- `feature_columns.pkl`
- `model_metadata.pkl`
- `scaler.pkl` and `imputer.pkl` (kept for compatibility in app loading)

## Run The App

```bash
streamlit run app.py
```

## Input Schema

Core input fields:

- `Offer_Price`
- `Total_Shares`
- `QIB`
- `HNI`
- `RII`
- `Proceeds_Total` (legacy alias `Issue_Size` is also supported)
- `Sector`

Date fields:

- Provide either `Year`, `Month`, `Quarter`
- Or provide `Listing_Date` (the app/pipeline derives date parts automatically)

Legacy column names are normalized in `ml_pipeline.py`, so older CSVs can still work if they map to supported aliases.

## Target Definition

Training target is resolved from available columns in this order:

1. `Listing_Gain > 0`
2. `Ipo_Success > 0` (also supports `Ipo_Sucess` and `IPOSuccess` variants)

Positive condition maps to class `1` (Success), otherwise class `0` (Failure).

## Batch Prediction CSV Notes

- Upload CSV in the **Batch Analysis** tab.
- Output includes original columns plus:
  - `Prediction`
  - `Success_Probability`
  - `Failure_Probability`
  - `Confidence`
- Results can be downloaded directly from the app.

## Disclaimer

This project is for educational and informational purposes only and does not constitute financial or investment advice.
