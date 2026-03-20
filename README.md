# IPO Success Analyzer

Fresh Streamlit + machine learning project rebuilt on the final dataset `_ipo_success_predictor.csv`.

## What Changed

- Old cleaned dataset dependency removed.
- New full training pipeline added for the final dataset.
- Training now uses a single tuned Random Forest model for maximum practical accuracy.
- App inference updated to use the new schema and saved best pipeline.

## Project Structure

```text
IPO-Success-Analyzer/
|-- app.py
|-- ml_pipeline.py
|-- retrain_model.py
|-- style.css
|-- _ipo_success_predictor.csv
|-- ipo_success_model.pkl
|-- scaler.pkl
|-- imputer.pkl
|-- feature_columns.pkl
|-- model_metadata.pkl
|-- README.md
```

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install streamlit pandas numpy scikit-learn joblib plotly
```

## Retrain The Model

Run this whenever dataset or logic changes:

```bash
python retrain_model.py
```

This command regenerates:
- `ipo_success_model.pkl` (best model pipeline)
- `scaler.pkl`
- `imputer.pkl`
- `feature_columns.pkl`
- `model_metadata.pkl`

## Run The App

```bash
streamlit run app.py
```

## Dataset Target Definition

During training:
- `Success = 1` when `Ipo_Success > 0` (primary)
- fallback supported: `Listing_Gain > 0`

## Required Batch Input Columns

- `Offer_Price`
- `Total_Shares`
- `QIB`
- `HNI`
- `RII`
- `Issue_Size` or `Proceeds_Total`
- `Sector`
- and either:
  - `Year`, `Month`, `Quarter`
  - or `Listing_Date`

## Disclaimer

This project is for educational use only and does not provide investment advice.
