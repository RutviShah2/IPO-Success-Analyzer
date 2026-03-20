import numpy as np
import pandas as pd


RAW_FEATURE_COLUMNS = [
    "Offer_Price",
    "Total_Shares",
    "QIB",
    "HNI",
    "RII",
    "Proceeds_Total",
    "Sector",
    "Year",
    "Month",
    "Quarter",
]


NUMERIC_FEATURES = [
    "Offer_Price",
    "Total_Shares",
    "QIB",
    "HNI",
    "RII",
    "Proceeds_Total",
    "Year",
    "Month",
    "Quarter",
    "Total_Subscription",
    "QIB_to_Total_Ratio",
    "HNI_to_Total_Ratio",
    "RII_to_Total_Ratio",
    "Strong_Subscription",
    "Very_Strong_Subscription",
    "Balanced_Subscription",
    "Demand_Excess",
    "Demand_Excess_Ratio",
    "Investor_Concentration",
    "Retail_vs_Inst_Ratio",
    "Log_Proceeds_Total",
    "Month_Sin",
    "Month_Cos",
]


CATEGORICAL_FEATURES = ["Sector"]


MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


LEGACY_COLUMN_ALIASES = {
    "Offer Price": "Offer_Price",
    "QIB(Qualified Institutional Buyers)": "QIB",
    "HNI(High Net-Worth Individuals)": "HNI",
    "RII(Retail Individual Investors)": "RII",
    "Issue_Size": "Proceeds_Total",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize old and new column names to a single schema."""
    normalized = df.copy()
    rename_map = {c: LEGACY_COLUMN_ALIASES[c] for c in normalized.columns if c in LEGACY_COLUMN_ALIASES}
    if rename_map:
        normalized = normalized.rename(columns=rename_map)

    if "Listing_Date" in normalized.columns and ("Year" not in normalized.columns or "Month" not in normalized.columns):
        parsed_date = pd.to_datetime(normalized["Listing_Date"], errors="coerce")
        normalized["Year"] = normalized.get("Year", parsed_date.dt.year)
        normalized["Month"] = normalized.get("Month", parsed_date.dt.month)
        normalized["Quarter"] = normalized.get("Quarter", parsed_date.dt.quarter)

    if "Quarter" not in normalized.columns and "Month" in normalized.columns:
        normalized["Quarter"] = ((pd.to_numeric(normalized["Month"], errors="coerce") - 1) // 3) + 1

    if "Year" not in normalized.columns or normalized["Year"].isna().all():
        normalized["Year"] = 2026

    if "Month" not in normalized.columns or normalized["Month"].isna().all():
        normalized["Month"] = 1

    if "Quarter" not in normalized.columns or normalized["Quarter"].isna().all():
        normalized["Quarter"] = 1

    return normalized


def build_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create model-ready features from raw input columns."""
    prepared = normalize_columns(df)

    for col in RAW_FEATURE_COLUMNS:
        if col not in prepared.columns:
            prepared[col] = np.nan

    numeric_raw = [c for c in RAW_FEATURE_COLUMNS if c != "Sector"]
    for col in numeric_raw:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce")

    total_demand = prepared["QIB"] + prepared["HNI"] + prepared["RII"]
    safe_total_shares = prepared["Total_Shares"].replace(0, np.nan)
    safe_total_demand = total_demand.replace(0, np.nan)

    prepared["Total_Subscription"] = total_demand / safe_total_shares
    prepared["QIB_to_Total_Ratio"] = prepared["QIB"] / safe_total_demand
    prepared["HNI_to_Total_Ratio"] = prepared["HNI"] / safe_total_demand
    prepared["RII_to_Total_Ratio"] = prepared["RII"] / safe_total_demand
    prepared["Demand_Excess"] = total_demand - prepared["Total_Shares"]
    prepared["Demand_Excess_Ratio"] = prepared["Demand_Excess"] / safe_total_shares

    ratio_cols = ["QIB_to_Total_Ratio", "HNI_to_Total_Ratio", "RII_to_Total_Ratio"]
    prepared["Investor_Concentration"] = prepared[ratio_cols].max(axis=1)

    institutional_total = (prepared["QIB"] + prepared["HNI"]).replace(0, np.nan)
    prepared["Retail_vs_Inst_Ratio"] = prepared["RII"] / institutional_total

    prepared["Log_Proceeds_Total"] = np.log1p(prepared["Proceeds_Total"].clip(lower=0))

    month_numeric = pd.to_numeric(prepared["Month"], errors="coerce")
    month_angle = 2 * np.pi * (month_numeric / 12.0)
    prepared["Month_Sin"] = np.sin(month_angle)
    prepared["Month_Cos"] = np.cos(month_angle)

    prepared["Strong_Subscription"] = (prepared["Total_Subscription"] > 1.5).astype(int)
    prepared["Very_Strong_Subscription"] = (prepared["Total_Subscription"] > 2.5).astype(int)
    prepared["Balanced_Subscription"] = (
        (prepared["QIB_to_Total_Ratio"] > 0.20)
        & (prepared["HNI_to_Total_Ratio"] > 0.10)
        & (prepared["RII_to_Total_Ratio"] > 0.20)
    ).astype(int)

    prepared["Sector"] = prepared["Sector"].astype("string").fillna("Unknown")

    feature_df = prepared[MODEL_FEATURES].copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    return feature_df
