"""
model_training.py
-----------------
Trains a Logistic Regression model on the IPO dataset and saves it as ipo_model.pkl.
Run this script ONCE before starting the Flask app.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ── 1. LOAD DATASET ──────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("dataset.csv")

# ── 2. SELECT RELEVANT COLUMNS ───────────────────────────────────────────────
df = df[df["Issue Size"].notna()].copy()
df = df[["Issue Size", "Offer Price", "QIB", "HNI", "RII", "Total", "IPO Success"]]
df.columns = ["Issue_Size", "Offer_Price", "QIB", "HNI", "RII", "Total", "IPO_Success"]

# ── 3. HANDLE MISSING VALUES ─────────────────────────────────────────────────
sub_cols = ["QIB", "HNI", "RII", "Total"]
df = df.dropna(subset=sub_cols, how="all")
df[sub_cols] = df[sub_cols].fillna(0)
df["Issue_Size"] = df["Issue_Size"].fillna(df["Issue_Size"].median())
df["Offer_Price"] = df["Offer_Price"].fillna(df["Offer_Price"].median())

# ── 4. COMPUTE TOTAL IF MISSING ──────────────────────────────────────────────
mask = df["Total"] == 0
df.loc[mask, "Total"] = df.loc[mask, "QIB"] + df.loc[mask, "HNI"] + df.loc[mask, "RII"]

# ── 5. PREPARE FEATURES & TARGET ─────────────────────────────────────────────
feature_cols = ["Issue_Size", "Offer_Price", "QIB", "HNI", "RII", "Total"]
X = df[feature_cols]
y = df["IPO_Success"].astype(int)

print(f"\nDataset shape : {df.shape}")
print(f"Class distribution:\n{y.value_counts()}")

# ── 6. TRAIN / TEST SPLIT ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# ── 7. SCALE FEATURES ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 8. TRAIN LOGISTIC REGRESSION ─────────────────────────────────────────────
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# ── 9. EVALUATE ──────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)
print("\n─── Model Evaluation ───────────────────────────────")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")
print("\nFull Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Fail", "Success"]))

print("─── Feature Coefficients ───────────────────────────")
for feat, coef in zip(feature_cols, model.coef_[0]):
    print(f"  {feat:<15} : {coef:+.4f}")

# ── 10. SAVE MODEL + SCALER ──────────────────────────────────────────────────
payload = {"model": model, "scaler": scaler, "features": feature_cols}
with open("ipo_model.pkl", "wb") as f:
    pickle.dump(payload, f)

print("\nModel saved to ipo_model.pkl")