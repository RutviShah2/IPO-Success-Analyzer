from datetime import datetime
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_pipeline import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, build_model_features


RANDOM_STATE = 42
DATASET_PATH = "_ipo_success_predictor.csv"
TUNE_ITERATIONS = 12


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.005)),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def get_candidate_models() -> dict:
    # Single-model setup as requested: Random Forest only.
    return {
        "RandomForest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
    }


def get_param_distributions(model_name: str) -> dict:
    spaces = {
        "RandomForest": {
            "model__n_estimators": randint(250, 900),
            "model__max_depth": [None, 8, 12, 16, 24],
            "model__min_samples_split": randint(2, 12),
            "model__min_samples_leaf": randint(1, 6),
            "model__max_features": ["sqrt", "log2", None],
            "model__bootstrap": [True, False],
        },
    }
    return spaces.get(model_name, {})


def tune_candidate_model(name: str, model, preprocessor, X_train, y_train):
    param_dist = get_param_distributions(name)
    if not param_dist:
        return model, None

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=TUNE_ITERATIONS,
        scoring="accuracy",
        n_jobs=-1,
        cv=cv,
        random_state=RANDOM_STATE,
        refit=True,
        verbose=0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_train, y_train)

    tuned_model = search.best_estimator_.named_steps["model"]
    return tuned_model, float(search.best_score_)


def evaluate_model(name: str, model, preprocessor, X_train, y_train, X_test, y_test) -> dict:
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    return {
        "name": name,
        "pipeline": pipeline,
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
        "test_accuracy": float(test_accuracy),
        "report": classification_report(y_test, y_pred, output_dict=False, zero_division=0),
    }


def _resolve_target(df: pd.DataFrame) -> pd.Series:
    cols_by_lower = {col.lower(): col for col in df.columns}

    if "listing_gain" in cols_by_lower:
        col = cols_by_lower["listing_gain"]
        return (pd.to_numeric(df[col], errors="coerce") > 0).astype(int)

    for key in ("ipo_success", "ipo_sucess", "iposuccess"):
        if key in cols_by_lower:
            col = cols_by_lower[key]
            return (pd.to_numeric(df[col], errors="coerce") > 0).astype(int)

    raise ValueError(
        "Target column not found. Expected one of: Listing_Gain, Ipo_Success, IPO_Success."
    )


def train() -> None:
    df = pd.read_csv(DATASET_PATH)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()

    y = _resolve_target(df)
    X = build_model_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    name, model = list(get_candidate_models().items())[0]

    preprocessor = build_preprocessor()
    tuned_model, tuned_cv = tune_candidate_model(name, model, preprocessor, X_train, y_train)

    preprocessor = build_preprocessor()
    result = evaluate_model(name, tuned_model, preprocessor, X_train, y_train, X_test, y_test)
    result["tuned_cv_score"] = tuned_cv
    result["model_params"] = tuned_model.get_params()

    print(
        f"{name}: test_acc={result['test_accuracy']:.4f}, "
        f"cv_acc={result['cv_mean']:.4f} +/- {result['cv_std']:.4f}, tuned_cv={tuned_cv:.4f}"
    )

    best_pipeline = result["pipeline"]

    joblib.dump(best_pipeline, "ipo_success_model.pkl")
    joblib.dump(None, "scaler.pkl")
    joblib.dump(None, "imputer.pkl")

    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(MODEL_FEATURES, f)

    sectors = sorted(df["Sector"].dropna().astype(str).unique().tolist()) if "Sector" in df.columns else ["Unknown"]
    metadata = {
        "model_name": result["name"],
        "accuracy": result["test_accuracy"],
        "cv_accuracy": result["cv_mean"],
        "cv_std": result["cv_std"],
        "features": MODEL_FEATURES,
        "target_definition": "Success = Ipo_Success > 0 or Listing_Gain > 0",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": DATASET_PATH,
        "dataset_rows": int(len(df)),
        "class_balance_success": float(y.mean()),
        "sectors": sectors,
        "model_comparison": [
            {
                "name": result["name"],
                "test_accuracy": round(result["test_accuracy"], 4),
                "cv_accuracy": round(result["cv_mean"], 4),
                "cv_std": round(result["cv_std"], 4),
            }
        ],
    }
    with open("model_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("\nSelected model:", result["name"])
    print("Test accuracy:", f"{result['test_accuracy']:.4f}")
    print("CV accuracy:", f"{result['cv_mean']:.4f} +/- {result['cv_std']:.4f}")
    print("\nClassification report:\n", result["report"])
    print("\nModel artifacts saved successfully.")


if __name__ == "__main__":
    train()
