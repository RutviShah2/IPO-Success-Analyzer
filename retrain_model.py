from datetime import datetime
import pickle
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import loguniform, randint, uniform

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

from ml_pipeline import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, build_model_features


RANDOM_STATE = 42
DATASET_PATH = "synthetic_ipo_dataset.csv"
TUNE_ITERATIONS = 20


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
    models = {
        "LogisticRegression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=900,
            max_depth=None,
            min_samples_leaf=2,
            min_samples_split=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=1200,
            max_depth=None,
            min_samples_leaf=2,
            min_samples_split=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            learning_rate=0.05,
            max_iter=400,
            min_samples_leaf=20,
            l2_regularization=0.02,
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            random_state=RANDOM_STATE,
            n_estimators=700,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            eval_metric="logloss",
            objective="binary:logistic",
            tree_method="hist",
            n_jobs=-1,
        )

    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            random_state=RANDOM_STATE,
            n_estimators=700,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="binary",
            n_jobs=-1,
            verbosity=-1,
        )

    return models


def get_param_distributions(model_name: str) -> dict:
    spaces = {
        "LogisticRegression": {
            "model__C": loguniform(1e-3, 50),
        },
        "RandomForest": {
            "model__n_estimators": randint(500, 1800),
            "model__max_depth": [None, 8, 12, 16, 24],
            "model__min_samples_split": randint(2, 20),
            "model__min_samples_leaf": randint(1, 12),
            "model__max_features": ["sqrt", "log2", None],
        },
        "ExtraTrees": {
            "model__n_estimators": randint(500, 2000),
            "model__max_depth": [None, 8, 12, 16, 24],
            "model__min_samples_split": randint(2, 20),
            "model__min_samples_leaf": randint(1, 12),
            "model__max_features": ["sqrt", "log2", None],
        },
        "HistGradientBoosting": {
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__max_iter": randint(150, 900),
            "model__max_depth": [None, 4, 6, 8, 10],
            "model__min_samples_leaf": randint(10, 120),
            "model__l2_regularization": uniform(0.0, 2.0),
        },
        "XGBoost": {
            "model__n_estimators": randint(300, 1800),
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__max_depth": randint(3, 12),
            "model__min_child_weight": randint(1, 12),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__reg_alpha": loguniform(1e-6, 10),
            "model__reg_lambda": loguniform(1e-3, 100),
        },
        "LightGBM": {
            "model__n_estimators": randint(300, 1800),
            "model__learning_rate": loguniform(0.01, 0.3),
            "model__num_leaves": randint(16, 256),
            "model__min_child_samples": randint(5, 120),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__reg_alpha": loguniform(1e-6, 10),
            "model__reg_lambda": loguniform(1e-3, 100),
        },
    }
    return spaces.get(model_name, {})


def tune_candidate_model(name: str, model, preprocessor, X_train, y_train):
    param_dist = get_param_distributions(name)
    if not param_dist:
        return model, None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
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
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=None)

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


def train() -> None:
    df = pd.read_csv(DATASET_PATH)
    if "Listing_Gain" not in df.columns:
        raise ValueError("Listing_Gain column is required to build success target.")

    y = (pd.to_numeric(df["Listing_Gain"], errors="coerce") > 0).astype(int)
    X = build_model_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    candidates = get_candidate_models()
    evaluated = []

    for name, model in candidates.items():
        preprocessor = build_preprocessor()
        tuned_model, tuned_cv = tune_candidate_model(name, model, preprocessor, X_train, y_train)

        preprocessor = build_preprocessor()
        result = evaluate_model(name, tuned_model, preprocessor, X_train, y_train, X_test, y_test)
        result["tuned_cv_score"] = tuned_cv
        result["model_params"] = tuned_model.get_params()
        evaluated.append(result)

        tuned_suffix = f", tuned_cv={tuned_cv:.4f}" if tuned_cv is not None else ""
        print(
            f"{name}: test_acc={result['test_accuracy']:.4f}, "
            f"cv_acc={result['cv_mean']:.4f} +/- {result['cv_std']:.4f}{tuned_suffix}"
        )

    tuned_candidates = {r["name"]: r["pipeline"].named_steps["model"] for r in evaluated}

    ranked = sorted(evaluated, key=lambda r: (r["test_accuracy"], r["cv_mean"]), reverse=True)
    top_three = ranked[:3]

    ensemble_estimators = [(r["name"], tuned_candidates[r["name"]]) for r in top_three]

    preprocessor = build_preprocessor()
    voting_model = VotingClassifier(estimators=ensemble_estimators, voting="soft")
    voting_result = evaluate_model("SoftVotingEnsemble", voting_model, preprocessor, X_train, y_train, X_test, y_test)
    evaluated.append(voting_result)

    preprocessor = build_preprocessor()
    stacking_model = StackingClassifier(
        estimators=ensemble_estimators,
        final_estimator=LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        cv=5,
        n_jobs=-1,
        passthrough=True,
    )
    stacking_result = evaluate_model("StackingEnsemble", stacking_model, preprocessor, X_train, y_train, X_test, y_test)
    evaluated.append(stacking_result)

    best = sorted(evaluated, key=lambda r: (r["test_accuracy"], r["cv_mean"]), reverse=True)[0]
    best_pipeline = best["pipeline"]

    joblib.dump(best_pipeline, "ipo_success_model.pkl")
    joblib.dump(None, "scaler.pkl")
    joblib.dump(None, "imputer.pkl")

    with open("feature_columns.pkl", "wb") as f:
        pickle.dump(MODEL_FEATURES, f)

    sectors = sorted(df["Sector"].dropna().astype(str).unique().tolist()) if "Sector" in df.columns else []
    metadata = {
        "model_name": best["name"],
        "accuracy": best["test_accuracy"],
        "cv_accuracy": best["cv_mean"],
        "cv_std": best["cv_std"],
        "features": MODEL_FEATURES,
        "target_definition": "Success = Listing_Gain > 0",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": DATASET_PATH,
        "dataset_rows": int(len(df)),
        "class_balance_success": float(y.mean()),
        "sectors": sectors,
        "model_comparison": [
            {
                "name": r["name"],
                "test_accuracy": round(r["test_accuracy"], 4),
                "cv_accuracy": round(r["cv_mean"], 4),
                "cv_std": round(r["cv_std"], 4),
            }
            for r in sorted(evaluated, key=lambda r: (r["test_accuracy"], r["cv_mean"]), reverse=True)
        ],
    }
    with open("model_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("\nSelected model:", best["name"])
    print("Test accuracy:", f"{best['test_accuracy']:.4f}")
    print("CV accuracy:", f"{best['cv_mean']:.4f} +/- {best['cv_std']:.4f}")
    print("\nClassification report:\n", best["report"])
    print("\nModel artifacts saved successfully.")


if __name__ == "__main__":
    train()
