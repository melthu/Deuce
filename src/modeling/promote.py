"""
Production model selection + promotion.

Benchmarks every candidate model type (Optuna-tuned params where available)
on a temporal holdout — train < latest season, validate on the latest
season — then retrains the WINNER on all completed matches and writes it to
models/best_model.pkl (the dashboard's preloaded model for upcoming
tournaments).

Run weekly by .github/workflows/update-data.yml, so the production model
type is re-decided automatically as the current season's validation data
grows.

    python3 src/modeling/promote.py
"""
import sys
import os
import json
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # repo root

import numpy as np
from sklearn.metrics import roc_auc_score

from src.modeling.dataset import encode_split, fit_preprocessors, load_training_frame

DATA_PATH       = "data/processed/final_training_data.csv"
BEST_MODEL_PATH = "models/best_model.pkl"
PARAMS_PATH     = "models/best_params.json"

# Fallbacks when best_params.json lacks an entry
DEFAULT_PARAMS = {
    "xgb":      {"n_estimators": 1000, "learning_rate": 0.03, "max_depth": 6,
                 "subsample": 0.8, "colsample_bytree": 0.8},
    "lgbm":     {"n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 63},
    "catboost": {"iterations": 1000, "learning_rate": 0.05, "depth": 6},
}

MIN_VAL_ROWS = 100   # latest season must have this many rows to be a holdout


def build_model(name: str, params: dict):
    if name == "xgb":
        import xgboost as xgb
        return xgb.XGBClassifier(**params, random_state=42, eval_metric="auc",
                                 tree_method="hist", verbosity=0)
    if name == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params, random_seed=42,
                                  eval_metric="AUC", verbose=0)
    raise ValueError(f"unknown model type: {name}")


def load_all_params() -> dict:
    tuned = {}
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            tuned = json.load(f)
    return {name: tuned.get(name, DEFAULT_PARAMS[name]) for name in DEFAULT_PARAMS}


def encode_xy(df, preprocessors):
    cat, cont, y = encode_split(df, preprocessors)
    return np.hstack([cat.astype(np.float64), cont]), y


def main():
    df = load_training_frame(DATA_PATH)   # completed matches only

    # Temporal holdout: latest season with enough data validates the candidates
    years = sorted(df["start_date"].dt.year.unique())
    val_year = years[-1]
    if (df["start_date"].dt.year == val_year).sum() < MIN_VAL_ROWS and len(years) > 1:
        val_year = years[-2]
    train_df = df[df["start_date"].dt.year < val_year]
    val_df   = df[df["start_date"].dt.year >= val_year]
    print(f"Benchmark: train < {val_year} ({len(train_df):,} rows) | "
          f"val = {val_year}+ ({len(val_df):,} rows)")

    preprocessors, _ = fit_preprocessors(train_df)
    X_train, y_train = encode_xy(train_df, preprocessors)
    X_val,   y_val   = encode_xy(val_df,   preprocessors)

    all_params = load_all_params()
    aucs = {}
    for name, params in all_params.items():
        model = build_model(name, params)
        model.fit(X_train, y_train)
        aucs[name] = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        print(f"  {name:<10} holdout AUC = {aucs[name]:.4f}")

    winner = max(aucs, key=aucs.get)
    print(f"\nWinner: {winner} ({aucs[winner]:.4f}) — retraining on all data...")

    # Retrain the winner on everything, with preprocessors fit on everything
    full_prep, _ = fit_preprocessors(df)
    X_full, y_full = encode_xy(df, full_prep)
    model = build_model(winner, all_params[winner])
    model.fit(X_full, y_full)

    payload = {
        "type":            "single",
        "model":           model,
        "name":            winner,
        "trained_through": str(df["start_date"].max().date()),
        "n_train_rows":    int(len(df)),
        "benchmark_auc":   float(aucs[winner]),
    }
    os.makedirs("models", exist_ok=True)
    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"Promoted {winner} (trained on {len(df):,} rows through "
          f"{payload['trained_through']}) to {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
