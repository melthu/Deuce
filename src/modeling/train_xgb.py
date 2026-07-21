import sys
import os
import json
import pickle
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # repo root

import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from src.modeling.dataset import extract_numpy, get_train_val_datasets, load_training_frame, \
    fit_preprocessors, encode_split
import numpy as np

DATA_PATH       = "data/processed/final_training_data.csv"
MODEL_PATH      = "models/best_xgb.pkl"
BEST_MODEL_PATH = "models/best_model.pkl"
PARAMS_PATH     = "models/best_params.json"

# Fallback hyperparameters when models/best_params.json has no xgb entry
DEFAULT_PARAMS = {
    "n_estimators":     2000,
    "learning_rate":    0.03,
    "max_depth":        6,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
}


def load_tuned_params() -> tuple[dict, bool]:
    """Return (params, tuned) — Optuna params from best_params.json if present."""
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            all_params = json.load(f)
        if "xgb" in all_params:
            return dict(all_params["xgb"]), True
    return dict(DEFAULT_PARAMS), False


def train(full_data: bool = False, promote: bool = False):
    params, tuned = load_tuned_params()
    print(f"Hyperparameters: {'Optuna-tuned (best_params.json)' if tuned else 'defaults'}")

    if full_data:
        # Train on every completed match — used by the scheduled retrain that
        # refreshes the preloaded model for upcoming tournaments. No holdout.
        df = load_training_frame(DATA_PATH)
        preprocessors, _ = fit_preprocessors(df)
        cat, cont, y_train = encode_split(df, preprocessors)
        X_train = np.hstack([cat.astype(np.float64), cont])
        trained_through = str(df["start_date"].max().date())
        print(f"Training on ALL data: {len(df)} rows through {trained_through}")

        model = xgb.XGBClassifier(**params, random_state=42, eval_metric="auc", verbosity=0)
        model.fit(X_train, y_train, verbose=False)
        val_auc = None
    else:
        # Standard benchmark run: train ≤2025, evaluate on the 2026 holdout
        train_ds, val_ds, vocab_sizes, _ = get_train_val_datasets(DATA_PATH)
        print(f"Train size : {len(train_ds)}  |  Val size : {len(val_ds)}")
        X_train, y_train = extract_numpy(train_ds)
        X_val,   y_val   = extract_numpy(val_ds)
        trained_through = "2025-12-31"

        fit_kwargs = {}
        if not tuned:
            # Untuned defaults rely on early stopping against the val set
            params = {**params, "early_stopping_rounds": 100}
            fit_kwargs = {"eval_set": [(X_val, y_val)]}

        model = xgb.XGBClassifier(**params, random_state=42, eval_metric="auc", verbosity=0)
        model.fit(X_train, y_train, verbose=False, **fit_kwargs)

        val_probs = model.predict_proba(X_val)[:, 1]
        val_auc   = roc_auc_score(y_val, val_probs)
        val_acc   = ((val_probs >= 0.5).astype(int) == y_val).mean()

        print(f"\n{'='*42}")
        print(f"  XGBoost Results")
        print(f"{'='*42}")
        print(f"  Val Accuracy : {val_acc:.4f}")
        print(f"  Val ROC-AUC  : {val_auc:.4f}")
        print(f"{'='*42}")

    os.makedirs("models", exist_ok=True)
    if not full_data:
        # best_xgb.pkl is strictly the benchmark artifact (train ≤2025) so
        # ensemble selection on the 2026 holdout stays leak-free; the
        # full-data production model only ever lives in best_model.pkl.
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        print(f"\nModel saved to: {MODEL_PATH}")

    if promote:
        payload = {
            "type":            "single",
            "model":           model,
            "name":            "xgb",
            "trained_through": trained_through,
            "n_train_rows":    int(len(y_train)),
            "val_auc":         val_auc,
        }
        with open(BEST_MODEL_PATH, "wb") as f:
            pickle.dump(payload, f)
        print(f"Promoted to:    {BEST_MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost match predictor")
    parser.add_argument("--full-data", action="store_true",
                        help="Train on all completed matches (no holdout) — "
                             "used by the scheduled retrain")
    parser.add_argument("--promote", action="store_true",
                        help="Also write the model to models/best_model.pkl "
                             "as the app's preloaded model")
    args = parser.parse_args()
    train(full_data=args.full_data, promote=args.promote)
