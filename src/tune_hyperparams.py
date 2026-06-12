"""
Optuna hyperparameter search for XGBoost, LightGBM, and/or CatBoost.

Uses fold[-2] (penultimate year) as the objective so the latest year
remains a clean hold-out for final evaluation.

CLI flags:
  --model  {xgb|lgbm|catboost|all}  which model(s) to tune  (default: all)
  --trials N                 Optuna trials per model  (default: 50)
  --retrain                  after tuning, retrain on the full training split
                             and overwrite the per-model pickle

Best params are always saved to models/best_params.json.
"""
import sys
import os
import json
import pickle
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import roc_auc_score

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError(
        "optuna is required for hyperparameter tuning.\n"
        "Install it with:  pip install optuna --break-system-packages"
    )

from src.temporal_cv import get_temporal_folds
from src.dataset import extract_numpy, get_train_val_datasets

DATA_PATH   = "data/processed/final_training_data.csv"
PARAMS_PATH = "models/best_params.json"


# ── XGBoost objective ─────────────────────────────────────────────────────────

def tune_xgb(n_trials: int, X_train, y_train, X_val, y_val) -> tuple[dict, float]:
    import xgboost as xgb

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1500),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "random_state": 42,
            "eval_metric":  "auc",
            "early_stopping_rounds": 50,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value


# ── LightGBM objective ────────────────────────────────────────────────────────

def tune_lgbm(n_trials: int, X_train, y_train, X_val, y_val) -> tuple[dict, float]:
    import lightgbm as lgb

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 1500),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 200),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction":  trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction":  trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq":      trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1":         trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
            "lambda_l2":         trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
            "random_state": 42,
            "verbose":      -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value


# ── CatBoost objective ────────────────────────────────────────────────────────

def tune_catboost(n_trials: int, X_train, y_train, X_val, y_val) -> tuple[dict, float]:
    from catboost import CatBoostClassifier

    def objective(trial):
        params = {
            "iterations":          trial.suggest_int("iterations", 300, 2000),
            "learning_rate":       trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "depth":               trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
            "random_strength":     trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_seed":  42,
            "eval_metric":  "AUC",
            "verbose":      0,
            "early_stopping_rounds": 50,
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
        return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params, study.best_value


# ── retrain on full train split ───────────────────────────────────────────────

def retrain_best(model_type: str, best_params: dict, X_train, y_train):
    """Fit best_params on the full training split and save the model."""
    if model_type == "xgb":
        import xgboost as xgb
        model = xgb.XGBClassifier(**best_params, random_state=42, eval_metric="auc")
        model.fit(X_train, y_train, verbose=False)
        path = "models/best_xgb.pkl"
    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(**best_params, random_seed=42,
                                   eval_metric="AUC", verbose=0)
        model.fit(X_train, y_train)
        path = "models/best_catboost.pkl"
    else:
        import lightgbm as lgb
        model = lgb.LGBMClassifier(**best_params, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        path = "models/best_lgbm.pkl"

    os.makedirs("models", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Retrained {model_type.upper()} saved to {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning for XGBoost / LightGBM"
    )
    parser.add_argument("--model",   choices=["xgb", "lgbm", "catboost", "all"], default="all",
                        help="Model(s) to tune  (default: all)")
    parser.add_argument("--trials",  type=int, default=50,
                        help="Optuna trials per model  (default: 50)")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain on full 2021-2025 data after tuning and overwrite "
                             "models/best_xgb.pkl / models/best_lgbm.pkl")
    args = parser.parse_args()

    # ── Use fold[-2] (penultimate year) as the tuning objective ──────────────
    print("Loading temporal CV folds...")
    folds = get_temporal_folds(DATA_PATH)
    if len(folds) < 2:
        raise RuntimeError(f"Need ≥ 2 folds; got {len(folds)}")
    train_ds, val_ds, _, _, fold_label = folds[-2]
    print(f"  Tuning fold: {fold_label}")

    X_train, y_train = extract_numpy(train_ds)
    X_val,   y_val   = extract_numpy(val_ds)
    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}")

    models_to_tune = ["xgb", "lgbm", "catboost"] if args.model == "all" else [args.model]
    all_best_params: dict = {}

    tuners = {"xgb": tune_xgb, "lgbm": tune_lgbm, "catboost": tune_catboost}
    for mtype in models_to_tune:
        print(f"\nTuning {mtype.upper()} ({args.trials} trials)...")
        best_params, best_auc = tuners[mtype](args.trials, X_train, y_train, X_val, y_val)

        all_best_params[mtype] = best_params
        print(f"  Best AUC: {best_auc:.4f}")
        print(f"  Best params: {json.dumps(best_params, indent=4)}")

    # ── Save params (merge — tuning one model must not wipe the others) ──────
    os.makedirs("models", exist_ok=True)
    merged = {}
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            merged = json.load(f)
    merged.update(all_best_params)
    with open(PARAMS_PATH, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nBest params saved to {PARAMS_PATH}")

    # ── Optional retrain on the full training split ───────────────────────────
    if args.retrain:
        print("\nRetraining on full train split...")
        train_full_ds, _, _, _ = get_train_val_datasets(DATA_PATH)
        X_full, y_full = extract_numpy(train_full_ds)
        for mtype, best_params in all_best_params.items():
            retrain_best(mtype, best_params, X_full, y_full)

    print("\nDone.")


if __name__ == "__main__":
    main()
