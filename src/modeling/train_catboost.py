import sys
import os
import json
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # repo root

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from src.modeling.dataset import extract_numpy, get_train_val_datasets

DATA_PATH   = "data/processed/final_training_data.csv"
MODEL_PATH  = "models/best_catboost.pkl"
PARAMS_PATH = "models/best_params.json"

DEFAULT_PARAMS = {
    "iterations":    2000,
    "learning_rate": 0.05,
    "depth":         6,
}


def load_tuned_params() -> tuple[dict, bool]:
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH) as f:
            all_params = json.load(f)
        if "catboost" in all_params:
            return dict(all_params["catboost"]), True
    return dict(DEFAULT_PARAMS), False


def train():
    train_ds, val_ds, vocab_sizes, _ = get_train_val_datasets(DATA_PATH)

    print(f"Train size : {len(train_ds)}  |  Val size : {len(val_ds)}")
    print(f"Vocab sizes: {vocab_sizes}\n")

    X_train, y_train = extract_numpy(train_ds)
    X_val,   y_val   = extract_numpy(val_ds)

    print(f"X_train shape: {X_train.shape}  |  X_val shape: {X_val.shape}\n")

    # Note: the first 4 columns are integer-encoded categoricals that have been
    # hstacked with float continuous features, producing a float64 array.
    # CatBoost treats them as numerical here; its ordered boosting still handles
    # low-cardinality integer features well.
    params, tuned = load_tuned_params()
    print(f"Hyperparameters: {'Optuna-tuned (best_params.json)' if tuned else 'defaults'}")
    if not tuned:
        params["early_stopping_rounds"] = 100

    # allow_writing_files: keeps CatBoost from writing catboost_info/ into the
    # repo root; nothing reads that training log.
    model = CatBoostClassifier(**params, eval_metric="AUC", random_seed=42,
                               verbose=0, allow_writing_files=False)
    model.fit(X_train, y_train, eval_set=(X_val, y_val) if not tuned else None)

    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    val_acc = (val_preds == y_val).mean()
    val_auc = roc_auc_score(y_val, val_probs)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to: {MODEL_PATH}\n")

    print(f"\n{'='*42}")
    print(f"  CatBoost Results")
    print(f"{'='*42}")
    print(f"  Val Accuracy : {val_acc:.4f}")
    print(f"  Val ROC-AUC  : {val_auc:.4f}")
    print(f"{'='*42}")


if __name__ == "__main__":
    train()
