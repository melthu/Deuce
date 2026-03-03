import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from src.dataset import extract_numpy, get_train_val_datasets

DATA_PATH   = "data/processed/final_training_data.csv"
MODEL_PATH  = "models/best_catboost.pkl"
LGBM_AUC    = 0.7375   # previous best


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
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.05,
        depth=6,
        eval_metric="AUC",
        early_stopping_rounds=100,
        random_seed=42,
        verbose=100,
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
    )

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
    print(f"\n  Benchmark Comparison")
    print(f"  LightGBM AUC : {LGBM_AUC:.4f}")
    print(f"  CatBoost AUC : {val_auc:.4f}  ({'▲ better' if val_auc > LGBM_AUC else '▼ worse'})")
    print(f"{'='*42}")


if __name__ == "__main__":
    train()
