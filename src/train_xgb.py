import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from src.dataset import extract_numpy, get_train_val_datasets

DATA_PATH   = "data/processed/final_training_data.csv"
MODEL_PATH  = "models/best_xgb.pkl"
DEEPFM_AUC  = 0.7011
LGBM_AUC    = 0.7375


def train():
    # ------------------------------------------------------------------
    # Data — identical split/scaling/vocab as DeepFM and LightGBM runs
    # ------------------------------------------------------------------
    train_ds, val_ds, vocab_sizes, _ = get_train_val_datasets(DATA_PATH)

    print(f"Train size : {len(train_ds)}  |  Val size : {len(val_ds)}")
    print(f"Vocab sizes: {vocab_sizes}\n")

    X_train, y_train = extract_numpy(train_ds)
    X_val,   y_val   = extract_numpy(val_ds)

    print(f"X_train shape: {X_train.shape}  |  X_val shape: {X_val.shape}\n")

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------
    model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=100,
        eval_metric="auc",
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(f"Best iteration : {model.best_iteration}")
    print(f"Best val AUC   : {model.best_score:.4f}\n")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    val_probs = model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)

    val_acc = (val_preds == y_val).mean()
    val_auc = roc_auc_score(y_val, val_probs)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to: {MODEL_PATH}\n")

    def tag(auc):
        if auc > LGBM_AUC:
            return "▲ best so far"
        if auc > DEEPFM_AUC:
            return "▲ beats DeepFM"
        return "▼ below DeepFM"

    print(f"{'='*42}")
    print(f"  XGBoost Results")
    print(f"{'='*42}")
    print(f"  Val Accuracy : {val_acc:.4f}")
    print(f"  Val ROC-AUC  : {val_auc:.4f}")
    print(f"\n  Benchmark Comparison")
    print(f"  DeepFM   AUC : {DEEPFM_AUC:.4f}")
    print(f"  LightGBM AUC : {LGBM_AUC:.4f}")
    print(f"  XGBoost  AUC : {val_auc:.4f}  ({tag(val_auc)})")
    print(f"{'='*42}")


if __name__ == "__main__":
    train()
