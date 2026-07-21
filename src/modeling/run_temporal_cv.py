"""
Run the tuned tree models through rolling 3-fold temporal CV and report
per-fold + mean AUC. One-off reporting driver (not part of the pipeline).

    python3 src/modeling/run_temporal_cv.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # repo root

import numpy as np
from sklearn.metrics import roc_auc_score

from src.modeling.temporal_cv import get_temporal_folds
from src.modeling.promote import build_model, load_all_params


def to_xy(ds):
    X = np.hstack([ds.cat.astype(np.float64), ds.cont.astype(np.float64)])
    return X, np.asarray(ds.labels)


def main():
    folds = get_temporal_folds()
    params = load_all_params()
    model_names = ["xgb", "lgbm", "catboost"]

    # results[model] = list of (fold_label, auc)
    results = {m: [] for m in model_names}

    for train_ds, val_ds, _vocab, _prep, label in folds:
        X_tr, y_tr = to_xy(train_ds)
        X_va, y_va = to_xy(val_ds)
        for m in model_names:
            model = build_model(m, params[m])
            model.fit(X_tr, y_tr)
            auc = roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])
            results[m].append((label, auc))

    fold_labels = [lbl for _, _, _, _, lbl in folds]
    print("\nRolling 3-fold temporal CV (each year validated once):\n")
    for lbl in fold_labels:
        print(f"  {lbl}")
    print()

    header = f"{'model':<10}" + "".join(f"  fold{i+1}" for i in range(len(folds))) + "     mean"
    print(header)
    print("-" * len(header))
    for m in model_names:
        aucs = [a for _, a in results[m]]
        cells = "".join(f"  {a:.4f}" for a in aucs)
        print(f"{m:<10}{cells}   {np.mean(aucs):.4f}")


if __name__ == "__main__":
    main()
