import sys
import os
import pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.dataset import get_train_val_datasets

DATA_PATH  = "data/processed/final_training_data.csv"
MODEL_PATHS = {
    "lgbm":     "models/best_lgbm.pkl",
    "catboost": "models/best_catboost.pkl",
    "xgb":      "models/best_xgb.pkl",
}
BEST_MODEL_PATH = "models/best_model.pkl"


def extract_numpy(dataset):
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    cat, cont, labels = next(iter(loader))
    X = np.hstack([cat.numpy(), cont.numpy()])
    y = labels.numpy().ravel()
    return X, y


def load_models():
    models = {}
    for name, path in MODEL_PATHS.items():
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
        print(f"  Loaded {name} from {path}")
    return models


def train():
    _, val_ds, _, _ = get_train_val_datasets(DATA_PATH)
    X_val, y_val = extract_numpy(val_ds)

    print("Loading saved models...")
    models = load_models()

    # ------------------------------------------------------------------
    # Individual AUCs
    # ------------------------------------------------------------------
    print("\nIndividual model AUCs:")
    individual_probs = {}
    individual_aucs  = {}
    for name, model in models.items():
        probs = model.predict_proba(X_val)[:, 1]
        auc   = roc_auc_score(y_val, probs)
        individual_probs[name] = probs
        individual_aucs[name]  = auc
        print(f"  {name:<12} AUC = {auc:.4f}")

    best_single_name = max(individual_aucs, key=individual_aucs.get)
    best_single_auc  = individual_aucs[best_single_name]

    # ------------------------------------------------------------------
    # Equal-weight ensemble
    # ------------------------------------------------------------------
    weights = [1 / len(models)] * len(models)
    ensemble_probs = sum(
        w * individual_probs[name]
        for w, name in zip(weights, models.keys())
    )
    ensemble_auc = roc_auc_score(y_val, ensemble_probs)

    print(f"\n  {'ensemble':<12} AUC = {ensemble_auc:.4f}")

    # ------------------------------------------------------------------
    # Select and save the winner
    # ------------------------------------------------------------------
    print(f"\n{'='*46}")
    print(f"  Ensemble Results")
    print(f"{'='*46}")
    for name, auc in individual_aucs.items():
        print(f"  {name:<12}: {auc:.4f}")
    print(f"  {'ensemble':<12}: {ensemble_auc:.4f}")
    print(f"{'='*46}")

    if ensemble_auc >= best_single_auc:
        payload = {
            "type":    "ensemble",
            "models":  models,
            "weights": weights,
            "names":   list(models.keys()),
        }
        saved_auc  = ensemble_auc
        saved_name = "ensemble"
    else:
        payload = {
            "type":  "single",
            "model": models[best_single_name],
            "name":  best_single_name,
        }
        saved_auc  = best_single_auc
        saved_name = best_single_name

    with open(BEST_MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n  Winner : {saved_name}  (AUC = {saved_auc:.4f})")
    print(f"  Saved  : {BEST_MODEL_PATH}")
    print(f"{'='*46}")


if __name__ == "__main__":
    train()
