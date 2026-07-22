"""
Point-in-time model training, used by the static exporter.

A point-in-time model is fit only on matches that finished strictly before a
given tournament's date - vocabulary, scaler and estimator alike - so its
predictions for that tournament have never seen the tournament's own future,
nor anything that happened after it.

This is the single definition of a point-in-time fit, so a retrospective
prediction on the site is reproducible from the same code that made it.
"""
import pandas as pd

from src.modeling.dataset import encode_split, fit_preprocessors
from src.modeling.train_xgb import load_tuned_params

# Below this many completed rows a point-in-time fit is not worth trusting;
# callers fall back to the preloaded model.
MIN_PIT_ROWS = 1000


def train_point_in_time(df: pd.DataFrame, tour_date: str):
    """
    Train an XGBoost on every completed match strictly before tour_date.

    df must be the mirrored training frame with pending rows still present
    (they are excluded here). Returns (payload, preprocessors), or None when
    there is too little history.
    """
    import xgboost as xgb

    train_df = df[(df["start_date"] < pd.Timestamp(tour_date)) & (df["is_pending"] == 0)]
    if len(train_df) < MIN_PIT_ROWS:
        return None

    preprocessors, _ = fit_preprocessors(train_df)
    cat, cont, y = encode_split(train_df, preprocessors)
    X = _stack(cat, cont)

    params, _ = load_tuned_params()
    model = xgb.XGBClassifier(**params, random_state=42, eval_metric="auc",
                              tree_method="hist", verbosity=0)
    model.fit(X, y, verbose=False)

    payload = {
        "type":            "single",
        "model":           model,
        "name":            "xgb (point-in-time)",
        "trained_through": str(train_df["start_date"].max().date()),
        "n_train_rows":    int(len(train_df)),
    }
    return payload, preprocessors


def _stack(cat, cont):
    """Categorical ids and scaled continuous columns in the model's column order."""
    import numpy as np
    return np.hstack([cat.astype(np.float64), cont])
