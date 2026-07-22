"""The candidates that survived screening, plus the model-level changes.

What is left worth testing:

  * **tuned Elo** - the only change with a confirmed effect on anything. It is
    worth +0.027 AUC as a standalone rating; this asks whether the trees, which
    already see `elo_diff` and a dozen form features, can use it.
  * **+QUALITY** - strength of schedule and upset rate. The only candidate
    group that was positive on all three columns of the fair search, by a
    margin the noise floor may well swallow.
  * **a blend** - averaging LightGBM, XGBoost and CatBoost. Nothing to do with
    features; the three sit within 0.01 AUC of each other and decorrelate,
    which is the cheapest reliable gain in tabular work.
  * **calibration** - the Monte Carlo consumes probabilities, not rankings, so
    logloss and Brier matter as much as AUC. Isotonic regression fit on an
    inner temporal split, never on the evaluation year.

Same protocol as run_search.py: per-set random search, select on 2022-24,
report on 2025-26.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from experiments.harness import load_frame, feature_cols, evaluate, EVAL_YEARS
from experiments.candidate_features import load_or_build as load_cand, GROUPS
from experiments.tuned_elo import load_or_build as load_tuned_elo, COLS as ELO_T
from experiments.run_search import search, sample_config, SELECT_YEARS, REPORT_YEARS
from experiments.models import factory, tuned_params

pd.set_option("display.width", 170)

SHIPPED_ELO = ["player_a_elo", "player_b_elo", "elo_diff"]
RESULTS = "experiments/results/final.json"


class Blend:
    """Equal-weight average of several fitted classifiers' P(class 1)."""

    def __init__(self, makers):
        self.models = [m() for m in makers]

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        return np.column_stack([1 - p, p])


class Calibrated:
    """Isotonic calibration fit on the tail of the training slice.

    The inner split is chronological and comes out of train, so the evaluation
    year is untouched - calibrating on it would be leakage dressed up as a
    metric improvement.
    """

    def __init__(self, maker, holdout=0.15):
        self.maker = maker
        self.holdout = holdout

    def fit(self, X, y):
        from sklearn.isotonic import IsotonicRegression
        cut = int(len(X) * (1 - self.holdout))
        inner = self.maker().fit(X[:cut], y[:cut])
        self.iso = IsotonicRegression(out_of_bounds="clip").fit(
            inner.predict_proba(X[cut:])[:, 1], y[cut:])
        self.model = self.maker().fit(X, y)
        return self

    def predict_proba(self, X):
        p = self.iso.predict(self.model.predict_proba(X)[:, 1])
        return np.column_stack([1 - p, p])


def search_wrapped(df, cols, label, wrap):
    """run_search.search, but the winning config is wrapped before reporting."""
    rng = np.random.default_rng(0)
    best_cfg, best_auc = None, -1.0
    for _ in range(24):
        cfg = sample_config(rng)
        auc = evaluate(df, _lgbm(cfg), cols, years=SELECT_YEARS).loc["MEAN", "auc"]
        if auc > best_auc:
            best_auc, best_cfg = auc, cfg
    make = wrap(best_cfg)
    rep = evaluate(df, make, cols, years=REPORT_YEARS).loc["MEAN"]
    allt = evaluate(df, make, cols, years=EVAL_YEARS).loc["MEAN"]
    print(f"{label:<20} n_feat={len(cols):<3} select={best_auc:.4f} "
          f"report={rep['auc']:.4f} all={allt['auc']:.4f} ll={allt['logloss']:.4f}")
    return {"label": label, "n_features": len(cols), "select_auc": float(best_auc),
            "report_auc": float(rep["auc"]), "all_auc": float(allt["auc"]),
            "all_logloss": float(allt["logloss"]), "all_acc": float(allt["acc"]),
            "config": best_cfg}


def _lgbm(cfg):
    import lightgbm as lgb
    return lambda: lgb.LGBMClassifier(**cfg, random_state=42, verbose=-1, n_jobs=4)


def main():
    df = load_frame(extra=pd.concat([load_cand(), load_tuned_elo()], axis=1))
    all_cand = set().union(*GROUPS.values())
    base = feature_cols(df, exclude=all_cand | set(ELO_T))
    quality = [c for c in GROUPS["QUALITY"] if c in df.columns]
    no_elo = [c for c in base if c not in SHIPPED_ELO]

    # The fitted rating has landed in src/, so `base` here IS the tuned-elo
    # feature set - data/interim is rebuilt from the new pipeline. The
    # tuned_elo.py columns would now be near-duplicates of it and are excluded.
    # What is left to test is model-level, on top of the new baseline.
    out = [search(df, base, "baseline (fitted elo)"),
           search(df, base + quality, "+quality")]

    out.append(search_wrapped(df, base, "blend lgb+xgb+cat", lambda cfg: (
        lambda: Blend([_lgbm(cfg),
                       factory("xgb", tuned_params("xgb")),
                       factory("catboost", tuned_params("catboost"))]))))
    out.append(search_wrapped(df, base, "isotonic-calibrated",
                              lambda cfg: (lambda: Calibrated(_lgbm(cfg)))))

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)

    tab = pd.DataFrame(out)[["label", "n_features", "select_auc", "report_auc",
                             "all_auc", "all_logloss", "all_acc"]].set_index("label")
    for c in ("report_auc", "all_auc", "all_logloss"):
        tab["d_" + c] = tab[c] - tab.loc["baseline", c]
    print("\n=== surviving candidates ===")
    print(tab.round(4).to_string())


if __name__ == "__main__":
    main()
