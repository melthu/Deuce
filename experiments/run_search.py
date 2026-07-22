"""Feature-set comparison with per-set hyperparameter search.

The first ablation added 36 columns to a LightGBM whose `feature_fraction` was
tuned for 30, and every group looked harmful. That is not a fair test: the
config is part of what is being compared. So each feature set gets its own
random search here, with identical budget and identical search space.

Selection and reporting are split so the winner is not chosen on the numbers it
is then reported with:

    SELECT_YEARS  2022-2024   pick the config
    REPORT_YEARS  2025-2026   score it, untouched by selection

Results are written to experiments/results/search.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from experiments.harness import load_frame, feature_cols, evaluate, EVAL_YEARS
from experiments.candidate_features import load_or_build, GROUPS

pd.set_option("display.width", 170)

SELECT_YEARS = [2022, 2023, 2024]
REPORT_YEARS = [2025, 2026]
N_CONFIGS = 24
RESULTS = "experiments/results/search.json"


def sample_config(rng) -> dict:
    return {
        "n_estimators":      int(rng.integers(300, 1500)),
        "learning_rate":     float(np.exp(rng.uniform(np.log(0.01), np.log(0.08)))),
        "num_leaves":        int(rng.integers(7, 64)),
        "min_child_samples": int(rng.integers(20, 150)),
        "feature_fraction":  float(rng.uniform(0.2, 1.0)),
        "bagging_fraction":  float(rng.uniform(0.6, 1.0)),
        "bagging_freq":      int(rng.integers(1, 8)),
        "lambda_l1":         float(rng.uniform(0, 5)),
        "lambda_l2":         float(rng.uniform(0, 5)),
    }


def lgbm(cfg):
    import lightgbm as lgb
    return lambda: lgb.LGBMClassifier(**cfg, random_state=42, verbose=-1, n_jobs=4)


def search(df, cols, label, n_configs=N_CONFIGS, seed=0):
    """Random search selected on SELECT_YEARS, then scored on REPORT_YEARS."""
    rng = np.random.default_rng(seed)
    best, best_auc, best_cfg = None, -1.0, None
    t0 = time.time()
    for i in range(n_configs):
        cfg = sample_config(rng)
        t = evaluate(df, lgbm(cfg), cols, years=SELECT_YEARS)
        auc = t.loc["MEAN", "auc"]
        if auc > best_auc:
            best_auc, best_cfg, best = auc, cfg, t
    report = evaluate(df, lgbm(best_cfg), cols, years=REPORT_YEARS)
    full = evaluate(df, lgbm(best_cfg), cols, years=EVAL_YEARS)
    print(f"{label:<16} n_feat={len(cols):<3} select={best_auc:.4f} "
          f"report={report.loc['MEAN', 'auc']:.4f} "
          f"all={full.loc['MEAN', 'auc']:.4f} ({time.time() - t0:.0f}s)")
    return {
        "label": label,
        "n_features": len(cols),
        "select_auc": float(best_auc),
        "report_auc": float(report.loc["MEAN", "auc"]),
        "report_logloss": float(report.loc["MEAN", "logloss"]),
        "all_auc": float(full.loc["MEAN", "auc"]),
        "all_logloss": float(full.loc["MEAN", "logloss"]),
        "all_acc": float(full.loc["MEAN", "acc"]),
        "config": best_cfg,
        "columns": cols,
    }


def main():
    cand = load_or_build()
    df = load_frame(extra=cand)
    all_cand = set().union(*GROUPS.values())
    base = feature_cols(df, exclude=all_cand)

    sets = {"baseline": base}
    for name, cols in GROUPS.items():
        sets[f"+{name}"] = base + [c for c in cols if c in df.columns]
    sets["+ALL"] = base + sorted(c for c in all_cand if c in df.columns)

    out = [search(df, cols, label) for label, cols in sets.items()]

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)

    tab = pd.DataFrame(out)[["label", "n_features", "select_auc", "report_auc",
                             "all_auc", "all_logloss", "all_acc"]]
    tab = tab.set_index("label")
    for c in ("select_auc", "report_auc", "all_auc"):
        tab["d_" + c.split("_")[0]] = tab[c] - tab.loc["baseline", c]
    print("\n=== per-set search (each set tuned with the same budget) ===")
    print(tab.round(4).to_string())


if __name__ == "__main__":
    main()
