"""How big does a difference have to be before it means anything?

The feature search produced spreads of ±0.005 AUC and no group that was
consistently positive. That number is meaningless without knowing how much the
harness moves on its own, so this holds the feature set and the search budget
fixed and varies only the seeds - the random-search seed and the model seed.
Whatever spread that produces is the floor: any feature-set difference smaller
than it is not evidence of anything.

Cheap by design: one feature set, one budget, N seeds.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from experiments.harness import load_frame, feature_cols, evaluate, EVAL_YEARS
from experiments.run_search import sample_config, SELECT_YEARS, REPORT_YEARS

RESULTS = "experiments/results/noise.json"
N_SEEDS = 8
N_CONFIGS = 12


def lgbm(cfg, seed):
    import lightgbm as lgb
    return lambda: lgb.LGBMClassifier(**cfg, random_state=seed, verbose=-1, n_jobs=4)


def main():
    df = load_frame()
    cols = feature_cols(df)

    rows = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        best_cfg, best_auc = None, -1.0
        for _ in range(N_CONFIGS):
            cfg = sample_config(rng)
            auc = evaluate(df, lgbm(cfg, seed), cols, years=SELECT_YEARS).loc["MEAN", "auc"]
            if auc > best_auc:
                best_auc, best_cfg = auc, cfg
        rep = evaluate(df, lgbm(best_cfg, seed), cols, years=REPORT_YEARS).loc["MEAN", "auc"]
        allt = evaluate(df, lgbm(best_cfg, seed), cols, years=EVAL_YEARS).loc["MEAN"]
        rows.append({"seed": seed, "select_auc": float(best_auc),
                     "report_auc": float(rep), "all_auc": float(allt["auc"]),
                     "all_logloss": float(allt["logloss"])})
        print(rows[-1])

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(rows, f, indent=2)

    print("\n=== noise floor: identical features, seeds varied ===")
    print(out.describe().loc[["mean", "std", "min", "max"]].round(4).to_string())
    print(f"\nspread (max - min) of all_auc  : {out['all_auc'].max() - out['all_auc'].min():.4f}")
    print(f"spread (max - min) of report_auc: {out['report_auc'].max() - out['report_auc'].min():.4f}")
    print("\nA feature-set difference below that spread is not evidence.")


if __name__ == "__main__":
    main()
