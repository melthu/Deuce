"""Does the tuned rating make the tree model better, or only the rating?

A rating that predicts matches better on its own is not automatically a better
model input - the trees may already be recovering the same information from
`elo_diff` plus form. So each variant gets its own hyperparameter search, on
the same budget and the same split as run_search.py.

  baseline        the shipped 34 features
  replace         shipped Elo columns swapped for the tuned ones
  add             tuned Elo columns alongside the shipped ones
  tuned-only      drop the shipped Elo entirely, keep everything else
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from experiments.harness import load_frame, feature_cols
from experiments.tuned_elo import load_or_build as load_tuned_elo, COLS as ELO_T
from experiments.run_search import search

pd.set_option("display.width", 170)

SHIPPED_ELO = ["player_a_elo", "player_b_elo", "elo_diff"]
RESULTS = "experiments/results/elo_model.json"


def main():
    df = load_frame(extra=load_tuned_elo())
    base = feature_cols(df, exclude=ELO_T)
    no_elo = [c for c in base if c not in SHIPPED_ELO]

    sets = {
        "baseline":   base,
        "replace":    no_elo + ELO_T,
        "add":        base + ELO_T,
        "tuned-only": no_elo + ELO_T,
    }
    # "replace" and "tuned-only" are the same column list; keep one.
    sets.pop("tuned-only")

    out = [search(df, cols, label) for label, cols in sets.items()]
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)

    tab = pd.DataFrame(out)[["label", "n_features", "select_auc", "report_auc",
                             "all_auc", "all_logloss", "all_acc"]].set_index("label")
    tab["d_report"] = tab["report_auc"] - tab.loc["baseline", "report_auc"]
    tab["d_all"] = tab["all_auc"] - tab.loc["baseline", "all_auc"]
    print("\n=== tuned Elo as a model input ===")
    print(tab.round(4).to_string())


if __name__ == "__main__":
    main()
