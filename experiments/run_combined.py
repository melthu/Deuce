"""The two confirmed levers, together.

By this point two things have survived a fair test on their own: the tuned Elo
(worth about +0.027 AUC as a standalone rating on years its fit never saw) and
the ranking-points proxy (Spearman 0.854 against the real published BWF list).
This asks whether they compound, or whether the trees see them as the same
signal twice.

Same protocol as run_search.py: every set gets its own hyperparameter search on
the same budget, selected on 2022-2024 and reported on 2025-2026.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from experiments.harness import load_frame, feature_cols
from experiments.candidate_features import load_or_build as load_cand, GROUPS
from experiments.tuned_elo import load_or_build as load_tuned_elo, COLS as ELO_T
from experiments.run_search import search

pd.set_option("display.width", 170)

SHIPPED_ELO = ["player_a_elo", "player_b_elo", "elo_diff"]
RESULTS = "experiments/results/combined.json"
N_CONFIGS = 32


def main():
    cand = load_cand()
    elo_t = load_tuned_elo()
    df = load_frame(extra=pd.concat([cand, elo_t], axis=1))

    all_cand = set().union(*GROUPS.values())
    base = feature_cols(df, exclude=all_cand | set(ELO_T))
    rank = [c for c in GROUPS["RANK"] if c in df.columns]
    no_elo = [c for c in base if c not in SHIPPED_ELO]

    sets = {
        "baseline":            base,
        "+rank":               base + rank,
        "tuned-elo":           no_elo + ELO_T,
        "tuned-elo +rank":     no_elo + ELO_T + rank,
        "both-elo +rank":      base + ELO_T + rank,
    }

    out = [search(df, cols, label, n_configs=N_CONFIGS) for label, cols in sets.items()]
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)

    tab = pd.DataFrame(out)[["label", "n_features", "select_auc", "report_auc",
                             "all_auc", "all_logloss", "all_acc"]].set_index("label")
    tab["d_report"] = tab["report_auc"] - tab.loc["baseline", "report_auc"]
    tab["d_all"] = tab["all_auc"] - tab.loc["baseline", "all_auc"]
    print("\n=== tuned Elo x ranking proxy ===")
    print(tab.round(4).to_string())


if __name__ == "__main__":
    main()
