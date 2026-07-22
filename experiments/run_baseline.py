"""Baseline: the shipped 34-feature set, scored on the rolling harness.

Every later experiment is compared against the numbers this prints.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from experiments.harness import load_frame, feature_cols, evaluate, summarise
from experiments.models import factory, tuned_params

pd.set_option("display.width", 140)


def main():
    df = load_frame()
    cont = feature_cols(df)
    print(f"{len(df):,} completed matches | {len(cont)} continuous features "
          f"+ 4 categorical\n")

    rows = []
    for name in ["lgbm", "xgb", "catboost"]:
        table = evaluate(df, factory(name, tuned_params(name)), cont)
        print(f"--- {name} (tuned params) ---")
        print(table.round(4).to_string(), "\n")
        rows.append(summarise(name, table))

    print("=== baseline summary (mean over eval years) ===")
    print(pd.DataFrame(rows).set_index("experiment").round(4).to_string())


if __name__ == "__main__":
    main()
