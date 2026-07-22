"""Feature-group ablation: does each candidate group earn its place?

Screened with LightGBM (the strongest baseline) on the rolling harness. Each
group is added to the shipped feature set on its own, then all together, so a
group that only helps in the presence of another still shows up in ALL.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from experiments.harness import load_frame, feature_cols, evaluate, summarise
from experiments.candidate_features import load_or_build, GROUPS
from experiments.models import factory, tuned_params

pd.set_option("display.width", 160)


def main(model_name: str = "lgbm"):
    cand = load_or_build()
    df = load_frame(extra=cand)
    make = factory(model_name, tuned_params(model_name))

    base = feature_cols(df, exclude=set().union(*GROUPS.values()))
    print(f"{len(df):,} matches | baseline {len(base)} continuous features | model={model_name}\n")

    rows = []
    table = evaluate(df, make, base)
    rows.append(summarise("baseline", table))
    print("baseline", table.loc["MEAN"].round(4).to_dict())

    for name, cols in GROUPS.items():
        cols = [c for c in cols if c in df.columns]
        t = evaluate(df, make, base + cols)
        rows.append(summarise(f"+{name} ({len(cols)})", t))
        print(f"+{name}", t.loc["MEAN"].round(4).to_dict())

    all_cols = base + [c for c in set().union(*GROUPS.values()) if c in df.columns]
    t = evaluate(df, make, sorted(all_cols))
    rows.append(summarise(f"+ALL ({len(all_cols) - len(base)})", t))
    print("+ALL", t.loc["MEAN"].round(4).to_dict())

    out = pd.DataFrame(rows).set_index("experiment")
    out["d_auc"] = out["auc"] - out.loc["baseline", "auc"]
    out["d_logloss"] = out["logloss"] - out.loc["baseline", "logloss"]
    print("\n=== feature-group ablation (mean over eval years) ===")
    print(out.round(4).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lgbm")
