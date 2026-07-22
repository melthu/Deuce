"""Is the shipped model over-parameterised?

Every candidate feature group made the shipped LightGBM worse, which is not how
a useful feature behaves - it is how a model that is already fitting noise
behaves. This sweeps capacity and regularisation on the shipped feature set,
and separately asks whether the two integer-encoded player-ID columns are
carrying signal or just giving the trees 705 arbitrary numbers to split on.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from experiments.harness import load_frame, feature_cols, evaluate, summarise
from experiments.models import tuned_params

pd.set_option("display.width", 160)

SWEEP = {
    "shipped (1000x63, lr.05)": dict(n_estimators=1000, num_leaves=63, learning_rate=0.05),
    "600x31 lr.03":             dict(n_estimators=600, num_leaves=31, learning_rate=0.03),
    "400x15 lr.03":             dict(n_estimators=400, num_leaves=15, learning_rate=0.03),
    "300x7 lr.03":              dict(n_estimators=300, num_leaves=7, learning_rate=0.03),
    "300x7 lr.03 mcs50":        dict(n_estimators=300, num_leaves=7, learning_rate=0.03,
                                     min_child_samples=50),
    "200x7 lr.05 mcs100 sub.8": dict(n_estimators=200, num_leaves=7, learning_rate=0.05,
                                     min_child_samples=100, subsample=0.8,
                                     subsample_freq=1, colsample_bytree=0.8),
    "150x3 lr.05 mcs100":       dict(n_estimators=150, num_leaves=3, learning_rate=0.05,
                                     min_child_samples=100),
    "300x7 lr.03 l2=10":        dict(n_estimators=300, num_leaves=7, learning_rate=0.03,
                                     min_child_samples=50, reg_lambda=10.0),
}


def lgbm(**p):
    import lightgbm as lgb
    return lambda: lgb.LGBMClassifier(**p, random_state=42, verbose=-1)


def main():
    df = load_frame()
    cont = feature_cols(df)

    rows = []
    for label, params in SWEEP.items():
        for ids in (True, False):
            t = evaluate(df, lgbm(**params), cont, use_player_ids=ids)
            rows.append(summarise(f"{label} | player-ids={'y' if ids else 'n'}", t))
            print(rows[-1])

    # Reference points: how much of this is just Elo?
    from sklearn.linear_model import LogisticRegression
    lr = lambda: LogisticRegression(max_iter=2000, C=0.1)
    for label, cols in [("logreg all features", cont),
                        ("logreg elo_diff only", ["elo_diff"])]:
        t = evaluate(df, lr, cols, use_player_ids=False)
        rows.append(summarise(label, t))
        print(rows[-1])

    out = pd.DataFrame(rows).set_index("experiment").sort_values("auc", ascending=False)
    print("\n=== capacity sweep (mean over eval years, best AUC first) ===")
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
