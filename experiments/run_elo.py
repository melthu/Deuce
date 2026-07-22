"""Tune the Elo rating itself.

Elo is the single strongest input the model has - logistic regression on
`elo_diff` alone reaches 0.703 AUC against 0.733 for the full 34-feature
LightGBM. Its constants were set by hand: K by tier, the 400-point scale, a
1500 start for everyone. None of them have ever been fit to the corpus.

This searches the rating's own parameters against out-of-sample match outcomes,
scoring the raw Elo expectancy with no model on top, so the result is a
straight statement about the rating and not about the trees. Four knobs the
shipped version does not have:

  * `mov`          margin-of-victory scaling, so a 21-5 win moves the rating
                   further than 22-20 - badminton scorelines are informative and
                   the shipped rating throws them away;
  * `provisional`  a larger K for a player's first few matches, so a newcomer
                   converges from 1500 quickly instead of over a season;
  * `decay`        regression toward the mean during a layoff;
  * `tier_alpha`   the tier-to-K curve, currently a hand-written lookup.

Evaluated on 2022-2026 with a strict chronological pass: each match is scored
using the rating that existed before it, so there is nothing to leak.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

from experiments.candidate_features import parse_score

RAW_PATH = "data/raw/raw_matches.csv"
RESULTS = "experiments/results/elo.json"

# Split so the parameters are never chosen on the years used to report them.
SELECT_YEARS = (2019, 2020, 2021, 2022, 2023)
REPORT_YEARS = (2024, 2025, 2026)
EVAL_YEARS = SELECT_YEARS + REPORT_YEARS

SHIPPED = dict(k_base=28.0, tier_alpha=1.0, scale=400.0, mov=0.0,
               provisional_k=0.0, provisional_n=0, decay=0.0, start=1500.0)


def run_elo(df, k_base=28.0, tier_alpha=1.0, scale=400.0, mov=0.0,
            provisional_k=0.0, provisional_n=0, decay=0.0, start=1500.0):
    """One chronological pass. Returns (expected_a, y, year) for scored rows."""
    elo, n_played, last_seen = {}, {}, {}
    exp_out, y_out, yr_out = [], [], []

    tiers = df["tier"].to_numpy()
    dates = df["start_date"].to_numpy()
    pa_arr = df["player_a"].to_numpy()
    pb_arr = df["player_b"].to_numpy()
    won_arr = df["player_a_won"].to_numpy()
    skip_arr = ((df["is_pending"] == 1) | (df["is_walkover"] == 1)).to_numpy()
    score_arr = df["score"].fillna("").to_numpy()
    years = df["start_date"].dt.year.to_numpy()

    for i in range(len(df)):
        pa, pb, tier, date = pa_arr[i], pb_arr[i], tiers[i], dates[i]

        ra, rb = elo.get(pa, start), elo.get(pb, start)
        if decay:
            for p, r in ((pa, ra), (pb, rb)):
                prev = last_seen.get(p)
                if prev is not None:
                    idle = (date - prev) / np.timedelta64(1, "D")
                    if idle > 60:
                        pull = min(1.0, decay * (idle - 60) / 365.0)
                        r = r + pull * (start - r)
                        elo[p] = r
            ra, rb = elo.get(pa, start), elo.get(pb, start)

        exp_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / scale))

        if not skip_arr[i]:
            if years[i] in EVAL_YEARS:
                exp_out.append(exp_a)
                y_out.append(int(won_arr[i]))
                yr_out.append(years[i])

            a_won = int(won_arr[i])
            k_tier = k_base * (float(tier) / 500.0) ** tier_alpha
            mult = 1.0
            if mov:
                parsed = parse_score(score_arr[i], a_won)
                if parsed:
                    margin = abs(parsed[2] - parsed[3])
                    mult = 1.0 + mov * np.log1p(margin) / np.log1p(21.0)

            for p, r, opp_exp, res in ((pa, ra, exp_a, a_won),
                                       (pb, rb, 1.0 - exp_a, 1 - a_won)):
                k = k_tier
                if provisional_n and n_played.get(p, 0) < provisional_n:
                    k = provisional_k
                elo[p] = r + k * mult * (res - opp_exp)
                n_played[p] = n_played.get(p, 0) + 1
                last_seen[p] = date

    return np.array(exp_out), np.array(y_out), np.array(yr_out)


def score(df, years=EVAL_YEARS, **params) -> dict:
    p, y, yr = run_elo(df, **params)
    keep = np.isin(yr, years)
    p, y, yr = np.clip(p[keep], 1e-6, 1 - 1e-6), y[keep], yr[keep]
    per_year = [roc_auc_score(y[yr == u], p[yr == u]) for u in sorted(set(yr))]
    return {"auc": float(np.mean(per_year)),
            "auc_pooled": float(roc_auc_score(y, p)),
            "logloss": float(log_loss(y, p)),
            "n": int(len(y))}


def sample(rng) -> dict:
    return dict(
        k_base=float(rng.uniform(8, 60)),
        tier_alpha=float(rng.uniform(0.0, 1.2)),
        scale=float(rng.uniform(200, 700)),
        mov=float(rng.uniform(0.0, 6.0)),
        provisional_k=float(rng.uniform(20, 120)),
        provisional_n=int(rng.integers(0, 25)),
        decay=float(rng.uniform(0.0, 0.6)),
        start=1500.0,
    )


BOUNDS = {"k_base": (5, 70), "tier_alpha": (0.0, 1.2), "scale": (200, 800),
          "mov": (0.0, 6.0), "provisional_k": (10, 140), "provisional_n": (0, 30),
          "decay": (0.0, 0.8)}


def refine(df, params, rounds=4, steps=9):
    """Coordinate descent on the selection years, one parameter at a time."""
    best = dict(params)
    best_ll = score(df, years=SELECT_YEARS, **best)["logloss"]
    for _ in range(rounds):
        improved = False
        for key, (lo, hi) in BOUNDS.items():
            cur = best[key]
            span = (hi - lo) / 6.0
            for v in np.linspace(max(lo, cur - span), min(hi, cur + span), steps):
                trial = dict(best)
                trial[key] = int(round(v)) if key == "provisional_n" else float(v)
                ll = score(df, years=SELECT_YEARS, **trial)["logloss"]
                if ll < best_ll - 1e-6:
                    best, best_ll, improved = trial, ll, True
        if not improved:
            break
    return best, best_ll


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["start_date"] = pd.to_datetime(df["start_date"])
    for col, default in [("score", ""), ("is_pending", 0), ("is_walkover", 0)]:
        if col not in df.columns:
            df[col] = default
    return df.sort_values("start_date", kind="stable").reset_index(drop=True)


def main(n=400):
    df = load_raw()
    t0 = time.time()

    rng = np.random.default_rng(7)
    best_params, best_ll = SHIPPED, score(df, years=SELECT_YEARS, **SHIPPED)["logloss"]
    for _ in range(n):
        params = sample(rng)
        ll = score(df, years=SELECT_YEARS, **params)["logloss"]
        if ll < best_ll:
            best_params, best_ll = params, ll
    print(f"random search: {n} configs, best select logloss {best_ll:.4f}")

    best_params, best_ll = refine(df, best_params)
    print(f"refined:       select logloss {best_ll:.4f}  ({time.time() - t0:.0f}s)\n")

    for label, years in (("SELECT 2019-23", SELECT_YEARS), ("REPORT 2024-26", REPORT_YEARS)):
        b = score(df, years=years, **SHIPPED)
        t = score(df, years=years, **best_params)
        print(f"{label}  shipped auc={b['auc']:.4f} ll={b['logloss']:.4f}  |  "
              f"tuned auc={t['auc']:.4f} ll={t['logloss']:.4f}  |  "
              f"d_auc={t['auc'] - b['auc']:+.4f} d_ll={t['logloss'] - b['logloss']:+.4f}")

    print("\ntuned parameters:")
    for k, v in best_params.items():
        print(f"  {k:<14} {v:.4g}")

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump({
            "shipped": {"params": SHIPPED,
                        "report": score(df, years=REPORT_YEARS, **SHIPPED)},
            "tuned":   {"params": best_params,
                        "report": score(df, years=REPORT_YEARS, **best_params)},
        }, f, indent=2)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
