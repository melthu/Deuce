"""The tuned rating, as feature columns aligned to raw_matches.csv.

`run_elo.py` fits Elo's own constants against out-of-sample outcomes and lands
on a rating that is worth about +0.027 AUC over the shipped one on years the
fit never saw. This re-runs that rating with the winning parameters and emits
it as columns, so the tree model can be asked whether a better input makes it
a better model.

Emitted per row, all strictly pre-match:

    player_{a,b}_elo_t      the tuned rating
    elo_diff_t              A minus B                (pair-level, negates)
    elo_expected_t          the rating's own P(A wins) (pair-level, 1 - x)

Params live in experiments/results/elo.json, written by run_elo.py.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from experiments.candidate_features import parse_score
from experiments.run_elo import load_raw, RESULTS as ELO_RESULTS

CACHE = "data/interim/tuned_elo.csv"
COLS = ["player_a_elo_t", "player_b_elo_t", "elo_diff_t", "elo_expected_t"]


def tuned_params(path: str = ELO_RESULTS) -> dict:
    with open(path) as f:
        return json.load(f)["tuned"]["params"]


def build(df: pd.DataFrame, k_base, tier_alpha, scale, mov,
          provisional_k, provisional_n, decay, start=1500.0) -> pd.DataFrame:
    elo, n_played, last_seen = {}, {}, {}
    rows = []

    tiers = df["tier"].to_numpy()
    dates = df["start_date"].to_numpy()
    pa_arr = df["player_a"].to_numpy()
    pb_arr = df["player_b"].to_numpy()
    won_arr = df["player_a_won"].to_numpy()
    skip_arr = ((df["is_pending"] == 1) | (df["is_walkover"] == 1)).to_numpy()
    score_arr = df["score"].fillna("").to_numpy()

    for i in range(len(df)):
        pa, pb, tier, date = pa_arr[i], pb_arr[i], tiers[i], dates[i]

        if decay:
            for p in (pa, pb):
                prev = last_seen.get(p)
                if prev is not None:
                    idle = (date - prev) / np.timedelta64(1, "D")
                    if idle > 60:
                        pull = min(1.0, decay * (idle - 60) / 365.0)
                        elo[p] = elo[p] + pull * (start - elo[p])
        ra, rb = elo.get(pa, start), elo.get(pb, start)
        exp_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / scale))

        rows.append({"player_a_elo_t": round(ra, 2), "player_b_elo_t": round(rb, 2),
                     "elo_diff_t": round(ra - rb, 2), "elo_expected_t": round(exp_a, 6)})

        if skip_arr[i]:
            continue

        a_won = int(won_arr[i])
        k_tier = k_base * (float(tier) / 500.0) ** tier_alpha
        mult = 1.0
        if mov:
            parsed = parse_score(score_arr[i], a_won)
            if parsed:
                mult = 1.0 + mov * np.log1p(abs(parsed[2] - parsed[3])) / np.log1p(21.0)

        for p, r, opp_exp, res in ((pa, ra, exp_a, a_won),
                                   (pb, rb, 1.0 - exp_a, 1 - a_won)):
            k = provisional_k if (provisional_n and n_played.get(p, 0) < provisional_n) else k_tier
            elo[p] = r + k * mult * (res - opp_exp)
            n_played[p] = n_played.get(p, 0) + 1
            last_seen[p] = date

    return pd.DataFrame(rows)


def load_or_build(cache: str = CACHE, rebuild: bool = False) -> pd.DataFrame:
    if not rebuild and os.path.exists(cache):
        return pd.read_csv(cache)
    out = build(load_raw(), **tuned_params())
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    out.to_csv(cache, index=False)
    return out


if __name__ == "__main__":
    out = load_or_build(rebuild=True)
    print(f"{len(out):,} rows -> {CACHE}")
    print(out.describe().T.round(3).to_string())
