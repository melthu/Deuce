"""
Monte Carlo tournament simulation engine.

Used by the Streamlit dashboard and runnable as a CLI:

    python3 src/serving/simulate.py --date 2026-02-24 --tier 300 --sims 10000

The engine is vectorised: each bracket round batches every pending match
across all simulations into a single predict_proba call (both slot
directions stacked into one matrix, for order-invariance), which is ~100x
faster than simulating brackets one at a time.

Completed matches of a live/partially-played tournament can be passed in
via `fixed_results` so simulations are conditioned on real outcomes.
"""
import sys
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # repo root

import argparse
import pickle

import numpy as np
import pandas as pd

from src.modeling.dataset import get_train_val_datasets, load_training_frame
from src.pipeline.feature_engineering import K_BY_TIER, EMA_ALPHA

DATA_PATH  = "data/processed/final_training_data.csv"
MODEL_PATH = "models/best_model.pkl"

DEFAULT_TOUR_DATE = "2026-02-24"   # German Open 2026
DEFAULT_TIER      = 300
DEFAULT_N_SIMS    = 10_000

ROUND_ORDER = ["first round", "second round", "third round",
               "quarter-finals", "semi-finals", "final"]


def round_sequence(n_first_round_matches: int) -> list[str]:
    """Round names for a knockout bracket that opens with the given number of
    first-round matches. 16 matches → 5 rounds, 32 matches → 6 rounds, etc.

    Rounds up: a draw missing a match (a walkover the scraper never recorded,
    say) leaves e.g. 15 openers, and truncating there returns one round too
    few - the bracket then never resolves to a single winner.
    """
    n_rounds = max(1, int(np.ceil(np.log2(max(1, n_first_round_matches)))) + 1)
    if n_rounds <= 3:
        return ROUND_ORDER[-n_rounds:]
    return ROUND_ORDER[:n_rounds - 3] + ROUND_ORDER[-3:]

# Order of the per-player slice of CONT_COLS held in the `static` matrix
STAT_KEYS = ["is_home", "matches_14d", "days_since", "recent_win_rate",
             "win_streak", "matches_7d", "avg_point_diff", "avg_games_pm",
             "rubber_game_rate", "avg_margin", "seed"]


def load_model(model_path: str = MODEL_PATH):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def get_n_features(payload):
    """Return the primary model's expected feature count, or None if unknown.
    CatBoost fitted on a plain numpy array leaves n_features_in_ at 0,
    fall back to feature_names_ in that case."""
    m = (payload["model"] if payload["type"] == "single"
         else next(iter(payload["models"].values())))
    n = getattr(m, "n_features_in_", None)
    if not n:
        names = getattr(m, "feature_names_", None)
        n = len(names) if names else None
    return n


def model_predict_proba(payload, X):
    """Supports both single-model and ensemble payloads.
    Auto-trims X to the model's expected feature count for backward compat."""
    n = get_n_features(payload)
    X_in = X[:, :n] if n is not None else X
    if payload["type"] == "ensemble":
        return sum(
            w * m.predict_proba(X_in)[:, 1]
            for w, m in zip(payload["weights"], payload["models"].values())
        )
    return payload["model"].predict_proba(X_in)[:, 1]


def build_time_zero_state(df, tour_date, tier=None):
    """
    Extract the tournament's first-round matchups plus each participant's
    pre-tournament stats exactly as they appear on Day 1.

    Player stats are collected from ALL of the tournament's rows (any round,
    completed or pending) so late-entering qualifiers are covered too.
    Mirrored duplicates are dropped: one canonical row per player pair.
    """
    day = df[df["start_date"] == pd.Timestamp(tour_date)]
    r1  = day[day["round"] == "first round"].copy()

    # Drop mirrored duplicates: keep the first row seen per unordered pair
    seen, keep = set(), []
    for _, row in r1.iterrows():
        pair = frozenset((row["player_a"], row["player_b"]))
        if pair not in seen:
            seen.add(pair)
            keep.append(row)
    r1_unique = pd.DataFrame(keep).reset_index(drop=True)

    player_stats = {}
    for _, row in day.iterrows():
        for side in ("a", "b"):
            name = row[f"player_{side}"]
            if name not in player_stats:
                player_stats[name] = {
                    "is_home":          int(row[f"player_{side}_is_home"]),
                    "matches_14d":      int(row[f"player_{side}_matches_last_14_days"]),
                    "days_since":       float(row[f"player_{side}_days_since_last_match"]),
                    "recent_win_rate":  float(row[f"player_{side}_recent_win_rate"]),
                    "elo":              float(row[f"player_{side}_elo"]),
                    "ema_form":         float(row[f"player_{side}_ema_form"]),
                    "win_streak":       int(row[f"player_{side}_win_streak"]),
                    "matches_7d":       int(row[f"player_{side}_matches_last_7_days"]),
                    "avg_point_diff":   float(row.get(f"player_{side}_avg_point_diff", 0.0)),
                    "avg_games_pm":     float(row.get(f"player_{side}_avg_games_per_match", 2.0)),
                    "rubber_game_rate": float(row.get(f"player_{side}_rubber_game_rate", 0.0)),
                    "avg_margin":       float(row.get(f"player_{side}_avg_victory_margin", 0.0)),
                    "seed":             float(row.get(f"player_{side}_seed", 0.0)),
                }

    return r1_unique, player_stats


def build_fixed_results(day_rows: pd.DataFrame) -> dict:
    """
    Map (round_name, frozenset({player_a, player_b})) → actual winner for
    every completed match of a tournament. Passed to run_monte_carlo so
    simulations of a live tournament are conditioned on real results.
    """
    out = {}
    for _, r in day_rows.iterrows():
        if int(r.get("is_pending", 0)) == 1:
            continue
        winner = r["player_a"] if r["player_a_won"] == 1 else r["player_b"]
        out[(r["round"], frozenset((r["player_a"], r["player_b"])))] = winner
    return out


def build_h2h_lookups(df, tour_date):
    """
    Pre-compute two H2H signals from all completed rows strictly before the
    given tournament date.

    Returns:
        h2h_rate_fn(pa, pb)  → float win rate of pa vs pb in [0, 1]
        h2h_last_fn(pa, pb)  → 1.0 if pa won last meeting, 0.0 if pb did, 0.5 if none
    """
    hist = df[df["start_date"] < pd.Timestamp(tour_date)]
    if "is_pending" in hist.columns:
        hist = hist[hist["is_pending"] == 0]
    # Walkovers are kept in the frame for bracket topology but were never
    # counted as history during feature engineering - match that here.
    if "is_walkover" in hist.columns:
        hist = hist[hist["is_walkover"] == 0]
    hist = hist.sort_values("start_date")

    # One pass over history, indexed by UNORDERED pair. These used to be two
    # closures that scanned the whole ~20k-row frame per call and memoised on
    # the ordered pair (pa, pb) - so a caller asking both directions of a
    # matchup, which every order-invariant prediction does, missed the cache on
    # every single call. The static export asks for 26k pairs both ways: 105k
    # full-frame scans, 191 s of the 237 s run. Indexing here makes each lookup
    # a dict read.
    #
    # `key` is the pair sorted, and `wins` counts them for the FIRST name in
    # that key, so both directions read off one entry.
    index = {}
    for pa, pb, won in zip(
        hist["player_a"].to_numpy(),
        hist["player_b"].to_numpy(),
        hist["player_a_won"].to_numpy(),
    ):
        key = (pa, pb) if pa <= pb else (pb, pa)
        winner = pa if won == 1 else pb
        entry = index.get(key)
        if entry is None:
            index[key] = [1 if winner == key[0] else 0, 1, winner]
        else:
            entry[0] += winner == key[0]
            entry[1] += 1
            entry[2] = winner          # hist is date-sorted, so last row wins

    def h2h_rate(pa, pb):
        key = (pa, pb) if pa <= pb else (pb, pa)
        entry = index.get(key)
        if entry is None:
            return 0.5
        wins = entry[0] if pa == key[0] else entry[1] - entry[0]
        return float(wins / entry[1])

    def h2h_last(pa, pb):
        key = (pa, pb) if pa <= pb else (pb, pa)
        entry = index.get(key)
        if entry is None:
            return 0.5
        return 1.0 if entry[2] == pa else 0.0

    return h2h_rate, h2h_last


def _same_nationality(pa, pb, nat_map):
    if not nat_map:
        return 0.0
    na, nb = nat_map.get(pa), nat_map.get(pb)
    return 1.0 if (na is not None and na == nb) else 0.0


def _cont_matrix(SA, SB, eA, eB, mA, mB, same, rate, last):
    """Assemble the (R, 30) continuous-feature matrix in CONT_COLS order.
    SA/SB are (R, 11) static-stat slices in STAT_KEYS order."""
    return np.column_stack([
        same, rate,
        SA[:, 0], SA[:, 1], SA[:, 2], SA[:, 3],
        SB[:, 0], SB[:, 1], SB[:, 2], SB[:, 3],
        eA, eB, eA - eB, mA, mB, last,
        SA[:, 4], SB[:, 4], SA[:, 5], SB[:, 5],
        SA[:, 6], SB[:, 6], SA[:, 7], SB[:, 7],
        SA[:, 8], SB[:, 8], SA[:, 9], SB[:, 9],
        SA[:, 10], SB[:, 10],
    ])


def _predict_one_direction(
    pa, pb, round_name, player_stats,
    h2h_rate_fn, h2h_last_fn,
    scaler, player_to_id, tier_to_id, round_to_id,
    model_payload, tier=None, nat_map=None,
):
    """Raw model call with pa in the player_a slot (30-feature cont vector)."""
    t        = DEFAULT_TIER if tier is None else tier
    tier_id  = tier_to_id.get(t, 0)
    round_id = round_to_id.get(round_name, 0)
    pa_id    = player_to_id.get(pa, 0)
    pb_id    = player_to_id.get(pb, 0)

    sa = player_stats[pa]
    sb = player_stats[pb]
    SA = np.array([[sa[k] for k in STAT_KEYS]], dtype=np.float64)
    SB = np.array([[sb[k] for k in STAT_KEYS]], dtype=np.float64)

    cont_raw = _cont_matrix(
        SA, SB,
        np.array([sa["elo"]]), np.array([sb["elo"]]),
        np.array([sa["ema_form"]]), np.array([sb["ema_form"]]),
        np.array([_same_nationality(pa, pb, nat_map)]),
        np.array([h2h_rate_fn(pa, pb)]),
        np.array([h2h_last_fn(pa, pb)]),
    )

    cont_scaled = scaler.transform(cont_raw)
    cat = np.array([[tier_id, round_id, pa_id, pb_id]], dtype=np.int64)
    X   = np.hstack([cat, cont_scaled])
    return float(model_predict_proba(model_payload, X)[0])


def predict_match(
    pa, pb, round_name, player_stats,
    h2h_rate_fn, h2h_last_fn,
    scaler, player_to_id, tier_to_id, round_to_id,
    model_payload, tier=None, nat_map=None,
):
    """
    Order-invariant win probability for pa beating pb.
    Averages both slot assignments so P(A beats B) == 1 - P(B beats A) exactly.
    """
    p_ab = _predict_one_direction(
        pa, pb, round_name, player_stats, h2h_rate_fn, h2h_last_fn,
        scaler, player_to_id, tier_to_id, round_to_id, model_payload, tier, nat_map,
    )
    p_ba = _predict_one_direction(
        pb, pa, round_name, player_stats, h2h_rate_fn, h2h_last_fn,
        scaler, player_to_id, tier_to_id, round_to_id, model_payload, tier, nat_map,
    )
    return (p_ab + (1.0 - p_ba)) / 2.0


def run_monte_carlo(
    n_sims, r1_matchups, player_stats,
    h2h_rate_fn, h2h_last_fn,
    scaler, player_to_id, tier_to_id, round_to_id,
    model_payload, rng, tier=None,
    nat_map=None, fixed_results=None, progress_cb=None,
    return_rounds=False,
):
    """
    Vectorised Monte Carlo over n_sims brackets.

    Per round, every match across all simulations is batched into one
    predict_proba call (both slot directions stacked, order-invariant
    averaging). In-bracket Elo/EMA updates are applied per simulation via
    (n_sims, n_players) arrays so form carries into later rounds.

    fixed_results: {(round_name, frozenset({a, b})): winner} - real outcomes
    of already-played matches; these override the model and are applied
    deterministically in every simulation.

    progress_cb(round_name, round_idx, n_rounds): optional UI hook.

    return_rounds: also return how often each player *reached* each round.
    The simulation already knows this - the slot array at the top of a round
    is exactly its entrants - but the title count alone throws it away.

    Returns: {player_name: n_titles_won}, or that plus
    {round_name: {player_name: n_sims_reached}} when return_rounds is set.
    """
    t = DEFAULT_TIER if tier is None else tier
    K = K_BY_TIER.get(t, 24)
    fixed_results = fixed_results or {}

    players = sorted(player_stats)
    P = len(players)
    pidx = {p: i for i, p in enumerate(players)}

    static   = np.array([[player_stats[p][k] for k in STAT_KEYS] for p in players],
                        dtype=np.float64)
    vocab_id = np.array([player_to_id.get(p, 0) for p in players], dtype=np.int64)

    E = np.tile(np.array([player_stats[p]["elo"] for p in players]), (n_sims, 1))
    M = np.tile(np.array([player_stats[p]["ema_form"] for p in players]), (n_sims, 1))

    slots = []
    for _, row in r1_matchups.iterrows():
        slots += [pidx[row["player_a"]], pidx[row["player_b"]]]
    current = np.tile(np.array(slots, dtype=np.int64), (n_sims, 1))

    tier_id = tier_to_id.get(t, 0)
    rounds = round_sequence(len(r1_matchups))
    n_rounds_total = len(rounds)

    # None until a round actually reduces the bracket to one slot. Seeding this
    # with current[:, 0] would report the first player of the first match as a
    # 100%-certain champion whenever the bracket fails to resolve - a confident
    # wrong answer instead of a visible failure.
    champions = None
    reached = {}
    for round_i, round_name in enumerate(rounds):
        # Entrants of this round, counted before any carry trim. A player holds
        # at most one slot per simulation - winners of distinct matches are
        # distinct people - so a plain bincount over the slot array is the
        # number of simulations in which they got this far.
        if return_rounds:
            reached[round_name] = np.bincount(current.ravel(), minlength=P)

        # Defensive: an odd slot count means the bracket is malformed
        # (a dropped slot somewhere) - give the trailing player a bye
        # rather than crashing on mismatched pairing arrays.
        carry = None
        if current.shape[1] % 2 == 1:
            carry = current[:, -1]
            current = current[:, :-1]
        n_matches = current.shape[1] // 2
        if n_matches == 0:
            break

        A = current[:, 0::2].ravel()          # (R,) player indices in slot a
        B = current[:, 1::2].ravel()
        R = A.shape[0]
        sim_idx = np.repeat(np.arange(n_sims), n_matches)

        # Pair-level features computed once per unique (a, b) pair
        key = A.astype(np.int64) * P + B
        uniq, inv = np.unique(key, return_inverse=True)
        n_u = len(uniq)
        rate_ab = np.empty(n_u); last_ab = np.empty(n_u)
        rate_ba = np.empty(n_u); last_ba = np.empty(n_u)
        same_u  = np.empty(n_u)
        fixed_u = np.full(n_u, -1.0)          # -1 = no real result on record
        for k, kk in enumerate(uniq):
            a, b = divmod(int(kk), P)
            pa_n, pb_n = players[a], players[b]
            rate_ab[k] = h2h_rate_fn(pa_n, pb_n)
            last_ab[k] = h2h_last_fn(pa_n, pb_n)
            rate_ba[k] = h2h_rate_fn(pb_n, pa_n)
            last_ba[k] = h2h_last_fn(pb_n, pa_n)
            same_u[k]  = _same_nationality(pa_n, pb_n, nat_map)
            winner = fixed_results.get((round_name, frozenset((pa_n, pb_n))))
            if winner is not None:
                fixed_u[k] = 1.0 if winner == pa_n else 0.0

        SA, SB = static[A], static[B]
        eA, eB = E[sim_idx, A], E[sim_idx, B]
        mA, mB = M[sim_idx, A], M[sim_idx, B]

        cont1 = _cont_matrix(SA, SB, eA, eB, mA, mB, same_u[inv], rate_ab[inv], last_ab[inv])
        cont2 = _cont_matrix(SB, SA, eB, eA, mB, mA, same_u[inv], rate_ba[inv], last_ba[inv])

        round_id = round_to_id.get(round_name, 0)
        cat1 = np.column_stack([np.full(R, tier_id), np.full(R, round_id),
                                vocab_id[A], vocab_id[B]])
        cat2 = np.column_stack([np.full(R, tier_id), np.full(R, round_id),
                                vocab_id[B], vocab_id[A]])

        cont = scaler.transform(np.vstack([cont1, cont2]))
        X    = np.hstack([np.vstack([cat1, cat2]).astype(np.float64), cont])
        probs = model_predict_proba(model_payload, X)
        p = (probs[:R] + (1.0 - probs[R:])) / 2.0

        # Real results (live/finished tournaments) override the model
        fx = fixed_u[inv]
        p = np.where(fx >= 0.0, fx, p)

        a_wins  = rng.random(R) < p
        winners = np.where(a_wins, A, B)
        losers  = np.where(a_wins, B, A)

        # In-bracket Elo/EMA updates (each player plays once per round/sim,
        # so the fancy-indexed assignments never collide)
        elo_w, elo_l = E[sim_idx, winners], E[sim_idx, losers]
        exp_w = 1.0 / (1.0 + 10.0 ** ((elo_l - elo_w) / 400.0))
        E[sim_idx, winners] = elo_w + K * (1.0 - exp_w)
        E[sim_idx, losers]  = elo_l - K * (1.0 - exp_w)
        M[sim_idx, winners] = EMA_ALPHA + (1 - EMA_ALPHA) * M[sim_idx, winners]
        M[sim_idx, losers]  = (1 - EMA_ALPHA) * M[sim_idx, losers]

        current = winners.reshape(n_sims, n_matches)
        if carry is not None:
            current = np.column_stack([current, carry])
        if progress_cb:
            progress_cb(round_name, round_i + 1, n_rounds_total)
        if current.shape[1] == 1:
            champions = current[:, 0]
            break

    if champions is None:
        raise ValueError(
            f"Bracket never resolved to a single winner: {len(r1_matchups)} "
            f"first-round matchups left {current.shape[1]} slots after "
            f"{len(rounds)} rounds. The draw is incomplete."
        )

    idx, counts = np.unique(champions, return_counts=True)
    titles = {players[i]: int(c) for i, c in zip(idx, counts)}
    if not return_rounds:
        return titles
    return titles, {
        rnd: {players[i]: int(c) for i, c in enumerate(vec) if c}
        for rnd, vec in reached.items()
    }


def run(tour_date: str, tier: int, n_sims: int,
        data_path: str = DATA_PATH, model_path: str = MODEL_PATH,
        condition: bool = False):
    print("Loading data and model...")
    df = load_training_frame(data_path, drop_pending=False)

    model_payload = load_model(model_path)
    model_name    = model_payload.get("name", model_payload.get("type", "unknown"))
    print(f"Model: {model_name}")

    _, _, _, preprocessors = get_train_val_datasets(data_path)

    r1_matchups, player_stats = build_time_zero_state(df, tour_date, tier)
    if r1_matchups.empty:
        print(f"ERROR: no first-round rows found for {tour_date}. "
              f"Is the tournament in {data_path}?")
        sys.exit(1)

    h2h_rate_fn, h2h_last_fn = build_h2h_lookups(df, tour_date)
    fixed = {}
    if condition:
        day   = df[df["start_date"] == pd.Timestamp(tour_date)]
        fixed = build_fixed_results(day)
        print(f"Conditioning on {len(fixed)} real results already on record.")

    print(f"\n{'='*62}")
    print(f"  {tour_date} - First Round Bracket ({len(r1_matchups)} matchups)")
    print(f"{'='*62}")
    for _, row in r1_matchups.iterrows():
        p = predict_match(
            row["player_a"], row["player_b"], "first round",
            player_stats, h2h_rate_fn, h2h_last_fn,
            preprocessors["scaler"], preprocessors["player_to_id"],
            preprocessors["tier_to_id"], preprocessors["round_to_id"],
            model_payload, tier=tier,
        )
        print(f"  {row['player_a']:30s} vs {row['player_b']:30s}  | P(A wins)={p:.3f}")
    print(f"{'='*62}")

    print(f"\nRunning {n_sims:,} simulations...")
    rng = np.random.default_rng(42)
    win_counts = run_monte_carlo(
        n_sims, r1_matchups, player_stats,
        h2h_rate_fn, h2h_last_fn,
        preprocessors["scaler"], preprocessors["player_to_id"],
        preprocessors["tier_to_id"], preprocessors["round_to_id"],
        model_payload, rng, tier=tier, fixed_results=fixed,
        progress_cb=lambda name, i, n: print(f"  [{i}/{n}] {name} simulated"),
    )

    leaderboard = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"\n{'='*54}")
    print(f"  Championship Probability Leaderboard ({n_sims:,} sims)")
    print(f"{'='*54}")
    print(f"  {'Player':<32} {'Win %':>7}")
    print(f"  {'-'*32}  {'-'*7}")
    for name, wins in leaderboard:
        print(f"  {name:<32} {wins/n_sims*100:>6.2f}%")
    print(f"{'='*54}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo BWF tournament simulation")
    parser.add_argument("--date",  default=DEFAULT_TOUR_DATE,
                        help=f"Tournament start date YYYY-MM-DD (default: {DEFAULT_TOUR_DATE})")
    parser.add_argument("--tier",  type=int, default=DEFAULT_TIER,
                        help=f"Tournament tier (default: {DEFAULT_TIER})")
    parser.add_argument("--sims",  type=int, default=DEFAULT_N_SIMS,
                        help=f"Number of simulations (default: {DEFAULT_N_SIMS:,})")
    parser.add_argument("--data",  default=DATA_PATH)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--condition", action="store_true",
                        help="Fix already-played matches to their real outcome "
                             "(default: pure pre-tournament forecast)")
    args = parser.parse_args()
    run(args.date, args.tier, args.sims, args.data, args.model, args.condition)
