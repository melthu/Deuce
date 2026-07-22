"""
Candidate features, computed in one chronological pass over raw_matches.csv.

Output rows align positionally with data/interim/engineered_matches.csv (both
come from raw_matches.csv sorted stably by start_date), so the harness can
concatenate them column-wise without a join key.

Every value is the player's state STRICTLY BEFORE the match, under the same
rules the shipped pipeline uses: pending rows and walkovers are read but never
update any state.

Groups, so an experiment can add them one at a time:

  RANK      a BWF-shaped ranking-points proxy. NOT the published BWF ranking -
            it is computed from the real results already in the corpus, by
            awarding each player the points their finishing round would earn at
            that tournament's tier and rolling the best 10 events over 52
            weeks. See docstring on award_points for what that does and does
            not share with the official list.
  ELO       derivatives of the existing rating: the explicit logistic expectancy
            trees have to approximate piecewise, 90-day momentum, distance from
            career peak, and a rating built only from Super 750+ matches.
  H2H       how much head-to-head history exists and how old it is - the shipped
            h2h features report 0.5 both for "never met" and for "split 1-1".
  FATIGUE   longer workload windows, and load counted in games rather than
            matches.
  QUALITY   strength of schedule and results against higher-rated opponents.
  SCORING   game-level win rate and deciding-game record.
  CONTEXT   how deep in the draw the match is, and experience at this tier.
"""
import os
import re
import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_PATH = "data/raw/raw_matches.csv"

_GAME_RE = re.compile(r"(\d+)\s*[–\-]\s*(\d+)")

# Winner's points at each tier, from BWF's World Tour table; the multipliers
# below are the ratios that table uses for earlier finishes. Only the ordering
# and rough spacing matter to a tree - this is a ranking-strength signal, not an
# attempt to reproduce a player's exact published point total.
TIER_BASE = {100: 5500, 300: 7000, 500: 9200, 750: 11000, 1000: 12000, 1500: 12000}
ROUND_MULT = {1: 1.00, 2: 0.70, 3: 0.55, 4: 0.40, 5: 0.25, 6: 0.15, 7: 0.10}
RUNNER_UP_MULT = 0.85

RANKING_WINDOW_DAYS = 364   # BWF's rolling 52 weeks
RANKING_BEST_N = 10         # BWF counts a player's best 10 tournaments
ELO_MOMENTUM_DAYS = 90
TOP_TIER = 750              # "Elo among the good events" cutoff
RECENT_N = 10               # rolling window for form-quality stats
LONG_N = 20                 # rolling window for the rarer events (deciders)

K_BY_TIER = {100: 20, 300: 24, 500: 28, 750: 32, 1000: 40, 1500: 50}


def parse_score(score_str, player_a_won: int):
    """(a_games_won, b_games_won, a_pts, b_pts, n_games) from player A's POV."""
    if not isinstance(score_str, str) or not score_str.strip():
        return None
    games = _GAME_RE.findall(score_str)
    if not games:
        return None
    a_g = b_g = a_p = b_p = 0
    for w, l in games:
        w, l = int(w), int(l)
        # raw scores are stored winner-first, so flip when B won the match
        pa, pb = (w, l) if player_a_won else (l, w)
        a_p += pa
        b_p += pb
        a_g += int(pa > pb)
        b_g += int(pb > pa)
    return a_g, b_g, a_p, b_p, len(games)


def rounds_left_map(df: pd.DataFrame) -> pd.Series:
    """Rounds remaining including this one: final=1, SF=2, QF=3, R16=4, ...

    Derived from how many matches the round holds rather than its name, because
    "first round" means R32 at one event and R64 at another, and the corpus
    spells the same round six ways.
    """
    counts = df.groupby(["tournament", "round"]).size()
    depth = np.maximum(1, np.round(np.log2(counts.clip(lower=1))) + 1).astype(int)
    return df.set_index(["tournament", "round"]).index.map(depth).to_numpy()


def award_points(df: pd.DataFrame) -> dict:
    """tournament -> {player: points earned}, from the round each player reached.

    A player's finish is the deepest round they appear in; the final's winner
    gets the champion's award and the loser the runner-up's. Walkovers still
    place a player in the draw, so they count toward a finish.
    """
    out = {}
    for (tour, tier, date), g in df.groupby(["tournament", "tier", "start_date"], sort=False):
        best = {}
        for _, r in g.iterrows():
            for p, won in ((r["player_a"], r["player_a_won"] == 1),
                           (r["player_b"], r["player_a_won"] == 0)):
                d = int(r["_rounds_left"])
                # champion only if they won the last round of this draw
                champ = won and d == 1
                cur = best.get(p)
                if cur is None or d < cur[0] or (d == cur[0] and champ):
                    best[p] = (d, champ)
        base = TIER_BASE.get(int(tier), 7000)
        pts = {}
        for p, (d, champ) in best.items():
            if d == 1:
                mult = 1.00 if champ else RUNNER_UP_MULT
            else:
                mult = ROUND_MULT.get(d, 0.10)
            pts[p] = base * mult
        out[(tour, date)] = pts
    return out


def _rolling_points(events, now) -> float:
    """Best-N total inside the rolling window, BWF style."""
    cutoff = now - pd.Timedelta(days=RANKING_WINDOW_DAYS)
    vals = sorted((p for d, p in events if cutoff <= d < now), reverse=True)
    return float(sum(vals[:RANKING_BEST_N]))


def build(raw_path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df["start_date"] = pd.to_datetime(df["start_date"])
    for col, default in [("score", ""), ("player_a_seed", 0), ("player_b_seed", 0),
                         ("is_pending", 0), ("is_walkover", 0)]:
        if col not in df.columns:
            df[col] = default
    df = df.sort_values("start_date", kind="stable").reset_index(drop=True)
    df["_rounds_left"] = rounds_left_map(df)

    points_by_event = award_points(df)
    # Events become visible only once their tournament has started, and the
    # strict `<` in _rolling_points keeps a tournament out of its own feature.
    event_queue = sorted(
        ((date, player, pts)
         for (_tour, date), players in points_by_event.items()
         for player, pts in players.items()),
        key=lambda e: e[0],
    )
    qi = 0
    rank_events = defaultdict(list)      # player -> [(date, points)]

    elo = defaultdict(lambda: 1500.0)
    elo_top = defaultdict(lambda: 1500.0)
    elo_peak = defaultdict(lambda: 1500.0)
    elo_hist = defaultdict(list)         # player -> [(date, elo)] post-match
    career = defaultdict(int)
    match_log = defaultdict(list)        # player -> [(date, n_games)]
    opp_elo = defaultdict(lambda: deque(maxlen=RECENT_N))
    beat_stronger = defaultdict(lambda: deque(maxlen=LONG_N))   # (won, was_underdog)
    game_log = defaultdict(lambda: deque(maxlen=RECENT_N))      # (games_won, games_played)
    decider_log = defaultdict(lambda: deque(maxlen=LONG_N))     # (went_3, won_it)
    tier_log = defaultdict(lambda: defaultdict(lambda: [0, 0])) # player -> tier -> [played, won]

    # Ranks are shared across all players, so they are recomputed once per date
    # rather than once per row.
    rank_cache: dict = {}

    rows = []
    for i, r in df.iterrows():
        date = r["start_date"]
        pa, pb = r["player_a"], r["player_b"]
        tier = int(r["tier"])
        pending = int(r["is_pending"]) == 1 or int(r["is_walkover"]) == 1

        # release every ranking event that happened strictly before today
        while qi < len(event_queue) and event_queue[qi][0] < date:
            d, p, pts = event_queue[qi]
            rank_events[p].append((d, pts))
            qi += 1

        pts_a = _rolling_points(rank_events[pa], date)
        pts_b = _rolling_points(rank_events[pb], date)

        if date not in rank_cache:
            standings = sorted(
                ((_rolling_points(ev, date), p) for p, ev in rank_events.items()),
                reverse=True,
            )
            rank_cache = {date: {p: i + 1 for i, (v, p) in enumerate(standings) if v > 0}}
        ranks = rank_cache[date]
        UNRANKED = 200
        rank_a = ranks.get(pa, UNRANKED)
        rank_b = ranks.get(pb, UNRANKED)

        elo_a, elo_b = elo[pa], elo[pb]
        exp_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))

        def momentum(p):
            hist = elo_hist[p]
            if not hist:
                return 0.0
            cutoff = date - pd.Timedelta(days=ELO_MOMENTUM_DAYS)
            past = [e for d, e in hist if d <= cutoff]
            return float(elo[p] - (past[-1] if past else 1500.0))

        def load(p, days):
            cutoff = date - pd.Timedelta(days=days)
            return [(d, g) for d, g in match_log[p] if d >= cutoff and d < date]

        def mean_or(seq, default):
            return float(np.mean(seq)) if len(seq) else default

        def tier_stats(p):
            played = won = 0
            for t, (pl, w) in tier_log[p].items():
                if t >= tier:
                    played += pl
                    won += w
            return played, (won / played if played else 0.5)

        feats = {"round_ordinal": int(r["_rounds_left"])}
        for tag, p, pts, rank in (("a", pa, pts_a, rank_a), ("b", pb, pts_b, rank_b)):
            l14 = load(p, 14)
            l28 = load(p, 28)
            t_played, t_rate = tier_stats(p)
            gl = game_log[p]
            dec = [w for went, w in decider_log[p] if went]
            seed = int(r[f"player_{tag}_seed"])
            feats.update({
                f"player_{tag}_rank_points":        round(pts, 1),
                f"player_{tag}_log_rank_points":    round(float(np.log1p(pts)), 4),
                f"player_{tag}_rank":               rank,
                f"player_{tag}_elo_momentum_90d":   round(momentum(p), 2),
                f"player_{tag}_elo_vs_peak":        round(float(elo[p] - elo_peak[p]), 2),
                f"player_{tag}_elo_top_tier":       round(float(elo_top[p]), 2),
                f"player_{tag}_career_matches":     career[p],
                f"player_{tag}_matches_last_28_days": len(l28),
                f"player_{tag}_games_last_14_days": int(sum(g for _, g in l14)),
                f"player_{tag}_avg_opp_elo":        round(mean_or(opp_elo[p], 1500.0), 2),
                f"player_{tag}_upset_rate":         round(
                    mean_or([w for w, under in beat_stronger[p] if under], 0.5), 4),
                f"player_{tag}_game_win_rate":      round(
                    (sum(w for w, _ in gl) / sum(t for _, t in gl)) if gl and sum(t for _, t in gl) else 0.5, 4),
                f"player_{tag}_decider_win_rate":   round(mean_or(dec, 0.5), 4),
                f"player_{tag}_tier_experience":    t_played,
                f"player_{tag}_tier_win_rate":      round(t_rate, 4),
                f"player_{tag}_seed_rank":          seed if seed > 0 else 33,
            })
        feats["elo_expected_a"] = round(float(exp_a), 6)
        feats["rank_points_diff"] = round(pts_a - pts_b, 1)
        feats["rank_diff"] = rank_a - rank_b
        rows.append(feats)

        if pending:
            continue

        # ---------------- post-match state updates ----------------
        a_won = int(r["player_a_won"])
        parsed = parse_score(r.get("score", ""), a_won)
        n_games = parsed[4] if parsed else 2

        K = K_BY_TIER.get(tier, 24)
        elo[pa] = elo_a + K * (a_won - exp_a)
        elo[pb] = elo_b + K * ((1 - a_won) - (1 - exp_a))
        for p in (pa, pb):
            elo_peak[p] = max(elo_peak[p], elo[p])
            elo_hist[p].append((date, elo[p]))
        if tier >= TOP_TIER:
            ea, eb = elo_top[pa], elo_top[pb]
            ex = 1.0 / (1.0 + 10.0 ** ((eb - ea) / 400.0))
            elo_top[pa] = ea + K * (a_won - ex)
            elo_top[pb] = eb + K * ((1 - a_won) - (1 - ex))

        for p, opp_rating, won in ((pa, elo_b, a_won), (pb, elo_a, 1 - a_won)):
            career[p] += 1
            match_log[p].append((date, n_games))
            opp_elo[p].append(opp_rating)
            own = elo_a if p == pa else elo_b
            beat_stronger[p].append((float(won), opp_rating > own))
            tier_log[p][tier][0] += 1
            tier_log[p][tier][1] += won
            decider_log[p].append((n_games >= 3, float(won)))
        if parsed:
            a_g, b_g, _, _, ng = parsed
            game_log[pa].append((a_g, ng))
            game_log[pb].append((b_g, ng))

    out = pd.DataFrame(rows)
    return out


GROUPS = {
    "RANK":    ["player_a_rank_points", "player_b_rank_points",
                "player_a_log_rank_points", "player_b_log_rank_points",
                "player_a_rank", "player_b_rank",
                "rank_points_diff", "rank_diff"],
    "ELO":     ["elo_expected_a",
                "player_a_elo_momentum_90d", "player_b_elo_momentum_90d",
                "player_a_elo_vs_peak", "player_b_elo_vs_peak",
                "player_a_elo_top_tier", "player_b_elo_top_tier"],
    "FATIGUE": ["player_a_matches_last_28_days", "player_b_matches_last_28_days",
                "player_a_games_last_14_days", "player_b_games_last_14_days"],
    "QUALITY": ["player_a_avg_opp_elo", "player_b_avg_opp_elo",
                "player_a_upset_rate", "player_b_upset_rate"],
    "SCORING": ["player_a_game_win_rate", "player_b_game_win_rate",
                "player_a_decider_win_rate", "player_b_decider_win_rate"],
    "CONTEXT": ["round_ordinal",
                "player_a_career_matches", "player_b_career_matches",
                "player_a_tier_experience", "player_b_tier_experience",
                "player_a_tier_win_rate", "player_b_tier_win_rate",
                "player_a_seed_rank", "player_b_seed_rank"],
}

CACHE = "data/interim/candidate_features.csv"


def load_or_build(cache: str = CACHE, rebuild: bool = False) -> pd.DataFrame:
    if not rebuild and os.path.exists(cache):
        return pd.read_csv(cache)
    out = build()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    out.to_csv(cache, index=False)
    return out


if __name__ == "__main__":
    out = load_or_build(rebuild=True)
    print(f"{len(out):,} rows x {out.shape[1]} candidate columns -> {CACHE}\n")
    print(out.describe().T.round(3).to_string())
