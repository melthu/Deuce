"""The Elo rating, with its constants fitted rather than guessed.

Elo is the single strongest input the model has: logistic regression on
`elo_diff` alone reaches 0.703 AUC where the full 34-feature LightGBM reaches
0.733. Until now its constants were hand-set - K by tier, a 400-point scale,
1500 for everyone - and had never been fit to anything.

`experiments/run_elo.py` fits them against out-of-sample match outcomes,
scoring the raw expectancy with no model on top. Selected on 2019-2023 and
reported on 2024-2026, which the fit never saw:

    shipped   AUC 0.6868   logloss 0.6342
    fitted    AUC 0.7135   logloss 0.6197

Three of the four knobs below did not exist in the shipped rating at all:

  MOV           A 21-5 win moves a rating further than a 22-20 win. The
                pipeline was already parsing scorelines for other features and
                then discarding them here.
  PROVISIONAL   A larger K for a player's first PROVISIONAL_N matches, so a
                newcomer converges off 1500 within a few events instead of
                over a season.
  DECAY         Regression toward the mean during a layoff, after a 60-day
                grace period.

The fourth is the tier curve, and its fitted value is the interesting one:
TIER_ALPHA lands near 0.06, which means the tier of an event barely changes how
much a result should move a rating. The old K_BY_TIER lookup spread K from 20
at a Super 100 to 50 at the World Tour Finals; that spread was doing almost
nothing, and most of the gain here comes from margin of victory instead.

Re-fit with `python3 experiments/run_elo.py` if the corpus grows a lot; the
parameters land in experiments/results/elo.json.
"""
import math
import re

START = 1500.0
K_BASE = 12.3
TIER_ALPHA = 0.0589      # K scales with (tier / 500) ** TIER_ALPHA
SCALE = 569.6            # rating points per decade of odds (Elo's classic 400)
MOV = 3.774              # margin-of-victory weight
PROVISIONAL_K = 28.77
PROVISIONAL_N = 23
DECAY = 0.0201           # pull toward START per year idle, after the grace
DECAY_GRACE_DAYS = 60

_MOV_NORM = math.log1p(21.0)
_GAME_RE = re.compile(r"(\d+)\s*[–\-]\s*(\d+)")


def point_margin(score_str) -> int | None:
    """Absolute total-point margin of a scoreline, or None if unparseable.

    Direction does not matter: the update below scales K by how one-sided the
    match was, and applies the same multiplier to both players.
    """
    if not isinstance(score_str, str) or not score_str.strip():
        return None
    games = _GAME_RE.findall(score_str)
    if not games:
        return None
    won = sum(int(w) for w, _ in games)
    lost = sum(int(l) for _, l in games)
    return abs(won - lost)


def expected(rating_a: float, rating_b: float) -> float:
    """P(A wins) implied by the two ratings alone."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / SCALE))


def k_for(tier, n_played: int) -> float:
    """The update size for one player in one match."""
    if n_played < PROVISIONAL_N:
        return PROVISIONAL_K
    return K_BASE * (float(tier) / 500.0) ** TIER_ALPHA


def decayed(rating: float, idle_days: float) -> float:
    """A rating pulled back toward START after a layoff."""
    if idle_days <= DECAY_GRACE_DAYS:
        return rating
    pull = min(1.0, DECAY * (idle_days - DECAY_GRACE_DAYS) / 365.0)
    return rating + pull * (START - rating)


def mov_multiplier(score_str) -> float:
    """How much a scoreline amplifies the update. 1.0 when no score is known."""
    margin = point_margin(score_str)
    if margin is None:
        return 1.0
    return 1.0 + MOV * math.log1p(margin) / _MOV_NORM


def update(rating: float, k: float, mult: float, result: float, expectation: float) -> float:
    return rating + k * mult * (result - expectation)
