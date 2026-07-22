"""
The serving path: prediction, simulation, and SHAP attribution.

This is where the retired Streamlit dashboard used to act as a second opinion.
With `app.py` gone the exporter is the only implementation, so these tests
assert the invariants that a disagreement between the two would have exposed.
"""
import numpy as np
import pandas as pd
import pytest

from src.serving.export_static import FEATURE_NAMES, DRIVER_OF, group_shap
from src.serving.simulate import (
    ROUND_ORDER, build_fixed_results, predict_match, round_sequence, run_monte_carlo,
)

SIMS = 200  # enough to check invariants; the real export uses 10,000


def _predict(f, pa, pb, rnd="first round"):
    return predict_match(
        pa, pb, rnd, f["stats"], f["h2h_rate"], f["h2h_last"],
        f["pre"]["scaler"], f["pre"]["player_to_id"], f["pre"]["tier_to_id"],
        f["pre"]["round_to_id"], f["payload"], tier=f["tier"], nat_map=f["nat_map"],
    )


def test_prediction_is_order_invariant(fitted):
    """
    P(A beats B) must equal 1 - P(B beats A) exactly. The model itself is not
    symmetric, so this holds only because predict_match averages both slot
    assignments; if that averaging is dropped, the same match gets two
    different answers depending on which way the scraper happened to store it.
    """
    for _, row in fitted["day"].head(12).iterrows():
        pa, pb, rnd = row["player_a"], row["player_b"], row["round"]
        assert _predict(fitted, pa, pb, rnd) + _predict(fitted, pb, pa, rnd) == pytest.approx(1.0)


def test_predictions_are_probabilities(fitted):
    for _, row in fitted["day"].iterrows():
        p = _predict(fitted, row["player_a"], row["player_b"], row["round"])
        assert 0.0 <= p <= 1.0


def test_same_nationality_is_actually_reaching_the_model(fitted):
    """
    Omitting nat_map silently zeroes the same_nationality feature instead of
    failing, so the only way to notice is that predictions stop moving.
    """
    f = fitted
    pairs = [(r["player_a"], r["player_b"]) for _, r in f["day"].iterrows()]
    with_nat = [_predict(f, a, b) for a, b in pairs]
    without = [predict_match(a, b, "first round", f["stats"], f["h2h_rate"], f["h2h_last"],
                             f["pre"]["scaler"], f["pre"]["player_to_id"],
                             f["pre"]["tier_to_id"], f["pre"]["round_to_id"],
                             f["payload"], tier=f["tier"], nat_map=None)
               for a, b in pairs]
    assert with_nat != without, "nat_map made no difference - same_nationality is not wired up"


@pytest.mark.parametrize("n_first_round, expected_rounds", [
    (16, 5), (32, 6), (8, 4), (4, 3), (2, 2),
    # A draw missing a match must round *up*: truncating leaves the bracket
    # one round short and it never resolves to a single winner.
    (15, 5), (31, 6),
])
def test_round_sequence_resolves_to_one_winner(n_first_round, expected_rounds):
    seq = round_sequence(n_first_round)
    assert len(seq) == expected_rounds
    assert seq[-1] == ROUND_ORDER[-1]
    assert all(r in ROUND_ORDER for r in seq)


def test_monte_carlo_is_a_distribution_over_the_draw(fitted):
    f = fitted
    counts = run_monte_carlo(
        SIMS, f["r1"], f["stats"], f["h2h_rate"], f["h2h_last"],
        f["pre"]["scaler"], f["pre"]["player_to_id"], f["pre"]["tier_to_id"],
        f["pre"]["round_to_id"], f["payload"], np.random.default_rng(42),
        tier=f["tier"], nat_map=f["nat_map"],
    )
    assert sum(counts.values()) == SIMS, "championship probabilities must sum to 1"

    entrants = set(f["r1"].to_numpy().ravel()) if hasattr(f["r1"], "to_numpy") else set()
    if entrants:
        assert set(counts) <= entrants, "a player who is not in the draw won it"

    # The exact failure of the old seeding bug: `champions = current[:, 0]` left
    # an unresolved bracket reporting the first player as a 100% champion.
    assert max(counts.values()) < SIMS, "one player won every simulation - bracket did not resolve"


def test_monte_carlo_round_counts_fill_every_slot(fitted):
    """
    `return_rounds` reports who reached each round. A round has a fixed number
    of slots and every one of them is occupied in every simulation, so the
    counts for a round must total exactly slots x sims - the check that
    separates "entrants of this round" from "winners of this round", which is
    an off-by-one that would otherwise look plausible.
    """
    f = fitted
    titles, reached = run_monte_carlo(
        SIMS, f["r1"], f["stats"], f["h2h_rate"], f["h2h_last"],
        f["pre"]["scaler"], f["pre"]["player_to_id"], f["pre"]["tier_to_id"],
        f["pre"]["round_to_id"], f["payload"], np.random.default_rng(42),
        tier=f["tier"], nat_map=f["nat_map"], return_rounds=True,
    )
    assert sum(titles.values()) == SIMS

    slots = 2 * len(f["r1"])
    for rnd in ROUND_ORDER:
        if rnd not in reached:
            continue
        assert sum(reached[rnd].values()) == slots * SIMS, (
            f"{rnd}: counted {sum(reached[rnd].values())} appearances, "
            f"expected {slots * SIMS} ({slots} slots x {SIMS} sims)")
        slots //= 2

    # Reaching the final is implied by winning it.
    final = reached.get("final", {})
    for player, n in titles.items():
        assert final.get(player, 0) >= n, f"{player} won more finals than they reached"


def test_monte_carlo_is_deterministic_under_a_fixed_seed(fitted):
    """The exporter seeds with 42; a rerun that shifts the numbers would churn
    every shard's fingerprint and republish the whole site."""
    f = fitted
    args = (SIMS, f["r1"], f["stats"], f["h2h_rate"], f["h2h_last"],
            f["pre"]["scaler"], f["pre"]["player_to_id"], f["pre"]["tier_to_id"],
            f["pre"]["round_to_id"], f["payload"])
    kw = dict(tier=f["tier"], nat_map=f["nat_map"])
    a = run_monte_carlo(*args, np.random.default_rng(42), **kw)
    b = run_monte_carlo(*args, np.random.default_rng(42), **kw)
    assert a == b


def test_fixed_results_override_the_model(fitted):
    """
    Conditioning a finished tournament on its own results must return the real
    champion at 100% - that is what makes a live draw's odds trustworthy.
    """
    f = fitted
    fixed = build_fixed_results(f["day"])
    counts = run_monte_carlo(
        SIMS, f["r1"], f["stats"], f["h2h_rate"], f["h2h_last"],
        f["pre"]["scaler"], f["pre"]["player_to_id"], f["pre"]["tier_to_id"],
        f["pre"]["round_to_id"], f["payload"], np.random.default_rng(42),
        tier=f["tier"], nat_map=f["nat_map"], fixed_results=fixed,
    )
    final = f["day"][f["day"]["round"] == ROUND_ORDER[-1]].iloc[0]
    champion = final["player_a"] if final["player_a_won"] == 1 else final["player_b"]
    assert counts.get(champion, 0) == SIMS, (
        f"expected {champion} to win every conditioned simulation, got {counts}"
    )


def test_every_feature_maps_to_exactly_one_driver():
    """
    A feature added to CONT_COLS without a driver raises KeyError mid-export,
    after work is already on disk. Catch it here instead.
    """
    missing = [f for f in FEATURE_NAMES if f not in DRIVER_OF]
    assert not missing, f"features with no SHAP driver: {missing}"
    # 35 since elo_expected joined the Rating driver; update deliberately, so
    # that a feature appearing by accident still trips this.
    assert len(FEATURE_NAMES) == 35


def test_grouped_shap_is_exact():
    """SHAP is additive, so grouping must preserve the total exactly."""
    rng = np.random.default_rng(0)
    sv = rng.normal(size=len(FEATURE_NAMES))
    grouped = group_shap(sv)
    assert sum(g["s"] for g in grouped) == pytest.approx(sv.sum(), abs=1e-3)
    assert [abs(g["s"]) for g in grouped] == sorted((abs(g["s"]) for g in grouped), reverse=True)
