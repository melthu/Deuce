"""
Point-in-time honesty, and sanity checks on a built payload.

The payload tests skip when `site/data` has not been built, so the suite stays
runnable without a 20-minute export.
"""
import glob
import json
import os

import pandas as pd
import pytest

from src.serving.export_static import (
    EXPORT_VERSION, fingerprint, is_placeholder, slugify,
)

OUT = "site/data"


def test_point_in_time_model_never_sees_its_own_tournament(fitted, df):
    """
    The whole claim behind a retrospective prediction. Vocabulary, scaler and
    estimator must all be fit on matches that finished strictly before the
    tournament started — not on or after its start date.
    """
    cutoff = pd.Timestamp(fitted["date_key"])
    assert pd.Timestamp(fitted["payload"]["trained_through"]) < cutoff

    eligible = df[(df["start_date"] < cutoff) & (df["is_pending"] == 0)]
    assert fitted["payload"]["n_train_rows"] == len(eligible), (
        "point-in-time training set does not match the pre-cutoff completed rows"
    )
    # The players it knows must also come only from before the cutoff.
    known = set(fitted["pre"]["player_to_id"])
    future = df[df["start_date"] >= cutoff]
    only_future = (set(future["player_a"]) | set(future["player_b"])) - (
        set(eligible["player_a"]) | set(eligible["player_b"]))
    assert not (known & only_future), (
        "vocabulary contains players who had not yet played: "
        f"{sorted(known & only_future)[:5]}"
    )


def test_fingerprint_changes_with_the_export_version(df):
    """
    Forgetting to bump EXPORT_VERSION after changing the payload means every
    fingerprint still matches and reruns silently ship stale files.
    """
    from src.serving import export_static

    day = df.head(40)
    hist = df.head(400)
    before = fingerprint(day, hist)
    export_static.EXPORT_VERSION = EXPORT_VERSION + 1
    try:
        assert fingerprint(day, hist) != before
    finally:
        export_static.EXPORT_VERSION = EXPORT_VERSION


def test_fingerprint_changes_when_a_result_lands(df):
    """What makes a live tournament re-export on its own as rounds complete."""
    day = df.head(40).copy()
    hist = df.head(400)
    before = fingerprint(day, hist)
    day.loc[day.index[0], "player_a_won"] = 1 - day.iloc[0]["player_a_won"]
    assert fingerprint(day, hist) != before


# ----------------------------------------------------------------------
# built payload
# ----------------------------------------------------------------------
def _shards(kind):
    paths = sorted(glob.glob(os.path.join(OUT, kind, "*.json")))
    if not paths:
        pytest.skip(f"{OUT}/{kind} not built — run `make export`")
    return paths


def test_tournament_payloads_are_well_formed():
    for path in _shards("tournament"):
        doc = json.loads(open(path).read())
        assert doc["slug"] == os.path.basename(path)[:-5]
        assert doc["status"] in {"upcoming", "live", "complete"}
        assert doc["matches"], f"{doc['slug']} has an empty bracket"
        for m in doc["matches"]:
            assert 0.0 <= m["p"] <= 1.0, f"{doc['slug']}: p out of range"
            assert m["pending"] or m["a_won"] is not None
            assert abs(sum(g["s"] for g in m["shap"])) < 50, "implausible SHAP magnitude"


def test_leaderboards_are_distributions_over_real_players():
    placeholder_seen = []
    for path in _shards("tournament"):
        doc = json.loads(open(path).read())
        entrants = {p for m in doc["matches"] for p in (m["a"], m["b"])}
        for key in ("leaderboard", "leaderboard_live"):
            lb = doc.get(key)
            if not lb:
                continue
            total = sum(e["p"] for e in lb)
            assert total == pytest.approx(1.0, abs=0.02), f"{doc['slug']}/{key} sums to {total}"
            for entry in lb:
                assert entry["name"] in entrants, (
                    f"{doc['slug']}/{key}: {entry['name']!r} is not in the draw")
                if is_placeholder(entry["name"]):
                    placeholder_seen.append(f"{doc['slug']}/{key}: {entry['name']}")
    assert not placeholder_seen, (
        "unfilled draw slots reached a championship leaderboard: " + ", ".join(placeholder_seen)
    )


def test_every_player_shard_has_a_matchup_shard():
    players = {os.path.basename(p) for p in _shards("player")}
    matchups = {os.path.basename(p) for p in _shards("matchup")}
    assert players == matchups, (
        "player and matchup shards disagree — a stale shard from an old slug, "
        f"or a missing one: {sorted(players ^ matchups)[:5]}"
    )


def test_every_shard_is_reachable_from_the_index():
    """
    The index is the only entry point, so a shard missing from it is dead
    weight the site can never open, and a duplicate slug means two tournaments
    overwrite each other's file. The reverse direction is not asserted: a draw
    Wikipedia never filled in is indexed without a shard on purpose.
    """
    index_path = os.path.join(OUT, "tournaments.json")
    if not os.path.exists(index_path):
        pytest.skip("index not built")
    index = json.loads(open(index_path).read())

    slugs = [e["slug"] for e in index]
    assert len(slugs) == len(set(slugs)), "duplicate slugs in the index"
    for entry in index:
        assert entry["slug"] == slugify(entry["name"])

    on_disk = {os.path.basename(p)[:-5] for p in _shards("tournament")}
    orphans = on_disk - set(slugs)
    assert not orphans, f"shards not reachable from the index: {sorted(orphans)[:5]}"
