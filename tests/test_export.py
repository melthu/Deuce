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
    EXPORT_VERSION, ROUND_ORDER, fingerprint, is_placeholder, slugify,
)

OUT = "site/data"


def test_point_in_time_model_never_sees_its_own_tournament(fitted, df):
    """
    The whole claim behind a retrospective prediction. Vocabulary, scaler and
    estimator must all be fit on matches that finished strictly before the
    tournament started - not on or after its start date.
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
        pytest.skip(f"{OUT}/{kind} not built - run `make export`")
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


def test_retirements_are_marked_and_not_scored():
    """
    A retirement leaves a partial score behind ("18-21, 6-2"). Without a flag
    the frontend renders that as a completed result, and the accuracy figure
    credits the model for a match decided by injury. The rest of the pipeline
    already excludes these rows from Elo, form, h2h and training.
    """
    for path in _shards("tournament"):
        doc = json.loads(open(path).read())
        for m in doc["matches"]:
            assert isinstance(m["wo"], bool), f"{doc['slug']}: `wo` missing"
        judged = [m for m in doc["matches"] if not m["pending"] and not m["wo"]]
        assert doc["accuracy"]["n"] == len(judged), (
            f"{doc['slug']}: accuracy denominator counts uncontested matches")
        assert doc["accuracy"]["hit"] <= doc["accuracy"]["n"]


def test_exported_walkover_flags_match_the_scrape(raw):
    """
    Guards the join, not the rule: `wo` is read off the engineered frame while
    the score beside it comes from the raw scrape, so a shift in either would
    show up as a match labelled retired that never was, or the reverse.
    """
    flagged = raw[raw["is_walkover"] == 1]
    if flagged.empty:
        pytest.skip("no walkovers in the corpus")
    expected = {
        (t, frozenset((r["player_a"], r["player_b"])))
        for t, g in flagged.groupby("tournament") for _, r in g.iterrows()
    }
    found = set()
    for path in _shards("tournament"):
        doc = json.loads(open(path).read())
        for m in doc["matches"]:
            if m["wo"]:
                found.add((doc["tournament"], frozenset((m["a"], m["b"]))))
    # Tournaments before FIRST_YEAR are never exported, so only check the
    # direction that can be checked: nothing is marked that the scrape did not.
    assert not (found - expected), (
        f"matches marked as walkovers that the raw data does not flag: "
        f"{sorted(str(x) for x in (found - expected))[:5]}")


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


def test_advancement_odds_are_structurally_sound():
    """
    Two checks the numbers cannot pass by accident. A round has a fixed number
    of slots, so "reached it" summed over every player must equal that count,
    16, 8, 4, 2 for a 32-draw. And reaching a later round implies reaching every
    earlier one, so a player's odds can only fall. An off-by-one in which slot
    array gets counted breaks the first; counting winners instead of entrants
    breaks both.
    """
    for path in _shards("tournament"):
        doc = json.loads(open(path).read())
        rounds = doc["adv_rounds"]
        for key in ("leaderboard", "leaderboard_live"):
            lb = doc.get(key)
            if not lb:
                continue
            for entry in lb:
                assert len(entry["adv"]) == len(rounds), (
                    f"{doc['slug']}/{key}: {entry['name']} has "
                    f"{len(entry['adv'])} advancement odds for {len(rounds)} rounds")
                assert all(a >= b - 1e-9 for a, b in zip(entry["adv"], entry["adv"][1:])), (
                    f"{doc['slug']}/{key}: {entry['name']}'s odds rise in a later round")
                assert entry["adv"][-1] >= entry["p"] - 1e-9, (
                    f"{doc['slug']}/{key}: {entry['name']} wins more often than "
                    f"they reach the final")
            # Every slot in a round is filled in every simulation, so summing
            # "reached it" over the board must come out at the round's slot
            # count - a whole number, halving each round.
            #
            # The expected count is taken from the board's own first column
            # rather than from the document's match list: the simulation's
            # topology comes from build_time_zero_state, and on an irregular
            # draw the two disagree. Akita Masters 2019 simulates a 12-player
            # bracket while its page lists 38 names. The chain also stops at
            # the first odd round, where run_monte_carlo hands out a bye and
            # the halving legitimately breaks.
            if key != "leaderboard" or any(
                    is_placeholder(p) for m in doc["matches"] for p in (m["a"], m["b"])):
                continue
            sums = [sum(e["adv"][i] for e in lb) for i in range(len(rounds))]
            assert sums and sums[0] == pytest.approx(round(sums[0]), abs=0.02), (
                f"{doc['slug']}: {rounds[0]} reached by {sums[0]:.2f} players, "
                f"which is not a whole number of slots")
            for i, rnd in enumerate(rounds):
                expect = sums[0] / 2 ** i
                if expect < 2 or abs(expect - round(expect)) > 1e-9:
                    break             # a bye splits the chain; stop here
                assert sums[i] == pytest.approx(expect, abs=0.02), (
                    f"{doc['slug']}: {rnd} reached by {sums[i]:.2f} players, "
                    f"but the round has {expect:g} slots")


def test_every_player_shard_has_a_matchup_shard():
    players = {os.path.basename(p) for p in _shards("player")}
    matchups = {os.path.basename(p) for p in _shards("matchup")}
    assert players == matchups, (
        "player and matchup shards disagree - a stale shard from an old slug, "
        f"or a missing one: {sorted(players ^ matchups)[:5]}"
    )


def test_index_status_agrees_with_the_shard():
    """
    The index and the shard used to derive status independently, and drifted:
    eleven finished tournaments were still labelled "live" in the index after
    the shards had been corrected. The index is what the sidebar renders, so
    that is the one the user actually sees.
    """
    index_path = os.path.join(OUT, "tournaments.json")
    if not os.path.exists(index_path):
        pytest.skip("index not built")
    index = {e["slug"]: e["status"] for e in json.loads(open(index_path).read())}

    mismatches = []
    for path in _shards("tournament"):
        doc = json.loads(open(path).read())
        if doc["slug"] in index and index[doc["slug"]] != doc["status"]:
            mismatches.append(f"{doc['slug']}: index={index[doc['slug']]} shard={doc['status']}")
    assert not mismatches, "index and shard disagree on status: " + ", ".join(mismatches)


def test_only_current_tournaments_are_live():
    """A draw weeks past its start date with an unplayed match is an incomplete
    page, not a live event - and "live" drives the conditioned leaderboard."""
    from src.serving.export_static import STALE_AFTER_DAYS

    index_path = os.path.join(OUT, "tournaments.json")
    if not os.path.exists(index_path):
        pytest.skip("index not built")
    today = pd.Timestamp.today().normalize()
    stale_live = [
        e["slug"] for e in json.loads(open(index_path).read())
        if e["status"] == "live" and (today - pd.Timestamp(e["date"])).days > STALE_AFTER_DAYS
    ]
    assert not stale_live, f"finished tournaments labelled live: {stale_live}"


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
