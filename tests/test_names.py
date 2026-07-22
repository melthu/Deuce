"""
Player identity: folding, slugs, and the integrity of the alias map.

Getting this wrong does not raise — it silently splits one player's Elo, form
and head-to-head across two identities, or worse, fuses two real players.
"""
import pandas as pd
import pytest

from src.pipeline.data_checks import find_name_collisions
from src.pipeline.player_names import ALIASES, REVIEWED_DISTINCT, canonical, fold_ascii
from src.serving.export_static import slugify


@pytest.mark.parametrize("raw_name, folded", [
    # Stroke and slash letters are single codepoints that NFKD does not
    # decompose, so a plain encode("ascii", "ignore") deletes them outright.
    # These four are the exact cases that hid split players and mangled slugs.
    ("Nguyễn Hải Đăng", "Nguyen Hai Dang"),
    ("Mikołaj Szymanowski", "Mikolaj Szymanowski"),
    ("Ditlev Jæger Holm", "Ditlev Jaeger Holm"),
    ("Ville Lång", "Ville Lang"),
    # Ordinary combining marks must keep working too.
    ("Arnaud Merklé", "Arnaud Merkle"),
    ("Pablo Abián", "Pablo Abian"),
])
def test_fold_ascii_preserves_letters(raw_name, folded):
    assert fold_ascii(raw_name) == folded


def test_folding_never_drops_a_letter():
    """A name must not lose characters to folding — that is how 'Đăng' became 'ang'."""
    for name in list(ALIASES) + list(ALIASES.values()):
        assert len(fold_ascii(name)) >= len(name.replace("æ", "").replace("ß", "")), name
        assert fold_ascii(name).strip(), name


def test_slugs_are_nonempty_and_unique_per_identity():
    canon = sorted(set(ALIASES.values()))
    slugs = {}
    for name in canon:
        s = slugify(name)
        assert s and s.strip("-"), f"{name!r} slugified to {s!r}"
        assert s not in slugs, f"{name!r} and {slugs[s]!r} share the slug {s!r}"
        slugs[s] = name


def test_alias_map_has_no_chains():
    """Every canonical name must be a fixed point: canonical(canonical(x)) == canonical(x)."""
    for alias, target in ALIASES.items():
        assert target not in ALIASES, (
            f"{alias!r} -> {target!r}, but {target!r} is itself an alias — "
            "canonicalisation is applied once, so a chain silently half-resolves"
        )
        assert canonical(canonical(alias)) == canonical(alias)


def test_reviewed_distinct_pairs_are_not_also_merged():
    """A pair cannot be both judged distinct and folded together."""
    for pair in REVIEWED_DISTINCT:
        a, b = tuple(pair)
        assert canonical(a) != canonical(b), f"{a!r} and {b!r} are merged and marked distinct"


def test_no_merged_pair_ever_played_each_other(raw):
    """
    The decisive disqualifier. If two spellings met on court they are two
    people, and merging them fabricates a player who beat themselves.
    """
    met = {frozenset(p) for p in raw[["player_a", "player_b"]].itertuples(index=False)}
    for alias, target in ALIASES.items():
        assert frozenset((alias, target)) not in met, (
            f"{alias!r} and {target!r} are merged but played each other"
        )


def test_raw_data_is_canonicalised(raw):
    """
    The scraper canonicalises on write, so no alias should survive in the CSV.
    A hit here means something wrote the raw file bypassing that choke point.
    """
    present = (set(raw["player_a"]) | set(raw["player_b"])) & set(ALIASES)
    assert not present, f"un-canonicalised spellings in raw_matches.csv: {sorted(present)}"


def test_no_unreviewed_collisions(raw):
    """
    Mirrors the warning `data_checks.py` prints. Not fatal there, because a new
    spelling is a review task rather than corruption — but the suite should say
    so out loud rather than let it accumulate.
    """
    collisions = find_name_collisions(raw)
    assert not collisions, (
        "unreviewed name collisions — add to ALIASES or REVIEWED_DISTINCT: "
        + ", ".join(f"{a!r}<->{b!r}" for a, b in collisions)
    )


def test_collision_finder_excludes_players_who_met():
    """Huang Yu / Huang Yu-kai fold to the same tokens but are different people."""
    df = pd.DataFrame({
        "player_a": ["Huang Yu"], "player_b": ["Huang Yu-kai"], "player_a_won": [1],
    })
    assert find_name_collisions(df) == []
