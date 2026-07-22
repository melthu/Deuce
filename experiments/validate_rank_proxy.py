"""Check the ranking proxy against the real published BWF ranking.

The proxy in candidate_features.py is derived from results, not downloaded, so
it is worth knowing how close it actually lands to the list it stands in for.

The only real BWF men's-singles ranking history that can be fetched without an
account is github.com/raywan/bwf-data, which covers 2015 w1 to 2016 w7 - about
60 weekly snapshots. Far too short to train on, but enough to answer "is this
proxy the same quantity, roughly?".

Per weekly snapshot this reports:
  * Spearman rank correlation over players present in both lists
  * how many of the real top 10 the proxy also puts in its top 10
  * median absolute rank error

Players are matched on ASCII-folded surname/given-name token sets, because the
BWF list writes "Jan O JORGENSEN" and the corpus writes "Jan Ø Jørgensen".

    python3 experiments/validate_rank_proxy.py
"""
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.pipeline.player_names import fold_ascii
from experiments.candidate_features import (
    award_points, rounds_left_map, _rolling_points, RAW_PATH)

BASE = "https://raw.githubusercontent.com/raywan/bwf-data/master/data/ms/"
CACHE_DIR = "data/interim/bwf_rankings"


def week_dates(year: int, week: int) -> pd.Timestamp:
    """BWF publishes on Thursdays; ISO week `week` of `year`."""
    return pd.Timestamp.fromisocalendar(year, max(1, min(week, 52)), 4)


def fetch_week(year: int, week: int) -> pd.DataFrame | None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"bwf_ms_{year}w{week}.csv")
    if not os.path.exists(path):
        url = f"{BASE}bwf_ms_{year}w{week}.csv"
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = r.read()
        except Exception:
            return None
        with open(path, "wb") as f:
            f.write(data)
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def name_key(name: str) -> frozenset:
    """Order- and case-insensitive token set, so word order cannot split a player."""
    # fold_ascii strips diacritics but preserves case, and the BWF list
    # upper-cases surnames ("CHEN Long") where the corpus does not.
    # Wikipedia link targets carry disambiguators - "Lin Chun-yi (badminton)".
    name = re.sub(r"\([^)]*\)", " ", str(name))
    toks = fold_ascii(name).lower().replace("-", " ").split()
    return frozenset(t for t in toks if len(t) > 1)


def proxy_standings(raw_path: str = RAW_PATH):
    """(date -> {player: points}) machinery, reused from the candidate builder."""
    df = pd.read_csv(raw_path)
    df["start_date"] = pd.to_datetime(df["start_date"])
    for col, default in [("is_pending", 0), ("is_walkover", 0)]:
        if col not in df.columns:
            df[col] = default
    df = df.sort_values("start_date", kind="stable").reset_index(drop=True)
    df["_rounds_left"] = rounds_left_map(df)

    events = sorted(
        ((date, player, pts)
         for (_t, date), players in award_points(df).items()
         for player, pts in players.items()),
        key=lambda e: e[0],
    )
    return events


def proxy_rank_at(events, when: pd.Timestamp) -> dict:
    by_player: dict[str, list] = {}
    for d, p, pts in events:
        if d >= when:
            break
        by_player.setdefault(p, []).append((d, pts))
    standings = sorted(((_rolling_points(ev, when), p) for p, ev in by_player.items()),
                       reverse=True)
    return {p: i + 1 for i, (v, p) in enumerate(standings) if v > 0}


WIKI_API = ("https://en.wikipedia.org/w/api.php?action=parse&page=BWF_World_Ranking"
            "&prop=wikitext&format=json&section=9")
WIKI_CACHE = "data/interim/bwf_wiki_current_ms.json"


def fetch_wikipedia_top20() -> tuple[pd.Timestamp, pd.DataFrame] | None:
    """The current men's-singles top 20 from the BWF World Ranking article.

    The article carries only a single current snapshot - no weekly series - so
    this cannot feed a feature. It is worth having anyway: the raywan mirror
    validates the proxy in 2015-16, and this validates it in the era the site
    actually serves.
    """
    os.makedirs(os.path.dirname(WIKI_CACHE), exist_ok=True)
    if os.path.exists(WIKI_CACHE):
        with open(WIKI_CACHE) as f:
            wikitext = json.load(f)["wikitext"]
    else:
        # Wikipedia rejects urllib's default User-Agent outright.
        req = urllib.request.Request(
            WIKI_API, headers={"User-Agent": "Deuce/1.0 (BWF match model; research)"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                wikitext = json.loads(r.read())["parse"]["wikitext"]["*"]
        except Exception as exc:
            print(f"  (fetch failed: {exc})")
            return None
        with open(WIKI_CACHE, "w") as f:
            json.dump({"wikitext": wikitext}, f)

    m = re.search(r"\{\{as of\|(\d{4})\|(\d{2})\|(\d{2})", wikitext)
    as_of = pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None

    rows = []
    # Each entry is "! <rank>" followed by a [[Player]] link and a points cell.
    for block in re.split(r"\n\|-\n", wikitext):
        rank = re.search(r"^!\s*(\d+)\s*$", block, re.M)
        names = re.findall(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", block)
        pts = re.search(r"align=\"right\"\s*\|\s*([\d,]+)\s*$", block, re.M)
        if not (rank and names and pts):
            continue
        # the first link is the country, the last is the player
        player = names[-1] if len(names) > 1 else names[0]
        rows.append({"RANK": int(rank.group(1)), "PLAYER": player,
                     "POINTS": int(pts.group(1).replace(",", ""))})
    if not rows:
        return None
    return as_of, pd.DataFrame(rows).drop_duplicates("RANK").sort_values("RANK")


def validate_current(events):
    got = fetch_wikipedia_top20()
    if got is None:
        print("\ncould not fetch the Wikipedia current-ranking snapshot - skipping")
        return
    as_of, real = got
    proxy = proxy_rank_at(events, as_of)
    proxy_by_key = {}
    for p, r in proxy.items():
        proxy_by_key.setdefault(name_key(p), r)

    print(f"\n=== proxy vs Wikipedia's current top 20, as of {as_of.date()} ===")
    rows, pairs = [], []
    for _, r in real.iterrows():
        pr = proxy_by_key.get(name_key(r["PLAYER"]))
        rows.append({"real": int(r["RANK"]), "player": r["PLAYER"],
                     "proxy": pr if pr else "-",
                     "delta": (pr - int(r["RANK"])) if pr else None})
        if pr:
            pairs.append((int(r["RANK"]), pr))
    print(pd.DataFrame(rows).to_string(index=False))
    if len(pairs) >= 8:
        a = np.array([x for x, _ in pairs])
        b = np.array([y for _, y in pairs])
        top10 = {name_key(r["PLAYER"]) for _, r in real[real["RANK"] <= 10].iterrows()}
        prox10 = {k for k, v in proxy_by_key.items() if v <= 10}
        print(f"\n  matched {len(pairs)}/20 | Spearman {spearmanr(a, b).statistic:.3f} | "
              f"top-10 overlap {len(top10 & prox10)}/10 | "
              f"median abs err {np.median(np.abs(a - b)):.1f} places")
    else:
        print(f"\n  only {len(pairs)}/20 names matched - too few to correlate")


def validate_xlsx(events):
    """The official weekly XLSX rankings, wherever they could be obtained."""
    from experiments.bwf_rankings import load_all
    snaps = load_all()
    if not snaps:
        print("\nno official XLSX snapshots available - skipping")
        return

    rows = []
    for as_of, real in snaps:
        proxy = proxy_rank_at(events, as_of)
        proxy_by_key = {}
        for p, r in proxy.items():
            proxy_by_key.setdefault(name_key(p), r)
        pairs = [(int(r["rank"]), proxy_by_key[name_key(r["player"])])
                 for _, r in real.iterrows() if name_key(r["player"]) in proxy_by_key]
        if len(pairs) < 30:
            continue
        a = np.array([x for x, _ in pairs])
        b = np.array([y for _, y in pairs])
        top10 = {name_key(r["player"]) for _, r in real[real["rank"] <= 10].iterrows()}
        prox10 = {k for k, v in proxy_by_key.items() if v <= 10}
        rows.append({"date": as_of.date(), "matched": len(pairs),
                     "spearman": spearmanr(a, b).statistic,
                     "top10_overlap": len(top10 & prox10),
                     "median_abs_err": float(np.median(np.abs(a - b)))})

    if not rows:
        print("\nofficial XLSX snapshots found but too few names matched")
        return
    out = pd.DataFrame(rows)
    print(f"\n=== proxy vs official BWF weekly XLSX "
          f"({out['date'].min()} .. {out['date'].max()}) ===")
    print(f"  snapshots compared        : {len(out)}")
    print(f"  players matched per week  : {out['matched'].median():.0f} median")
    print(f"  Spearman rho              : {out['spearman'].mean():.3f} "
          f"(min {out['spearman'].min():.3f}, max {out['spearman'].max():.3f})")
    print(f"  real top-10 also in proxy top-10 : {out['top10_overlap'].mean():.1f} / 10")
    print(f"  median absolute rank error       : {out['median_abs_err'].mean():.1f} places")


def main():
    events = proxy_standings()

    weeks = [(2015, w) for w in range(1, 53)] + [(2016, w) for w in range(1, 8)]
    rows = []
    for year, week in weeks:
        real = fetch_week(year, week)
        if real is None or "PLAYER" not in real.columns:
            continue
        when = week_dates(year, week)
        proxy = proxy_rank_at(events, when)
        proxy_by_key = {}
        for p, r in proxy.items():
            proxy_by_key.setdefault(name_key(p), r)

        pairs = []
        for _, r in real.iterrows():
            k = name_key(r["PLAYER"])
            if k in proxy_by_key:
                pairs.append((int(r["RANK"]), proxy_by_key[k]))
        if len(pairs) < 30:
            continue
        real_r = np.array([a for a, _ in pairs])
        prox_r = np.array([b for _, b in pairs])

        real_top10 = {name_key(r["PLAYER"]) for _, r in real.nsmallest(10, "RANK").iterrows()}
        prox_top10 = {k for k, v in proxy_by_key.items() if v <= 10}

        rows.append({
            "week": f"{year}w{week:02d}",
            "matched": len(pairs),
            "spearman": spearmanr(real_r, prox_r).statistic,
            "top10_overlap": len(real_top10 & prox_top10),
            "median_abs_err": float(np.median(np.abs(real_r - prox_r))),
        })

    if not rows:
        print("no weekly snapshots could be fetched - skipping validation")
        return

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print("\n=== proxy vs published BWF men's singles ranking ===")
    print(f"  weekly snapshots compared : {len(out)}")
    print(f"  players matched per week  : {out['matched'].mean():.0f} median")
    print(f"  Spearman rho              : {out['spearman'].mean():.3f} "
          f"(min {out['spearman'].min():.3f}, max {out['spearman'].max():.3f})")
    print(f"  real top-10 also in proxy top-10 : {out['top10_overlap'].mean():.1f} / 10")
    print(f"  median absolute rank error       : {out['median_abs_err'].mean():.1f} places")

    validate_xlsx(events)
    validate_current(events)


if __name__ == "__main__":
    main()
