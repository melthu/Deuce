"""
Publish gate for the static site.

Every bug this project has hit — the truncated calendar, the broken scrapers,
the brackets that never resolved — failed silently and confidently: the run
went green and shipped something plausible but wrong. This is the check that
stands between a bad export and the live site.

    python3 src/serving/check_export.py [site/data]

Exits non-zero with a reason. Thresholds are deliberately loose; they catch
"the export collapsed", not "the numbers moved a bit".
"""
import glob
import json
import os
import sys

MIN_TOURNAMENTS = 180     # 222 in the index as of 2026-07; a big drop is a bug
MIN_PLAYERS     = 150
MIN_TOTAL_MB    = 3.0
MAX_TOTAL_MB    = 80.0
MAX_UNSIMULATED = 0.15    # share of draws allowed to ship without a forecast


def fail(msg: str):
    print(f"FAIL: {msg}")
    sys.exit(1)


def main(out_dir: str = "site/data"):
    if not os.path.isdir(out_dir):
        fail(f"{out_dir} does not exist — run `make export` first")

    index_path = os.path.join(out_dir, "tournaments.json")
    if not os.path.exists(index_path):
        fail("tournaments.json is missing; the site cannot paint without it")
    with open(index_path) as f:
        index = json.load(f)
    if len(index) < MIN_TOURNAMENTS:
        fail(f"index has {len(index)} tournaments, expected at least {MIN_TOURNAMENTS}")

    files = glob.glob(os.path.join(out_dir, "tournament", "*.json"))
    if len(files) < MIN_TOURNAMENTS:
        fail(f"{len(files)} tournament shards, expected at least {MIN_TOURNAMENTS}")

    players = glob.glob(os.path.join(out_dir, "player", "*.json"))
    matchups = glob.glob(os.path.join(out_dir, "matchup", "*.json"))
    if len(players) < MIN_PLAYERS:
        fail(f"{len(players)} player cards, expected at least {MIN_PLAYERS}")
    if len(matchups) != len(players):
        fail(f"{len(matchups)} matchup shards vs {len(players)} player cards — "
             "every active player needs both")

    total = sum(os.path.getsize(p) for p in
                glob.glob(os.path.join(out_dir, "**", "*.json"), recursive=True))
    mb = total / 1e6
    if not (MIN_TOTAL_MB <= mb <= MAX_TOTAL_MB):
        fail(f"payload is {mb:.1f} MB, expected {MIN_TOTAL_MB}–{MAX_TOTAL_MB} MB")

    # Structural checks on the shards themselves. A file that parses but has no
    # matches, or a bracket the simulator could not resolve, is the failure mode
    # that reaches the browser looking fine.
    empty, unsimulated, bad = [], [], []
    for path in files:
        try:
            with open(path) as f:
                doc = json.load(f)
        except ValueError:
            bad.append(os.path.basename(path))
            continue
        if not doc.get("matches"):
            empty.append(doc.get("slug", os.path.basename(path)))
        if doc.get("leaderboard") is None:
            unsimulated.append(doc.get("slug", os.path.basename(path)))

    if bad:
        fail(f"{len(bad)} shard(s) are not valid JSON: {', '.join(bad[:5])}")
    if empty:
        fail(f"{len(empty)} tournament(s) have no matches: {', '.join(empty[:5])}")

    share = len(unsimulated) / len(files)
    if share > MAX_UNSIMULATED:
        fail(f"{len(unsimulated)} of {len(files)} draws ({share:.0%}) have no "
             f"simulation, above the {MAX_UNSIMULATED:.0%} ceiling — this is "
             "what a bracket-topology regression looks like")

    print(f"OK: {len(files)} tournaments · {len(players)} players · {mb:.1f} MB")
    if unsimulated:
        print(f"    {len(unsimulated)} draw(s) ship without a forecast "
              f"(incomplete on Wikipedia): {', '.join(unsimulated[:5])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "site/data"))
