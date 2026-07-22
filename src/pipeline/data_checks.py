"""
Sanity gate for the scheduled data refresh.

Run after scraping and before committing: exits non-zero if the raw CSV
looks corrupted, so a bad scrape never reaches the deployed app.

    python3 src/pipeline/data_checks.py
"""
import itertools
import os
import re
import sys
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # repo root

from src.pipeline.player_names import ALIASES, REVIEWED_DISTINCT, fold_ascii

RAW_PATH    = "data/raw/raw_matches.csv"
CONFIG_PATH = "data/config/tournaments_config.csv"

# Thinnest real season on record is 2021 (16) in the World Tour era and 13 in
# the Super Series era; a year that comes back under this lost tournaments.
MIN_PER_YEAR = 10
FIRST_YEAR   = 2010

REQUIRED_COLS = [
    "tournament", "tier", "round", "start_date", "host_country",
    "player_a", "player_b", "player_a_won",
]
MIN_ROWS              = 9_000   # dataset only ever grows
MAX_WALKOVER_FRACTION = 0.05
MAX_PENDING_FRACTION  = 0.05


def _name_tokens(name: str) -> frozenset:
    return frozenset(t for t in re.split(r"[^a-z0-9]+", fold_ascii(name).lower()) if t)


def find_name_collisions(df: pd.DataFrame) -> list:
    """
    Spellings that probably refer to one player but are not in ALIASES.

    Reported, never auto-merged: two real players can normalise to the same
    string (Huang Yu and Huang Yu-kai played each other), so a human decides.
    Pairs that ever met are excluded - that is proof they are different people.
    """
    names = [n for n in set(df["player_a"]) | set(df["player_b"])
             if isinstance(n, str) and n.strip() and n.strip().upper() != "TBD"]
    toks = {n: _name_tokens(n) for n in names}
    nat = {}
    for side in ("a", "b"):
        col = f"player_{side}_nat"
        if col in df.columns:
            for n, v in df[[f"player_{side}", col]].dropna().itertuples(index=False):
                nat.setdefault(n, v)
    met = {frozenset(p) for p in
           df[["player_a", "player_b"]].itertuples(index=False)}

    out = []
    for a, b in itertools.combinations(sorted(names), 2):
        ta, tb = toks[a], toks[b]
        if not (ta == tb or ((ta < tb or tb < ta) and len(ta & tb) >= 2)):
            continue
        if frozenset((a, b)) in met:
            continue
        if a in nat and b in nat and nat[a] != nat[b]:
            continue
        if ALIASES.get(a, a) == ALIASES.get(b, b):
            continue          # already folded together
        if frozenset((a, b)) in REVIEWED_DISTINCT:
            continue          # checked, judged distinct
        out.append((a, b))
    return out


def check_config() -> list:
    """
    The calendar must cover every season, not just some of them.

    `build_config.py` refuses to overwrite a config that shrank by more than 5%,
    but that only catches a scrape that collapses in one go. A year whose
    Wikipedia page changed shape returns zero rows quietly while the other
    sixteen years carry the total, so the calendar stays large enough to pass
    and the season simply goes missing. Check each year, not the total.

    The lookahead year (`date.today().year + 1`) is not required: its page
    genuinely does not exist for most of the season.
    """
    if not os.path.exists(CONFIG_PATH):
        return [f"{CONFIG_PATH} is missing - run build_config.py"]

    cfg = pd.read_csv(CONFIG_PATH)
    errors = []

    years = pd.to_datetime(cfg["start_date"], errors="coerce").dt.year
    if years.isna().any():
        errors.append(f"{int(years.isna().sum())} unparseable start_date values in the config")

    counts = years.dropna().astype(int).value_counts()
    this_year = date.today().year
    for year in range(FIRST_YEAR, this_year + 1):
        n = int(counts.get(year, 0))
        if n == 0:
            errors.append(f"config has no tournaments for {year} - that season's page failed to scrape")
        elif n < MIN_PER_YEAR:
            errors.append(f"config has only {n} tournaments for {year} (expected >= {MIN_PER_YEAR})")

    stray = sorted(y for y in counts.index if y < FIRST_YEAR or y > this_year + 1)
    if stray:
        errors.append(f"config has tournaments dated outside {FIRST_YEAR}-{this_year + 1}: {stray}")

    dup = cfg["url"].duplicated().sum()
    if dup:
        errors.append(f"{dup} duplicate draw URLs in the config")

    return errors


def main() -> int:
    df = pd.read_csv(RAW_PATH)
    errors = check_config()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")

    if len(df) < MIN_ROWS:
        errors.append(f"row count {len(df)} < {MIN_ROWS} - scrape lost data")

    if not missing:
        for col in ["tournament", "player_a", "player_b", "start_date"]:
            n_null = df[col].isna().sum()
            if n_null:
                errors.append(f"{n_null} null values in '{col}'")

        bad_dates = pd.to_datetime(df["start_date"], errors="coerce").isna().sum()
        if bad_dates:
            errors.append(f"{bad_dates} unparseable start_date values")

        dup_keys = df.duplicated(
            subset=["tournament", "round", "player_a", "player_b"]
        ).sum()
        if dup_keys:
            errors.append(f"{dup_keys} duplicate (tournament, round, pair) rows")

        if "is_walkover" in df.columns:
            frac = df["is_walkover"].mean()
            if frac > MAX_WALKOVER_FRACTION:
                errors.append(f"walkover fraction {frac:.3f} > {MAX_WALKOVER_FRACTION}")

        if "is_pending" in df.columns:
            frac = df["is_pending"].mean()
            if frac > MAX_PENDING_FRACTION:
                errors.append(f"pending fraction {frac:.3f} > {MAX_PENDING_FRACTION}")

    if errors:
        print("DATA CHECKS FAILED:")
        for e in errors:
            print(f"  ✗ {e}")
        return 1

    n_pending = int(df["is_pending"].sum()) if "is_pending" in df.columns else 0
    print(f"Data checks passed: {len(df)} rows ({n_pending} pending), "
          f"{df['tournament'].nunique()} tournaments.")

    # A warning, not a failure: an unreviewed variant splits one player's
    # history in two, which is bad but not corrupt. It needs a human to look
    # at it, so it must be visible rather than silent.
    collisions = find_name_collisions(df)
    if collisions:
        print(f"\nWARNING: {len(collisions)} unreviewed player-name collision(s). "
              f"Add to ALIASES in src/pipeline/player_names.py, or record why not:")
        for a, b in collisions:
            print(f"  ? {a!r}  <->  {b!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
