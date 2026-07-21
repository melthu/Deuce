"""
Sanity gate for the scheduled data refresh.

Run after scraping and before committing: exits non-zero if the raw CSV
looks corrupted, so a bad scrape never reaches the deployed app.

    python3 src/pipeline/data_checks.py
"""
import sys

import pandas as pd

RAW_PATH = "data/raw/raw_matches.csv"

REQUIRED_COLS = [
    "tournament", "tier", "round", "start_date", "host_country",
    "player_a", "player_b", "player_a_won",
]
MIN_ROWS              = 9_000   # dataset only ever grows
MAX_WALKOVER_FRACTION = 0.05
MAX_PENDING_FRACTION  = 0.05


def main() -> int:
    df = pd.read_csv(RAW_PATH)
    errors = []

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")

    if len(df) < MIN_ROWS:
        errors.append(f"row count {len(df)} < {MIN_ROWS} — scrape lost data")

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
