"""
Rolling 3-fold temporal cross-validation.

The last 3 distinct calendar years in the dataset each serve as a validation fold:
  Fold 1: train < year[-3], val == year[-3]
  Fold 2: train < year[-2], val == year[-2]
  Fold 3: train < year[-1], val == year[-1]

Each fold fits its own vocab + StandardScaler on its training slice only,
so there is zero leakage between folds.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))  # repo root

from src.modeling.dataset import (
    BWFDataset,
    encode_split,
    fit_preprocessors,
    load_training_frame,
)

DATA_PATH = "data/processed/final_training_data.csv"


def get_temporal_folds(csv_path: str = DATA_PATH):
    """
    Build rolling 3-fold temporal CV datasets.

    Returns:
        list of 3 tuples (train_ds, val_ds, vocab_sizes, preprocessors, fold_label)
        where each element is:
          train_ds / val_ds : BWFDataset
          vocab_sizes       : dict with num_players, num_tiers, num_rounds
          preprocessors     : dict with scaler, player_to_id, tier_to_id, round_to_id
          fold_label        : human-readable string summarising the fold
    """
    df = load_training_frame(csv_path)

    years = sorted(df["start_date"].dt.year.unique())
    if len(years) < 4:
        raise ValueError(
            f"Need ≥ 4 distinct years for 3-fold CV; dataset only has {years}"
        )

    val_years = years[-3:]   # e.g. [2024, 2025, 2026]
    folds = []

    for val_year in val_years:
        train_df = df[df["start_date"].dt.year < val_year].copy()
        val_df   = df[df["start_date"].dt.year == val_year].copy()

        if train_df.empty or val_df.empty:
            continue

        # ── Vocab + scaler fit on train only, shared encode logic ───────
        preprocessors, vocab_sizes = fit_preprocessors(train_df)

        train_cat, train_cont, train_labels = encode_split(train_df, preprocessors)
        val_cat,   val_cont,   val_labels   = encode_split(val_df,   preprocessors)

        train_dataset = BWFDataset(train_cat, train_cont, train_labels)
        val_dataset   = BWFDataset(val_cat,   val_cont,   val_labels)

        fold_label = (
            f"val={val_year}  "
            f"(train={len(train_df):,} rows, val={len(val_df):,} rows)"
        )
        folds.append((train_dataset, val_dataset, vocab_sizes, preprocessors, fold_label))

    return folds


if __name__ == "__main__":
    folds = get_temporal_folds()
    print(f"Temporal CV: {len(folds)} folds\n")
    for i, (train_ds, val_ds, vocab, _, label) in enumerate(folds, 1):
        print(f"  Fold {i}: {label}")
        print(f"    num_players={vocab['num_players']}  "
              f"num_tiers={vocab['num_tiers']}  "
              f"num_rounds={vocab['num_rounds']}")
