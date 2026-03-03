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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.dataset import BWFDataset, CONT_COLS, UNK_ID, fill_missing_cont_cols

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
    df = pd.read_csv(csv_path)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["round"] = df["round"].str.lower()
    fill_missing_cont_cols(df)

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

        # ── Vocabularies built from train only ──────────────────────────
        train_players = sorted(
            set(train_df["player_a"].unique()) | set(train_df["player_b"].unique())
        )
        player_to_id = {name: idx + 1 for idx, name in enumerate(train_players)}

        tiers      = sorted(train_df["tier"].unique())
        tier_to_id = {t: i for i, t in enumerate(tiers)}

        rounds      = sorted(train_df["round"].unique())
        round_to_id = {r: i for i, r in enumerate(rounds)}

        vocab_sizes = {
            "num_players": len(player_to_id) + 1,   # +1 for UNK slot
            "num_tiers":   len(tier_to_id),
            "num_rounds":  len(round_to_id),
        }

        # ── Scaler fit on train only ─────────────────────────────────────
        scaler = StandardScaler()
        scaler.fit(train_df[CONT_COLS].values)

        # ── Encode + scale ───────────────────────────────────────────────
        def encode(split_df: pd.DataFrame):
            cat = np.column_stack([
                split_df["tier"].map(tier_to_id).values,
                split_df["round"].map(round_to_id).fillna(0).values,
                split_df["player_a"].map(player_to_id).fillna(UNK_ID).values,
                split_df["player_b"].map(player_to_id).fillna(UNK_ID).values,
            ])
            cont   = scaler.transform(split_df[CONT_COLS].values)
            labels = split_df["player_a_won"].values
            return cat, cont, labels

        train_cat, train_cont, train_labels = encode(train_df)
        val_cat,   val_cont,   val_labels   = encode(val_df)

        train_dataset = BWFDataset(train_cat, train_cont, train_labels)
        val_dataset   = BWFDataset(val_cat,   val_cont,   val_labels)

        preprocessors = {
            "scaler":       scaler,
            "player_to_id": player_to_id,
            "tier_to_id":   tier_to_id,
            "round_to_id":  round_to_id,
        }

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
