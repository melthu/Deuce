import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:        # torch is optional - only the DeepFM/TabNet paths need it
    torch = None
    _TorchDataset = object

DATA_PATH = "data/processed/final_training_data.csv"

CONT_COLS = [
    # Original 10
    "same_nationality",
    "h2h_win_rate_a_vs_b",
    "player_a_is_home",
    "player_a_matches_last_14_days",
    "player_a_days_since_last_match",
    "player_a_recent_win_rate",
    "player_b_is_home",
    "player_b_matches_last_14_days",
    "player_b_days_since_last_match",
    "player_b_recent_win_rate",
    # New 10
    "player_a_elo",
    "player_b_elo",
    "elo_diff",
    "player_a_ema_form",
    "player_b_ema_form",
    "h2h_last_winner",
    "player_a_win_streak",
    "player_b_win_streak",
    "player_a_matches_last_7_days",
    "player_b_matches_last_7_days",
    # Score-derived 4
    "player_a_avg_point_diff",
    "player_b_avg_point_diff",
    "player_a_avg_games_per_match",
    "player_b_avg_games_per_match",
    # New 6: rubber-game rate, victory margin, seeding
    "player_a_rubber_game_rate",
    "player_b_rubber_game_rate",
    "player_a_avg_victory_margin",
    "player_b_avg_victory_margin",
    "player_a_seed",
    "player_b_seed",
]

UNK_ID = 0  # reserved for players not seen during training

# Wikipedia round-name variants → canonical names used everywhere downstream
ROUND_ALIASES = {
    "1st round":      "first round",
    "2nd round":      "second round",
    "3rd round":      "third round",
    "first round[2]": "first round",
    "quarterfinals":  "quarter-finals",
    "semifinals":     "semi-finals",
    "finals":         "final",
}


def extract_numpy(dataset):
    """
    Return (X, y) numpy arrays from a BWFDataset with cat and cont features
    concatenated horizontally. Works without torch installed.
    """
    X = np.hstack([dataset.cat.astype(np.float64), dataset.cont.astype(np.float64)])
    y = dataset.labels.astype(np.float32).ravel()
    return X, y


def fill_missing_cont_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Fill any CONT_COLS absent from df with 0.0 (backward-compatibility helper)."""
    for col in CONT_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df


def load_training_frame(csv_path: str = DATA_PATH, drop_pending: bool = True,
                        drop_walkover: bool | None = None) -> pd.DataFrame:
    """
    Load the mirrored training CSV with standard cleaning applied:
    parsed dates, lower-cased rounds, backward-compat column fills.

    Two kinds of row carry no usable outcome and are dropped for training or
    eval, but kept when the caller needs the full bracket (the dashboard and
    the static exporter pass drop_pending=False to display draws):

      * pending   - a published draw match that has not been played yet
      * walkover  - a retirement or no-show; it fills a real bracket slot, so
                    it must stay visible for topology, but its scoreline is
                    partial and its result uncontested

    drop_walkover defaults to whatever drop_pending is, so every existing
    training caller keeps excluding them and every display caller keeps them.
    """
    if drop_walkover is None:
        drop_walkover = drop_pending
    df = pd.read_csv(csv_path)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["round"] = df["round"].str.lower().replace(ROUND_ALIASES)
    if "is_pending" not in df.columns:
        df["is_pending"] = 0
    if "is_walkover" not in df.columns:
        df["is_walkover"] = 0
    if drop_pending:
        df = df[df["is_pending"] != 1].reset_index(drop=True)
    if drop_walkover:
        df = df[df["is_walkover"] != 1].reset_index(drop=True)
    fill_missing_cont_cols(df)
    return df


def fit_preprocessors(train_df: pd.DataFrame):
    """
    Fit vocabularies (players, tiers, rounds) and a StandardScaler on the
    given training slice ONLY - the caller guarantees the slice contains no
    future data relative to what will be predicted.

    Returns:
        preprocessors : dict with scaler, player_to_id, tier_to_id, round_to_id
        vocab_sizes   : dict with num_players, num_tiers, num_rounds
    """
    train_players = sorted(
        set(train_df["player_a"].unique()) | set(train_df["player_b"].unique())
    )
    player_to_id = {name: idx + 1 for idx, name in enumerate(train_players)}
    # UNK_ID (0) is implicitly assigned to any name not in the dict

    tier_to_id  = {t: i for i, t in enumerate(sorted(train_df["tier"].unique()))}
    round_to_id = {r: i for i, r in enumerate(sorted(train_df["round"].unique()))}

    scaler = StandardScaler()
    scaler.fit(train_df[CONT_COLS].values)

    preprocessors = {
        "scaler":       scaler,
        "player_to_id": player_to_id,
        "tier_to_id":   tier_to_id,
        "round_to_id":  round_to_id,
    }
    vocab_sizes = {
        "num_players": len(player_to_id) + 1,   # +1 for UNK slot
        "num_tiers":   len(tier_to_id),
        "num_rounds":  len(round_to_id),
    }
    return preprocessors, vocab_sizes


def encode_split(split_df: pd.DataFrame, preprocessors: dict):
    """Encode a dataframe slice with already-fit preprocessors.
    Unseen tiers/rounds/players all map to id 0."""
    cat = np.column_stack([
        split_df["tier"].map(preprocessors["tier_to_id"]).fillna(0).values,
        split_df["round"].map(preprocessors["round_to_id"]).fillna(0).values,
        split_df["player_a"].map(preprocessors["player_to_id"]).fillna(UNK_ID).values,
        split_df["player_b"].map(preprocessors["player_to_id"]).fillna(UNK_ID).values,
    ])
    cont   = preprocessors["scaler"].transform(split_df[CONT_COLS].values)
    labels = split_df["player_a_won"].values
    return cat, cont, labels


class BWFDataset(_TorchDataset):
    """
    Thin wrapper that holds pre-encoded numpy arrays and serves tensors.
    Encoding and scaling are handled externally by get_train_val_datasets().
    """

    def __init__(self, cat: np.ndarray, cont: np.ndarray, labels: np.ndarray):
        self.cat    = cat.astype(np.int64)
        self.cont   = cont.astype(np.float32)
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        Returns:
            cat_features  : LongTensor  (4,)  - [tier, round, player_a, player_b]
            cont_features : FloatTensor (30,) - scaled continuous features
            label         : FloatTensor (1,)  - player_a_won
        """
        if torch is None:
            raise ImportError("torch is required to iterate BWFDataset as tensors")
        return (
            torch.tensor(self.cat[idx],         dtype=torch.long),
            torch.tensor(self.cont[idx],        dtype=torch.float32),
            torch.tensor([self.labels[idx]],    dtype=torch.float32),
        )


def get_train_val_datasets(csv_path: str = DATA_PATH):
    """
    Load final_training_data.csv, split chronologically, and apply
    strictly leakage-free preprocessing:

      - Vocabularies (players, tiers, rounds) built from train only.
      - StandardScaler fit on train only, then applied to both splits.
      - 2026 players unseen in training receive UNK_ID (0).

    Returns:
        train_dataset : BWFDataset
        val_dataset   : BWFDataset
        vocab_sizes   : dict with num_players, num_tiers, num_rounds
        preprocessors : dict with scaler, player_to_id, tier_to_id, round_to_id
    """
    df = load_training_frame(csv_path)

    train_df = df[df["start_date"].dt.year <= 2025].copy()
    val_df   = df[df["start_date"].dt.year >= 2026].copy()

    preprocessors, vocab_sizes = fit_preprocessors(train_df)

    train_cat, train_cont, train_labels = encode_split(train_df, preprocessors)
    val_cat,   val_cont,   val_labels   = encode_split(val_df,   preprocessors)

    train_dataset = BWFDataset(train_cat, train_cont, train_labels)
    val_dataset   = BWFDataset(val_cat,   val_cont,   val_labels)

    return train_dataset, val_dataset, vocab_sizes, preprocessors


if __name__ == "__main__":
    train_ds, val_ds, vocab_sizes, _ = get_train_val_datasets()

    print("=== Split Sizes ===")
    print(f"  Train rows : {len(train_ds)}")
    print(f"  Val rows   : {len(val_ds)}")

    print("\n=== Vocabulary Sizes (fit on train) ===")
    print(f"  num_players : {vocab_sizes['num_players']}  (includes 1 UNK slot)")
    print(f"  num_tiers   : {vocab_sizes['num_tiers']}")
    print(f"  num_rounds  : {vocab_sizes['num_rounds']}")

    X, y = extract_numpy(train_ds)
    print(f"\n=== extract_numpy(train) ===")
    print(f"  X: {X.shape} {X.dtype}  |  y: {y.shape} {y.dtype}")
