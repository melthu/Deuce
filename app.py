import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import pandas as pd
import streamlit as st

from src.dataset import get_train_val_datasets
from src.simulate_german_open import (
    ROUND_ORDER,
    model_predict_proba,
    build_time_zero_state,
    build_h2h_lookups,
    predict_match,
    simulate_bracket,
)

DATA_PATH  = "data/processed/final_training_data.csv"
CONFIG_PATH = "data/config/tournaments_config.csv"
MODEL_PATH = "models/best_model.pkl"

st.set_page_config(
    page_title="BWF Match Predictor",
    page_icon="🏸",
    layout="wide",
)


# ------------------------------------------------------------------
# Cached resource loaders
# ------------------------------------------------------------------

@st.cache_resource
def load_resources():
    with open(MODEL_PATH, "rb") as f:
        model_payload = pickle.load(f)

    _, _, _, preprocessors = get_train_val_datasets(DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["round"] = df["round"].str.lower()

    return model_payload, preprocessors, df


@st.cache_data(show_spinner=False)
def get_2026_tournaments():
    cfg = pd.read_csv(CONFIG_PATH)
    cfg2026 = cfg[cfg["start_date"].str.startswith("2026")].copy()
    cfg2026 = cfg2026.sort_values("start_date")
    return cfg2026[["tournament_name", "tier", "start_date"]].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def run_simulation(tour_date: str, tier: int, n_sims: int, seed: int = 42):
    """Cached simulation — re-runs only when parameters change."""
    model_payload, preprocessors, df = load_resources()

    scaler       = preprocessors["scaler"]
    player_to_id = preprocessors["player_to_id"]
    tier_to_id   = preprocessors["tier_to_id"]
    round_to_id  = preprocessors["round_to_id"]

    r32_matchups, player_stats = build_time_zero_state(df, tour_date, tier)
    h2h_rate_fn, h2h_last_fn  = build_h2h_lookups(df, tour_date)

    if r32_matchups.empty:
        return None, None

    # Build bracket display table
    bracket_rows = []
    for _, row in r32_matchups.iterrows():
        p = predict_match(
            row["player_a"], row["player_b"], "first round",
            player_stats, h2h_rate_fn, h2h_last_fn,
            scaler, player_to_id, tier_to_id, round_to_id, model_payload,
        )
        bracket_rows.append({
            "Player A":      row["player_a"],
            "Player B":      row["player_b"],
            "P(A wins)":     round(p, 3),
        })
    bracket_df = pd.DataFrame(bracket_rows)

    # Monte Carlo
    rng = np.random.default_rng(seed)
    win_counts = {}
    for _ in range(n_sims):
        champion = simulate_bracket(
            r32_matchups, player_stats,
            h2h_rate_fn, h2h_last_fn,
            scaler, player_to_id, tier_to_id, round_to_id,
            model_payload, rng,
        )
        win_counts[champion] = win_counts.get(champion, 0) + 1

    leaderboard = (
        pd.DataFrame(win_counts.items(), columns=["Player", "Wins"])
        .sort_values("Wins", ascending=False)
        .reset_index(drop=True)
    )
    leaderboard["Win %"] = (leaderboard["Wins"] / n_sims * 100).round(2)

    return bracket_df, leaderboard


# ------------------------------------------------------------------
# App layout
# ------------------------------------------------------------------

st.title("🏸 BWF Men's Singles — Monte Carlo Bracket Simulator")

tournaments = get_2026_tournaments()

with st.sidebar:
    st.header("Simulation Settings")

    options = tournaments["tournament_name"].tolist()
    selected = st.selectbox("Tournament", options, index=len(options) - 1)

    row = tournaments[tournaments["tournament_name"] == selected].iloc[0]
    tour_date = row["start_date"]
    tier      = int(row["tier"])

    st.caption(f"Date: {tour_date}  |  Tier: {tier}")

    n_sims = st.slider("Simulations", min_value=1_000, max_value=50_000, value=10_000, step=1_000)

    run_btn = st.button("▶  Run Simulation", use_container_width=True, type="primary")

# Show a placeholder until the button is clicked
if not run_btn:
    st.info("Select a tournament in the sidebar and click **▶ Run Simulation** to begin.")
    st.stop()

# Run
with st.spinner(f"Running {n_sims:,} simulations for {selected}..."):
    bracket_df, leaderboard = run_simulation(tour_date, tier, n_sims)

if bracket_df is None:
    st.error(f"No first-round data found for {selected} ({tour_date}). The dataset may not include this tournament.")
    st.stop()

# ------------------------------------------------------------------
# Main panel — two columns
# ------------------------------------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader(f"First Round Bracket ({len(bracket_df)} matchups)")
    styled = bracket_df.style.format({"P(A wins)": "{:.3f}"}).background_gradient(
        subset=["P(A wins)"], cmap="RdYlGn", vmin=0.3, vmax=0.7
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

with col_right:
    st.subheader(f"Championship Probability ({n_sims:,} simulations)")

    top_n = min(16, len(leaderboard))
    chart_data = leaderboard.head(top_n).set_index("Player")["Win %"]
    st.bar_chart(chart_data, horizontal=True)

    st.dataframe(
        leaderboard[["Player", "Win %"]].style.format({"Win %": "{:.2f}%"}).background_gradient(
            subset=["Win %"], cmap="Blues"
        ),
        use_container_width=True,
        hide_index=True,
    )

    model_name = load_resources()[0].get("name", load_resources()[0].get("type", "?"))
    st.caption(f"Model: **{model_name}**  |  Val AUC benchmark: 0.7754")
