import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import streamlit as st

from src.dataset import get_train_val_datasets, CONT_COLS
from src.simulate_german_open import (
    ROUND_ORDER,
    model_predict_proba,
    build_time_zero_state,
    build_h2h_lookups,
    predict_match,
    simulate_bracket,
    _predict_one_direction,
)

DATA_PATH   = "data/processed/final_training_data.csv"
CONFIG_PATH = "data/config/tournaments_config.csv"
MODEL_PATH  = "models/best_model.pkl"

# Human-readable names for the 34 model features (4 cat IDs + 30 cont)
FEATURE_NAMES = [
    "tier_id", "round_id", "player_a_id", "player_b_id",
] + CONT_COLS

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


@st.cache_resource
def get_shap_explainer():
    model_payload, _, _ = load_resources()

    def _pick_model(payload):
        if payload["type"] == "single":
            return payload["model"]
        models = payload["models"]
        for name in ("xgb", "lgbm", "catboost"):
            if name in models:
                return models[name]
        for m in models.values():
            if not hasattr(m, "parameters"):
                return m
        return None

    model = _pick_model(model_payload)
    if model is None or hasattr(model, "parameters"):
        return None
    return shap.TreeExplainer(model)


@st.cache_data(show_spinner=False)
def get_all_tournaments():
    cfg = pd.read_csv(CONFIG_PATH)
    cfg["start_date"] = pd.to_datetime(cfg["start_date"], errors="coerce")
    cfg = cfg.dropna(subset=["start_date"])
    cfg = cfg.sort_values("start_date", ascending=False)
    return cfg[["tournament_name", "tier", "start_date"]].reset_index(drop=True)


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

    bracket_rows = []
    for _, row in r32_matchups.iterrows():
        p = predict_match(
            row["player_a"], row["player_b"], "first round",
            player_stats, h2h_rate_fn, h2h_last_fn,
            scaler, player_to_id, tier_to_id, round_to_id, model_payload, tier,
        )
        bracket_rows.append({
            "Player A":  row["player_a"],
            "Player B":  row["player_b"],
            "P(A wins)": round(p, 3),
        })
    bracket_df = pd.DataFrame(bracket_rows)

    rng = np.random.default_rng(seed)
    win_counts = {}
    for _ in range(n_sims):
        champion = simulate_bracket(
            r32_matchups, player_stats,
            h2h_rate_fn, h2h_last_fn,
            scaler, player_to_id, tier_to_id, round_to_id,
            model_payload, rng, tier,
        )
        win_counts[champion] = win_counts.get(champion, 0) + 1

    leaderboard = (
        pd.DataFrame(win_counts.items(), columns=["Player", "Wins"])
        .sort_values("Wins", ascending=False)
        .reset_index(drop=True)
    )
    leaderboard["Win %"] = (leaderboard["Wins"] / n_sims * 100).round(2)
    return bracket_df, leaderboard


def get_actual_winner(df, tour_date: str):
    """Find the actual tournament winner from the final-round row."""
    final_rows = df[
        (df["start_date"] == pd.Timestamp(tour_date)) &
        (df["round"] == "final")
    ]
    if final_rows.empty:
        return None
    row = final_rows.iloc[0]
    return row["player_a"] if row["player_a_won"] == 1 else row["player_b"]


def build_shap_input(pa, pb, round_name, player_stats, h2h_rate_fn, h2h_last_fn,
                     scaler, player_to_id, tier_to_id, round_to_id, tier):
    """
    Build the 30-element continuous feature vector for one direction (pa → slot_a).
    Returns a (1, 34) array: 4 cat IDs + 30 scaled cont features.
    """
    UNK      = 0
    tier_id  = tier_to_id.get(tier, 0)
    round_id = round_to_id.get(round_name, 0)
    pa_id    = player_to_id.get(pa, UNK)
    pb_id    = player_to_id.get(pb, UNK)
    sa, sb   = player_stats[pa], player_stats[pb]

    cont_raw = np.array([[
        0.0,
        h2h_rate_fn(pa, pb),
        float(sa["is_home"]),
        float(sa["matches_14d"]),
        float(sa["days_since"]),
        float(sa["recent_win_rate"]),
        float(sb["is_home"]),
        float(sb["matches_14d"]),
        float(sb["days_since"]),
        float(sb["recent_win_rate"]),
        float(sa["elo"]),
        float(sb["elo"]),
        float(sa["elo"] - sb["elo"]),
        float(sa["ema_form"]),
        float(sb["ema_form"]),
        h2h_last_fn(pa, pb),
        float(sa["win_streak"]),
        float(sb["win_streak"]),
        float(sa["matches_7d"]),
        float(sb["matches_7d"]),
        float(sa["avg_point_diff"]),
        float(sb["avg_point_diff"]),
        float(sa["avg_games_pm"]),
        float(sb["avg_games_pm"]),
        float(sa["rubber_game_rate"]),
        float(sb["rubber_game_rate"]),
        float(sa["avg_margin"]),
        float(sb["avg_margin"]),
        float(sa["seed"]),
        float(sb["seed"]),
    ]], dtype=np.float32)

    cont_scaled = scaler.transform(cont_raw)
    cat = np.array([[tier_id, round_id, pa_id, pb_id]], dtype=np.float64)
    X = np.hstack([cat, cont_scaled])
    return X


# ------------------------------------------------------------------
# App layout
# ------------------------------------------------------------------

st.title("🏸 BWF Men's Singles — Point-in-Time Historical Backtester")

tournaments = get_all_tournaments()

with st.sidebar:
    st.header("Settings")
    labels = [
        f"{row['tournament_name']} ({row['start_date'].strftime('%Y-%m-%d')})"
        for _, row in tournaments.iterrows()
    ]
    selected_label = st.selectbox("Tournament", labels, index=0)
    selected_idx   = labels.index(selected_label)

    t_row     = tournaments.iloc[selected_idx]
    selected  = t_row["tournament_name"]
    tour_date = t_row["start_date"].strftime("%Y-%m-%d")
    tier      = int(t_row["tier"])

    st.caption(f"Date: {tour_date}  |  Tier: {tier}")
    n_sims  = st.slider("Simulations", min_value=1_000, max_value=50_000,
                        value=10_000, step=1_000)
    run_btn = st.button("▶  Run Simulation", use_container_width=True, type="primary")

# Pre-load the tournament roster (fast — just a DataFrame filter)
model_payload, preprocessors, df = load_resources()
scaler       = preprocessors["scaler"]
player_to_id = preprocessors["player_to_id"]
tier_to_id   = preprocessors["tier_to_id"]
round_to_id  = preprocessors["round_to_id"]

r32_matchups, player_stats = build_time_zero_state(df, tour_date, tier)
h2h_rate_fn, h2h_last_fn  = build_h2h_lookups(df, tour_date)
roster = sorted(player_stats.keys())

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------
tab_sim, tab_shap = st.tabs(["🏆 Monte Carlo Bracket", "🔍 Matchup Explainer"])

# ── Tab 1: Monte Carlo Bracket ─────────────────────────────────────
with tab_sim:
    if not run_btn:
        st.info("Select a tournament in the sidebar and click **▶ Run Simulation** to begin.")
    else:
        if r32_matchups.empty:
            st.error(f"No first-round data found for **{selected}** ({tour_date}). "
                     "The dataset may not cover this tournament.")
        else:
            with st.status(f"Running Point-in-Time Engine for {selected}...",
                           expanded=True) as status:
                st.write("Slicing historical data to pre-tournament state...")
                st.write("Loading Point-in-Time player features...")
                st.write(f"Running {n_sims:,} Monte Carlo simulations...")
                bracket_df, leaderboard = run_simulation(tour_date, tier, n_sims)
                st.write("Preparing results...")
                status.update(
                    label=f"Simulation complete — {selected}",
                    state="complete",
                    expanded=False,
                )

            # Reality Check — annotate the actual tournament winner
            actual_winner = get_actual_winner(df, tour_date)
            if actual_winner:
                leaderboard["Actual Result"] = leaderboard["Player"].apply(
                    lambda p: "🥇 Winner" if p == actual_winner else ""
                )

            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader(f"First Round Bracket ({len(bracket_df)} matchups)")
                styled = (
                    bracket_df.style
                    .format({"P(A wins)": "{:.3f}"})
                    .background_gradient(subset=["P(A wins)"], cmap="RdYlGn",
                                         vmin=0.3, vmax=0.7)
                )
                st.dataframe(styled, use_container_width=True, hide_index=True)

            with col_right:
                st.subheader(f"Championship Probability ({n_sims:,} sims)")
                top_n      = min(16, len(leaderboard))
                chart_data = leaderboard.head(top_n).set_index("Player")["Win %"]
                st.bar_chart(chart_data, horizontal=True)

                display_cols = ["Player", "Win %"]
                if "Actual Result" in leaderboard.columns:
                    display_cols.append("Actual Result")

                st.dataframe(
                    leaderboard[display_cols]
                    .style.format({"Win %": "{:.2f}%"})
                    .background_gradient(subset=["Win %"], cmap="Blues"),
                    use_container_width=True,
                    hide_index=True,
                )
                model_name = model_payload.get("name", model_payload.get("type", "?"))
                st.caption(f"Model: **{model_name}**  |  Val AUC: 0.7872")

# ── Tab 2: Matchup Explainer ────────────────────────────────────────
with tab_shap:
    st.subheader("SHAP Matchup Explainer")
    st.markdown(
        "Select any two players from the tournament roster to see exactly which "
        "features push the model's prediction above or below its baseline."
    )

    if not roster:
        st.warning(f"No bracket data found for **{selected}**. "
                   "Select a different tournament.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            player_a = st.selectbox("Player A", roster, index=0, key="shap_pa")
        with col_b:
            default_b = min(1, len(roster) - 1)
            player_b  = st.selectbox("Player B", roster, index=default_b, key="shap_pb")

        analyze_btn = st.button("🔍 Analyze Matchup", type="primary")

        if player_a == player_b:
            st.warning("Please select two **different** players.")
        elif analyze_btn:
            sa = player_stats[player_a]
            sb = player_stats[player_b]

            # ── Tale of the Tape ──────────────────────────────────
            st.subheader("Tale of the Tape")
            tape_df = pd.DataFrame({
                "Stat": [
                    "Elo Rating", "EMA Form", "Win Streak",
                    "Days Since Last Match", "Matches (Last 14d)",
                    "H2H Win Rate", "Avg Point Diff", "Seed",
                ],
                player_a: [
                    f"{sa['elo']:.0f}",
                    f"{sa['ema_form']:.3f}",
                    f"{sa['win_streak']}",
                    f"{sa['days_since']:.0f}",
                    f"{sa['matches_14d']}",
                    f"{h2h_rate_fn(player_a, player_b):.3f}",
                    f"{sa['avg_point_diff']:+.2f}",
                    f"{int(sa['seed'])}",
                ],
                player_b: [
                    f"{sb['elo']:.0f}",
                    f"{sb['ema_form']:.3f}",
                    f"{sb['win_streak']}",
                    f"{sb['days_since']:.0f}",
                    f"{sb['matches_14d']}",
                    f"{h2h_rate_fn(player_b, player_a):.3f}",
                    f"{sb['avg_point_diff']:+.2f}",
                    f"{int(sb['seed'])}",
                ],
            })
            st.dataframe(tape_df, use_container_width=True, hide_index=True)
            st.divider()

            # ── Build feature vector ──────────────────────────────
            X = build_shap_input(
                player_a, player_b, "first round",
                player_stats, h2h_rate_fn, h2h_last_fn,
                scaler, player_to_id, tier_to_id, round_to_id, tier,
            )

            # ── Win probability (order-invariant) ─────────────────
            p_win = predict_match(
                player_a, player_b, "first round",
                player_stats, h2h_rate_fn, h2h_last_fn,
                scaler, player_to_id, tier_to_id, round_to_id, model_payload, tier,
            )

            st.metric(
                label=f"P({player_a} beats {player_b})",
                value=f"{p_win*100:.1f}%",
                delta=f"{(p_win - 0.5)*100:+.1f}pp vs 50/50",
            )

            # ── SHAP analysis ─────────────────────────────────────
            explainer = get_shap_explainer()
            if explainer is None:
                st.info("SHAP waterfall not available — the best model is a neural "
                        "network (DeepFM). Tree-based SHAP requires a tree model.")
            else:
                with st.spinner("Computing SHAP values..."):
                    _m = (model_payload["model"] if model_payload["type"] == "single"
                          else next(iter(model_payload["models"].values())))
                    _n = getattr(_m, "n_features_in_", X.shape[1])
                    X_shap = X[:, :_n]
                    feat_names_shap = FEATURE_NAMES[:_n]
                    shap_vals = explainer(X_shap)
                    shap_vals.feature_names = feat_names_shap

                st.markdown(
                    f"**Waterfall plot** — how each feature shifts the model's output "
                    f"(log-odds) from the base value to the final prediction for "
                    f"**{player_a}** in the Player A slot."
                )

                shap.plots.waterfall(shap_vals[0], max_display=20, show=False)
                fig = plt.gcf()
                fig.set_size_inches(10, 8)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                with st.expander("Raw feature values"):
                    feat_df = pd.DataFrame({
                        "Feature":      feat_names_shap,
                        "Scaled value": X_shap[0].tolist(),
                        "SHAP value":   shap_vals.values[0].tolist(),
                    })
                    feat_df["SHAP value"]   = feat_df["SHAP value"].round(4)
                    feat_df["Scaled value"] = feat_df["Scaled value"].round(4)
                    feat_df = feat_df.sort_values("SHAP value", key=abs, ascending=False)
                    st.dataframe(feat_df, use_container_width=True, hide_index=True)

            # ── Player Form — Last 5 Matches ──────────────────────
            st.divider()
            st.subheader("Player Form — Last 5 Matches")
            st.caption(
                "Win probability shown is how likely each player would have been "
                "to beat their actual opponent (based on their stats at match time)."
            )

            def build_form_chart(player: str, tour_date_str: str):
                cutoff = pd.Timestamp(tour_date_str)
                mask = (
                    ((df["player_a"] == player) | (df["player_b"] == player)) &
                    (df["start_date"] < cutoff)
                )
                hist = df[mask].sort_values("start_date").tail(5).reset_index(drop=True)
                if hist.empty:
                    return None

                records = []
                for _, mrow in hist.iterrows():
                    m_date   = mrow["start_date"]
                    is_a     = (mrow["player_a"] == player)
                    opp      = mrow["player_b"] if is_a else mrow["player_a"]
                    won      = bool(mrow["player_a_won"] == 1) if is_a else bool(mrow["player_a_won"] == 0)
                    side     = "a" if is_a else "b"
                    opp_side = "b" if is_a else "a"

                    mini_stats = {
                        player: {
                            "is_home":          int(mrow.get(f"player_{side}_is_home", 0)),
                            "matches_14d":      int(mrow.get(f"player_{side}_matches_last_14_days", 0)),
                            "days_since":       float(mrow.get(f"player_{side}_days_since_last_match", 100)),
                            "recent_win_rate":  float(mrow.get(f"player_{side}_recent_win_rate", 0.5)),
                            "elo":              float(mrow.get(f"player_{side}_elo", 1500)),
                            "ema_form":         float(mrow.get(f"player_{side}_ema_form", 0.5)),
                            "win_streak":       int(mrow.get(f"player_{side}_win_streak", 0)),
                            "matches_7d":       int(mrow.get(f"player_{side}_matches_last_7_days", 0)),
                            "avg_point_diff":   float(mrow.get(f"player_{side}_avg_point_diff", 0.0)),
                            "avg_games_pm":     float(mrow.get(f"player_{side}_avg_games_per_match", 2.0)),
                            "rubber_game_rate": float(mrow.get(f"player_{side}_rubber_game_rate", 0.0)),
                            "avg_margin":       float(mrow.get(f"player_{side}_avg_victory_margin", 0.0)),
                            "seed":             float(mrow.get(f"player_{side}_seed", 0.0)),
                        },
                        opp: {
                            "is_home":          int(mrow.get(f"player_{opp_side}_is_home", 0)),
                            "matches_14d":      int(mrow.get(f"player_{opp_side}_matches_last_14_days", 0)),
                            "days_since":       float(mrow.get(f"player_{opp_side}_days_since_last_match", 100)),
                            "recent_win_rate":  float(mrow.get(f"player_{opp_side}_recent_win_rate", 0.5)),
                            "elo":              float(mrow.get(f"player_{opp_side}_elo", 1500)),
                            "ema_form":         float(mrow.get(f"player_{opp_side}_ema_form", 0.5)),
                            "win_streak":       int(mrow.get(f"player_{opp_side}_win_streak", 0)),
                            "matches_7d":       int(mrow.get(f"player_{opp_side}_matches_last_7_days", 0)),
                            "avg_point_diff":   float(mrow.get(f"player_{opp_side}_avg_point_diff", 0.0)),
                            "avg_games_pm":     float(mrow.get(f"player_{opp_side}_avg_games_per_match", 2.0)),
                            "rubber_game_rate": float(mrow.get(f"player_{opp_side}_rubber_game_rate", 0.0)),
                            "avg_margin":       float(mrow.get(f"player_{opp_side}_avg_victory_margin", 0.0)),
                            "seed":             float(mrow.get(f"player_{opp_side}_seed", 0.0)),
                        },
                    }
                    m_tier  = int(mrow.get("tier", tier))
                    m_round = str(mrow.get("round", "first round")).lower()

                    hist_before = df[df["start_date"] < m_date]
                    h2h_r, h2h_l = build_h2h_lookups(
                        hist_before.assign(start_date=hist_before["start_date"]),
                        m_date.strftime("%Y-%m-%d"),
                    )
                    p = predict_match(
                        player, opp, m_round, mini_stats,
                        h2h_r, h2h_l,
                        scaler, player_to_id, tier_to_id, round_to_id, model_payload, m_tier,
                    )
                    records.append({
                        "Date":     m_date.strftime("%Y-%m-%d"),
                        "Opponent": opp,
                        "Win Prob": round(p, 3),
                        "Result":   "W" if won else "L",
                    })
                return pd.DataFrame(records)

            form_col_a, form_col_b = st.columns(2)
            for col_widget, player_name in [(form_col_a, player_a), (form_col_b, player_b)]:
                with col_widget:
                    st.markdown(f"**{player_name}**")
                    form_df = build_form_chart(player_name, tour_date)
                    if form_df is None or form_df.empty:
                        st.caption("No historical match data found before this tournament.")
                    else:
                        fig2, ax = plt.subplots(figsize=(5, 3))
                        colors = ["green" if r == "W" else "red"
                                  for r in form_df["Result"]]
                        ax.plot(range(len(form_df)), form_df["Win Prob"].values,
                                color="steelblue", linewidth=1.5, zorder=1)
                        ax.scatter(range(len(form_df)), form_df["Win Prob"].values,
                                   c=colors, s=60, zorder=2)
                        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
                        ax.set_ylim(0, 1)
                        ax.set_xticks(range(len(form_df)))
                        ax.set_xticklabels(form_df["Date"].tolist(),
                                           rotation=30, ha="right", fontsize=7)
                        ax.set_ylabel("Win Probability")
                        ax.set_title(f"Last {len(form_df)} matches")
                        fig2.tight_layout()
                        st.pyplot(fig2)
                        plt.close(fig2)
                        st.dataframe(
                            form_df.style.applymap(
                                lambda v: "color: green" if v == "W" else "color: red",
                                subset=["Result"],
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
