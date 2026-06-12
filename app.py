import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import pickle
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import streamlit as st
from streamlit_calendar import calendar as st_calendar

from src.dataset import (
    CONT_COLS,
    encode_split,
    fit_preprocessors,
    get_train_val_datasets,
    load_training_frame,
)
from src.simulate import (
    ROUND_ORDER,
    STAT_KEYS,
    _cont_matrix,
    _same_nationality,
    build_fixed_results,
    build_time_zero_state,
    get_n_features,
    predict_match,
    round_sequence,
    run_monte_carlo,
)
from src.train_xgb import load_tuned_params

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

DATA_PATH   = "data/processed/final_training_data.csv"
RAW_PATH    = "data/raw/raw_matches.csv"
CONFIG_PATH = "data/config/tournaments_config.csv"
MODEL_PATH  = "models/best_model.pkl"

FEATURE_NAMES = ["tier_id", "round_id", "player_a_id", "player_b_id"] + CONT_COLS

MIN_PIT_ROWS = 1000   # minimum completed rows needed to train a point-in-time model

TIER_LABELS: dict[int, str] = {
    1500: "Finals",
    1000: "Super 1000",
    750:  "Super 750",
    500:  "Super 500",
    300:  "Super 300",
    100:  "Super 100",
}

ROUND_LABELS = {
    "first round":    "Round 1",
    "second round":   "Round 2",
    "third round":    "Round 3",
    "quarter-finals": "Quarter-finals",
    "semi-finals":    "Semi-finals",
    "final":          "Final",
    "group stage":    "Group Stage",
}

# Country → flag emoji, used for both player nationalities and host countries
COUNTRY_FLAGS: dict = {
    "Algeria": "🇩🇿", "Australia": "🇦🇺", "Austria": "🇦🇹", "Azerbaijan": "🇦🇿",
    "Bahrain": "🇧🇭", "Bangladesh": "🇧🇩", "Belgium": "🇧🇪", "Brazil": "🇧🇷",
    "Bulgaria": "🇧🇬", "Canada": "🇨🇦", "Chile": "🇨🇱", "China": "🇨🇳",
    "Chinese Taipei": "🇹🇼", "Croatia": "🇭🇷", "Cuba": "🇨🇺", "Czech Republic": "🇨🇿",
    "Denmark": "🇩🇰", "Egypt": "🇪🇬", "El Salvador": "🇸🇻",
    "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Estonia": "🇪🇪", "Finland": "🇫🇮", "France": "🇫🇷",
    "Germany": "🇩🇪", "Guatemala": "🇬🇹", "Hong Kong": "🇭🇰", "Hungary": "🇭🇺",
    "India": "🇮🇳", "Indonesia": "🇮🇩", "Ireland": "🇮🇪", "Israel": "🇮🇱",
    "Italy": "🇮🇹", "Japan": "🇯🇵", "Kazakhstan": "🇰🇿", "Korea": "🇰🇷",
    "Lithuania": "🇱🇹", "Luxembourg": "🇱🇺", "Macau": "🇲🇴", "Malaysia": "🇲🇾",
    "Mauritius": "🇲🇺", "Mexico": "🇲🇽", "Mongolia": "🇲🇳", "Myanmar": "🇲🇲",
    "Nepal": "🇳🇵", "Netherlands": "🇳🇱", "New Zealand": "🇳🇿", "Norway": "🇳🇴",
    "Philippines": "🇵🇭", "Poland": "🇵🇱", "Portugal": "🇵🇹",
    "Republic of Ireland": "🇮🇪", "Russia": "🇷🇺", "Saudi Arabia": "🇸🇦",
    "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Singapore": "🇸🇬", "Slovakia": "🇸🇰",
    "Slovenia": "🇸🇮", "South Korea": "🇰🇷", "Spain": "🇪🇸", "Sri Lanka": "🇱🇰",
    "Sweden": "🇸🇪", "Switzerland": "🇨🇭", "Syria": "🇸🇾", "Taiwan": "🇹🇼",
    "Thailand": "🇹🇭", "Trinidad and Tobago": "🇹🇹", "Turkey": "🇹🇷",
    "Ukraine": "🇺🇦", "United Arab Emirates": "🇦🇪", "United Kingdom": "🇬🇧",
    "United States": "🇺🇸", "Vietnam": "🇻🇳", "Zambia": "🇿🇲",
}

TODAY = date.today()


# ------------------------------------------------------------------
# Cached resource loaders
# ------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_df() -> pd.DataFrame:
    """Full mirrored dataset, pending rows included (needed for live draws)."""
    return load_training_frame(DATA_PATH, drop_pending=False)


@st.cache_resource(show_spinner=False)
def load_pretrained():
    """Preloaded model (refreshed by the scheduled GitHub Actions retrain)
    plus the global preprocessors it was trained with."""
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    _, _, _, preprocessors = get_train_val_datasets(DATA_PATH)
    return payload, preprocessors


@st.cache_resource(max_entries=4, show_spinner=False)
def get_point_in_time_model(tour_date: str):
    """
    Train an XGBoost on every completed match strictly before tour_date,
    with vocab + scaler fit on that same slice — a true point-in-time model.
    Returns (payload, preprocessors), or None when there is too little history.
    """
    import xgboost as xgb

    df = load_df()
    train_df = df[(df["start_date"] < pd.Timestamp(tour_date)) & (df["is_pending"] == 0)]
    if len(train_df) < MIN_PIT_ROWS:
        return None

    preprocessors, _ = fit_preprocessors(train_df)
    cat, cont, y = encode_split(train_df, preprocessors)
    X = np.hstack([cat.astype(np.float64), cont])

    params, _ = load_tuned_params()
    model = xgb.XGBClassifier(**params, random_state=42, eval_metric="auc",
                              tree_method="hist", verbosity=0)
    model.fit(X, y, verbose=False)

    payload = {
        "type":            "single",
        "model":           model,
        "name":            "xgb (point-in-time)",
        "trained_through": str(train_df["start_date"].max().date()),
        "n_train_rows":    int(len(train_df)),
    }
    return payload, preprocessors


@st.cache_data(show_spinner=False)
def load_player_nats() -> dict:
    """player_name → nationality, from the raw scrape (used for the
    same_nationality feature and for flag emojis)."""
    try:
        raw = pd.read_csv(RAW_PATH)
    except Exception:
        return {}
    result = {}
    for side in ("a", "b"):
        sub = raw[[f"player_{side}", f"player_{side}_nat"]].dropna()
        for name, nat in sub.itertuples(index=False):
            result.setdefault(name, nat)
    return result


@st.cache_resource(max_entries=4, show_spinner=False)
def get_shap_explainer(model_key: str, _payload=None):
    """TreeExplainer for the active model. model_key makes the cache entry
    unique per model; _payload is excluded from hashing by convention."""
    m = _payload["model"] if _payload["type"] == "single" else None
    if m is None and _payload.get("models"):
        for name in ("xgb", "lgbm", "catboost"):
            if name in _payload["models"]:
                m = _payload["models"][name]
                break
    try:
        return shap.TreeExplainer(m) if m is not None else None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_all_tournaments() -> pd.DataFrame:
    cfg = pd.read_csv(CONFIG_PATH)
    cfg["start_date"] = pd.to_datetime(cfg["start_date"], errors="coerce")
    cfg = cfg.dropna(subset=["start_date"]).sort_values("start_date", ascending=False)
    return cfg[["tournament_name", "tier", "start_date", "host_country"]].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _get_h2h_hist(tour_date: str) -> pd.DataFrame:
    """Completed-match history strictly before the tournament (for H2H)."""
    df = load_df()
    hist = df[(df["start_date"] < pd.Timestamp(tour_date)) & (df["is_pending"] == 0)].copy()
    return hist.sort_values("start_date").reset_index(drop=True)


def _make_h2h_fns(hist: pd.DataFrame):
    """Memoised H2H closures over a pre-filtered historical slice."""
    rate_cache: dict = {}
    last_cache: dict = {}

    def h2h_rate(pa, pb):
        key = (pa, pb)
        if key not in rate_cache:
            rows_a = hist[(hist["player_a"] == pa) & (hist["player_b"] == pb)]
            rows_b = hist[(hist["player_a"] == pb) & (hist["player_b"] == pa)]
            wins  = rows_a["player_a_won"].sum() + (1 - rows_b["player_a_won"]).sum()
            total = len(rows_a) + len(rows_b)
            rate_cache[key] = float(wins / total) if total > 0 else 0.5
        return rate_cache[key]

    def h2h_last(pa, pb):
        key = (pa, pb)
        if key not in last_cache:
            meetings = hist[
                ((hist["player_a"] == pa) & (hist["player_b"] == pb)) |
                ((hist["player_a"] == pb) & (hist["player_b"] == pa))
            ].sort_values("start_date")
            if meetings.empty:
                last_cache[key] = 0.5
            else:
                lr = meetings.iloc[-1]
                last_cache[key] = (
                    float(lr["player_a_won"]) if lr["player_a"] == pa
                    else float(1 - lr["player_a_won"])
                )
        return last_cache[key]

    return h2h_rate, h2h_last


@st.cache_data(show_spinner=False)
def _get_cached_tournament_state(tour_date: str, tier: int):
    """First-round matchups + Day-1 player stats per (tour_date, tier)."""
    return build_time_zero_state(load_df(), tour_date, tier)


@st.cache_data(show_spinner=False)
def get_tournament_rows(tour_date: str) -> pd.DataFrame:
    """All of a tournament's rows (completed + pending), mirrored dups removed."""
    df = load_df()
    day = df[df["start_date"] == pd.Timestamp(tour_date)]
    seen, keep = set(), []
    for _, row in day.iterrows():
        key = (row["round"], frozenset((row["player_a"], row["player_b"])))
        if key not in seen:
            seen.add(key)
            keep.append(row)
    return pd.DataFrame(keep).reset_index(drop=True) if keep else pd.DataFrame()


# ------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------

def format_name(name: str) -> str:
    # NAT_MAP is bound once at bootstrap (st.cache_data returns a fresh copy
    # per call, so calling load_player_nats() here would copy the dict every time)
    flag = COUNTRY_FLAGS.get(NAT_MAP.get(name, ""), "🏸")
    return f"{flag} {name}"


def format_tier(tier: int) -> str:
    return TIER_LABELS.get(tier, f"Tier {tier}")


def get_actual_winner(day_rows: pd.DataFrame):
    finals = day_rows[(day_rows["round"] == "final") & (day_rows["is_pending"] == 0)]
    if finals.empty:
        return None
    r = finals.iloc[0]
    return r["player_a"] if r["player_a_won"] == 1 else r["player_b"]


def build_calendar_events(all_tours, selected_key, today) -> list:
    """FullCalendar event dicts: selected = green, past = gray, upcoming = blue."""
    events = []
    for _, r in all_tours.iterrows():
        tour_dt  = r["start_date"]
        tour_key = tour_dt.strftime("%Y-%m-%d")
        end_key  = (tour_dt + pd.Timedelta(days=6)).strftime("%Y-%m-%d")
        flag     = COUNTRY_FLAGS.get(str(r["host_country"]), "🌐")
        is_past  = tour_dt.date() < today
        is_sel   = tour_key == selected_key

        if is_sel:
            bg, border = "#2e7d32", "#1b5e20"
        elif is_past:
            bg, border = "#9e9e9e", "#757575"
        else:
            bg, border = "#1e88e5", "#1565c0"

        title = f"★ {flag} {r['tournament_name']}" if is_sel else f"{flag} {r['tournament_name']}"
        events.append({
            "id":              tour_key,
            "title":           title,
            "start":           tour_key,
            "end":             end_key,
            "backgroundColor": bg,
            "borderColor":     border,
        })
    return events


def build_model_input(pa, pb, round_name, player_stats, h2h_rate_fn, h2h_last_fn,
                      scaler, player_to_id, tier_to_id, round_to_id, tier, nat_map):
    """(1, 34) feature array with pa in the player_a slot, plus the human-
    readable (unscaled) value of every feature for SHAP display."""
    sa, sb = player_stats[pa], player_stats[pb]
    SA = np.array([[sa[k] for k in STAT_KEYS]], dtype=np.float64)
    SB = np.array([[sb[k] for k in STAT_KEYS]], dtype=np.float64)
    cont_raw = _cont_matrix(
        SA, SB,
        np.array([sa["elo"]]), np.array([sb["elo"]]),
        np.array([sa["ema_form"]]), np.array([sb["ema_form"]]),
        np.array([_same_nationality(pa, pb, nat_map)]),
        np.array([h2h_rate_fn(pa, pb)]),
        np.array([h2h_last_fn(pa, pb)]),
    )
    cont_scaled = scaler.transform(cont_raw)
    cat = np.array([[tier_to_id.get(tier, 0), round_to_id.get(round_name, 0),
                     player_to_id.get(pa, 0), player_to_id.get(pb, 0)]], dtype=np.float64)
    display_vals = (
        [format_tier(tier), ROUND_LABELS.get(round_name, round_name.title()), pa, pb]
        + [float(v) for v in cont_raw[0]]
    )
    return np.hstack([cat, cont_scaled]), display_vals


# Human-readable names for the SHAP chart
_PRETTY_GLOBAL = {
    "tier_id":             "Tournament tier",
    "round_id":            "Round",
    "same_nationality":    "Same nationality",
    "h2h_win_rate_a_vs_b": "Head-to-head win rate",
    "elo_diff":            "Elo difference",
    "h2h_last_winner":     "Won the last head-to-head",
}
_PRETTY_STAT = {
    "is_home":               "home advantage",
    "matches_last_14_days":  "matches in last 14 days",
    "days_since_last_match": "days since last match",
    "recent_win_rate":       "win rate (last 180 days)",
    "elo":                   "Elo rating",
    "ema_form":              "form (EMA)",
    "win_streak":            "win streak",
    "matches_last_7_days":   "matches in last 7 days",
    "avg_point_diff":        "avg point differential",
    "avg_games_per_match":   "avg games per match",
    "rubber_game_rate":      "deciding-game rate",
    "avg_victory_margin":    "avg victory margin",
    "seed":                  "seed",
}


def pretty_feature(name: str, pa: str, pb: str) -> str:
    if name == "player_a_id":
        return f"{pa} (who they are)"
    if name == "player_b_id":
        return f"{pb} (who they are)"
    if name in _PRETTY_GLOBAL:
        return _PRETTY_GLOBAL[name]
    for prefix, who in (("player_a_", pa), ("player_b_", pb)):
        if name.startswith(prefix):
            key = name[len(prefix):]
            return f"{who} — {_PRETTY_STAT.get(key, key.replace('_', ' '))}"
    return name


def _fmt_val(v) -> str:
    if isinstance(v, str):
        return v
    if float(v).is_integer() and abs(v) < 10_000:
        return f"{int(v)}"
    if abs(v) >= 100:
        return f"{v:,.0f}"
    return f"{v:.2f}"


def build_shap_figure(shap_row, base_value, feat_names, display_vals,
                      pa, pb, top_n: int = 12) -> go.Figure:
    """Theme-aware horizontal bar chart of SHAP contributions: blue bars push
    the prediction toward pa, red toward pb; labels show raw feature values."""
    order = np.argsort(np.abs(shap_row))[::-1]
    top, rest = order[:top_n], order[top_n:]

    labels = [f"{pretty_feature(feat_names[i], pa, pb)}  =  {_fmt_val(display_vals[i])}"
              for i in top]
    values = [float(shap_row[i]) for i in top]
    if len(rest):
        labels.append(f"{len(rest)} other features")
        values.append(float(shap_row[rest].sum()))

    labels, values = labels[::-1], values[::-1]   # biggest bar on top
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line_width=0),
        text=[f"{v:+.2f}" for v in values],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>impact: %{x:+.3f}<extra></extra>",
    ))
    p_base = 1.0 / (1.0 + np.exp(-base_value))
    p_out  = 1.0 / (1.0 + np.exp(-(base_value + float(shap_row.sum()))))
    fig.update_layout(
        title=dict(
            text=f"Model view from the Player A slot: "
                 f"<b>{p_out*100:.1f}%</b> for {pa} (baseline {p_base*100:.1f}%)",
            font=dict(size=14),
        ),
        height=34 * len(labels) + 130,
        margin=dict(l=10, r=55, t=55, b=10),
        xaxis=dict(title="impact on prediction (log-odds)", zeroline=False),
        yaxis=dict(automargin=True),
        showlegend=False,
        bargap=0.25,
    )
    fig.add_vline(x=0, line_width=1, line_color="rgba(128,128,128,0.6)")
    return fig


def render_bracket_figure(round_winners: dict[str, list[str]]) -> go.Figure:
    """Plotly table for the most-likely bracket path (post-simulation)."""
    LABELS = {
        "first round":    "R1",
        "second round":   "R2",
        "third round":    "R3",
        "quarter-finals": "QF",
        "semi-finals":    "SF",
        "final":          "🏆 Final",
    }
    rounds_present = [r for r in ROUND_ORDER if r in round_winners]
    headers    = [LABELS.get(r, r.title()) for r in rounds_present]
    max_rows   = max(len(round_winners[r]) for r in rounds_present)

    col_data, col_colors = [], []
    for rnd in rounds_present:
        players  = [format_name(p) for p in round_winners[rnd]]
        n        = len(players)
        is_final = rnd == rounds_present[-1]
        colors   = [
            ("#fff3cd" if is_final else ("#dceefb" if j % 2 == 0 else "#ffffff"))
            if j < n else "#f5f5f5"
            for j in range(max_rows)
        ]
        col_data.append(players + [""] * (max_rows - n))
        col_colors.append(colors)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in headers],
            fill_color="#1a3a5c", font=dict(color="white", size=13),
            align="center", height=34,
        ),
        cells=dict(
            values=col_data, fill_color=col_colors,
            align="center", font=dict(size=11), height=26,
        ),
    )])
    fig.update_layout(margin=dict(l=0, r=0, t=4, b=0),
                      height=max(180, 30 * max_rows + 70))
    return fig


def build_radar_chart(pa: str, pb: str, player_stats: dict) -> go.Figure:
    extractors = {
        "Elo":       lambda s: s["elo"],
        "Form":      lambda s: s["ema_form"],
        "Streak":    lambda s: s["win_streak"],
        "Pt Diff":   lambda s: s["avg_point_diff"],
        "Freshness": lambda s: -s["days_since"],
    }

    def _norm(key, player):
        vals = [extractors[key](s) for s in player_stats.values()]
        lo, hi = min(vals), max(vals)
        v = extractors[key](player_stats[player])
        return (v - lo) / (hi - lo) if hi > lo else 0.5

    dims = list(extractors.keys())
    fig  = go.Figure()
    for player, color, fill in [
        (pa, "#1f77b4", "rgba(31,119,180,0.20)"),
        (pb, "#d62728", "rgba(214,39,40,0.20)"),
    ]:
        r_vals = [_norm(d, player) for d in dims] + [_norm(dims[0], player)]
        fig.add_trace(go.Scatterpolar(
            r=r_vals, theta=dims + [dims[0]], fill="toself",
            name=format_name(player),
            line=dict(color=color), fillcolor=fill,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, legend=dict(x=0.8, y=1.1),
        margin=dict(l=30, r=30, t=30, b=30), height=380,
    )
    return fig


def render_match_explainer(pa, pb, round_name, ctx):
    """Win-prob metric + SHAP waterfall for one matchup under the active model."""
    p_win = predict_match(
        pa, pb, round_name, ctx["player_stats"],
        ctx["h2h_rate_fn"], ctx["h2h_last_fn"],
        ctx["scaler"], ctx["player_to_id"], ctx["tier_to_id"], ctx["round_to_id"],
        ctx["payload"], ctx["tier"], ctx["nat_map"],
    )
    st.metric(
        label=f"P({format_name(pa)} beats {format_name(pb)})",
        value=f"{p_win * 100:.1f}%",
        delta=f"{(p_win - 0.5) * 100:+.1f}pp vs 50/50",
    )

    explainer = get_shap_explainer(ctx["model_key"], _payload=ctx["payload"])
    if explainer is None:
        st.info("SHAP unavailable — no tree model loaded.")
        return

    X, display_vals = build_model_input(
        pa, pb, round_name, ctx["player_stats"],
        ctx["h2h_rate_fn"], ctx["h2h_last_fn"],
        ctx["scaler"], ctx["player_to_id"], ctx["tier_to_id"], ctx["round_to_id"],
        ctx["tier"], ctx["nat_map"],
    )
    with st.spinner("Computing SHAP values…"):
        n = get_n_features(ctx["payload"]) or X.shape[1]
        X_shap     = X[:, :n]
        feat_names = FEATURE_NAMES[:n]
        shap_vals  = explainer(X_shap)

    base = float(np.ravel(shap_vals.base_values)[0])
    fig = build_shap_figure(
        shap_vals.values[0], base, feat_names, display_vals[:n], pa, pb,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"🔵 pushes the prediction toward **{format_name(pa)}** · "
        f"🔴 toward **{format_name(pb)}**. Values shown are the real "
        f"(unscaled) inputs. The headline probability above averages both "
        f"slot orders; this chart explains the Player A slot only."
    )

    with st.expander("All feature values"):
        feat_df = pd.DataFrame({
            "Feature":    [pretty_feature(f, pa, pb) for f in feat_names],
            "Value":      [_fmt_val(v) for v in display_vals[:n]],
            "SHAP value": shap_vals.values[0].round(4).tolist(),
        }).sort_values("SHAP value", key=abs, ascending=False)
        st.dataframe(feat_df, use_container_width=True, hide_index=True)


def build_form_chart(player: str, tour_date_str: str, ctx):
    """Re-predict the player's last 5 completed matches point-in-time."""
    df = load_df()
    cutoff = pd.Timestamp(tour_date_str)
    mask = (
        ((df["player_a"] == player) | (df["player_b"] == player)) &
        (df["start_date"] < cutoff) & (df["is_pending"] == 0)
    )
    hist = df[mask].sort_values("start_date").tail(5).reset_index(drop=True)
    if hist.empty:
        return None

    records = []
    for _, mrow in hist.iterrows():
        m_date = mrow["start_date"]
        is_a   = mrow["player_a"] == player
        opp    = mrow["player_b"] if is_a else mrow["player_a"]
        won    = bool(mrow["player_a_won"] == 1) if is_a else bool(mrow["player_a_won"] == 0)
        side, os_ = ("a", "b") if is_a else ("b", "a")

        def _stats(s):
            return {
                "is_home":          int(mrow.get(f"player_{s}_is_home", 0)),
                "matches_14d":      int(mrow.get(f"player_{s}_matches_last_14_days", 0)),
                "days_since":       float(mrow.get(f"player_{s}_days_since_last_match", 100)),
                "recent_win_rate":  float(mrow.get(f"player_{s}_recent_win_rate", 0.5)),
                "elo":              float(mrow.get(f"player_{s}_elo", 1500)),
                "ema_form":         float(mrow.get(f"player_{s}_ema_form", 0.5)),
                "win_streak":       int(mrow.get(f"player_{s}_win_streak", 0)),
                "matches_7d":       int(mrow.get(f"player_{s}_matches_last_7_days", 0)),
                "avg_point_diff":   float(mrow.get(f"player_{s}_avg_point_diff", 0.0)),
                "avg_games_pm":     float(mrow.get(f"player_{s}_avg_games_per_match", 2.0)),
                "rubber_game_rate": float(mrow.get(f"player_{s}_rubber_game_rate", 0.0)),
                "avg_margin":       float(mrow.get(f"player_{s}_avg_victory_margin", 0.0)),
                "seed":             float(mrow.get(f"player_{s}_seed", 0.0)),
            }

        mini = {player: _stats(side), opp: _stats(os_)}
        h2h_r, h2h_l = _make_h2h_fns(_get_h2h_hist(m_date.strftime("%Y-%m-%d")))
        p = predict_match(
            player, opp, str(mrow.get("round", "first round")).lower(), mini,
            h2h_r, h2h_l,
            ctx["scaler"], ctx["player_to_id"], ctx["tier_to_id"], ctx["round_to_id"],
            ctx["payload"], int(mrow.get("tier", ctx["tier"])), ctx["nat_map"],
        )
        records.append({
            "Date":     m_date.strftime("%Y-%m-%d"),
            "Opponent": format_name(opp),
            "Win Prob": round(p, 3),
            "Result":   "W" if won else "L",
        })
    return pd.DataFrame(records)


# ------------------------------------------------------------------
# App bootstrap
# ------------------------------------------------------------------

st.set_page_config(page_title="ShuttleCast", page_icon="🏸", layout="wide")
st.markdown(
    "<style>div[data-testid='stStatusWidget']{display:none!important}</style>",
    unsafe_allow_html=True,
)

all_tours = get_all_tournaments()
NAT_MAP   = load_player_nats()

# Default to the current or most recent tournament (next week at the latest) —
# the furthest-future entries usually have no draw to show yet.
_near = all_tours[all_tours["start_date"] <= pd.Timestamp(TODAY) + pd.Timedelta(days=7)]
_default_row = _near.iloc[0] if not _near.empty else all_tours.iloc[0]
_default_key = _default_row["start_date"].strftime("%Y-%m-%d")

if "sim_results"       not in st.session_state:
    st.session_state["sim_results"]       = {}
if "selected_tour_key" not in st.session_state:
    st.session_state["selected_tour_key"] = _default_key
if "cal_initial_date"  not in st.session_state:
    st.session_state["cal_initial_date"]  = _default_row["start_date"].strftime("%Y-%m-01")

st.title("🏸 ShuttleCast")

tour_date = st.session_state["selected_tour_key"]
t_match   = all_tours[all_tours["start_date"] == pd.Timestamp(tour_date)]
if t_match.empty:
    t_match = all_tours.iloc[0:1]
t_row_sel = t_match.iloc[0]
selected  = t_row_sel["tournament_name"]
tier      = int(t_row_sel["tier"])
host_flag = COUNTRY_FLAGS.get(str(t_row_sel["host_country"]), "🌐")

# ------------------------------------------------------------------
# Sidebar — calendar + tournament picker
# ------------------------------------------------------------------

with st.sidebar:
    _years     = sorted(all_tours["start_date"].dt.year.unique().tolist())
    _mon_names = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
    _cur_init  = st.session_state["cal_initial_date"]
    _cur_year  = int(_cur_init[:4])
    _cur_mo    = int(_cur_init[5:7]) - 1

    _jy, _jm = st.columns(2)
    with _jy:
        _sel_year = st.selectbox(
            "Year", _years,
            index=_years.index(_cur_year) if _cur_year in _years else len(_years) - 1,
            key="nav_year", label_visibility="collapsed",
        )
    with _jm:
        _sel_mon_name = st.selectbox(
            "Month", _mon_names, index=_cur_mo,
            key="nav_month", label_visibility="collapsed",
        )
    _new_initial = f"{_sel_year}-{_mon_names.index(_sel_mon_name) + 1:02d}-01"
    if _new_initial != _cur_init:
        st.session_state["cal_initial_date"] = _new_initial
        st.rerun()

    cal_events = build_calendar_events(
        all_tours, st.session_state["selected_tour_key"], TODAY
    )
    cal_options = {
        "initialView":   "dayGridMonth",
        "initialDate":   st.session_state["cal_initial_date"],
        "headerToolbar": {"left": "prev,next today", "center": "title", "right": ""},
        "height":       420,
        "navLinks":     False,
        "editable":     False,
        "selectable":   False,
        "dayMaxEvents": 2,
    }
    cal_state = st_calendar(
        events=cal_events,
        options=cal_options,
        callbacks=["eventClick"],
        custom_css="""
            .fc-event { cursor: pointer; font-size: 10px; }
            .fc-toolbar-title { font-size: 1rem !important; }
            .fc-button { font-size: 0.72rem !important; padding: 2px 6px !important; }
            .fc-daygrid-event-dot { display: none; }
            .fc-daygrid-day-frame { min-height: 48px !important; }
            .fc-daygrid-day-top { padding: 1px 2px !important; }
            .fc-daygrid-event { margin: 0 !important; }
        """,
        key=f"bwf_cal_{st.session_state['cal_initial_date']}",
    )

    if cal_state and cal_state.get("eventClick"):
        clicked_id = cal_state["eventClick"]["event"]["id"]
        if clicked_id != st.session_state["selected_tour_key"]:
            clicked_dt = pd.Timestamp(clicked_id)
            st.session_state["cal_initial_date"]  = clicked_dt.strftime("%Y-%m-01")
            st.session_state["selected_tour_key"] = clicked_id
            st.rerun()

    _is_up  = pd.Timestamp(tour_date).date() > TODAY
    _status = "🔮 Upcoming" if _is_up else "📜 Past / Live"
    st.markdown(
        f"<div style='background:#f0faf0;border-left:4px solid #2e7d32;"
        f"border-radius:4px;padding:10px 12px;margin:8px 0'>"
        f"<div style='font-size:0.72rem;color:#555;font-weight:600;letter-spacing:.04em'>"
        f"SELECTED TOURNAMENT</div>"
        f"<div style='font-size:1.05rem;font-weight:700;margin:3px 0'>"
        f"{host_flag} {selected}</div>"
        f"<div style='font-size:0.8rem;color:#444'>{format_tier(tier)}</div>"
        f"<div style='font-size:0.75rem;color:#777;margin-top:2px'>"
        f"{tour_date} &nbsp;·&nbsp; {_status}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.72rem;color:#999;margin:2px 0 6px'>"
        "🔵 Upcoming &nbsp;·&nbsp; ⚫ Past &nbsp;·&nbsp; 🟢 Selected"
        "</div>",
        unsafe_allow_html=True,
    )
    n_sims  = st.slider("Monte Carlo Simulations", 100, 10_000, 1_000, 100)
    run_btn = st.button("▶ Run Simulation", use_container_width=True, type="primary")

# ------------------------------------------------------------------
# Active model: preloaded for upcoming tournaments, point-in-time for past
# ------------------------------------------------------------------

nat_map = NAT_MAP
is_future = pd.Timestamp(tour_date).date() > TODAY

if is_future:
    payload, preprocessors = load_pretrained()
    model_key  = f"pretrained|{payload.get('trained_through', 'static')}"
    model_desc = (
        f"**Preloaded XGBoost** — trained through "
        f"{payload.get('trained_through', '2025 (initial release)')}"
    )
else:
    with st.spinner(f"Training point-in-time model on all matches before {tour_date} "
                    f"(cached after first run)…"):
        pit = get_point_in_time_model(tour_date)
    if pit is None:
        payload, preprocessors = load_pretrained()
        model_key  = f"pretrained|{payload.get('trained_through', 'static')}"
        model_desc = ("**Preloaded XGBoost** — too little history before this "
                      "tournament for a point-in-time model")
    else:
        payload, preprocessors = pit
        model_key  = f"pit|{tour_date}"
        model_desc = (
            f"**Point-in-time XGBoost** — trained on "
            f"{payload['n_train_rows']:,} rows strictly before {tour_date} (no leakage)"
        )

scaler       = preprocessors["scaler"]
player_to_id = preprocessors["player_to_id"]
tier_to_id   = preprocessors["tier_to_id"]
round_to_id  = preprocessors["round_to_id"]

r1_matchups, player_stats = _get_cached_tournament_state(tour_date, tier)
h2h_rate_fn, h2h_last_fn  = _make_h2h_fns(_get_h2h_hist(tour_date))
day_rows = get_tournament_rows(tour_date)
# TBD qualifiers hold bracket slots but aren't analyzable players
roster   = sorted(n for n in player_stats if not n.startswith("TBD ("))

n_done    = int((day_rows["is_pending"] == 0).sum()) if not day_rows.empty else 0
n_pending = int((day_rows["is_pending"] == 1).sum()) if not day_rows.empty else 0
# "Live" = mixed played/unplayed matches AND the event is actually this week.
# (Old tournaments can carry a few pending rows too — group-stage matches where
# Wikipedia never marks a per-match winner.)
_days_ago = (TODAY - pd.Timestamp(tour_date).date()).days
is_live   = n_done > 0 and n_pending > 0 and 0 <= _days_ago <= 14

ctx = {
    "payload": payload, "model_key": model_key,
    "scaler": scaler, "player_to_id": player_to_id,
    "tier_to_id": tier_to_id, "round_to_id": round_to_id,
    "player_stats": player_stats,
    "h2h_rate_fn": h2h_rate_fn, "h2h_last_fn": h2h_last_fn,
    "tier": tier, "nat_map": nat_map,
}

st.markdown(f"🧠 {model_desc}")

# ------------------------------------------------------------------
# Tabs
# ------------------------------------------------------------------

tab_draw, tab_sim, tab_matchup = st.tabs(
    ["📋 Draw & Predictions", "🎲 Monte Carlo", "⚡ Matchup Analyzer"]
)

# ── Tab 1: Draw & Predictions ──────────────────────────────────────

with tab_draw:
    if day_rows.empty:
        st.info(
            f"No draw available yet for **{selected}** ({tour_date}).  \n"
            "Draws appear automatically once published on Wikipedia and picked "
            "up by the weekly data refresh."
        )
    else:
        if is_live:
            st.subheader(f"{host_flag} {selected} · 🔴 In progress")
        elif n_pending and not n_done:
            st.subheader(f"{host_flag} {selected} · 🔮 Draw published")
        else:
            st.subheader(f"{host_flag} {selected} · ✅ Completed")
            if n_pending:
                st.caption(f"⚠️ {n_pending} matches have no recorded winner on "
                           "Wikipedia (typically group-stage rows) and are "
                           "shown with model probabilities instead.")

        winner = get_actual_winner(day_rows)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Players", len(roster))
        m2.metric("Matches played", n_done)
        m3.metric("To play", n_pending)
        m4.metric("Champion", format_name(winner) if winner else "TBD")

        _present = set(day_rows["round"])
        rounds_present = ([r for r in ROUND_ORDER if r in _present]
                          + sorted(_present - set(ROUND_ORDER)))  # group stage etc.
        for rnd in rounds_present:
            rnd_rows = day_rows[day_rows["round"] == rnd]
            table = []
            for _, m in rnd_rows.iterrows():
                pa, pb = m["player_a"], m["player_b"]
                prob, result = None, ""
                if int(m["is_pending"]) == 0:
                    w = pa if m["player_a_won"] == 1 else pb
                    result = f"🏆 {format_name(w)}"
                elif pa in player_stats and pb in player_stats:
                    p = predict_match(
                        pa, pb, rnd, player_stats, h2h_rate_fn, h2h_last_fn,
                        scaler, player_to_id, tier_to_id, round_to_id,
                        payload, tier, nat_map,
                    )
                    prob = round(p * 100, 1)
                    result = "🔮 to play"
                table.append({
                    "Player A":  format_name(pa),
                    "Player B":  format_name(pb),
                    "P(A wins)": prob,
                    "Result":    result,
                })
            n_open = sum(1 for t in table if t["P(A wins)"] is not None)
            label = f"**{ROUND_LABELS.get(rnd, rnd.title())}** ({len(table)} matches"
            label += f", {n_open} to play)" if n_open else ")"
            with st.expander(label, expanded=(rnd == rounds_present[0] or n_open > 0)):
                st.dataframe(
                    pd.DataFrame(table),
                    column_config={
                        "P(A wins)": st.column_config.ProgressColumn(
                            "P(A wins)", format="%.1f%%",
                            min_value=0, max_value=100,
                        ),
                    },
                    use_container_width=True, hide_index=True,
                )

        # ── Per-match feature attribution ───────────────────────────
        st.divider()
        st.subheader("🔍 Explain a match")
        pending_first = pd.concat([
            day_rows[day_rows["is_pending"] == 1],
            day_rows[day_rows["is_pending"] == 0],
        ])
        options = [
            (m["round"], m["player_a"], m["player_b"])
            for _, m in pending_first.iterrows()
            if m["player_a"] in player_stats and m["player_b"] in player_stats
        ]
        if options:
            sel = st.selectbox(
                "Match", options,
                format_func=lambda o: (
                    f"{ROUND_LABELS.get(o[0], o[0].title())}: {o[1]} vs {o[2]}"
                ),
            )
            render_match_explainer(sel[1], sel[2], sel[0], ctx)
        else:
            st.caption("No matches with full player data available to explain.")

# ── Tab 2: Monte Carlo ─────────────────────────────────────────────

with tab_sim:
    if not r1_matchups.empty and n_done > 0:
        # For live tournaments, condition simulations on real results by
        # default; for finished ones, default to the pure pre-tournament
        # forecast (otherwise the "simulation" trivially returns the winner).
        condition = st.toggle(
            f"Condition on the {n_done} match results already played",
            value=is_live, key=f"cond_{tour_date}",
            help="ON: already-played matches are fixed to their real outcome and "
                 "only unplayed matches are simulated. OFF: pure pre-tournament "
                 "forecast from Day-1 player stats.",
        )
    else:
        condition = False
    sim_key = f"{tour_date}|{tier}|{n_sims}|{condition}"

    if r1_matchups.empty:
        st.info(
            f"No first-round draw for **{selected}** yet — simulation needs the "
            "bracket. It appears automatically once the draw is published."
        )
    elif run_btn:
        fixed = build_fixed_results(day_rows) if condition else {}
        n_fixed = len(fixed)

        progress_ph = st.progress(0, text="Preparing bracket…")
        t0  = time.time()
        rng = np.random.default_rng(42)

        def _cb(round_name, i, n):
            progress_ph.progress(
                i / n, text=f"Simulated **{ROUND_LABELS.get(round_name, round_name)}** "
                            f"across {n_sims:,} brackets ({i}/{n} rounds)"
            )

        win_counts = run_monte_carlo(
            n_sims, r1_matchups, player_stats,
            h2h_rate_fn, h2h_last_fn,
            scaler, player_to_id, tier_to_id, round_to_id,
            payload, rng, tier,
            nat_map=nat_map, fixed_results=fixed, progress_cb=_cb,
        )
        elapsed = time.time() - t0
        progress_ph.empty()

        leaderboard = (
            pd.DataFrame(win_counts.items(), columns=["Player", "Wins"])
            .sort_values("Wins", ascending=False).reset_index(drop=True)
        )
        leaderboard["Win %"] = (leaderboard["Wins"] / n_sims * 100).round(2)
        actual_winner = get_actual_winner(day_rows)
        if actual_winner:
            leaderboard["Actual Result"] = leaderboard["Player"].apply(
                lambda p: "🥇 Winner" if p == actual_winner else ""
            )

        # Most-likely bracket path: greedy winner per match, conditioned on
        # real results where they exist
        round_winners: dict[str, list[str]] = {}
        current = [(r["player_a"], r["player_b"]) for _, r in r1_matchups.iterrows()]
        for rnd in round_sequence(len(r1_matchups)):
            if not current:
                break
            winners = []
            for pa, pb in current:
                real = fixed.get((rnd, frozenset((pa, pb))))
                if real is not None:
                    winners.append(real)
                else:
                    p = predict_match(pa, pb, rnd, player_stats,
                                      h2h_rate_fn, h2h_last_fn,
                                      scaler, player_to_id, tier_to_id, round_to_id,
                                      payload, tier, nat_map)
                    winners.append(pa if p >= 0.5 else pb)
            round_winners[rnd] = winners
            current = list(zip(winners[::2], winners[1::2]))

        st.session_state["sim_results"][sim_key] = {
            "leaderboard":   leaderboard,
            "round_winners": round_winners,
            "actual_winner": actual_winner,
            "elapsed":       elapsed,
            "n_fixed":       n_fixed,
        }
        st.rerun()

    elif sim_key in st.session_state["sim_results"]:
        res = st.session_state["sim_results"][sim_key]
        leaderboard, round_winners = res["leaderboard"], res["round_winners"]

        note = (f" · conditioned on {res['n_fixed']} real results"
                if res["n_fixed"] else "")
        st.success(
            f"✅ **{selected}** — {n_sims:,} sims in {res['elapsed']:.1f}s · "
            f"Model: **{payload.get('name', '?')}**{note}"
        )

        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.subheader(f"Championship Probability ({n_sims:,} sims)")
            chart_lb = leaderboard.head(min(12, len(leaderboard))).set_index("Player")["Win %"]
            chart_lb.index = chart_lb.index.map(format_name)
            st.bar_chart(chart_lb, horizontal=True)
        with col_right:
            st.subheader("Leaderboard")
            disp_cols = ["Player", "Win %"] + (
                ["Actual Result"] if "Actual Result" in leaderboard.columns else []
            )
            disp_lb = leaderboard[disp_cols].copy()
            disp_lb["Player"] = disp_lb["Player"].apply(format_name)
            st.dataframe(
                disp_lb.style
                .format({"Win %": "{:.2f}%"})
                .background_gradient(subset=["Win %"], cmap="Blues"),
                use_container_width=True, hide_index=True,
            )

        if round_winners:
            st.subheader("🌳 Most Likely Bracket Path")
            st.plotly_chart(render_bracket_figure(round_winners), use_container_width=True)

    else:
        st.subheader(f"{host_flag} {selected} · {format_tier(tier)}")
        st.caption(
            f"{len(roster)} players in bracket · choose a simulation count in the "
            "sidebar and click **▶ Run Simulation**. Live tournaments are "
            "automatically conditioned on results already played."
        )

# ── Tab 3: Matchup Analyzer ────────────────────────────────────────

with tab_matchup:
    st.subheader("⚡ Matchup Analyzer")

    if not roster:
        st.warning(f"No bracket data for **{selected}**. Pick another tournament.")
        st.stop()

    col_a, col_b = st.columns(2)
    with col_a:
        player_a = st.selectbox("Player A", roster, index=0,
                                format_func=format_name, key="shap_pa")
    with col_b:
        player_b = st.selectbox("Player B", roster, index=min(1, len(roster) - 1),
                                format_func=format_name, key="shap_pb")

    analyze_btn = st.button("🔍 Analyze Matchup", type="primary")
    shap_key    = f"{player_a}|{player_b}|{tour_date}"

    if analyze_btn:
        st.session_state["shap_analyzed"] = shap_key

    if player_a == player_b:
        st.warning("Select two **different** players.")
    elif st.session_state.get("shap_analyzed") == shap_key:
        sa, sb = player_stats[player_a], player_stats[player_b]

        render_match_explainer(player_a, player_b, "first round", ctx)

        st.subheader("📋 Stats")
        tape_df = pd.DataFrame({
            "Stat": [
                "Elo Rating", "EMA Form", "Win Streak",
                "Days Since Last Match", "Matches (Last 14d)",
                "H2H Win Rate", "Avg Point Diff", "Seed",
            ],
            format_name(player_a): [
                f"{sa['elo']:.0f}", f"{sa['ema_form']:.3f}", f"{sa['win_streak']}",
                f"{sa['days_since']:.0f}", f"{sa['matches_14d']}",
                f"{h2h_rate_fn(player_a, player_b):.3f}",
                f"{sa['avg_point_diff']:+.2f}", f"{int(sa['seed'])}",
            ],
            format_name(player_b): [
                f"{sb['elo']:.0f}", f"{sb['ema_form']:.3f}", f"{sb['win_streak']}",
                f"{sb['days_since']:.0f}", f"{sb['matches_14d']}",
                f"{h2h_rate_fn(player_b, player_a):.3f}",
                f"{sb['avg_point_diff']:+.2f}", f"{int(sb['seed'])}",
            ],
        })
        st.dataframe(tape_df, use_container_width=True, hide_index=True)

        st.subheader("🕸️ Stat Radar")
        st.plotly_chart(
            build_radar_chart(player_a, player_b, player_stats),
            use_container_width=True,
        )

        st.divider()
        st.subheader("📈 Recent Form — Last 5 Matches")
        st.caption("Win probability estimated strictly from pre-match data (no leakage).")
        form_col_a, form_col_b = st.columns(2)
        for col_w, player_name in [(form_col_a, player_a), (form_col_b, player_b)]:
            with col_w:
                st.markdown(f"**{format_name(player_name)}**")
                fdf = build_form_chart(player_name, tour_date, ctx)
                if fdf is None or fdf.empty:
                    st.caption("No match history before this tournament.")
                else:
                    fig2, ax = plt.subplots(figsize=(5, 3))
                    colors = ["green" if r == "W" else "red" for r in fdf["Result"]]
                    ax.plot(range(len(fdf)), fdf["Win Prob"].values,
                            color="steelblue", linewidth=1.5, zorder=1)
                    ax.scatter(range(len(fdf)), fdf["Win Prob"].values,
                               c=colors, s=60, zorder=2)
                    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
                    ax.set_ylim(0, 1)
                    ax.set_xticks(range(len(fdf)))
                    ax.set_xticklabels(fdf["Date"].tolist(), rotation=30,
                                       ha="right", fontsize=7)
                    ax.set_ylabel("Win Probability")
                    fig2.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)
                    st.dataframe(
                        fdf.style.map(
                            lambda v: "color: green" if v == "W" else "color: red",
                            subset=["Result"],
                        ),
                        use_container_width=True, hide_index=True,
                    )
    else:
        st.info("Select two players above and click **🔍 Analyze Matchup**.")
