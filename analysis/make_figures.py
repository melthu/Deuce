"""Regenerate the README's evaluation figures from the real corpus.

    python3 analysis/make_figures.py

Every number is measured here, so a figure cannot drift from the data behind it.
Re-run after the corpus grows. Needs matplotlib (the [research] extra).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score

from src.modeling.dataset import fit_preprocessors, load_training_frame
from src.modeling.promote import DATA_PATH, build_model, encode_xy, load_all_params

OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
TEST_YEAR = 2026
N_BOOT    = 2000
CANDIDATE = "lgbm"

# Blue/orange are validated categorical slots 1 and 2; red is the diverging
# counterpart to blue. All three pass the CVD and contrast checks on this
# surface, so keep them if the figures are restyled.
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
GRID, INK = "#dcdcd6", "#3d3d3a"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 11, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": GRID,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.dpi": 200,
})


def measure():
    df = load_training_frame(DATA_PATH)
    train_df = df[df["start_date"].dt.year < TEST_YEAR]
    test_df  = df[df["start_date"].dt.year >= TEST_YEAR].copy()

    prep, _ = fit_preprocessors(train_df)
    X_train, y_train = encode_xy(train_df, prep)
    X_test,  y_test  = encode_xy(test_df,  prep)

    d = test_df
    signals = {
        "Rating (elo_diff)":      d["elo_diff"].values,
        "Seeding":                (d.player_b_seed.replace(0, 99)
                                   - d.player_a_seed.replace(0, 99)).values,
        "Avg point differential": (d.player_a_avg_point_diff
                                   - d.player_b_avg_point_diff).values,
        "180-day win rate":       (d.player_a_recent_win_rate
                                   - d.player_b_recent_win_rate).values,
        "EMA form":               (d.player_a_ema_form - d.player_b_ema_form).values,
        "H2H last winner":        d["h2h_last_winner"].values,
        "Matches last 7 days":    (d.player_a_matches_last_7_days
                                   - d.player_b_matches_last_7_days).values,
        "H2H win rate":           d["h2h_win_rate_a_vs_b"].values,
        "Rubber-game rate":       (d.player_a_rubber_game_rate
                                   - d.player_b_rubber_game_rate).values,
        "Win streak":             (d.player_a_win_streak - d.player_b_win_streak).values,
        "Avg victory margin":     (d.player_a_avg_victory_margin
                                   - d.player_b_avg_victory_margin).values,
        "Days since last match":  (d.player_a_days_since_last_match
                                   - d.player_b_days_since_last_match).values,
    }
    signal_auc = {k: roc_auc_score(y_test, v) for k, v in signals.items()}

    model = build_model(CANDIDATE, load_all_params()[CANDIDATE])
    model.fit(X_train, y_train)
    p_model = model.predict_proba(X_test)[:, 1]
    p_elo   = np.clip(d["elo_expected"].values, 1e-6, 1 - 1e-6)

    # Resample matches, not rows: the frame is mirrored, so a match's two rows
    # are one observation and resampling rows would understate the interval.
    d = d.reset_index(drop=True)
    key = (d.tournament.astype(str) + "|" + d["round"].astype(str) + "|"
           + d.apply(lambda r: "".join(sorted([str(r.player_a), str(r.player_b)])), axis=1))
    groups = [np.asarray(idx) for idx in d.groupby(key.values).indices.values()]

    rng, deltas = np.random.default_rng(0), []
    for _ in range(N_BOOT):
        rows = np.concatenate([groups[i] for i in
                               rng.integers(0, len(groups), len(groups))])
        yb = y_test[rows]
        if yb.min() != yb.max():
            deltas.append(roc_auc_score(yb, p_model[rows])
                          - roc_auc_score(yb, p_elo[rows]))

    return dict(signal_auc=signal_auc, y=y_test, p_model=p_model, p_elo=p_elo,
                deltas=np.array(deltas), n_matches=len(groups))


def save(fig, name):
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", os.path.relpath(path))


def fig_signals(m):
    """Bars anchored at 0.5, the null. Anchoring at zero would imply 0 is a
    meaningful AUC, and would hide the two signals that rank backwards."""
    items = sorted(m["signal_auc"].items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    for i, (_, auc) in enumerate(items):
        ax.barh(i, auc - 0.5, left=0.5, height=0.66,
                color=BLUE if auc >= 0.5 else RED)
        right = auc >= 0.5
        ax.text(auc + (0.005 if right else -0.005), i, f"{auc:.3f}",
                va="center", ha="left" if right else "right", fontsize=9.5)

    ax.axvline(0.5, color=INK, lw=1.1)
    ax.set_yticks(range(len(items)), [n for n, _ in items])
    ax.set_xlim(0.40, 0.73)   # left room for the inverted bars' value labels
    ax.set_xlabel("AUC on the held-out season  (0.5 = no information)")
    ax.set_title("Single-signal AUC, held-out season")
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    save(fig, "signal_auc")


def fig_reliability(m):
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    edges = np.linspace(0, 1, 6)

    ax.plot([0, 1], [0, 1], color="#a9a9a3", lw=1.4, ls=(0, (4, 3)))
    for p, colour, label in ((m["p_model"], BLUE,   "Gradient-boosted model"),
                             (m["p_elo"],   ORANGE, "Elo expectancy alone")):
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (p >= lo) & (p < hi)
            if sel.sum() >= 10:
                xs.append(p[sel].mean())
                ys.append(m["y"][sel].mean())
        ax.plot(xs, ys, color=colour, lw=2.6, marker="o", markersize=8, label=label)

    ax.plot([], [], color="#a9a9a3", lw=1.4, ls=(0, (4, 3)),
            label="perfect calibration")
    ax.set(xlim=(0, 1), ylim=(0, 1),
           xlabel="Predicted P(A wins)", ylabel="Observed win rate")
    ax.set_title("Predicted vs realised win rate")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_axisbelow(True)
    save(fig, "calibration")


def fig_bootstrap(m):
    d = m["deltas"]
    lo, hi, mean = np.percentile(d, 2.5), np.percentile(d, 97.5), d.mean()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    counts, _, _ = ax.hist(d, bins=44, color="#9ec5f4")

    # Stop the marker lines at the top of the histogram rather than spanning the
    # axes, so the band above stays clear for the legend.
    top = counts.max() * 1.04
    ax.plot([0, 0], [0, top], color=INK, lw=1.6)
    ax.plot([mean, mean], [0, top], color=BLUE, lw=2.2)
    for x in (lo, hi):
        ax.plot([x, x], [0, top], color=BLUE, lw=1.4, ls=(0, (3, 2)))
    ax.set_ylim(0, counts.max() * 1.42)
    ax.legend(handles=[
        Line2D([], [], color=INK, lw=1.6, label="no improvement"),
        Line2D([], [], color=BLUE, lw=2.2, label=f"mean {mean:+.4f}"),
        Line2D([], [], color=BLUE, lw=1.4, ls=(0, (3, 2)),
               label=f"95% interval [{lo:+.4f}, {hi:+.4f}]"),
    ], loc="upper left", fontsize=9.5)

    ax.set_xlabel("Δ AUC per resample")
    ax.set_ylabel(f"Count of {len(d):,} resamples")
    ax.set_title("Bootstrap distribution of ΔAUC, model minus rating")
    ax.set_yticks([])
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    save(fig, "bootstrap_delta")


if __name__ == "__main__":
    m = measure()
    for old in ("signal_auc-dark", "calibration-dark", "bootstrap_delta-dark"):
        p = os.path.join(OUT_DIR, f"{old}.png")
        if os.path.exists(p):
            os.remove(p)
            print("removed", os.path.relpath(p))
    fig_signals(m)
    fig_reliability(m)
    fig_bootstrap(m)
