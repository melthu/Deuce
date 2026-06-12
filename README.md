# ShuttleCast

**ShuttleCast** is a point-in-time prediction engine for BWF Men's Singles badminton tournaments. It scrapes match data from Wikipedia (300+ tournaments, 2010–present), engineers 30 leakage-free temporal features, trains gradient-boosted tree models, and serves everything through an interactive Streamlit dashboard: pick any tournament from a calendar, run a vectorised Monte Carlo bracket simulation, and drill into SHAP explanations for individual matchups.

A scheduled GitHub Actions workflow keeps the deployment fresh: every Monday it scrapes newly finished tournaments (and newly published draws), re-engineers features, retrains the preloaded model, and commits — Streamlit Cloud redeploys automatically.

---

## How predictions work

| Tournament | Model used |
|------------|-----------|
| **Upcoming** (starts after today) | Preloaded XGBoost (`models/best_model.pkl`), retrained weekly on all completed matches |
| **Past or live** | Point-in-time XGBoost trained in-app on every match strictly before the tournament's start date — vocab, scaler, and model all fit on that slice only (no leakage), cached per tournament |

Live tournaments get special treatment: matches already played are taken as fixed results, and the Monte Carlo simulation is conditioned on them — championship odds update as the real bracket unfolds with each weekly data refresh.

### Benchmark results

Train ≤ 2025, validation = all 2026 matches to date (leak-free temporal holdout; June 2026 data snapshot):

| Model (Optuna-tuned) | Val AUC |
|----------------------|---------|
| XGBoost  | 0.7156  |
| LightGBM | 0.7190  |
| Ensemble | 0.7229  |
| CatBoost | **0.7233** |

The production model for upcoming tournaments is chosen by `src/promote.py`: every week it benchmarks all tuned candidates on the latest season, retrains the **winner** on all completed matches, and promotes it to `models/best_model.pkl` — so the model type is re-decided automatically as the season's validation data grows (currently CatBoost). Benchmark AUCs shift as the 2026 validation set grows.

---

## Setup

```bash
git clone https://github.com/melthu/ShuttleCast.git
cd ShuttleCast
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On Linux (e.g. Streamlit Cloud), `packages.txt` installs `libgomp1` for LightGBM automatically.

The repo already includes the scraped data (`data/raw/raw_matches.csv`), the processed training set, and trained model pickles — `make dashboard` works immediately.

## Quick Start

```bash
make dashboard   # launch Streamlit at http://localhost:8501

make update      # incremental scrape: new/pending/recent tournaments only
make data        # full rescrape of every tournament (~15 min)
make features    # re-engineer features from raw CSV (~1 min)
make train       # retrain LightGBM + CatBoost + XGBoost + ensemble selection
make tune        # Optuna hyperparameter search — 50 trials
make cv          # rolling 3-fold temporal cross-validation
make simulate ARGS="--date 2026-02-24 --tier 300 --sims 10000"
```

Or run the full pipeline end-to-end: `python3 run_pipeline.py --all`

---

## Dashboard

`app.py` has a calendar sidebar (click any tournament block to select it) and three tabs:

**📋 Draw & Predictions** — the full bracket, round by round. Completed matches show the real winner; unplayed matches show the model's win probability for each player. A match selector renders a SHAP waterfall explaining exactly which features drive any prediction.

**🎲 Monte Carlo** — simulate the bracket 100–10,000 times (vectorised: every match in a round across all simulations is one `predict_proba` call, so 10k sims take seconds). Outputs a championship-probability leaderboard, the most-likely bracket path, and a 🥇 reality-check marker when the actual winner is on record. Live tournaments are conditioned on results already played.

**⚡ Matchup Analyzer** — pick any two players in the draw: win probability, stat comparison table, radar chart, SHAP waterfall, and last-5-matches form charts with point-in-time win-probability estimates.

---

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1 | `src/build_config.py` | `data/config/tournaments_config.csv` — tournament calendar 2010→present (year range is dynamic; new seasons appear automatically) |
| 2 | `src/scraper_orchestrator.py` → `scraper_wiki_single.py` | `data/raw/raw_matches.csv` — matches in true bracket order with per-game scores, seeds, walkover + pending flags. `--incremental` merges only new/changed tournaments |
| 3 | `src/feature_engineering.py` | `data/interim/engineered_matches.csv` — 30 temporal features; walkovers dropped, pending matches get features but never update history |
| 4 | `src/data_loader.py` | `data/processed/final_training_data.csv` — every match mirrored A↔B for positional symmetry |

`src/data_checks.py` is the sanity gate the weekly workflow runs before committing scraped data (row counts, nulls, duplicate keys, walkover/pending fractions).

### Pending matches

The scraper marks drawn-but-unplayed matches (`is_pending=1`, no bolded winner on Wikipedia). They flow through feature engineering — so the dashboard can predict upcoming draws — but are excluded from all history, Elo updates, and training.

---

## Features

**4 categorical:** tier, round, player\_a ID, player\_b ID

**30 continuous (`CONT_COLS` in `dataset.py`):**

| Group | Count | Features |
|-------|-------|---------|
| Original | 10 | same\_nationality, h2h\_win\_rate, home advantage ×2, 14-day match count ×2, days since last match ×2, 180-day win rate ×2 |
| Elo / EMA | 10 | player\_a/b Elo (K scaled by tier), Elo difference, player\_a/b EMA form (α=0.3), H2H last winner, win streak ×2, matches in last 7 days ×2 |
| Score-derived | 4 | avg point differential ×2, avg games per match ×2 — rolling 10 matches |
| Bracket | 6 | rubber-game rate ×2, avg victory margin ×2, seeding ×2 |

**No data leakage:** all temporal features use strict `start_date < current_date` slicing, with pending matches additionally excluded from history.

**Elo K-factors by tier:** `{100: 20, 300: 24, 500: 28, 750: 32, 1000: 40, 1500: 50}`. Default Elo = 1500. EMA α = 0.3, default = 0.5.

---

## Models

- **XGBoost / LightGBM / CatBoost** (`src/train_xgb.py`, `src/train_lgbm.py`, `src/train_catboost.py`) — benchmark trainers; all read Optuna-tuned hyperparameters from `models/best_params.json`.
- **promote.py** — production selection: benchmarks all three tuned candidates on the latest season, retrains the winner on every completed match, writes `models/best_model.pkl`. Run weekly by CI.
- **TabNet** (`src/train_tabnet.py`) and **DeepFM** (`src/model.py` + `src/train.py`) — neural baselines.
- **Ensemble** (`src/train_ensemble.py`) — AUC-weighted average of all saved models, for benchmarking.

Hyperparameters for all three tree models are tuned with Optuna (`src/tune_hyperparams.py`) against the penultimate year so the final holdout stays clean.

---

## Monte Carlo Simulation

`src/simulate.py` (imported by the app, also a CLI):

1. Builds point-in-time player stats (Elo, EMA, streak, …) from data strictly before the tournament start.
2. Simulates all N brackets round-by-round. Each round batches every match across all simulations into a single `predict_proba` call, with both slot orders averaged — `P(A beats B) ≡ 1 − P(B beats A)` — to eliminate positional bias.
3. Applies in-bracket Elo/EMA updates per simulation so later-round predictions reflect tournament form.
4. Matches with real results on record are fixed to their actual outcome in every simulation.

```bash
python3 src/simulate.py --date 2026-02-24 --tier 300 --sims 10000
```

---

## Automation

`.github/workflows/update-data.yml` runs every Monday 06:00 UTC (and on demand via *workflow_dispatch*):

1. Rebuild the tournament calendar (picks up new seasons automatically)
2. `scraper_orchestrator.py --incremental` — scrape only missing/pending/recent tournaments and merge
3. `data_checks.py` — abort on anything suspicious before it can reach the deployed app
4. Re-engineer features + mirror the dataset
5. Re-select and retrain the production model (`promote.py` — best of tuned XGBoost/LightGBM/CatBoost on the latest season, retrained on all completed matches)
6. Commit & push → Streamlit Cloud redeploys

---

## Project Structure

```
ShuttleCast/
├── run_pipeline.py              # Master CLI: --scrape --features --train --tune --all
├── app.py                       # Streamlit dashboard
├── Makefile
├── requirements.txt             # full app/training deps
├── requirements-ci.txt          # minimal deps for the weekly refresh (no torch)
├── packages.txt                 # apt deps for Streamlit Cloud (libgomp1)
├── .github/workflows/update-data.yml   # weekly scrape + retrain + commit
├── src/
│   ├── build_config.py          # tournament calendar scraper (dynamic year range)
│   ├── scraper_wiki_single.py   # single-tournament scraper (bracket order, pending flags)
│   ├── scraper_orchestrator.py  # all tournaments; --incremental merge mode
│   ├── data_checks.py           # sanity gate for automated scrapes
│   ├── feature_engineering.py   # 30 temporal features, leakage-free
│   ├── data_loader.py           # A↔B mirroring
│   ├── dataset.py               # shared preprocessing: vocab/scaler fitting, encoding
│   ├── simulate.py              # vectorised Monte Carlo engine + CLI
│   ├── train_xgb.py             # production trainer (--full-data --promote)
│   ├── train_lgbm.py / train_catboost.py / train_tabnet.py / train.py
│   ├── train_ensemble.py        # AUC-weighted ensemble selection
│   ├── temporal_cv.py           # rolling 3-fold temporal cross-validation
│   ├── tune_hyperparams.py      # Optuna search
│   └── model.py                 # BWFDeepFM (PyTorch)
├── data/
│   ├── config/tournaments_config.csv   # tracked
│   ├── raw/raw_matches.csv             # tracked
│   ├── interim/                        # git-ignored
│   └── processed/final_training_data.csv  # tracked (needed by the app)
└── models/                      # best_model.pkl + per-model pickles (tracked)
```

---

## License

MIT
