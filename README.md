# ShuttleCast

**ShuttleCast** is a point-in-time prediction engine for BWF Men's Singles badminton tournaments. It scrapes match data from Wikipedia (241 tournaments, 2010–2026), engineers 34 temporal features with strict no-leakage slicing, trains an ensemble of gradient-boosted tree models, and exposes everything through an interactive Streamlit dashboard where you can select any tournament, run a Monte Carlo bracket simulation, and drill into SHAP explanations for individual matchups.

---

## Model Results

Validation set = all 2026 matches (334 rows). Training set = 2010–2025 (17,784 rows, mirrored).

| Model    | Val AUC |
|----------|---------|
| LightGBM | 0.7372  |
| CatBoost | 0.7821  |
| XGBoost  | **0.7872** ← best |
| TabNet   | 0.7472  |
| Ensemble | 0.7768  |

Best model: **XGBoost (single)** — saved to `models/best_model.pkl`.

---

## Setup

```bash
git clone https://github.com/melthu/ShuttleCast.git
cd ShuttleCast
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On Linux (e.g. Streamlit Cloud), `packages.txt` installs `libgomp1` for LightGBM automatically.

The repo already includes:
- `data/raw/raw_matches.csv` — 9,255 scraped match results
- `data/config/tournaments_config.csv` — 241 tournament metadata rows
- `models/best_model.pkl` + individual model `.pkl` files
- `models/best_params.json` — Optuna best hyperparameters

So you can go straight to `make features && make train` or just `make dashboard` if you want to use the pre-trained models.

---

## Quick Start

```bash
make dashboard   # launch Streamlit at http://localhost:8501

make features    # re-engineer features from raw CSV (~1 min)
make train       # retrain LightGBM + CatBoost + XGBoost + ensemble selection
make train_tabnet  # train TabNet and re-run ensemble selection
make tune        # Optuna hyperparameter search — 50 trials (~5 min)
make cv          # rolling 3-fold temporal cross-validation
make simulate    # Monte Carlo simulation (CLI, no UI)
make data        # re-scrape Wikipedia — ~15 min, use only if needed
```

Or run the full pipeline end-to-end:
```bash
python3 run_pipeline.py --all
```

---

## Dashboard

`app.py` is a Streamlit app with three tabs:

**Simulation tab**
- Interactive FullCalendar sidebar — click any tournament block to select it
- Year/month dropdowns for fast navigation; past events muted, upcoming events highlighted
- Point-in-time engine: filters the dataset to `start_date < tournament_start`, builds a fresh vocab + scaler on that subset, then loads the pre-trained best model
- Monte Carlo simulation (1,000–10,000 iterations) with a live progress bar, sims/s, ETA, and a running championship-probability leaderboard
- "Reality Check" overlay: marks the actual tournament winner with a gold medal if the event is in the past
- Stats panel comparing both finalists' point-in-time Elo, EMA form, win streak, and fatigue

**SHAP Explainer tab**
- Pick any two players from the active draw
- SHAP waterfall plot explaining the XGBoost prediction for that matchup
- Automatically trims the feature vector to the saved model's `n_features_in_` for backward compatibility

**Form Trends tab**
- Rolling Elo and win-rate charts for any player over their career

---

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1 | `src/build_config.py` | `data/config/tournaments_config.csv` — 241 tournaments 2010–2026, handles World Tour (2018+) and Super Series (2010–2017) Wikipedia formats |
| 2 | `src/scraper_orchestrator.py` → `scraper_wiki_single.py` | `data/raw/raw_matches.csv` — 9,255 matches with per-game scores, seeds, walkover flags, nationalities |
| 3 | `src/feature_engineering.py` | `data/interim/engineered_matches.csv` — 30 temporal features, walkovers filtered before any computation |
| 4 | `src/data_loader.py` | `data/processed/final_training_data.csv` — 18,118 rows (every match mirrored by swapping Player A ↔ B for positional symmetry) |

---

## Features

**4 categorical:** tier, round, player\_a ID, player\_b ID

**30 continuous (`CONT_COLS` in `dataset.py`):**

| Group | Count | Features |
|-------|-------|---------|
| Original | 10 | same\_nationality, h2h\_win\_rate, home advantage ×2, 14-day match count ×2, days since last match ×2, 180-day win rate ×2 |
| Elo / EMA | 10 | player\_a/b Elo (K scaled by tier), Elo difference, player\_a/b EMA form (α=0.3), H2H last winner, win streak ×2, matches in last 7 days ×2 |
| Score-derived | 4 | avg point differential ×2, avg games per match ×2 — rolling 10 matches |
| Phase 5 | 6 | rubber-game rate ×2, avg victory margin ×2, seeding ×2 |

**No data leakage:** all temporal features use strict `hist = df[df['start_date'] < current_date]`.

**Elo K-factors by tier:** `{100: 20, 300: 24, 500: 28, 750: 32, 1000: 40, 1500: 50}`. Default Elo = 1500. EMA α = 0.3, default = 0.5.

---

## Models

### Tree models
- **LightGBM** (`src/train_lgbm.py`) — saved to `models/best_lgbm.pkl`
- **CatBoost** (`src/train_catboost.py`) — saved to `models/best_catboost.pkl`
- **XGBoost** (`src/train_xgb.py`) — saved to `models/best_xgb.pkl`

Hyperparameters for XGBoost and LightGBM are searched via Optuna (`src/tune_hyperparams.py`) and stored in `models/best_params.json`. The search uses the penultimate validation year as the objective to keep the final 2026 split clean.

### TabNet
`src/train_tabnet.py` wraps `pytorch-tabnet`'s `TabNetClassifier` with categorical embeddings (dim=8) for the 4 categorical columns. Architecture: n_d=n_a=32, n_steps=5, γ=1.5, sparse λ=1e-4. Best epoch was 13/200 (early stopping at patience=25). Val AUC = 0.7472.

### DeepFM
`src/model.py` implements a custom `BWFDeepFM` in PyTorch: shared embedding layer (dim=32), FM interaction layer, and a deep MLP (256→128→64). `src/train.py` trains it with cosine-annealing LR, patience-5 early stopping. Val AUC = 0.7011.

### Ensemble
`src/train_ensemble.py` loads all available saved models and combines them with **AUC-weighted averaging** — each model's weight is proportional to (AUC − 0.5) so near-random models contribute almost nothing. The ensemble is saved only if it beats the best individual model.

---

## Monte Carlo Simulation

`src/simulate_german_open.py` (also called by `app.py`) runs N iterations of a tournament bracket:

1. Builds point-in-time player stats (Elo, EMA, streak, etc.) from data strictly before the tournament start date.
2. For each simulated match: constructs a 34-feature vector for both draw positions, averages `P(A | slot_a)` and `1 − P(B | slot_a)` to eliminate positional bias, then samples the outcome stochastically.
3. Applies in-bracket Elo and EMA updates after each match so later-round predictions reflect tournament form.
4. Player stats are deep-copied per iteration so updates don't bleed between simulations.

---

## Temporal Cross-Validation

`src/temporal_cv.py` runs rolling 3-fold CV where the last three distinct years each serve as the validation fold:

```
Fold 1: train < year[-3],  val == year[-3]
Fold 2: train < year[-2],  val == year[-2]
Fold 3: train < year[-1],  val == year[-1]
```

Each fold fits its own vocab and scaler on the training slice only to prevent leakage.

---

## Project Structure

```
ShuttleCast/
├── run_pipeline.py              # Master CLI: --scrape --features --train --tune --all
├── app.py                       # Streamlit dashboard (Monte Carlo + SHAP)
├── Makefile
├── requirements.txt
├── packages.txt                 # apt deps for Streamlit Cloud (libgomp1)
├── src/
│   ├── build_config.py          # Step 1 — BWF calendar scraper (2010-2026)
│   ├── scraper_wiki_single.py   # Step 2a — single-tournament Wikipedia scraper
│   ├── scraper_orchestrator.py  # Step 2b — orchestrates all tournaments
│   ├── feature_engineering.py  # Step 3 — temporal feature engineering (30 features)
│   ├── data_loader.py           # Step 4 — dataset mirroring for positional symmetry
│   ├── dataset.py               # PyTorch dataset, extract_numpy, preprocessing
│   ├── model.py                 # BWFDeepFM (PyTorch)
│   ├── train.py                 # DeepFM training loop (standalone)
│   ├── train_lgbm.py            # LightGBM trainer
│   ├── train_catboost.py        # CatBoost trainer
│   ├── train_xgb.py             # XGBoost trainer
│   ├── train_tabnet.py          # TabNet trainer + TabNetWrapper
│   ├── train_ensemble.py        # AUC-weighted ensemble selection + DeepFMWrapper
│   ├── temporal_cv.py           # Rolling 3-fold temporal cross-validation
│   ├── tune_hyperparams.py      # Optuna search for XGBoost + LightGBM
│   └── simulate_german_open.py  # Monte Carlo tournament simulation
├── data/
│   ├── config/tournaments_config.csv   # 241 tournaments 2010-2026 (tracked)
│   ├── raw/raw_matches.csv             # 9,255 match results (tracked)
│   ├── interim/                        # engineered features (git-ignored)
│   └── processed/                      # ML-ready mirrored dataset (git-ignored)
└── models/
    ├── best_params.json         # Optuna best hyperparameters (tracked)
    ├── best_model.pkl           # active best model or ensemble (tracked)
    ├── best_lgbm.pkl / best_catboost.pkl / best_xgb.pkl / best_tabnet.pkl
    └── best_deepfm.pt           # DeepFM checkpoint (git-ignored — large)
```

---

## License

MIT
