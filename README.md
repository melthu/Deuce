# BWF Men's Singles Match Prediction

Predicts men's singles match outcomes across BWF World Tour and Super Series tournaments (2010–2026) using temporal feature engineering and gradient-boosted tree models.

**Best model: XGBoost — 0.7872 ROC-AUC on 2026 hold-out data.**

---

## Model Results

| Model    | Val AUC |
|----------|---------|
| LightGBM | 0.7372  |
| CatBoost | 0.7821  |
| XGBoost  | **0.7872** |
| Ensemble | 0.7768  |

Val split = all 2026 matches (334 rows). Train split = 2010–2025 (17,784 rows, mirrored).

---

## Setup

```bash
git clone https://github.com/melthu/BWF-Prediction-Model.git
cd BWF-Prediction-Model
pip install -r requirements.txt
```

A virtual environment is recommended:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick Start

```bash
make data       # scrape Wikipedia — ~15 min, 241 tournaments 2010-2026
make features   # engineer features from raw matches
make train      # train LightGBM, CatBoost, XGBoost; select best
make dashboard  # launch Streamlit at http://localhost:8501
make simulate   # Monte Carlo simulation for a tournament
make tune       # Optuna hyperparameter search (50 trials, ~5 min)
make cv         # rolling 3-fold temporal cross-validation
```

Or run the full pipeline end-to-end:
```bash
python3 run_pipeline.py --all
```

---

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `src/build_config.py` | Scrapes BWF calendar pages 2010–2026. Handles both World Tour (2018+) and Super Series (2010–2017) Wikipedia formats. Outputs 241 tournaments to `data/config/tournaments_config.csv`. |
| 2 | `src/scraper_orchestrator.py` | Calls `scraper_wiki_single` for each tournament. Injects `start_date`, `host_country`, per-game scores, seeds, and walkover flags. Outputs `data/raw/raw_matches.csv`. |
| 3 | `src/feature_engineering.py` | Builds 30 temporal features with strict no-leakage slicing (`start_date < current`). Filters walkovers. Outputs `data/interim/engineered_matches.csv`. |
| 4 | `src/data_loader.py` | Mirrors every row by swapping Player A ↔ B (doubles the dataset). Enforces positional symmetry. Outputs `data/processed/final_training_data.csv`. |

---

## Features (30 continuous + 4 categorical)

**Categorical (4):** tier, round, player\_a ID, player\_b ID

**Original (10):** same\_nationality, h2h\_win\_rate, home advantage flags, 14-day match count, days since last match, 180-day win rate — for both players

**Elo / EMA (10):** player\_a/b Elo rating (K scaled by tier), Elo difference, player\_a/b EMA form (α=0.3), H2H last winner, player\_a/b win streak, player\_a/b matches in last 7 days

**Score-derived (4):** player\_a/b avg point differential, player\_a/b avg games per match — rolling over last 10 matches

**New (6):** player\_a/b rubber-game rate (% of 3-game matches), player\_a/b avg victory margin, player\_a/b seeding

---

## Data Notes

- `data/raw/raw_matches.csv` and `data/config/tournaments_config.csv` are checked in as snapshots — you do not need to re-scrape to run `make features` and `make train`.
- `data/interim/` and `data/processed/` are excluded from git (regenerable).
- Model `.pkl` files are excluded from git (regenerable via `make train`).
- `models/best_params.json` is tracked — it stores Optuna's best hyperparameters (hours of compute).

---

## Project Structure

```
bwfML/
├── run_pipeline.py          # Master CLI: --scrape, --features, --train, --tune, --all
├── app.py                   # Streamlit dashboard
├── Makefile
├── requirements.txt
├── src/
│   ├── build_config.py      # Step 1 — BWF calendar scraper
│   ├── scraper_wiki_single.py   # Step 2a — single-tournament Wikipedia scraper
│   ├── scraper_orchestrator.py  # Step 2b — orchestrates all tournaments
│   ├── feature_engineering.py  # Step 3 — temporal feature engineering
│   ├── data_loader.py       # Step 4 — dataset mirroring
│   ├── dataset.py           # PyTorch dataset, preprocessing, extract_numpy
│   ├── model.py             # DeepFM architecture (PyTorch)
│   ├── train.py             # DeepFM training loop (standalone)
│   ├── train_lgbm.py        # LightGBM trainer
│   ├── train_catboost.py    # CatBoost trainer
│   ├── train_xgb.py         # XGBoost trainer
│   ├── train_ensemble.py    # Ensemble selection
│   ├── temporal_cv.py       # Rolling 3-fold temporal cross-validation
│   ├── tune_hyperparams.py  # Optuna hyperparameter search
│   └── simulate_german_open.py  # Monte Carlo tournament simulation
├── data/
│   ├── config/tournaments_config.csv   # 241 tournaments (tracked)
│   ├── raw/raw_matches.csv             # 9,255 match results (tracked)
│   ├── interim/                        # engineered features (git-ignored)
│   └── processed/                      # ML-ready mirrored dataset (git-ignored)
└── models/
    ├── best_params.json     # Optuna results (tracked)
    └── *.pkl / *.pt         # trained artifacts (git-ignored, run make train)
```

---

## License

MIT
