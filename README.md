# ShuttleCast

**ShuttleCast** is a point-in-time prediction engine for BWF Men's Singles badminton tournaments. It scrapes match data from Wikipedia (300+ tournaments, 2010–present), engineers 30 leakage-free temporal features, trains gradient-boosted tree models, and publishes the results as a static site: pick any tournament, read the model's call on every match in the draw, simulate the bracket 10,000 times, and see which factors drove any individual prediction.

The point of "point-in-time" is that a past tournament is predicted by a model trained **only on matches before it started** — vocabulary, scaler and estimator all fit on that slice. It has never seen the tournament it is predicting, so its record on those draws is a genuine out-of-sample one, and the site shows where it was wrong as readily as where it was right.

A scheduled GitHub Actions workflow keeps it fresh: daily it scrapes newly finished tournaments and newly published draws, re-engineers features, re-selects and retrains the production model, re-exports only what changed, and deploys to GitHub Pages.

---

## How predictions work

| Tournament | Model used |
|------------|-----------|
| **Upcoming** (starts after today) | The promoted model (`models/best_model.pkl`) — re-selected and retrained weekly on all completed matches. Currently LightGBM; the payload's `name` field says which |
| **Past or live** | Point-in-time XGBoost trained on every match strictly before the tournament's start date — vocab, scaler, and model all fit on that slice only (no leakage), cached per tournament |

Live tournaments get special treatment: matches already played are taken as fixed results, and the Monte Carlo simulation is conditioned on them — championship odds update as the real bracket unfolds with each weekly data refresh.

### Benchmark results

Train ≤ 2025, validation = all 2026 matches to date (leak-free temporal holdout; July 2026 data snapshot):

| Model (Optuna-tuned) | Val AUC |
|----------------------|---------|
| CatBoost | 0.7134  |
| XGBoost  | 0.7196  |
| LightGBM | **0.7240** |

The production model for upcoming tournaments is chosen by `src/modeling/promote.py`: every week it benchmarks all tuned candidates on the latest season, retrains the **winner** on all completed matches, and promotes it to `models/best_model.pkl` — so the model type is re-decided automatically as the season's validation data grows. Both the ranking and the AUCs move week to week; don't hardcode an assumption about which model wins.

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

make export      # precompute the static site payload (incremental; ARGS="--force" to rebuild)
make site        # serve the built static site on http://localhost:8000
```

Or run the full pipeline end-to-end: `python3 run_pipeline.py --all`

---

## The site

The primary frontend is a static site on GitHub Pages — plain HTML/CSS/JS, no build step
and no cold start. **No model runs in the browser.** Every model output is over a bounded
set — a point-in-time model exists only to predict its own tournament's ~31 matches — so
`src/serving/export_static.py` precomputes all of them into JSON and the page just draws
the result. Shipping the models instead would be ~540 MB; shipping what they *said* is
about 9 MB, sharded so nobody downloads more than the screen they're on.

```
site/data/tournaments.json     index; first paint
site/data/tournament/<slug>    bracket, per-match predictions, grouped SHAP, leaderboards
site/data/player/<slug>        current-form card
site/data/matchup/<slug>       that player against every other active player
```

Each shard carries a fingerprint of the inputs behind it, so a rebuild only touches what
actually moved: a full export is ~20 minutes, an unchanged rerun is seconds. That is also
what keeps a live tournament current — its own rows change as results land, so the
fingerprint misses and the file re-exports on the next run, no special-casing needed.

`src/serving/check_export.py` gates publication on shard counts, payload size, empty
brackets and the share of draws with no simulation.

```bash
make export && make site    # build, then browse at http://localhost:8000
```

## Streamlit dashboard

`app.py` is kept as the correctness oracle for the static export. It has a calendar sidebar (click any tournament block to select it) and three tabs:

**📋 Draw & Predictions** — the full bracket, round by round. Completed matches show the real winner; unplayed matches show the model's win probability for each player. A match selector renders a SHAP waterfall explaining exactly which features drive any prediction.

**🎲 Monte Carlo** — simulate the bracket 100–10,000 times (vectorised: every match in a round across all simulations is one `predict_proba` call, so 10k sims take seconds). Outputs a championship-probability leaderboard, the most-likely bracket path, and a 🥇 reality-check marker when the actual winner is on record. Live tournaments are conditioned on results already played.

**⚡ Matchup Analyzer** — pick any two players in the draw: win probability, stat comparison table, radar chart, SHAP waterfall, and last-5-matches form charts with point-in-time win-probability estimates.

---

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1 | `src/pipeline/build_config.py` | `data/config/tournaments_config.csv` — tournament calendar 2010→present (year range is dynamic; new seasons appear automatically) |
| 2 | `src/pipeline/scraper_orchestrator.py` → `scraper_wiki_single.py` | `data/raw/raw_matches.csv` — matches in true bracket order with per-game scores, seeds, walkover + pending flags. `--incremental` merges only new/changed tournaments. Player names are canonicalised on write (`player_names.py`) |
| 3 | `src/pipeline/feature_engineering.py` | `data/interim/engineered_matches.csv` — 30 temporal features; walkovers and pending matches get features but never update history |
| 4 | `src/pipeline/data_loader.py` | `data/processed/final_training_data.csv` — every match mirrored A↔B for positional symmetry |

`src/pipeline/data_checks.py` is the sanity gate the weekly workflow runs before committing scraped data (row counts, nulls, duplicate keys, walkover/pending fractions).

### Player identity

Wikipedia spells the same player several ways — word order (`Kidambi Srikanth` /
`Srikanth Kidambi`), optional name parts (`Anthony Ginting` / `Anthony Sinisuka Ginting`),
case, hyphenation and diacritics. Every spelling was otherwise a separate player with its
own Elo, form and head-to-head: Parupalli Kashyap's career was split 121/65 and Prannoy's
four ways. `src/pipeline/player_names.py` folds 78 alternate spellings into 68 canonical
identities at the point the raw CSV is written.

The map is explicit rather than a normalisation rule, because normalising is not safe in
general: **Huang Yu and Huang Yu-kai reduce to the same string but played each other** in
the third round of Kaohsiung Masters 2023. Every merge was checked against nationality,
career span, and whether the two names ever shared a draw or met. `data_checks.py` reports
new collisions but never merges them.

### Pending matches and walkovers

The scraper marks drawn-but-unplayed matches (`is_pending=1`, no bolded winner on Wikipedia). They flow through feature engineering — so the dashboard can predict upcoming draws — but are excluded from all history, Elo updates, and training.

Walkovers (`is_walkover=1`) are handled the same way, and for the same reason they used to be
handled *differently*: dropping them left 76 of 222 post-2018 draws with a non-power-of-two
first round, which silently broke bracket topology in the Monte Carlo. They are now kept as
rows so the bracket resolves, but contribute nothing to Elo, EMA, head-to-head or training
(`load_training_frame(drop_walkover=...)`, defaulting to `drop_pending`).

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

- **XGBoost / LightGBM / CatBoost** (`src/modeling/train_xgb.py`, `src/modeling/train_lgbm.py`, `src/modeling/train_catboost.py`) — benchmark trainers; all read Optuna-tuned hyperparameters from `models/best_params.json`. Their per-candidate pickles are benchmark artifacts and stay untracked; only the promoted model is committed.
- **promote.py** — production selection: benchmarks all three tuned candidates on the latest season, retrains the winner on every completed match, writes `models/best_model.pkl`. Run weekly by CI.
- **TabNet** (`src/modeling/train_tabnet.py`) and **DeepFM** (`src/modeling/model.py` + `src/modeling/train.py`) — neural baselines.
- **Ensemble** (`src/modeling/train_ensemble.py`) — AUC-weighted average of all saved models, for benchmarking.

Hyperparameters for all three tree models are tuned with Optuna (`src/modeling/tune_hyperparams.py`) against the penultimate year so the final holdout stays clean.

---

## Monte Carlo Simulation

`src/serving/simulate.py` (imported by the app, also a CLI):

1. Builds point-in-time player stats (Elo, EMA, streak, …) from data strictly before the tournament start.
2. Simulates all N brackets round-by-round. Each round batches every match across all simulations into a single `predict_proba` call, with both slot orders averaged — `P(A beats B) ≡ 1 − P(B beats A)` — to eliminate positional bias.
3. Applies in-bracket Elo/EMA updates per simulation so later-round predictions reflect tournament form.
4. Matches with real results on record are fixed to their actual outcome in every simulation.

```bash
python3 src/serving/simulate.py --date 2026-02-24 --tier 300 --sims 10000
```

---

## Automation

`.github/workflows/update-data.yml` runs daily at 06:00 UTC (and on demand via *workflow_dispatch*), in two jobs.

**refresh** — scrape and retrain:

1. Rebuild the tournament calendar (picks up new seasons automatically)
2. `scraper_orchestrator.py --incremental` — scrape only missing/pending/recent tournaments and merge
3. `data_checks.py` — abort on anything suspicious before it can reach the deployed app
4. Re-engineer features + mirror the dataset
5. Re-select and retrain the production model (`promote.py` — best of tuned XGBoost/LightGBM/CatBoost on the latest season, retrained on all completed matches)
6. Commit & push → Streamlit Cloud redeploys

**publish** — export and deploy:

7. Restore the previous `site/data` from cache, then export incrementally
8. `check_export.py` — refuse to deploy a payload that looks collapsed
9. `upload-pages-artifact` → `deploy-pages`

Daily rather than weekly because a live tournament's predictions should move as its rounds
complete. Both the scrape and the export no-op cheaply when nothing has changed, so the
extra runs cost little. A push touching `site/**` or `src/serving/**` runs **publish** only.

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
│   ├── pipeline/                # data acquisition → training table
│   │   ├── build_config.py          # tournament calendar scraper (dynamic year range)
│   │   ├── scraper_wiki_single.py   # single-tournament scraper (bracket order, pending flags)
│   │   ├── scraper_orchestrator.py  # all tournaments; --incremental merge mode
│   │   ├── data_checks.py           # sanity gate for automated scrapes
│   │   ├── feature_engineering.py   # 30 temporal features, leakage-free
│   │   └── data_loader.py           # A↔B mirroring
│   ├── modeling/                # preprocessing, trainers, model selection
│   │   ├── dataset.py               # shared preprocessing: vocab/scaler fitting, encoding
│   │   ├── pit_model.py             # point-in-time trainer (app + exporter share it)
│   │   ├── promote.py               # weekly production model selection
│   │   ├── train_xgb.py / train_lgbm.py / train_catboost.py / train_tabnet.py / train.py
│   │   ├── train_ensemble.py        # AUC-weighted ensemble selection
│   │   ├── temporal_cv.py           # rolling 3-fold temporal cross-validation
│   │   ├── tune_hyperparams.py      # Optuna search
│   │   └── model.py                 # BWFDeepFM (PyTorch)
│   └── serving/                 # everything that turns a model into an answer
│       ├── simulate.py              # vectorised Monte Carlo engine + CLI
│       ├── export_static.py         # precomputes the static site payload
│       └── check_export.py          # publish gate for that payload
├── site/                        # static frontend (index.html + app.js + styles.css)
│   └── data/                        # generated by `make export`; git-ignored
├── data/
│   ├── config/tournaments_config.csv   # tracked
│   ├── raw/raw_matches.csv             # tracked
│   ├── interim/                        # git-ignored (regenerated by `make features`)
│   └── processed/final_training_data.csv  # tracked (needed by the app)
└── models/                      # only best_model.pkl + best_params.json are tracked
```

Every module bootstraps `sys.path` to the repo root, so scripts run the same whether
invoked as `python3 src/serving/simulate.py` or imported as `src.serving.simulate`.

---

## License

MIT
