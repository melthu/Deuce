.PHONY: data update features train train_tabnet dashboard export site all simulate cv tune help

data:        ## Scrape Wikipedia + rebuild raw CSV (full rescrape)
	python3 run_pipeline.py --scrape

update:      ## Incremental refresh: new/pending/recent tournaments only
	python3 src/pipeline/build_config.py && python3 src/pipeline/scraper_orchestrator.py --incremental && python3 src/pipeline/data_checks.py

features:    ## Re-engineer features + mirror dataset
	python3 run_pipeline.py --features

train:       ## Train all models, save best to models/best_model.pkl
	python3 src/modeling/train_lgbm.py && python3 src/modeling/train_catboost.py && python3 src/modeling/train_xgb.py && python3 src/modeling/train_ensemble.py

train_tabnet: ## Train TabNet and re-run ensemble selection
	python3 src/modeling/train_tabnet.py && python3 src/modeling/train_ensemble.py

dashboard:   ## Launch Streamlit app
	streamlit run app.py

export:      ## Precompute the static site payload (incremental; --force to rebuild)
	python3 src/serving/export_static.py $(ARGS)

site:        ## Serve the built static site locally on :8000
	cd site && python3 -m http.server 8000

all:         ## Full pipeline end-to-end
	python3 run_pipeline.py --all

simulate:    ## Monte Carlo simulation CLI (override: make simulate ARGS="--date 2026-02-24 --tier 300")
	python3 src/serving/simulate.py $(ARGS)

cv:          ## Run rolling 3-fold temporal cross-validation
	python3 src/modeling/temporal_cv.py

tune:        ## Optuna hyperparameter search (50 trials) + retrain best models
	python3 src/modeling/tune_hyperparams.py --model all --trials 50 --retrain

help:        ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
