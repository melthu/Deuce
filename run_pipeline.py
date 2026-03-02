import subprocess
import sys
import time
import argparse

SCRAPE_STEPS = [
    ("Building 5-year tournament config",  "src/build_config.py"),
    ("Scraping Wikipedia match results",   "src/scraper_orchestrator.py"),
]

FEATURE_STEPS = [
    ("Engineering temporal features",      "src/feature_engineering.py"),
    ("Mirroring dataset for ML readiness", "src/data_loader.py"),
]

TRAIN_STEPS = [
    ("Training LightGBM",   "src/train_lgbm.py"),
    ("Training CatBoost",   "src/train_catboost.py"),
    ("Training XGBoost",    "src/train_xgb.py"),
    ("Selecting best model (ensemble)", "src/train_ensemble.py"),
]


def run_steps(steps, step_offset=0, total_steps=None):
    if total_steps is None:
        total_steps = len(steps)
    for i, (label, script) in enumerate(steps, start=step_offset + 1):
        print(f"\n[{i}/{total_steps}] {label}...")
        step_start = time.time()

        result = subprocess.run([sys.executable, script], capture_output=False)

        elapsed = time.time() - step_start
        if result.returncode != 0:
            print(f"\n[ERROR] Step {i} failed (exit code {result.returncode}). Pipeline halted.")
            sys.exit(result.returncode)

        print(f"[{i}/{total_steps}] Done in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="BWF Men's Singles — Data & Training Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--scrape",   action="store_true", help="Scrape Wikipedia (steps 1-2)")
    parser.add_argument("--features", action="store_true", help="Engineer features + mirror dataset (steps 3-4)")
    parser.add_argument("--train",    action="store_true", help="Train all models and save best")
    parser.add_argument("--all",      action="store_true", help="Full pipeline: scrape + features + train")
    args = parser.parse_args()

    # Default to --all if no flag given
    if not any([args.scrape, args.features, args.train, args.all]):
        args.all = True

    print("=" * 60)
    print("  BWF Men's Singles — Data & Training Pipeline")
    print("=" * 60)
    pipeline_start = time.time()

    if args.all:
        all_steps = SCRAPE_STEPS + FEATURE_STEPS + TRAIN_STEPS
        run_steps(all_steps, step_offset=0, total_steps=len(all_steps))
    else:
        offset = 0
        if args.scrape:
            run_steps(SCRAPE_STEPS, step_offset=offset, total_steps=len(SCRAPE_STEPS))
            offset += len(SCRAPE_STEPS)
        if args.features:
            run_steps(FEATURE_STEPS, step_offset=0, total_steps=len(FEATURE_STEPS))
        if args.train:
            run_steps(TRAIN_STEPS, step_offset=0, total_steps=len(TRAIN_STEPS))

    total = time.time() - pipeline_start
    print("\n" + "=" * 60)
    print(f"  Pipeline complete!  Total time: {total:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
