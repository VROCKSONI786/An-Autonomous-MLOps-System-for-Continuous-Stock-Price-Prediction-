"""
retraining_scheduler.py
────────────────────────
Automated retraining pipeline that:
  1. Collects fresh data
  2. Re-runs preprocessing
  3. Checks drift (DriftDetector)
  4. Checks performance degradation (PerformanceMonitor)
  5. Retrains whichever symbols need it
  6. Can be called by cron / Jenkins or run directly

Usage
-----
    python retraining_scheduler.py               # checks all configured symbols
    python retraining_scheduler.py --force        # retrain all regardless of metrics
    python retraining_scheduler.py --symbol TCS.NS
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

import yaml

# ── Path bootstrap ────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(HERE)          # this script lives in src/, so parent is project root
PROJECT_ROOT = SRC_ROOT  # project root (adjusted)
for p in [SRC_ROOT, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data_collection.data_orchestrator import DataOrchestrator
from monitoring.drift_detector import DriftDetector
from monitoring.performance_monitor import PerformanceMonitor
from model.model_trainer import ModelTrainer
from preprocessing.data_preprocessor import DataPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RetrainingScheduler")


def _load_config() -> dict:
    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _symbols_needing_retrain(
    config: dict,
    symbols: list[str],
    force: bool = False,
) -> list[str]:
    """Return list of symbols that need retraining based on drift or performance."""
    if force:
        logger.info("--force flag: retraining all symbols.")
        return symbols

    needs_retrain: set[str] = set()

    # ── 1. Drift detection ────────────────────────────────────────────────────
    logger.info("\n[Step 1] Drift Detection")
    drift_threshold = config.get("drift", {}).get("detection_threshold", 0.1)
    ref_days = config.get("drift", {}).get("reference_period_days", 90)

    try:
        detector = DriftDetector(reference_period_days=ref_days)
        drift_results = detector.check_all_symbols(symbols)

        for dr in drift_results:
            if dr["drift_detected"] or dr["drift_score"] >= drift_threshold:
                sym = dr["symbol"]
                logger.info(
                    f"  ⚠ Drift detected for {sym} "
                    f"(score={dr['drift_score']:.4f}) → queued for retrain"
                )
                needs_retrain.add(sym)
            else:
                logger.info(f"  ✓ {dr['symbol']} — no drift (score={dr['drift_score']:.4f})")
    except Exception as e:
        logger.error(f"Drift detection error: {e}")

    # ── 2. Performance monitoring ─────────────────────────────────────────────
    logger.info("\n[Step 2] Performance Monitoring")
    try:
        monitor = PerformanceMonitor()
        _, degraded = monitor.monitor_all_models(symbols)
        for sym in degraded:
            logger.info(f"  ⚠ {sym} performance degraded → queued for retrain")
            needs_retrain.add(sym)
    except Exception as e:
        logger.error(f"Performance monitoring error: {e}")

    return sorted(needs_retrain)


def run_retraining(
    symbols_to_retrain: list[str],
    config: dict,
) -> dict:
    """Collect fresh data, preprocess, and retrain the given symbols."""
    if not symbols_to_retrain:
        logger.info("\nNo symbols require retraining. Pipeline complete.")
        return {}

    logger.info(f"\n[Step 3] Retraining: {symbols_to_retrain}")

    # ── Fresh data collection ─────────────────────────────────────────────────
    logger.info("\n  Collecting fresh data…")
    try:
        orchestrator = DataOrchestrator()
        orchestrator.collect_all_data()
        logger.info("  ✓ Data collection complete.")
    except Exception as e:
        logger.error(f"  Data collection failed: {e}")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    logger.info("\n  Preprocessing data…")
    try:
        preprocessor = DataPreprocessor()
        preprocessor.preprocess_pipeline()
        logger.info("  ✓ Preprocessing complete.")
    except Exception as e:
        logger.error(f"  Preprocessing failed: {e}")

    # ── Model training ────────────────────────────────────────────────────────
    trainer = ModelTrainer()
    trainer.symbols = symbols_to_retrain  # override to retrain only needed symbols

    results = {}
    for symbol in symbols_to_retrain:
        logger.info(f"\n  Training {symbol}…")
        try:
            model, metrics = trainer.train_for_symbol(symbol)
            results[symbol] = {
                "test_loss": metrics[0],
                "test_mae": metrics[1],
                "test_mse": metrics[2],
                "retrained_at": datetime.now().isoformat(),
            }
            logger.info(f"  ✓ {symbol} retrained. MAE={metrics[1]:.6f}")
        except Exception as e:
            logger.error(f"  Training failed for {symbol}: {e}", exc_info=True)
            results[symbol] = {"error": str(e)}

    return results


def _save_run_report(
    symbols_checked: list[str],
    symbols_retrained: list[str],
    results: dict,
) -> None:
    """Save a JSON run report to logs/."""
    logs_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "run_timestamp": datetime.now().isoformat(),
        "symbols_checked": symbols_checked,
        "symbols_retrained": symbols_retrained,
        "results": results,
    }
    path = os.path.join(logs_dir, f"retrain_run_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"\nRun report saved: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stock model retraining scheduler")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain all symbols regardless of drift/performance",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Retrain a specific symbol only (e.g. TCS.NS)",
    )
    args = parser.parse_args()

    config = _load_config()
    all_symbols = config["data_collection"]["stock_symbols"]

    if args.symbol:
        if args.symbol not in all_symbols:
            logger.error(f"Symbol {args.symbol!r} not in config.")
            sys.exit(1)
        symbols_to_check = [args.symbol]
    else:
        symbols_to_check = all_symbols

    logger.info("=" * 60)
    logger.info("Retraining Scheduler Started")
    logger.info(f"Symbols: {symbols_to_check}")
    logger.info("=" * 60)

    symbols_to_retrain = _symbols_needing_retrain(
        config, symbols_to_check, force=args.force
    )

    results = run_retraining(symbols_to_retrain, config)

    _save_run_report(symbols_to_check, symbols_to_retrain, results)

    logger.info("\n" + "=" * 60)
    logger.info("Retraining Scheduler Complete")
    logger.info("=" * 60)

    if results:
        print("\nRetrained models:")
        for sym, r in results.items():
            if "error" in r:
                print(f"  {sym}: FAILED — {r['error']}")
            else:
                print(f"  {sym}: MAE={r['test_mae']:.6f}")


if __name__ == "__main__":
    main()