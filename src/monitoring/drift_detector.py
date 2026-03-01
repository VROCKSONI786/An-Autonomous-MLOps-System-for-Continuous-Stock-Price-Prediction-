import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Evidently import with version guard ──────────────────────────────────────

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset
    from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
    EVIDENTLY_AVAILABLE = True
    logger.info("Evidently loaded successfully.")
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently not installed. Falling back to scipy-based drift detection.")


# ── Project root helper ──────────────────────────────────────────────────────

def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


# ── Scipy fallback ───────────────────────────────────────────────────────────

def _ks_drift(ref: pd.Series, cur: pd.Series, threshold: float = 0.05) -> dict:
    """Kolmogorov-Smirnov test as a fallback drift measure."""
    ref_clean = ref.dropna()
    cur_clean = cur.dropna()
    if len(ref_clean) < 5 or len(cur_clean) < 5:
        return {"drift_detected": False, "drift_score": 0.0, "p_value": 1.0}
    stat, p_value = stats.ks_2samp(ref_clean, cur_clean)
    return {
        "drift_detected": bool(p_value < threshold),
        "drift_score": round(float(stat), 4),
        "p_value": round(float(p_value), 4),
    }


# ── DriftDetector ────────────────────────────────────────────────────────────

class DriftDetector:
    """Detect data drift between reference and current windows."""

    def __init__(self, reference_period_days: int = 90, detection_threshold: float = 0.05):
        self.reference_period_days = reference_period_days
        self.detection_threshold = detection_threshold
        self.reports_dir = os.path.join(_project_root(), "data", "reports")
        os.makedirs(self.reports_dir, exist_ok=True)

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_data(self, filepath: str | None = None) -> pd.DataFrame | None:
        if filepath is None:
            filepath = os.path.join(
                _project_root(), "data", "processed", "preprocessed_data.csv"
            )
        if not os.path.exists(filepath):
            logger.error(f"Preprocessed data not found: {filepath}")
            return None
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df

    # ── Window splitting ──────────────────────────────────────────────────────

    def split_reference_current(
        self, df: pd.DataFrame, symbol: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        symbol_data = df[df["Symbol"] == symbol].copy().sort_index()

        end_date = symbol_data.index.max()
        ref_end = end_date - timedelta(days=self.reference_period_days)
        ref_start = ref_end - timedelta(days=self.reference_period_days)

        reference_df = symbol_data[
            (symbol_data.index >= ref_start) & (symbol_data.index < ref_end)
        ].drop(columns=["Symbol"], errors="ignore")

        current_df = symbol_data[symbol_data.index >= ref_end].drop(
            columns=["Symbol"], errors="ignore"
        )

        # Keep only numeric columns
        reference_df = reference_df.select_dtypes(include=[np.number])
        current_df = current_df.select_dtypes(include=[np.number])
        # Align columns
        common_cols = reference_df.columns.intersection(current_df.columns)
        reference_df = reference_df[common_cols]
        current_df = current_df[common_cols]

        logger.info(
            f"{symbol}: reference={len(reference_df)} rows, current={len(current_df)} rows"
        )
        return reference_df, current_df

    # ── Column-level drift ────────────────────────────────────────────────────

    def analyze_column_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        symbol: str,
    ) -> list[dict]:
        column_drifts = []

        if EVIDENTLY_AVAILABLE:
            for col in reference_df.columns:
                try:
                    report = Report(metrics=[ColumnDriftMetric(column_name=col)])
                    report.run(reference_data=reference_df, current_data=current_df)
                    rd = report.as_dict()
                    result = rd["metrics"][0]["result"]
                    column_drifts.append(
                        {
                            "column": col,
                            "drift_detected": bool(result.get("drift_detected", False)),
                            "drift_score": round(float(result.get("drift_score", 0.0)), 4),
                            "method": "evidently",
                        }
                    )
                except Exception as e:
                    logger.debug(f"Evidently column drift failed for {col}: {e}")
                    # Fall back to KS for this column
                    ks = _ks_drift(reference_df[col], current_df[col])
                    column_drifts.append({"column": col, "method": "ks", **ks})
        else:
            for col in reference_df.columns:
                ks = _ks_drift(reference_df[col], current_df[col])
                column_drifts.append({"column": col, "method": "ks", **ks})

        column_drifts.sort(key=lambda x: x["drift_score"], reverse=True)
        return column_drifts

    # ── Dataset-level drift ───────────────────────────────────────────────────

    def detect_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        symbol: str,
    ) -> dict | None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if EVIDENTLY_AVAILABLE:
            try:
                report = Report(metrics=[DataDriftPreset()])
                report.run(reference_data=reference_df, current_data=current_df)
                rd = report.as_dict()

                # Navigate the result dict safely
                metrics_result = rd.get("metrics", [{}])[0].get("result", {})
                drift_detected = bool(metrics_result.get("dataset_drift", False))
                drift_share = float(metrics_result.get("share_of_drifted_columns", 0.0))
                n_drifted = int(metrics_result.get("number_of_drifted_columns", 0))

                # Save HTML report
                report_path = os.path.join(
                    self.reports_dir, f"{symbol}_drift_{timestamp}.html"
                )
                report.save_html(report_path)
                logger.info(f"HTML report saved: {report_path}")

                return {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "drift_detected": drift_detected,
                    "drift_score": round(drift_share, 4),
                    "n_drifted_columns": n_drifted,
                    "report_path": report_path,
                    "reference_size": len(reference_df),
                    "current_size": len(current_df),
                    "method": "evidently",
                }
            except Exception as e:
                logger.warning(f"Evidently dataset drift failed for {symbol}: {e}. Falling back to KS.")

        # Scipy fallback — aggregate KS results
        col_results = self.analyze_column_drift(reference_df, current_df, symbol)
        if not col_results:
            return None

        drifted = [c for c in col_results if c["drift_detected"]]
        drift_share = round(len(drifted) / len(col_results), 4) if col_results else 0.0
        drift_detected = drift_share > 0.3  # >30 % columns drifted

        # Save JSON report as fallback
        report_path = os.path.join(
            self.reports_dir, f"{symbol}_drift_{timestamp}.json"
        )
        with open(report_path, "w") as f:
            json.dump(
                {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "drift_detected": drift_detected,
                    "drift_share": drift_share,
                    "column_results": col_results,
                },
                f,
                indent=2,
            )
        logger.info(f"JSON fallback report saved: {report_path}")

        return {
            "symbol": symbol,
            "timestamp": timestamp,
            "drift_detected": drift_detected,
            "drift_score": drift_share,
            "n_drifted_columns": len(drifted),
            "report_path": report_path,
            "reference_size": len(reference_df),
            "current_size": len(current_df),
            "method": "scipy-ks",
        }

    # ── Check all symbols ─────────────────────────────────────────────────────

    def check_all_symbols(self, symbols: list[str]) -> list[dict]:
        logger.info("=" * 60)
        logger.info("Starting Drift Detection")
        logger.info("=" * 60)

        df = self.load_data()
        if df is None:
            return []

        all_results = []

        for symbol in symbols:
            logger.info(f"\n--- {symbol} ---")
            try:
                ref_df, cur_df = self.split_reference_current(df, symbol)

                if ref_df.empty or cur_df.empty:
                    logger.warning(f"Insufficient data for {symbol}, skipping.")
                    continue

                drift_result = self.detect_drift(ref_df, cur_df, symbol)
                if drift_result is None:
                    continue

                col_drifts = self.analyze_column_drift(ref_df, cur_df, symbol)
                drift_result["column_drifts"] = col_drifts

                all_results.append(drift_result)

                status = "⚠ DRIFT" if drift_result["drift_detected"] else "✓ OK"
                logger.info(
                    f"{symbol}: {status} | score={drift_result['drift_score']:.4f} "
                    f"| {drift_result['n_drifted_columns']} cols drifted"
                )
                if col_drifts:
                    logger.info("  Top 5 drifting features:")
                    for c in col_drifts[:5]:
                        logger.info(f"    {c['column']}: {c['drift_score']:.4f}")

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}", exc_info=True)
                continue

        self._save_summary(all_results)
        logger.info("\nDrift Detection Complete")
        return all_results

    # ── Summary persistence ───────────────────────────────────────────────────

    def _save_summary(self, results: list[dict]) -> None:
        if not results:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.reports_dir, f"drift_summary_{timestamp}.json")
        # Remove non-serialisable objects
        clean = json.loads(json.dumps(results, default=str))
        with open(summary_path, "w") as f:
            json.dump(clean, f, indent=2)
        logger.info(f"Drift summary saved: {summary_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = _project_root()
    config_path = os.path.join(root, "config", "config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    symbols = config["data_collection"]["stock_symbols"]
    ref_days = config.get("drift", {}).get("reference_period_days", 90)

    detector = DriftDetector(reference_period_days=ref_days)
    results = detector.check_all_symbols(symbols)

    print("\n" + "=" * 60)
    print("DRIFT SUMMARY")
    print("=" * 60)
    for r in results:
        status = "⚠ DRIFT DETECTED" if r["drift_detected"] else "✓ No Drift"
        print(
            f"{r['symbol']}: {status} | score={r['drift_score']:.4f} "
            f"| method={r['method']}"
        )