import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import joblib
import yaml
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Set MLflow tracking URI to project root (BEFORE importing mlflow) ───
_project_root_temp = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_mlflow_db_path = os.path.join(_project_root_temp, "mlflow", "mlflow.db")
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_mlflow_db_path}"

import mlflow

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.lstm_model import LSTMStockModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLOSE_IDX = 3  # 'Close' is the 4th column in feature_columns


def _project_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


class PerformanceMonitor:

    def __init__(self):
        root = _project_root()
        self.models_dir  = os.path.join(root, "models", "saved_models")
        self.scaler_path = os.path.join(root, "models", "scaler.pkl")
        self.logs_dir    = os.path.join(root, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)

        self.scaler = None
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            logger.info("Scaler loaded.")
        else:
            logger.warning("Scaler not found — metrics will be on scaled data.")

    # ── Inverse transform ──────────────────────────────────────────────────────

    def _to_inr(self, scaled: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            return scaled
        n      = len(scaled)
        n_feat = self.scaler.n_features_in_
        dummy  = np.zeros((n, n_feat))
        dummy[:, CLOSE_IDX] = scaled.flatten()
        return self.scaler.inverse_transform(dummy)[:, CLOSE_IDX]

    # ── All accuracy metrics ───────────────────────────────────────────────────

    def _compute_metrics(
        self,
        y_actual: np.ndarray,
        y_pred: np.ndarray,
        symbol: str,
        model_path: str,
        test_samples: int,
    ) -> dict:
        # Standard regression metrics
        mae  = float(mean_absolute_error(y_actual, y_pred))
        mse  = float(mean_squared_error(y_actual, y_pred))
        rmse = float(np.sqrt(mse))
        r2   = float(r2_score(y_actual, y_pred))

        # MAPE — guard zero division
        safe_actual = np.where(np.abs(y_actual) < 1e-10, 1e-10, y_actual)
        mape = float(np.mean(np.abs((y_actual - y_pred) / safe_actual)) * 100)

        # ── Direction accuracy ──────────────────────────────────────────
        # "Did we correctly predict whether price went UP or DOWN today?"
        # This is the most important metric for trading decisions.
        if len(y_actual) > 1:
            actual_dir = np.sign(np.diff(y_actual))
            pred_dir   = np.sign(np.diff(y_pred))
            direction_acc = float(np.mean(actual_dir == pred_dir) * 100)
        else:
            direction_acc = 0.0

        # ── Threshold accuracy ──────────────────────────────────────────
        # "What % of predictions are within N% of the actual price?"
        abs_pct_err = np.abs((y_actual - y_pred) / safe_actual) * 100

        within_1pct  = float(np.mean(abs_pct_err <= 1.0) * 100)
        within_2pct  = float(np.mean(abs_pct_err <= 2.0) * 100)   # threshold_accuracy
        within_5pct  = float(np.mean(abs_pct_err <= 5.0) * 100)

        # ── Hit rate — combined score ───────────────────────────────────
        # Average of direction_accuracy and within_2pct → single number to watch
        hit_rate = round((direction_acc + within_2pct) / 2, 2)

        return {
            "symbol":                    symbol,
            "timestamp":                 datetime.now().isoformat(),
            "test_samples":              test_samples,
            # Core regression metrics
            "mae_inr":                   round(mae,  2),
            "rmse_inr":                  round(rmse, 2),
            "mse_inr":                   round(mse,  2),
            "mape_pct":                  round(mape, 4),
            "r2_score":                  round(r2,   6),
            # Accuracy metrics (most useful for trading)
            "direction_accuracy_pct":    round(direction_acc, 2),
            "threshold_accuracy_pct":    round(within_2pct,   2),   # ±2% band
            "within_1pct_pct":           round(within_1pct,   2),
            "within_5pct_pct":           round(within_5pct,   2),
            "hit_rate_pct":              hit_rate,
            # Price context
            "last_actual_price_inr":     round(float(y_actual[-1]),  2),
            "last_predicted_price_inr":  round(float(y_pred[-1]),    2),
            "avg_actual_price_inr":      round(float(np.mean(y_actual)), 2),
            "model_path":                model_path,
        }

    # ── Single symbol ──────────────────────────────────────────────────────────

    def evaluate_model(self, symbol: str, model_path: str = None) -> dict | None:
        logger.info(f"\nEvaluating {symbol}…")

        if model_path is None:
            candidates = [
                os.path.join(self.models_dir, f"{symbol}_lstm_model.keras"),
                os.path.join(self.models_dir, f"{symbol.replace('.NS','')}_lstm_model.keras"),
            ]
            model_path = next((p for p in candidates if os.path.exists(p)), None)

        if not model_path or not os.path.exists(model_path):
            logger.warning(f"No saved model found for {symbol} in {self.models_dir}")
            return None

        try:
            lstm = LSTMStockModel(sequence_length=60, n_features=21)
            lstm.load_model(model_path)

            _, _, X_test, _, _, y_test = lstm.prepare_data(
                symbol=symbol, test_size=0.2, validation_split=0.1
            )

            if len(X_test) == 0:
                logger.warning(f"No test data for {symbol}")
                return None

            y_pred_scaled = lstm.predict(X_test).flatten()
            y_actual      = self._to_inr(y_test)
            y_pred        = self._to_inr(y_pred_scaled)

            metrics = self._compute_metrics(
                y_actual, y_pred, symbol, model_path, len(y_test)
            )

            logger.info(
                f"  MAE: ₹{metrics['mae_inr']} | RMSE: ₹{metrics['rmse_inr']} | "
                f"MAPE: {metrics['mape_pct']:.2f}% | R²: {metrics['r2_score']:.4f}\n"
                f"  Dir Acc: {metrics['direction_accuracy_pct']:.1f}% | "
                f"Threshold Acc (±2%): {metrics['threshold_accuracy_pct']:.1f}% | "
                f"Hit Rate: {metrics['hit_rate_pct']:.1f}%"
            )
            return metrics

        except Exception as e:
            logger.error(f"Error evaluating {symbol}: {e}", exc_info=True)
            return None

    # ── All symbols ────────────────────────────────────────────────────────────

    def monitor_all_models(self, symbols: list[str]) -> tuple[list[dict], list[str]]:
        logger.info("\n" + "=" * 60)
        logger.info("Performance Monitoring")
        logger.info("=" * 60)

        all_metrics = []
        for sym in symbols:
            m = self.evaluate_model(sym)
            if m:
                all_metrics.append(m)

        self._save_metrics(all_metrics)
        degraded = self._check_degradation(all_metrics)

        if degraded:
            logger.warning(f"\n⚠ Models needing retraining: {degraded}")
        else:
            logger.info("\n✓ All models within thresholds.")

        return all_metrics, degraded

    # ── Degradation thresholds ─────────────────────────────────────────────────

    def _check_degradation(
        self,
        metrics: list[dict],
        mape_threshold:      float = 5.0,   # > 5% MAPE
        r2_threshold:        float = 0.80,  # R² < 0.80
        dir_acc_threshold:   float = 52.0,  # direction accuracy < 52% (barely better than coin flip)
        hit_rate_threshold:  float = 55.0,  # combined hit rate < 55%
    ) -> list[str]:
        degraded = []
        for m in metrics:
            reasons = []
            if m["mape_pct"]                  > mape_threshold:
                reasons.append(f"MAPE={m['mape_pct']:.2f}%")
            if m["r2_score"]                  < r2_threshold:
                reasons.append(f"R²={m['r2_score']:.4f}")
            if m["direction_accuracy_pct"]    < dir_acc_threshold:
                reasons.append(f"DirAcc={m['direction_accuracy_pct']:.1f}%")
            if m["hit_rate_pct"]              < hit_rate_threshold:
                reasons.append(f"HitRate={m['hit_rate_pct']:.1f}%")
            if reasons:
                degraded.append(m["symbol"])
                logger.warning(f"  ⚠ {m['symbol']} degraded: {', '.join(reasons)}")
        return degraded

    # ── Save metrics ───────────────────────────────────────────────────────────

    def _save_metrics(self, metrics: list[dict]) -> None:
        if not metrics:
            logger.warning("No metrics to save — all models missing or failed.")
            return

        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap    = os.path.join(self.logs_dir, f"performance_metrics_{ts}.json")
        history = os.path.join(self.logs_dir, "performance_history.jsonl")

        with open(snap, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Snapshot saved: {snap}")

        with open(history, "a") as f:
            for m in metrics:
                f.write(json.dumps(m) + "\n")
        logger.info(f"History appended: {history}")

    # ── MLflow logging ─────────────────────────────────────────────────────────

    def log_to_mlflow(
        self,
        metrics: list[dict],
        tracking_uri: str = "http://localhost:5000",
    ) -> None:
        try:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment("model_monitoring")
            for m in metrics:
                with mlflow.start_run(run_name=f"Monitor_{m['symbol']}"):
                    mlflow.log_metrics({
                        "mae_inr":                 m["mae_inr"],
                        "rmse_inr":                m["rmse_inr"],
                        "mape_pct":                m["mape_pct"],
                        "r2_score":                m["r2_score"],
                        "direction_accuracy_pct":  m["direction_accuracy_pct"],
                        "threshold_accuracy_pct":  m["threshold_accuracy_pct"],
                        "within_1pct_pct":         m["within_1pct_pct"],
                        "within_5pct_pct":         m["within_5pct_pct"],
                        "hit_rate_pct":            m["hit_rate_pct"],
                    })
                    mlflow.log_param("symbol",    m["symbol"])
                    mlflow.log_param("timestamp", m["timestamp"])
            logger.info("Metrics logged to MLflow.")
        except Exception as e:
            logger.error(f"MLflow logging failed: {e}")

    def load_history(self) -> pd.DataFrame:
        path = os.path.join(self.logs_dir, "performance_history.jsonl")
        if not os.path.exists(path):
            return pd.DataFrame()
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return pd.DataFrame(rows) if rows else pd.DataFrame()


if __name__ == "__main__":
    root = _project_root()
    with open(os.path.join(root, "config", "config.yaml")) as f:
        config = yaml.safe_load(f)

    symbols = config["data_collection"]["stock_symbols"]
    monitor = PerformanceMonitor()
    metrics, degraded = monitor.monitor_all_models(symbols)
    monitor.log_to_mlflow(metrics, config["mlflow"]["tracking_uri"])

    print("\n" + "=" * 75)
    print(f"{'Symbol':<20} {'MAE(₹)':>8} {'MAPE%':>7} {'R²':>7} "
          f"{'DirAcc%':>9} {'Thresh%':>9} {'HitRate%':>10}")
    print("-" * 75)
    for m in metrics:
        flag = " ⚠" if m["symbol"] in degraded else "  "
        print(
            f"{m['symbol']:<20} {m['mae_inr']:>8.2f} {m['mape_pct']:>7.2f} "
            f"{m['r2_score']:>7.4f} {m['direction_accuracy_pct']:>9.1f} "
            f"{m['threshold_accuracy_pct']:>9.1f} {m['hit_rate_pct']:>10.1f}{flag}"
        )
    if degraded:
        print(f"\n⚠ Retraining needed: {degraded}")