"""
model_trainer.py
─────────────────
Orchestrates training for all configured stock symbols.
Canonical model save path: {project_root}/models/saved_models/
All components (predictions.py, performance_monitor.py) load from this same path.

Fix history:
  - Unified save path to {project_root}/models/saved_models/ (was inconsistent)
  - Uses _project_root() helper so it works regardless of CWD
  - MLflow artifact logging preserved
  - Plots saved to logs/plots/
"""

import os
import sys
import yaml
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.lstm_model import LSTMStockModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _project_root() -> str:
    """Return absolute project root regardless of where script is called from."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


class ModelTrainer:

    def __init__(self, config_path: str = None):
        root = _project_root()
        cfg  = config_path or os.path.join(root, "config", "config.yaml")

        with open(cfg) as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config.get("model", {})
        self.model_config.setdefault("sequence_length", 60)
        self.model_config.setdefault("n_features",      21)
        self.model_config.setdefault("lstm_units",     128)
        self.model_config.setdefault("dropout_rate",   0.2)
        self.model_config.setdefault("learning_rate",  0.001)
        self.model_config.setdefault("batch_size",     32)
        self.model_config.setdefault("epochs",         100)
        self.model_config.setdefault("patience",       15)

        # ── Canonical paths ────────────────────────────────────────────────────
        # ALL components must use these exact paths.
        # predictions.py, performance_monitor.py, retraining_scheduler.py
        # all load from models_dir.
        self.models_dir = os.path.join(root, "models", "saved_models")
        self.plots_dir  = os.path.join(root, "logs", "plots")
        self.mlflow_uri = self.config.get("mlflow", {}).get(
            "tracking_uri", "http://localhost:5000"
        )

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.plots_dir,  exist_ok=True)

        logger.info(f"ModelTrainer initialised")
        logger.info(f"  Models dir: {self.models_dir}")
        logger.info(f"  Plots dir:  {self.plots_dir}")
        logger.info(f"  MLflow URI: {self.mlflow_uri}")

    # ── Train single symbol ────────────────────────────────────────────────────

    def train_for_symbol(self, symbol: str) -> tuple:
        """
        Train LSTM model for one symbol.
        Returns (model, test_metrics) where test_metrics = [loss, mae, mse].
        Model is saved to self.models_dir/{symbol}_lstm_model.keras
        """
        logger.info(f"\n{'='*55}")
        logger.info(f"Training: {symbol}")
        logger.info(f"{'='*55}")

        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment("stock_prediction")

        with mlflow.start_run(run_name=f"Train_{symbol}"):

            # ── Initialise model ───────────────────────────────────────────────
            lstm = LSTMStockModel(
                sequence_length=self.model_config["sequence_length"],
                n_features=self.model_config["n_features"],
            )
            lstm.model_config.update(self.model_config)

            # ── Prepare data ───────────────────────────────────────────────────
            X_train, X_val, X_test, y_train, y_val, y_test = lstm.prepare_data(
                symbol=symbol,
                test_size=0.2,
                validation_split=0.1,
            )

            # Update n_features from actual data shape (handles macro features)
            lstm.n_features = X_train.shape[2]

            # ── Build & train ──────────────────────────────────────────────────
            lstm.build_model()

            train_result = lstm.train(
                X_train, y_train,
                X_val,   y_val,
                symbol=symbol,
                save_dir=self.models_dir,
            )

            # ── Evaluate on test set ───────────────────────────────────────────
            test_metrics = lstm.evaluate(X_test, y_test)
            test_loss, test_mae, test_mse = test_metrics[0], test_metrics[1], test_metrics[2]

            logger.info(
                f"Test results — Loss: {test_loss:.6f} | "
                f"MAE: {test_mae:.6f} | MSE: {test_mse:.6f}"
            )

            # ── Log hyperparameters to MLflow ──────────────────────────────────
            mlflow.log_params({
                "symbol":          symbol,
                "sequence_length": self.model_config["sequence_length"],
                "n_features":      lstm.n_features,
                "lstm_units":      self.model_config["lstm_units"],
                "dropout_rate":    self.model_config["dropout_rate"],
                "learning_rate":   self.model_config["learning_rate"],
                "batch_size":      self.model_config["batch_size"],
                "epochs_max":      self.model_config["epochs"],
                "patience":        self.model_config["patience"],
            })

            # ── Log metrics to MLflow ──────────────────────────────────────────
            mlflow.log_metrics({
                "test_loss":       test_loss,
                "test_mae":        test_mae,
                "test_mse":        test_mse,
                "best_val_loss":   train_result["best_val_loss"],
                "best_val_mae":    train_result["best_val_mae"],
                "epochs_run":      train_result["epochs_run"],
            })

            # ── Training history plot ──────────────────────────────────────────
            history_plot = self._plot_training_history(
                train_result["history"], symbol
            )
            if history_plot:
                mlflow.log_artifact(history_plot)

            # ── Predictions plot ───────────────────────────────────────────────
            pred_plot = self._plot_predictions(
                lstm, X_test, y_test, symbol
            )
            if pred_plot:
                mlflow.log_artifact(pred_plot)

            # ── Log model artifact ─────────────────────────────────────────────
            model_path = train_result["model_path"]
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path)
                logger.info(f"Model artifact logged: {model_path}")

        logger.info(f"✓ {symbol} training complete → {model_path}")
        return lstm.model, [test_loss, test_mae, test_mse]

    # ── Train all symbols ──────────────────────────────────────────────────────

    def train_all(self) -> dict:
        symbols = self.config["data_collection"]["stock_symbols"]
        results = {}

        logger.info(f"\nTraining {len(symbols)} models: {symbols}")

        for symbol in symbols:
            try:
                model, metrics = self.train_for_symbol(symbol)
                results[symbol] = {
                    "status":    "success",
                    "test_loss": metrics[0],
                    "test_mae":  metrics[1],
                    "test_mse":  metrics[2],
                }
                logger.info(
                    f"✓ {symbol}: loss={metrics[0]:.6f} | "
                    f"MAE={metrics[1]:.6f} | MSE={metrics[2]:.6f}"
                )
            except Exception as e:
                logger.error(f"✗ {symbol} failed: {e}", exc_info=True)
                results[symbol] = {"status": "failed", "error": str(e)}

        self._print_summary(results)
        return results

    # ── Plots ──────────────────────────────────────────────────────────────────

    def _plot_training_history(
        self,
        history: dict,
        symbol: str,
    ) -> str | None:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"{symbol} — Training History", fontsize=14)

            # Loss
            axes[0].plot(history["loss"],     label="Train Loss", linewidth=2)
            axes[0].plot(history["val_loss"], label="Val Loss",   linewidth=2)
            axes[0].set_title("Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # MAE
            axes[1].plot(history["mae"],     label="Train MAE", linewidth=2)
            axes[1].plot(history["val_mae"], label="Val MAE",   linewidth=2)
            axes[1].set_title("MAE")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("MAE")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            path = os.path.join(self.plots_dir, f"{symbol}_history.png")
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"History plot saved: {path}")
            return path
        except Exception as e:
            logger.warning(f"History plot failed for {symbol}: {e}")
            return None

    def _plot_predictions(
        self,
        lstm: LSTMStockModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
        symbol: str,
        n_samples: int = 100,
    ) -> str | None:
        try:
            y_pred = lstm.predict(X_test).flatten()

            # Plot last n_samples for clarity
            n = min(n_samples, len(y_test))
            x_ax = range(n)
            actual = y_test[-n:]
            pred   = y_pred[-n:]

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(x_ax, actual, label="Actual",    linewidth=2, color="#2196F3")
            ax.plot(x_ax, pred,   label="Predicted", linewidth=2, color="#FF9800",
                    linestyle="--")
            ax.fill_between(x_ax, actual, pred, alpha=0.15, color="#9C27B0")
            ax.set_title(f"{symbol} — Test Set Predictions (scaled)", fontsize=13)
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Scaled Close Price")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            path = os.path.join(self.plots_dir, f"{symbol}_predictions.png")
            plt.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Prediction plot saved: {path}")
            return path
        except Exception as e:
            logger.warning(f"Prediction plot failed for {symbol}: {e}")
            return None

    # ── Summary ────────────────────────────────────────────────────────────────

    def _print_summary(self, results: dict) -> None:
        logger.info(f"\n{'='*60}")
        logger.info("Training Summary")
        logger.info(f"{'='*60}")

        successful = [s for s, r in results.items() if r["status"] == "success"]
        failed     = [s for s, r in results.items() if r["status"] == "failed"]

        for symbol, r in results.items():
            if r["status"] == "success":
                logger.info(
                    f"  ✓ {symbol:<20} "
                    f"loss={r['test_loss']:.6f} | "
                    f"MAE={r['test_mae']:.6f}"
                )
            else:
                logger.error(f"  ✗ {symbol:<20} FAILED: {r.get('error','?')}")

        logger.info(f"\n  Successful: {len(successful)}/{len(results)}")
        if failed:
            logger.warning(f"  Failed:     {failed}")

        logger.info(f"\nModels saved to: {self.models_dir}")
        saved = [f for f in os.listdir(self.models_dir) if f.endswith(".keras")]
        for f in sorted(saved):
            path = os.path.join(self.models_dir, f)
            size = os.path.getsize(path) / (1024 * 1024)
            logger.info(f"  {f}  ({size:.1f} MB)")


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.train_all()