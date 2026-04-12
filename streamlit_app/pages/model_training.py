import os
import sys
import yaml
import pandas as pd
import streamlit as st
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for p in [SRC_ROOT, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model.model_trainer import ModelTrainer
from preprocessing.data_preprocessor import DataPreprocessor

MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")


def _load_config() -> dict:
    with open(os.path.join(PROJECT_ROOT, "config", "config.yaml")) as f:
        return yaml.safe_load(f)


def show():
    st.header(" Model Training")
    st.write("Train Bidirectional LSTM models for stock price prediction with MLflow tracking.")

    config  = _load_config()
    symbols = config["data_collection"]["stock_symbols"]

    # ── Settings ──────────────────────────────────────────────────────────────
    st.subheader(" Training Settings")
    col1, col2, col3 = st.columns(3)

    with col1:
        epochs      = st.number_input("Epochs",     min_value=10,  max_value=200, value=50,  step=10)
        batch_size  = st.number_input("Batch Size", min_value=16,  max_value=128, value=32,  step=16)
    with col2:
        lstm_units  = st.number_input("LSTM Units", min_value=32,  max_value=256, value=128, step=32)
        dropout     = st.slider("Dropout Rate",     min_value=0.0, max_value=0.5, value=0.2, step=0.05)
    with col3:
        seq_length  = st.number_input("Sequence Length", min_value=30, max_value=120, value=60, step=10)
        # Show actual MLflow backend being used
        import socket
        _mlflow_cfg = config.get("mlflow", {}).get("tracking_uri", "")
        _server_up  = False
        if _mlflow_cfg.startswith("http"):
            try:
                from urllib.parse import urlparse as _up
                _p = _up(_mlflow_cfg)
                with socket.create_connection((_p.hostname or "localhost", _p.port or 5000), timeout=1):
                    _server_up = True
            except Exception:
                pass
        if _server_up:
            st.metric("MLflow", "Server", help=f"Connected to {_mlflow_cfg}")
        else:
            st.metric("MLflow", "SQLite", help="Using local mlflow/mlflow.db (no server needed on Streamlit Cloud)")

    # ── Stock selection ────────────────────────────────────────────────────────
    st.subheader(" Select Stocks to Train")
    selected = st.multiselect("Stocks", symbols, default=symbols)

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    st.subheader(" Step 1 — Preprocess Data")
    st.caption("Must run before training if data has changed.")

    if st.button(" Preprocess Data", width='stretch'):
        with st.spinner("Preprocessing… creating technical indicators and scaling data."):
            try:
                preprocessor = DataPreprocessor()
                result       = preprocessor.preprocess_pipeline()
                if result is not None:
                    st.success(f" Preprocessed data shape: {result.shape}")
                    st.dataframe(result.head(5), width="stretch")
                else:
                    st.error(" Preprocessing failed — ensure stock_prices_*.csv exists in data/raw/")
            except Exception as e:
                st.error(f" Preprocessing error: {e}")

    # ── Step 2: Train ─────────────────────────────────────────────────────────
    st.subheader(" Step 2 — Train Models")

    if st.button(" Start Training", width='stretch', type="primary"):
        if not selected:
            st.warning(" Select at least one stock.")
            return

        # Check preprocessed data exists
        prep_path = os.path.join(PROJECT_ROOT, "data", "processed", "preprocessed_data.csv")
        if not os.path.exists(prep_path):
            st.error(" preprocessed_data.csv not found. Run Preprocess Data first.")
            return

        progress = st.progress(0, text="Starting training…")
        log_area = st.empty()

        try:
            trainer = ModelTrainer()
            trainer.model_config["epochs"]          = int(epochs)
            trainer.model_config["batch_size"]       = int(batch_size)
            trainer.model_config["lstm_units"]       = int(lstm_units)
            trainer.model_config["dropout_rate"]     = float(dropout)
            trainer.model_config["sequence_length"]  = int(seq_length)

            results = {}

            for i, symbol in enumerate(selected):
                pct = int((i / len(selected)) * 100)
                progress.progress(pct, text=f"Training {symbol}… ({i+1}/{len(selected)})")
                log_area.info(f"Training {symbol}…")

                try:
                    model, metrics = trainer.train_for_symbol(symbol)
                    results[symbol] = metrics
                    log_area.success(f" {symbol} — MAE: {metrics[1]:.6f}")
                except Exception as e:
                    log_area.warning(f" {symbol} failed: {e}")

            progress.progress(100, text=" Training complete!")

            if results:
                st.success(f" Trained {len(results)}/{len(selected)} models successfully!")
                results_df = pd.DataFrame([
                    {
                        "Symbol":    sym,
                        "Test Loss": f"{m[0]:.6f}",
                        "Test MAE":  f"{m[1]:.6f}",
                        "Test MSE":  f"{m[2]:.6f}",
                    }
                    for sym, m in results.items()
                ])
                st.dataframe(results_df, width="stretch", hide_index=True)
            else:
                st.error(" No models trained successfully.")

        except Exception as e:
            st.error(f" Training error: {e}")

    # ── Saved models ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader(" Saved Models")

    if os.path.exists(MODELS_DIR):
        model_files = sorted(
            [f for f in os.listdir(MODELS_DIR) if f.endswith(".keras")],
            reverse=True,
        )
        if model_files:
            rows = []
            for f in model_files:
                fpath = os.path.join(MODELS_DIR, f)
                size  = os.path.getsize(fpath) / (1024 * 1024)
                mtime = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M")
                rows.append({
                    "Model File":  f,
                    "Symbol":      f.replace("_lstm_model.keras", ""),
                    "Size (MB)":   f"{size:.1f}",
                    "Last Trained": mtime,
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("No trained models found. Click 'Start Training' to create them.")
    else:
        st.info(f"Models directory not found: `{MODELS_DIR}`")