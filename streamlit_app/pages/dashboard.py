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

MODELS_DIR   = os.path.join(PROJECT_ROOT, "models", "saved_models")
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
LOGS_DIR     = os.path.join(PROJECT_ROOT, "logs")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "data", "reports")


def _load_config() -> dict:
    with open(os.path.join(PROJECT_ROOT, "config", "config.yaml")) as f:
        return yaml.safe_load(f)


def _count_files(directory: str, pattern: str) -> int:
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if pattern in f])


def show():
    st.header("🏠 Dashboard Overview")

    config  = _load_config()
    symbols = config["data_collection"]["stock_symbols"]

    # ── System status KPIs ────────────────────────────────────────────────────
    st.subheader("📊 System Status")

    data_files   = _count_files(DATA_RAW_DIR, "stock_prices_")
    model_files  = _count_files(MODELS_DIR, ".keras")
    drift_files  = _count_files(REPORTS_DIR, "drift_summary_")
    perf_files   = _count_files(LOGS_DIR, "performance_metrics_")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tracked Stocks", len(symbols))
    c2.metric("Data Snapshots", data_files,
              delta="Ready" if data_files else "Run Collection")
    c3.metric("Trained Models", f"{model_files}/{len(symbols)}",
              delta="✅ All trained" if model_files == len(symbols) else "⚠ Incomplete")
    c4.metric("Drift Reports", drift_files)
    c5.metric("Perf Snapshots", perf_files)

    st.divider()

    # ── Tracked stocks ────────────────────────────────────────────────────────
    st.subheader("📈 Tracked Stocks")
    cols = st.columns(len(symbols))
    for i, symbol in enumerate(symbols):
        company = symbol.replace(".NS", "")
        model_exists = os.path.exists(
            os.path.join(MODELS_DIR, f"{symbol}_lstm_model.keras")
        )
        status = "🟢 Trained" if model_exists else "🔴 Not trained"
        with cols[i]:
            st.metric(company, status, label_visibility="visible")

    st.divider()

    # ── Pipeline status ───────────────────────────────────────────────────────
    st.subheader("🔄 Pipeline Status")

    steps = [
        ("Data Collection",     os.path.exists(DATA_RAW_DIR) and data_files > 0,          "data/raw/stock_prices_*.csv"),
        ("Preprocessing",       os.path.exists(os.path.join(PROJECT_ROOT, "data", "processed", "preprocessed_data.csv")), "data/processed/preprocessed_data.csv"),
        ("Scaler",              os.path.exists(os.path.join(PROJECT_ROOT, "models", "scaler.pkl")), "models/scaler.pkl"),
        ("Model Training",      model_files > 0,                                           f"models/saved_models/ ({model_files} models)"),
        ("Performance Monitor", perf_files > 0,                                            "logs/performance_metrics_*.json"),
        ("Drift Detection",     drift_files > 0,                                           "data/reports/drift_summary_*.json"),
    ]

    rows = []
    for name, done, path in steps:
        rows.append({
            "Step":   name,
            "Status": "✅ Done" if done else "⏳ Pending",
            "Output": path,
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.divider()

    # ── Recent activity ───────────────────────────────────────────────────────
    st.subheader("📋 Recent Files")

    all_files = []
    for directory, label in [
        (DATA_RAW_DIR, "Data"),
        (LOGS_DIR, "Logs"),
        (REPORTS_DIR, "Reports"),
    ]:
        if os.path.exists(directory):
            for f in os.listdir(directory):
                fpath = os.path.join(directory, f)
                if os.path.isfile(fpath):
                    all_files.append({
                        "File":     f,
                        "Type":     label,
                        "Modified": datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M"),
                        "Size":     f"{os.path.getsize(fpath)/1024:.1f} KB",
                    })

    if all_files:
        df_files = pd.DataFrame(all_files)
        df_files = df_files.sort_values("Modified", ascending=False).head(15)
        st.dataframe(df_files, width="stretch", hide_index=True)
    else:
        st.info("No activity yet. Start by collecting data.")

    st.divider()

    # ── Quick actions ─────────────────────────────────────────────────────────
    st.subheader("⚡ Quick Actions")
    c1, c2, c3, c4 = st.columns(4)

    if c1.button("📥 Collect Data", width='stretch'):
        st.info("Go to **📥 Data Collection** in the sidebar.")
    if c2.button("🤖 Train Models", width='stretch'):
        st.info("Go to **🤖 Model Training** in the sidebar.")
    if c3.button("📈 Predictions", width='stretch'):
        st.info("Go to **📈 Predictions** in the sidebar.")
    if c4.button("📊 Monitor", width='stretch'):
        st.info("Go to **📊 Performance Monitor** in the sidebar.")