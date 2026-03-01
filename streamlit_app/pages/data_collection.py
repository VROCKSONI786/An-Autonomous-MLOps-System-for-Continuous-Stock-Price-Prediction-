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

from data_collection.data_orchestrator import DataOrchestrator


def _load_config() -> dict:
    with open(os.path.join(PROJECT_ROOT, "config", "config.yaml")) as f:
        return yaml.safe_load(f)


def show():
    st.header("📥 Data Collection")
    st.write("Collect stock price data, fundamentals, and news for analysis.")

    config  = _load_config()
    symbols = config["data_collection"]["stock_symbols"]
    data_dir = os.path.join(PROJECT_ROOT, "data", "raw")

    # ── Settings ──────────────────────────────────────────────────────────────
    st.subheader("⚙️ Collection Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Historical Period", ["1y", "2y", "5y", "max"], index=1, disabled=True,
                     help="Configured in config/config.yaml")
        st.selectbox("Data Interval", ["1d", "1wk", "1mo"], index=0, disabled=True,
                     help="Configured in config/config.yaml")
    with col2:
        st.info(
            "**Free-tier Gemini limits**\n\n"
            "• 15 requests/min\n"
            "• 1500 requests/day\n\n"
            "Sentiment analyzes **top 10 articles** with 5s delay between each."
        )

    st.subheader("📊 Configured Stocks")
    st.write(" | ".join([f"**{s.replace('.NS','')}**" for s in symbols]))

    # ── Collect button ────────────────────────────────────────────────────────
    if st.button("🚀 Start Data Collection", width='stretch', type="primary"):
        progress = st.progress(0, text="Initializing…")
        status   = st.empty()
        log_box  = st.empty()

        try:
            orchestrator = DataOrchestrator()

            status.text("📊 Collecting stock prices…")
            progress.progress(20, text="Fetching stock prices…")
            data = orchestrator.collect_all_data()
            progress.progress(100, text="✅ Complete")

            st.success("✅ Data collected successfully!")

            # Summary metrics
            st.subheader("📋 Collection Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price Records",    len(data["price_data"])       if data["price_data"]       is not None else 0)
            c2.metric("Companies",        len(data["fundamental_data"]) if data["fundamental_data"] is not None else 0)
            c3.metric("News Articles",    len(data["news_data"])        if data["news_data"]        is not None else 0)
            c4.metric("Sentiment Rows",   len(data["sentiment_data"])   if data["sentiment_data"]   is not None else 0)

            # Sample price data
            if data["price_data"] is not None:
                st.subheader("📈 Sample Price Data")
                st.dataframe(
                    data["price_data"].head(10),
                    width="stretch",
                    hide_index=False,
                )

            # Sample sentiment
            if data["sentiment_data"] is not None and not data["sentiment_data"].empty:
                st.subheader("🧠 Sentiment Sample")
                cols_to_show = ["title", "sentiment", "confidence", "impact", "reasoning"]
                show_cols = [c for c in cols_to_show if c in data["sentiment_data"].columns]
                st.dataframe(
                    data["sentiment_data"][show_cols].head(10),
                    width="stretch",
                )

        except Exception as e:
            st.error(f"❌ Error during data collection: {e}")
            progress.progress(0)

    # ── Existing files ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📁 Existing Data Files")

    if os.path.exists(data_dir):
        files = sorted(os.listdir(data_dir), reverse=True)
        csv_files = [f for f in files if f.endswith(".csv")]

        if csv_files:
            rows = []
            for f in csv_files:
                fpath = os.path.join(data_dir, f)
                size  = os.path.getsize(fpath)
                rows.append({
                    "Filename": f,
                    "Size":     f"{size/1024:.1f} KB",
                    "Modified": datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M"),
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("No data files found. Click 'Start Data Collection' to begin.")
    else:
        st.info("Data directory not found. Click 'Start Data Collection' to create it.")