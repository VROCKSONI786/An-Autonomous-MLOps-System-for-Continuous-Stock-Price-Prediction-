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
from config_manager import ConfigManager


def _load_config() -> dict:
    with open(os.path.join(PROJECT_ROOT, "config", "config.yaml")) as f:
        return yaml.safe_load(f)


def show():
    st.header(" Data Collection")
    st.write("Collect stock price data, fundamentals, and news for analysis.")

    # Use ConfigManager to get all stocks including custom ones
    cm = ConfigManager()
    symbols = cm.get_all_stocks()
    stats = cm.get_stats()
    data_dir = os.path.join(PROJECT_ROOT, "data", "raw")

    # ── Settings ──────────────────────────────────────────────────────────────
    st.subheader(" Collection Settings")
    col1, col2 = st.columns(2)
    with col1:
        st.selectbox("Historical Period", ["1y", "2y", "5y", "max"], index=1,
                     help="Configured in config/config.yaml")
        st.selectbox("Data Interval", ["1d", "1wk", "1mo"], index=0,
                     help="Configured in config/config.yaml")
    with col2:
        # removed Gemini rate limit warning; users can configure their own API limits elsewhere
        pass

    st.subheader(" Configured Stocks")
    
    # Show breakdown of stock types
    if stats['custom_stocks_count'] > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**📌 Default ({stats['default_stocks_count']}):** {', '.join([s.replace('.NS','') for s in stats['default_stocks']])}")
        with col2:
            st.write(f"**➕ Custom ({stats['custom_stocks_count']}):** {', '.join([s.replace('.NS','') for s in stats['custom_stocks']])}")
    else:
        st.write(" | ".join([f"**{s.replace('.NS','')}**" for s in symbols]))
    
    st.caption(f"Total: {len(symbols)} stocks tracked")

    # ── Collect button ────────────────────────────────────────────────────────
    if st.button(" Start Data Collection", width='stretch', type="primary"):
        progress = st.progress(0, text="Initializing…")
        status   = st.empty()
        log_box  = st.empty()

        try:
            orchestrator = DataOrchestrator()

            status.text(" Collecting stock prices…")
            progress.progress(20, text="Fetching stock prices…")
            data = orchestrator.collect_all_data()
            progress.progress(100, text=" Complete")

            st.success(" Data collected successfully!")

            # Summary metrics
            st.subheader(" Collection Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Price Records",    len(data["price_data"])       if data["price_data"]       is not None else 0)
            c2.metric("Companies",        len(data["fundamental_data"]) if data["fundamental_data"] is not None else 0)
            c3.metric("News Articles",    len(data["news_data"])        if data["news_data"]        is not None else 0)
            c4.metric("Sentiment Rows",   len(data["sentiment_data"])   if data["sentiment_data"]   is not None else 0)

            # Sample price data
            if data["price_data"] is not None:
                st.subheader(" Sample Price Data")
                st.dataframe(
                    data["price_data"].head(10),
                    width="stretch",
                    hide_index=False,
                )

            # Sample sentiment
            if data["sentiment_data"] is not None and not data["sentiment_data"].empty:
                st.subheader(" Sentiment Sample")
                cols_to_show = ["title", "sentiment", "confidence", "impact", "reasoning"]
                show_cols = [c for c in cols_to_show if c in data["sentiment_data"].columns]
                st.dataframe(
                    data["sentiment_data"][show_cols].head(10),
                    width="stretch",
                )

            # Market Sentiment Summary
            if data["sentiment_summary"]:
                st.divider()
                st.subheader(" Market Sentiment Summary")
                st.info(data["sentiment_summary"])

        except Exception as e:
            st.error(f" Error during data collection: {e}")
            progress.progress(0)

    # ── Existing files (hidden in expander) ────────────────────────────────────
    st.divider()
    
    with st.expander(" View Data Collection Sample", expanded=False):
        st.caption("Raw data files collected during each data collection run")
        
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
    
    # ── Display collected news in tabs ─────────────────────────────────────────
    st.divider()
    st.subheader(" Latest Collected News")
    
    _display_news_by_source(data_dir)


def _display_news_by_source(data_dir: str):
    """Display collected news articles organized by source in tabs."""
    
    # Find latest news CSV file
    if not os.path.exists(data_dir):
        st.info("No data directory yet. Collect data first.")
        return
    
    # Get the most recent news CSV file
    news_files = sorted(
        [f for f in os.listdir(data_dir) if f.startswith("news_") and f.endswith(".csv")],
        reverse=True
    )
    
    if not news_files:
        st.info("No news files found. Collect data first.")
        return
    
    latest_news_file = os.path.join(data_dir, news_files[0])
    
    try:
        df = pd.read_csv(latest_news_file)
        
        if df.empty:
            st.info("No news articles in the latest collection.")
            return
        
        # Group by source
        sources = df['source'].unique() if 'source' in df.columns else []
        
        # Exclude moneycontrol for now (articles are from 2016)
        sources = [s for s in sources if s != 'moneycontrol']
        
        if len(sources) == 0:
            st.info("No supported news sources in the latest collection.")
            return
        
        # Create tabs for each source
        tab_names = sorted([s.replace('_', ' ').title() for s in sources])
        tabs = st.tabs(tab_names)
        
        for tab_idx, (tab, source_name) in enumerate(zip(tabs, sorted(sources))):
            with tab:
                source_df = df[df['source'] == source_name].copy()
                
                if source_df.empty:
                    st.info(f"No articles from {source_name}")
                    continue
                
                st.caption(f" {len(source_df)} articles from {source_name}")
                
                # Display each article
                for idx, row in source_df.iterrows():
                    with st.container():
                        # Title + Link
                        title = row.get('title', 'No title')
                        link = row.get('link', '')
                        
                        if link:
                            st.markdown(f"**[{title[:100]}...]({link})**" if len(title) > 100 else f"**[{title}]({link})**")
                        else:
                            st.markdown(f"**{title[:100]}...**" if len(title) > 100 else f"**{title}**")
                        
                        # Summary
                        summary = row.get('summary', '')
                        if summary:
                            summary_text = str(summary)[:200] + ("..." if len(str(summary)) > 200 else "")
                            st.write(f"📝 {summary_text}")
                        
                        # Published date
                        published = row.get('published', '')
                        if published:
                            st.caption(f"🕐 {published}")
                        
                        st.divider()
    
    except Exception as e:
        st.error(f"Error displaying news: {e}")