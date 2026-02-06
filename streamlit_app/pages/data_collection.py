import streamlit as st
import sys
import os
import yaml
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data_collection.data_orchestrator import DataOrchestrator

def show():
    st.header("📥 Data Collection")
    
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    symbols = config['data_collection']['stock_symbols']
    
    st.write("Collect stock price data, fundamentals, and news for analysis")
    
    # Display configured stocks
    st.subheader("📊 Configured Stocks")
    st.write(", ".join([s.replace('.NS', '') for s in symbols]))
    
    # Data collection settings
    st.subheader("⚙️ Collection Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        period = st.selectbox("Historical Period", ["1y", "2y", "5y", "max"], index=1)
        interval = st.selectbox("Data Interval", ["1d", "1wk", "1mo"], index=0)
    
    with col2:
        collect_news = st.checkbox("Collect News", value=True)
        collect_fundamentals = st.checkbox("Collect Fundamentals", value=True)
    
    # Start collection button
    if st.button("🚀 Start Data Collection", use_container_width=True, type="primary"):
        with st.spinner("Collecting data... This may take a few minutes"):
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Initialize orchestrator
                orchestrator = DataOrchestrator()
                
                status_text.text("📊 Collecting stock prices...")
                progress_bar.progress(25)
                
                # Collect data
                data = orchestrator.collect_all_data()
                
                progress_bar.progress(100)
                status_text.text("✅ Data collection complete!")
                
                st.success("✅ Data collected successfully!")
                
                # Show summary
                st.subheader("📋 Collection Summary")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if data['price_data'] is not None:
                        st.metric("Price Records", len(data['price_data']))
                
                with col2:
                    if data['fundamental_data'] is not None:
                        st.metric("Companies Analyzed", len(data['fundamental_data']))
                
                with col3:
                    if data['news_data'] is not None:
                        st.metric("News Articles", len(data['news_data']))
                
                # Show sample data
                if data['price_data'] is not None:
                    st.subheader("📈 Sample Price Data")
                    st.dataframe(data['price_data'].head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error during data collection: {str(e)}")
                progress_bar.progress(0)
    
    # Show existing data
    st.subheader("📁 Existing Data Files")
    
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
    
    if os.path.exists(data_dir):
        files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        if files:
            files_df = pd.DataFrame({
                'Filename': files,
                'Modified': [datetime.fromtimestamp(os.path.getmtime(os.path.join(data_dir, f))).strftime('%Y-%m-%d %H:%M:%S') for f in files]
            })
            st.dataframe(files_df, use_container_width=True, hide_index=True)
        else:
            st.info("No data files found. Click 'Start Data Collection' to begin.")
    else:
        st.info("Data directory not found. Click 'Start Data Collection' to create it.")

import pandas as pd