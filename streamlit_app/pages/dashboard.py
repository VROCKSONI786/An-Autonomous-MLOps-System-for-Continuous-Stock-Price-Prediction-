import streamlit as st
import pandas as pd
import os
import sys
import yaml
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def show():
    st.header("🏠 Dashboard Overview")
    
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    symbols = config['data_collection']['stock_symbols']
    
    # System status
    st.subheader("📊 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tracked Stocks", len(symbols), "5 stocks")
    
    with col2:
        # Check if data exists
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
        data_files = [f for f in os.listdir(data_dir) if f.startswith('stock_prices_')] if os.path.exists(data_dir) else []
        st.metric("Data Files", len(data_files), "Ready" if data_files else "Not Ready")
    
    with col3:
        # Check if models exist
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "saved_models")
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')] if os.path.exists(models_dir) else []
        st.metric("Trained Models", len(model_files), f"{len(model_files)}/{len(symbols)}")
    
    with col4:
        st.metric("System Status", "🟢 Active", "Online")
    
    # Tracked stocks
    st.subheader("📈 Tracked Stocks")
    
    cols = st.columns(5)
    for idx, symbol in enumerate(symbols):
        with cols[idx]:
            company = symbol.replace('.NS', '')
            st.markdown(f"""
            <div class="metric-card">
                <h3>{company}</h3>
                <p style="font-size: 0.9rem; color: #666;">Indian Market</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Recent activity
    st.subheader("📋 Recent Activity")
    
    activities = []
    
    # Check data collection
    if data_files:
        latest_data = sorted(data_files)[-1]
        timestamp = latest_data.replace('stock_prices_', '').replace('.csv', '')
        activities.append({
            'Time': timestamp,
            'Activity': 'Data Collection',
            'Status': '✅ Completed',
            'Details': f'{len(symbols)} stocks updated'
        })
    
    # Check models
    if model_files:
        activities.append({
            'Time': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'Activity': 'Model Training',
            'Status': '✅ Completed',
            'Details': f'{len(model_files)} models trained'
        })
    
    if activities:
        df_activities = pd.DataFrame(activities)
        st.dataframe(df_activities, use_container_width=True, hide_index=True)
    else:
        st.info("No recent activities. Start by collecting data!")
    
    # Quick actions
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Collect Latest Data", use_container_width=True):
            st.info("Navigate to 'Data Collection' page to collect data")
    
    with col2:
        if st.button("🤖 Train Models", use_container_width=True):
            st.info("Navigate to 'Model Training' page to train models")
    
    with col3:
        if st.button("📈 View Predictions", use_container_width=True):
            st.info("Navigate to 'Predictions' page to see predictions")