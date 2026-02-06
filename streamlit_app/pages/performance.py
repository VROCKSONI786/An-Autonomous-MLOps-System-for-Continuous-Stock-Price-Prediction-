import streamlit as st
import sys
import os
import yaml
import pandas as pd
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.monitoring.performance_monitor import PerformanceMonitor

def show():
    st.header("📊 Performance Monitoring")
    
    st.write("Monitor model performance and detect degradation")
    
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    symbols = config['data_collection']['stock_symbols']
    
    # Monitor models button
    if st.button("🔍 Check Model Performance", use_container_width=True, type="primary"):
        with st.spinner("Evaluating models..."):
            try:
                monitor = PerformanceMonitor()
                metrics, degraded = monitor.monitor_all_models(symbols)
                
                if metrics:
                    st.success("✅ Performance check complete!")
                    
                    # Show metrics
                    st.subheader("📊 Performance Metrics")
                    
                    metrics_df = pd.DataFrame(metrics)
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                    
                    # Show degraded models
                    if degraded:
                        st.warning(f"⚠️ Models requiring retraining: {', '.join(degraded)}")
                    else:
                        st.success("✅ All models performing well!")
                    
                    # Visualize performance
                    st.subheader("📈 Performance Comparison")
                    
                    import plotly.graph_objects as go
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        x=[m['symbol'] for m in metrics],
                        y=[m['mae'] for m in metrics],
                        name='MAE',
                        marker_color='lightblue'
                    ))
                    
                    fig.update_layout(
                        title='Model MAE Comparison',
                        xaxis_title='Stock',
                        yaxis_title='MAE',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.warning("⚠️ No metrics available. Please train models first.")
                    
            except Exception as e:
                st.error(f"❌ Error monitoring performance: {str(e)}")
    
    # Show performance history
    st.subheader("📜 Performance History")
    
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
    history_file = os.path.join(logs_dir, "performance_history.jsonl")
    
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = [json.loads(line) for line in f]
        
        if history:
            history_df = pd.DataFrame(history[-20:])  # Last 20 records
            st.dataframe(history_df, use_container_width=True, hide_index=True)
        else:
            st.info("No performance history available yet.")
    else:
        st.info("No performance history file found. Run performance check first.")