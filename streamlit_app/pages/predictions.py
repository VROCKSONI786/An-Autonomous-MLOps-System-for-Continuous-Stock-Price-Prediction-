import streamlit as st
import sys
import os
import yaml
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.lstm_model import LSTMStockModel
from src.preprocessing.data_preprocessor import DataPreprocessor
import joblib

# def show():
#     st.header("📈 Stock Price Predictions")
    
#     # Load config
#     config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.yaml")
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)
    
#     symbols = config['data_collection']['stock_symbols']
    
#     # Check available models
#     models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "saved_models")
    
#     if not os.path.exists(models_dir):
#         st.warning("⚠️ No models found. Please train models first.")
#         return
    
#     model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
#     available_symbols = [f.replace('_lstm_model.keras', '') for f in model_files]
    
#     if not available_symbols:
#         st.warning("⚠️ No trained models found. Please train models first.")
#         return
    
#     # Select stock
#     selected_symbol = st.selectbox("Select Stock", available_symbols)
    
#     # Prediction settings
#     col1, col2 = st.columns(2)
    
#     with col1:
#         forecast_days = st.slider("Forecast Days", min_value=1, max_value=30, value=7)
    
#     with col2:
#         show_confidence = st.checkbox("Show Confidence Interval", value=True)
def show():
    st.header("📈 Stock Price Predictions")
    
    # 1. Load config
    # __file__ is in src/deployment/pages/, so we go up 3 levels to reach project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(project_root, "config", "config.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    symbols = config['data_collection']['stock_symbols']
    
    # 2. FIXED PATH: Match the structure shown in your image
    # The image shows: src -> model -> models -> saved_models
    models_dir = os.path.join(project_root, "src", "models", "saved_models")
    
    if not os.path.exists(models_dir):
        st.warning(f"⚠️ Models directory not found at {models_dir}")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
    available_symbols = [f.replace('_lstm_model.keras', '') for f in model_files]
    
    if not available_symbols:
        st.warning("⚠️ No trained models found in the directory. Please train models first.")
        return
    
    # 3. Select stock
    selected_symbol = st.selectbox("Select Stock", available_symbols)
    
    # Prediction settings
    col1, col2 = st.columns(2)
    with col1:
        forecast_days = st.slider("Forecast Days", min_value=1, max_value=30, value=7)
    with col2:
        show_confidence = st.checkbox("Show Confidence Interval", value=True)

    # Make predictions
    if st.button("🔮 Generate Predictions", use_container_width=True, type="primary"):
        with st.spinner(f"Generating predictions for {selected_symbol}..."):
            try:
                # Load model
                model_path = os.path.join(models_dir, f"{selected_symbol}_lstm_model.keras")
                lstm_model = LSTMStockModel(sequence_length=60, n_features=21)
                lstm_model.load_model(model_path)
                
                # Prepare data
                X_train, X_val, X_test, y_train, y_val, y_test = lstm_model.prepare_data(
                    symbol=selected_symbol,
                    test_size=0.2,
                    validation_split=0.1
                )
                
                # Make predictions on test set
                predictions = lstm_model.predict(X_test)
                
                # Load scaler
                scaler_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "scaler.pkl")
                scaler = joblib.load(scaler_path)
                
                # Inverse transform predictions
                # Create dummy array with all features
                dummy_features = np.zeros((len(predictions), 21))
                dummy_features[:, 3] = predictions.flatten()  # Close price is at index 3
                pred_original = scaler.inverse_transform(dummy_features)[:, 3]
                
                # Inverse transform actual values
                dummy_actual = np.zeros((len(y_test), 21))
                dummy_actual[:, 3] = y_test
                actual_original = scaler.inverse_transform(dummy_actual)[:, 3]
                
                # Plot predictions vs actual
                st.subheader("📊 Predictions vs Actual Prices")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    y=actual_original,
                    mode='lines',
                    name='Actual',
                    line=dict(color='blue', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    y=pred_original,
                    mode='lines',
                    name='Predicted',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f'{selected_symbol} - Price Predictions',
                    xaxis_title='Time Steps',
                    yaxis_title='Price (₹)',
                    hovermode='x unified',
                    height=500
                )
                
                st.plotly_chart(fig, width='stretch')
                
                # Show metrics
                st.subheader("📈 Prediction Metrics")
                
                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                
                mae = mean_absolute_error(actual_original, pred_original)
                rmse = np.sqrt(mean_squared_error(actual_original, pred_original))
                r2 = r2_score(actual_original, pred_original)
                mape = np.mean(np.abs((actual_original - pred_original) / actual_original)) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("MAE", f"₹{mae:.2f}")
                
                with col2:
                    st.metric("RMSE", f"₹{rmse:.2f}")
                
                with col3:
                    st.metric("R² Score", f"{r2:.4f}")
                
                with col4:
                    st.metric("MAPE", f"{mape:.2f}%")
                
                # Show prediction table
                st.subheader("📋 Recent Predictions")
                
                pred_df = pd.DataFrame({
                    'Time Step': range(len(actual_original[-20:])),
                    'Actual Price (₹)': actual_original[-20:],
                    'Predicted Price (₹)': pred_original[-20:],
                    'Error (₹)': actual_original[-20:] - pred_original[-20:]
                })
                
                st.dataframe(pred_df, width='stretch', hide_index=True)
                
                # Recommendation
                st.subheader("💡 Recommendation")
                
                last_actual = actual_original[-1]
                last_pred = pred_original[-1]
                trend = "📈 Bullish" if last_pred > last_actual else "📉 Bearish"
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Last Actual Price", f"₹{last_actual:.2f}")
                    st.metric("Last Predicted Price", f"₹{last_pred:.2f}")
                
                with col2:
                    st.metric("Price Change", f"₹{last_pred - last_actual:.2f}", f"{((last_pred - last_actual) / last_actual * 100):.2f}%")
                    st.metric("Trend", trend)
                
            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")
                import traceback
                st.code(traceback.format_exc())