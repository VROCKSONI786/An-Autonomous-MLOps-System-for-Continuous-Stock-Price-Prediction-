import streamlit as st
import sys
import os
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.model_trainer import ModelTrainer
from src.preprocessing.data_preprocessor import DataPreprocessor

def show():
    st.header("🤖 Model Training")
    
    st.write("Train LSTM models for stock price prediction")
    
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    symbols = config['data_collection']['stock_symbols']
    
    # Training settings
    st.subheader("⚙️ Training Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.number_input("Epochs", min_value=10, max_value=200, value=50, step=10)
        batch_size = st.number_input("Batch Size", min_value=16, max_value=128, value=32, step=16)
    
    with col2:
        lstm_units = st.number_input("LSTM Units", min_value=32, max_value=256, value=128, step=32)
        dropout = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
    
    with col3:
        sequence_length = st.number_input("Sequence Length", min_value=30, max_value=120, value=60, step=10)
    
    # Select stocks to train
    st.subheader("📊 Select Stocks to Train")
    selected_symbols = st.multiselect("Stocks", symbols, default=symbols)
    
    # Preprocessing step
    st.subheader("🔧 Data Preprocessing")
    
    if st.button("📊 Preprocess Data", use_container_width=True):
        with st.spinner("Preprocessing data..."):
            try:
                preprocessor = DataPreprocessor()
                preprocessed_data = preprocessor.preprocess_pipeline()
                
                if preprocessed_data is not None:
                    st.success(f"✅ Data preprocessed successfully! Shape: {preprocessed_data.shape}")
                    st.dataframe(preprocessed_data.head(), use_container_width=True)
                else:
                    st.error("❌ Preprocessing failed!")
                    
            except Exception as e:
                st.error(f"❌ Error during preprocessing: {str(e)}")
    
    # Training step
    st.subheader("🚀 Train Models")
    
    if st.button("🤖 Start Training", use_container_width=True, type="primary"):
        if not selected_symbols:
            st.warning("⚠️ Please select at least one stock to train")
            return
        
        with st.spinner("Training models... This may take several minutes"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Initialize trainer
                trainer = ModelTrainer()
                
                # Update config with user settings
                trainer.model_config['epochs'] = epochs
                trainer.model_config['batch_size'] = batch_size
                trainer.model_config['lstm_units'] = lstm_units
                trainer.model_config['dropout_rate'] = dropout
                trainer.model_config['sequence_length'] = sequence_length
                
                results = {}
                
                for idx, symbol in enumerate(selected_symbols):
                    status_text.text(f"Training {symbol}... ({idx+1}/{len(selected_symbols)})")
                    progress_bar.progress((idx) / len(selected_symbols))
                    
                    try:
                        model, metrics = trainer.train_for_symbol(symbol)
                        results[symbol] = metrics
                    except Exception as e:
                        st.warning(f"⚠️ Failed to train {symbol}: {str(e)}")
                        continue
                
                progress_bar.progress(100)
                status_text.text("✅ Training complete!")
                
                st.success(f"✅ Successfully trained {len(results)}/{len(selected_symbols)} models!")
                
                # Show results
                st.subheader("📊 Training Results")
                
                import pandas as pd
                results_df = pd.DataFrame([
                    {
                        'Symbol': symbol,
                        'Test Loss': f"{metrics[0]:.6f}",
                        'Test MAE': f"{metrics[1]:.6f}",
                        'Test MSE': f"{metrics[2]:.6f}"
                    }
                    for symbol, metrics in results.items()
                ])
                
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                progress_bar.progress(0)
    
    # # Show existing models
    # st.subheader("💾 Saved Models")
    
    # models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "saved_models")
    
    # if os.path.exists(models_dir):
    #     model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
        
    #     if model_files:
    #         import pandas as pd
    #         from datetime import datetime
            
    #         models_df = pd.DataFrame({
    #             'Model': model_files,
    #             'Symbol': [f.replace('_lstm_model.keras', '') for f in model_files],
    #             #'Modified': [datetime.fromtimestamp(os.path.getmtime(os.path.join(models_dir, f))).strftime('%Y-%m-%d %H:%M:%S') for f in model_files]
    #         })
    #         st.dataframe(models_df, use_container_width=True, hide_index=True)
    #     else:
    #         st.info("No trained models found. Click 'Start Training' to begin.")
    # else:
    #     st.info("Models directory not found.")
    # Show existing models
    st.subheader("💾 Saved Models")
    
    # FIXED PATH: Points to src/model/models/saved_models
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    models_dir = os.path.join(project_root, "src", "models", "saved_models")
    
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.keras')]
        
        if model_files:
            import pandas as pd
            
            models_df = pd.DataFrame({
                'Model': model_files,
                'Symbol': [f.replace('_lstm_model.keras', '') for f in model_files],
            })
            # Updated width='stretch' to resolve deprecation warning
            st.dataframe(models_df, width='stretch', hide_index=True)
        else:
            st.info("No trained models found. Click 'Start Training' to begin.")
    else:
        # Debugging: Show where it's looking if it still fails
        st.error(f"Models directory not found at: {models_dir}")