import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import logging
import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.lstm_model import LSTMStockModel
from preprocessing.data_preprocessor import DataPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor model performance over time"""
    
    def __init__(self):
        self.performance_history = []
        
    def evaluate_model(self, symbol, model_path=None):
        """Evaluate model on recent data"""
        logger.info(f"Evaluating model for {symbol}")
        
        try:
            # Load model
            lstm_model = LSTMStockModel(sequence_length=60, n_features=21)
            
            if model_path is None:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                model_path = os.path.join(
                    project_root, "models", "saved_models", f"{symbol}_lstm_model.keras"
                )
            
            if not os.path.exists(model_path):
                logger.warning(f"Model not found for {symbol}")
                return None
            
            lstm_model.load_model(model_path)
            
            # Prepare test data
            X_train, X_val, X_test, y_train, y_val, y_test = lstm_model.prepare_data(
                symbol=symbol,
                test_size=0.2,
                validation_split=0.1
            )
            
            # Make predictions
            y_pred = lstm_model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate MAPE
            mape = np.mean(np.abs((y_test - y_pred.flatten()) / (y_test + 1e-10))) * 100
            
            metrics = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2_score': float(r2),
                'mape': float(mape),
                'test_samples': len(y_test)
            }
            
            logger.info(f"Performance metrics for {symbol}:")
            logger.info(f"  MSE: {mse:.6f}")
            logger.info(f"  MAE: {mae:.6f}")
            logger.info(f"  RMSE: {rmse:.6f}")
            logger.info(f"  R²: {r2:.6f}")
            logger.info(f"  MAPE: {mape:.2f}%")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {symbol}: {str(e)}")
            return None
    
    def monitor_all_models(self, symbols):
        """Monitor performance of all models"""
        logger.info("\n" + "="*60)
        logger.info("Starting Performance Monitoring")
        logger.info("="*60 + "\n")
        
        all_metrics = []
        
        for symbol in symbols:
            metrics = self.evaluate_model(symbol)
            if metrics:
                all_metrics.append(metrics)
        
        # Save metrics
        self.save_metrics(all_metrics)
        
        # Check for degradation
        degraded_models = self.check_degradation(all_metrics)
        
        if degraded_models:
            logger.warning(f"\n⚠️ Models requiring retraining: {degraded_models}")
        else:
            logger.info("\n✓ All models performing well")
        
        logger.info("\n" + "="*60)
        logger.info("Performance Monitoring Complete")
        logger.info("="*60)
        
        return all_metrics, degraded_models
    
    def check_degradation(self, metrics, mae_threshold=0.1, r2_threshold=0.7):
        """Check if any models have degraded performance"""
        degraded = []
        
        for metric in metrics:
            if metric['mae'] > mae_threshold or metric['r2_score'] < r2_threshold:
                degraded.append(metric['symbol'])
                logger.warning(
                    f"{metric['symbol']} performance degraded: "
                    f"MAE={metric['mae']:.4f}, R²={metric['r2_score']:.4f}"
                )
        
        return degraded
    
    def save_metrics(self, metrics):
        """Save performance metrics to file"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        
        # Save individual metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(logs_dir, f"performance_metrics_{timestamp}.json")
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"\nMetrics saved to {metrics_file}")
        
        # Append to history
        history_file = os.path.join(logs_dir, "performance_history.jsonl")
        with open(history_file, 'a') as f:
            for metric in metrics:
                f.write(json.dumps(metric) + '\n')
        
        logger.info(f"Metrics appended to {history_file}")
    
    def log_to_mlflow(self, metrics):
        """Log metrics to MLflow"""
        try:
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("model_monitoring")
            
            for metric in metrics:
                with mlflow.start_run(run_name=f"Monitor_{metric['symbol']}"):
                    mlflow.log_metrics({
                        'mse': metric['mse'],
                        'mae': metric['mae'],
                        'rmse': metric['rmse'],
                        'r2_score': metric['r2_score'],
                        'mape': metric['mape']
                    })
                    mlflow.log_param('symbol', metric['symbol'])
                    mlflow.log_param('timestamp', metric['timestamp'])
            
            logger.info("Metrics logged to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging to MLflow: {str(e)}")

if __name__ == "__main__":
    import yaml
    
    # Load config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    config_path = os.path.join(project_root, "config", "config.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    symbols = config['data_collection']['stock_symbols']
    
    # Run monitoring
    monitor = PerformanceMonitor()
    metrics, degraded = monitor.monitor_all_models(symbols)
    
    # Log to MLflow
    monitor.log_to_mlflow(metrics)
    
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    for metric in metrics:
        print(f"{metric['symbol']}: MAE={metric['mae']:.6f}, R²={metric['r2_score']:.4f}")