import yaml
import mlflow
import mlflow.keras
import logging
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.lstm_model import LSTMStockModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train and track models using MLflow"""
    
    def __init__(self, config_path=None):
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            config_path = os.path.join(project_root, "config", "config.yaml")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])
        
        self.symbols = self.config['data_collection']['stock_symbols']
        self.model_config = self.config['model']
    
    def train_for_symbol(self, symbol):
        """Train model for a specific stock symbol"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training model for {symbol}")
        logger.info(f"{'='*60}\n")
        
        with mlflow.start_run(run_name=f"LSTM_{symbol}"):
            # Log parameters
            mlflow.log_params({
                'symbol': symbol,
                'sequence_length': self.model_config['sequence_length'],
                'lstm_units': self.model_config['lstm_units'],
                'dropout_rate': self.model_config['dropout_rate'],
                'epochs': self.model_config['epochs'],
                'batch_size': self.model_config['batch_size']
            })
            
            # Initialize model
            lstm_model = LSTMStockModel(
                sequence_length=self.model_config['sequence_length'],
                n_features=21  # Will be updated based on actual features
            )
            
            # Build model
            lstm_model.build_model(
                lstm_units=self.model_config['lstm_units'],
                dropout_rate=self.model_config['dropout_rate']
            )
            
            # Prepare data
            X_train, X_val, X_test, y_train, y_val, y_test = lstm_model.prepare_data(
                symbol=symbol,
                test_size=0.2,
                validation_split=0.1
            )
            
            # Train model
            history = lstm_model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=self.model_config['epochs'],
                batch_size=self.model_config['batch_size']
            )
            
            # Evaluate
            results = lstm_model.evaluate(X_test, y_test)
            
            # Log metrics
            mlflow.log_metrics({
                'test_loss': results[0],
                'test_mae': results[1],
                'test_mse': results[2],
                'train_loss': history.history['loss'][-1],
                'val_loss': history.history['val_loss'][-1]
            })
            
            # Plot training history
            self.plot_training_history(history, symbol)
            
            # Make predictions
            predictions = lstm_model.predict(X_test)
            self.plot_predictions(y_test, predictions, symbol)
            
            # Save model
            model_path = f"models/saved_models/{symbol}_lstm_model.keras"
            lstm_model.save_model(model_path)
            
            # Log model to MLflow
            mlflow.keras.log_model(lstm_model.model, "model")
            
            logger.info(f"\n✓ Training complete for {symbol}")
            logger.info(f"Test Loss: {results[0]:.6f}, Test MAE: {results[1]:.6f}\n")
            
            return lstm_model, results
    
    def plot_training_history(self, history, symbol):
        """Plot and save training history"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        plots_dir = os.path.join(project_root, "logs", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history.history['loss'], label='Train Loss')
        axes[0].plot(history.history['val_loss'], label='Val Loss')
        axes[0].set_title(f'{symbol} - Training Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE plot
        axes[1].plot(history.history['mae'], label='Train MAE')
        axes[1].plot(history.history['val_mae'], label='Val MAE')
        axes[1].set_title(f'{symbol} - Training MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plot_path = os.path.join(plots_dir, f'{symbol}_training_history.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        # Log to MLflow
        mlflow.log_artifact(plot_path)
    
    def plot_predictions(self, y_true, y_pred, symbol):
        """Plot and save predictions vs actual"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        plots_dir = os.path.join(project_root, "logs", "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        plt.figure(figsize=(15, 6))
        plt.plot(y_true, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.title(f'{symbol} - Predictions vs Actual')
        plt.xlabel('Time Steps')
        plt.ylabel('Scaled Price')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(plots_dir, f'{symbol}_predictions.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        # Log to MLflow
        mlflow.log_artifact(plot_path)
    
    def train_all_symbols(self):
        """Train models for all configured symbols"""
        logger.info("\n" + "="*60)
        logger.info("Starting training for all symbols")
        logger.info("="*60 + "\n")
        
        results = {}
        
        for symbol in self.symbols:
            try:
                model, metrics = self.train_for_symbol(symbol)
                results[symbol] = {
                    'model': model,
                    'test_loss': metrics[0],
                    'test_mae': metrics[1],
                    'test_mse': metrics[2]
                }
            except Exception as e:
                logger.error(f"Error training {symbol}: {str(e)}")
                continue
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("Training Summary")
        logger.info("="*60)
        
        for symbol, result in results.items():
            logger.info(f"{symbol}: Test MAE = {result['test_mae']:.6f}")
        
        logger.info("\n✓ All training complete!")
        
        return results

if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.train_all_symbols()